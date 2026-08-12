#!/usr/bin/env python3
"""A2: gold-conditioned agentic teacher trace generator (Path A).

For a set of *target* train examples (typically the all-wrong band from an A1
pass@K run), this script:

1. Loads the standard bare-tool rows (system+user prompt, NO gold).
2. Injects the gold SQL as a privileged internal reference into a TEACHER copy
   of the prompt (teacher_hint.inject_privileged_hint). The teacher is told to
   solve independently and never verbalize the reference.
3. Runs the teacher through the SAME agentic tool loop used at inference
   (draft -> sqlite_query verify -> final answer), capturing a STRUCTURED
   multi-turn transcript (assistant / tool turns) so downstream SFT can mask
   tool-response tokens.
4. Verifies the final SQL against gold with BIRD raw-row set equality
   (bird_result_match) and readonly safety.
5. Runs text-leakage detection (drop leaked traces) and copy/transcription
   detection (flag, keep, and report copy_rate).

Output: ``teacher_traces.jsonl`` (one row per target, verified or not, with the
structured transcript + verification + leakage/copy flags) and
``teacher_summary.json``. A3 turns the KEPT verified traces into hint-free
student RFT records.

This is a GPU/vLLM script; run it on a Ray worker node (see ray_teacher_job.sh).
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_DIR), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from nl2sql_gspo.inference_tool_executor import (  # noqa: E402
    configure_tool_env,
    execute_tool_calls,
    extract_tool_calls,
)
from nl2sql_gspo.sql_utils import (  # noqa: E402
    bird_get_gold_rows,
    bird_execute_sql,
    bird_result_match,
    extract_sql,
    is_safe_readonly_sql,
)

from run_inference_bird import (  # noqa: E402
    _async_vllm_generate_text,
    build_assistant_tool_message,
    gemma_tool_loop_stop_token_ids,
    get_generation_messages,
    keep_first_tool_call_only,
    prepare_rows_for_generation,
    render_prompt,
    resolve_vllm_tokenizer_source,
)

from teacher.teacher_hint import (  # noqa: E402
    inject_privileged_hint,
    detect_text_leakage,
    detect_copy,
    summarize_copy_rate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate gold-conditioned teacher traces for RFT (Path A2)."
    )
    p.add_argument(
        "--model_name_or_path",
        required=True,
        help="Teacher model / checkpoint (e.g. RUN A ckpt-20).",
    )
    p.add_argument(
        "--input_file",
        required=True,
        help="Bare-tool train jsonl (system+user prompt, gold_sql field).",
    )
    p.add_argument(
        "--database_dir",
        required=True,
        help="Train databases dir (get_database_path resolves <db_id>).",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--target_idx_file",
        type=str,
        default=None,
        help=(
            "Optional JSON list of source_idx to target "
            "(e.g. the all-wrong band). Default: all rows."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Cap number of target rows (after target filter).",
    )
    p.add_argument(
        "--hint_strategy",
        type=str,
        default="full_sql",
        choices=["full_sql", "none"],
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Teacher trajectories per target (keep best verified).",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_prompt_length", type=int, default=34000)
    p.add_argument("--max_new_tokens", type=int, default=8000)
    p.add_argument("--max_tool_rounds", type=int, default=8)
    p.add_argument("--eval_timeout", type=float, default=60.0)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=8)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--vllm_max_model_len", type=int, default=None)
    p.add_argument("--vllm_async_concurrency", type=int, default=16)
    p.add_argument("--near_copy_threshold", type=float, default=0.95)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_tool_rows(input_file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as handle:
        for source_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("source_idx", source_idx)
            rows.append(row)
    return rows


def select_targets(
    rows: List[Dict[str, Any]],
    target_idx_file: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    if target_idx_file:
        target_ids = set(int(x) for x in json.load(open(target_idx_file)))
        rows = [
            r
            for r in rows
            if int(r.get("source_idx", -1)) in target_ids
        ]

    rows = [
        r
        for r in rows
        if extract_sql(r.get("gold_sql", "")).strip()
    ]

    if limit is not None and limit >= 0:
        rows = rows[:limit]

    return rows


def build_teacher_row(
    row: Dict[str, Any],
    hint_strategy: str,
) -> Dict[str, Any]:
    """Return a copy of ``row`` prepared for generation.

    ``hint_strategy == "full_sql"`` injects the privileged gold reference
    (teacher mode). ``hint_strategy == "none"`` keeps the ORIGINAL hint-free
    prompt (self-trace mode) so the student's own verified trajectories can be
    harvested for the sometimes / all-correct bands; there is nothing to leak in
    this mode.
    """

    base_prompt = get_generation_messages(row)
    gold_sql = extract_sql(row.get("gold_sql", ""))

    if hint_strategy == "none":
        teacher_prompt = [dict(m) for m in base_prompt]
    else:
        teacher_prompt = inject_privileged_hint(
            base_prompt,
            gold_sql,
            strategy=hint_strategy,
        )

    teacher_row = dict(row)
    teacher_row["prompt"] = teacher_prompt
    teacher_row["messages"] = teacher_prompt
    return teacher_row


async def teacher_tool_loop(
    engine,
    sampling_params_cls,
    tokenizer,
    teacher_row: Dict[str, Any],
    *,
    max_new_tokens: int,
    max_model_len: int,
    max_tool_rounds: int,
    eval_timeout: float,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    """Agentic teacher loop that captures a STRUCTURED transcript.

    Returns dict with:
      - transcript: list of {role: assistant|tool, content} (assistant turns
        hold raw generated text incl. the call...{} line; tool turns hold the
        rendered <|tool_response>...)
      - prediction_text: flattened assistant+tool text (for SQL extraction/scan)
      - assistant_texts: list of assistant-turn strings (for leakage scan)
      - first_tool_sql: SQL argument of the first sqlite_query call (copy check)
      - had_tool_calls, tool_rounds, stop_reason
    """

    messages = [dict(m) for m in get_generation_messages(teacher_row)]
    tools = teacher_row.get("tools")
    transcript: List[Dict[str, str]] = []
    assistant_texts: List[str] = []
    generated_parts: List[str] = []
    first_tool_sql: Optional[str] = None
    completion_tokens = 0
    tool_rounds = 0
    stop_reason = "finished"
    request_prefix = f"teacher-idx{teacher_row.get('source_idx', -1)}"

    for round_index in range(max_tool_rounds + 1):
        remaining = max_new_tokens - completion_tokens
        if remaining <= 0:
            stop_reason = "max_new_tokens"
            break

        prompt_text = render_prompt(tokenizer, messages, tools)
        cur_prompt_tokens = len(
            tokenizer(prompt_text, truncation=False)["input_ids"]
        )
        available = max_model_len - cur_prompt_tokens
        if available <= 0:
            stop_reason = "context_length_exceeded"
            break

        req_max = min(remaining, available)

        # NOTE:
        # The screenshots do not show the lines immediately after req_max
        # and before the vLLM generation call.

        generated_text, round_tokens = await _async_vllm_generate_text(
            engine=engine,
            sampling_params_cls=sampling_params_cls,
            prompt_text=prompt_text,
            max_tokens=req_max,
            temperature=temperature,
            top_p=top_p,
            request_prefix=request_prefix,
            stop_token_ids=gemma_tool_loop_stop_token_ids(tokenizer),
        )

        tool_calls = extract_tool_calls(generated_text)

        if not tool_calls:
            completion_tokens += round_tokens
            if generated_text:
                generated_parts.append(generated_text)
                assistant_texts.append(generated_text)
                transcript.append(
                    {
                        "role": "assistant",
                        "content": generated_text,
                    }
                )
            stop_reason = (
                "max_new_tokens"
                if round_tokens >= remaining
                else "finished"
            )
            break

        if round_index >= max_tool_rounds:
            stop_reason = "max_tool_rounds"
            break

        generated_text, tool_calls = keep_first_tool_call_only(
            generated_text,
            tool_calls,
        )

        completion_tokens += len(
            tokenizer(
                generated_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
        )

        if generated_text:
            generated_parts.append(generated_text)
            assistant_texts.append(generated_text)
            transcript.append(
                {
                    "role": "assistant",
                    "content": generated_text,
                }
            )

        # record the first sqlite_query SQL argument for copy detection
        if first_tool_sql is None:
            for call in tool_calls:
                fn = call.get("function") or {}
                if fn.get("name") == "sqlite_query":
                    args = fn.get("arguments") or {}
                    first_tool_sql = (
                        str(args.get("sql", ""))
                        if isinstance(args, dict)
                        else ""
                    )
                    break

        tool_responses = await asyncio.to_thread(
            execute_tool_calls,
            tool_calls,
            eval_timeout,
        )

        for response in tool_responses:
            rendered = response.get("rendered", "")
            generated_parts.append(rendered)
            transcript.append(
                {
                    "role": "tool",
                    "content": rendered,
                }
            )

        messages.append(
            build_assistant_tool_message(
                generated_text,
                tool_calls,
                tool_responses,
            )
        )
        tool_rounds += 1

    prediction_text = "\n".join(
        part for part in generated_parts if part
    ).strip()

    return {
        "transcript": transcript,
        "prediction_text": prediction_text,
        "assistant_texts": assistant_texts,
        "first_tool_sql": first_tool_sql,
        "had_tool_calls": tool_rounds > 0,
        "tool_rounds": tool_rounds,
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from nl2sql_gspo.model_utils import load_tokenizer

    configure_tool_env(args.database_dir)

    tokenizer_source = resolve_vllm_tokenizer_source(
        args.model_name_or_path
    )
    tokenizer = load_tokenizer(tokenizer_source)

    all_rows = load_tool_rows(args.input_file)
    targets = select_targets(
        all_rows,
        args.target_idx_file,
        args.limit,
    )
    print(
        f"[teacher] loaded {len(all_rows)} rows | "
        f"targeting {len(targets)}"
    )

    teacher_rows = [
        build_teacher_row(r, args.hint_strategy)
        for r in targets
    ]

    teacher_rows, skipped = prepare_rows_for_generation(
        teacher_rows,
        tokenizer,
        args.max_prompt_length,
    )

    if skipped:
        (
            Path(args.output_dir) / "skipped_prompts.jsonl"
        ).write_text(
            "\n".join(json.dumps(s) for s in skipped),
            encoding="utf-8",
        )
                
        (
            Path(args.output_dir) / "skipped_prompts.jsonl"
        ).write_text(
            "\n".join(json.dumps(s) for s in skipped),
            encoding="utf-8",
        )

    max_model_len = args.vllm_max_model_len or (
        args.max_prompt_length + args.max_new_tokens
    )
    print(
        f"[teacher] async vLLM tp={args.vllm_tensor_parallel_size} "
        f"concurrency={args.vllm_async_concurrency} max_model_len={max_model_len}"
    )

    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        tokenizer=tokenizer_source,
        trust_remote_code=True,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    semaphore = asyncio.Semaphore(max(1, args.vllm_async_concurrency))

    gold_cache: Dict[int, Tuple[bool, Any, str]] = {}

    def gold_rows_for(row: Dict[str, Any]) -> Tuple[bool, Any, str]:
        idx = int(row.get("source_idx", -1))
        if idx not in gold_cache:
            gold_cache[idx] = bird_get_gold_rows(
                extract_sql(row.get("gold_sql", "")),
                row.get("db_id", ""),
                args.database_dir,
                timeout_s=args.eval_timeout,
            )
        return gold_cache[idx]

    total = len(teacher_rows) * max(1, args.num_samples)
    done = 0
    results: List[Dict[str, Any]] = []

    async def one(
        row: Dict[str, Any],
        sample_id: int,
    ) -> Dict[str, Any]:
        nonlocal done
        idx = int(row.get("source_idx", -1))
        gold_sql = extract_sql(row.get("gold_sql", ""))

        async with semaphore:
            err = ""
            try:
                loop_out = await teacher_tool_loop(
                    engine=engine,
                    sampling_params_cls=SamplingParams,
                    tokenizer=tokenizer,
                    teacher_row=row,
                    max_new_tokens=args.max_new_tokens,
                    max_model_len=max_model_len,
                    max_tool_rounds=args.max_tool_rounds,
                    eval_timeout=args.eval_timeout,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                loop_out = {
                    "transcript": [],
                    "prediction_text": "",
                    "assistant_texts": [],
                    "first_tool_sql": None,
                    "had_tool_calls": False,
                    "tool_rounds": 0,
                    "completion_tokens": 0,
                    "stop_reason": "generation_error",
                }

            final_sql = extract_sql(loop_out["prediction_text"])
            readonly = bool(final_sql) and is_safe_readonly_sql(final_sql)

            # verify against gold (raw-row set equality, BIRD semantics)
            verified = False
            exec_ok = False
            match_err = ""
            if readonly:
                try:
                    gold_exec, gold_rowset, gold_err = await asyncio.to_thread(
                        gold_rows_for,
                        row,
                    )
                    if gold_exec:
                        pred_exec, pred_rows, pred_err = await asyncio.to_thread(
                            bird_execute_sql,
                            final_sql,
                            row.get("db_id", ""),
                            args.database_dir,
                            args.eval_timeout,
                        )
                        exec_ok = bool(pred_exec)
                        if pred_exec:
                            verified = bool(
                                bird_result_match(pred_rows, gold_rowset)
                            )
                        else:
                            match_err = f"pred_exec_failed: {pred_err[:120]}"
                    else:
                        match_err = f"gold_exec_failed: {gold_err[:120]}"
                except Exception as exc:
                    match_err = f"verify_error: {type(exc).__name__}: {exc}"

            leak = (
                detect_text_leakage(
                    loop_out["assistant_texts"],
                    gold_sql,
                )
                if args.hint_strategy != "none"
                else []
            )

            copy = detect_copy(
                gold_sql,
                loop_out.get("first_tool_sql"),
                final_sql,
                had_tool_calls=loop_out["had_tool_calls"],
                near_copy_threshold=args.near_copy_threshold,
            )

            kept = bool(verified and not leak)

            done += 1
            if done == 1 or done == total or done % 25 == 0:
                print(
                    f"[teacher] {done}/{total} idx={idx} sample={sample_id} "
                    f"verified={verified} kept={kept} leak={leak} "
                    f"copy_first={copy['copy_first_call']} "
                    f"stop={loop_out['stop_reason']}"
                )

            return {
                "idx": idx,
                "sample_id": sample_id,
                "db_id": row.get("db_id", ""),
                "question": row.get("question", ""),
                "gold_sql": gold_sql,
                "final_sql": final_sql,
                "verified": verified,
                "exec_ok": exec_ok,
                "readonly": readonly,
                "kept": kept,
                "leak_reasons": leak,
                "copy_flags": copy,
                "tool_rounds": loop_out["tool_rounds"],
                "stop_reason": loop_out["stop_reason"],
                "match_error": match_err,
                "generation_error": err,
                "transcript": loop_out["transcript"],
                "prediction_text": loop_out["prediction_text"],
            }

    try:
        tasks = [
            one(row, s)
            for row in teacher_rows
            for s in range(max(1, args.num_samples))
        ]
        results = await asyncio.gather(*tasks)
    finally:
        engine.shutdown()

    return {
        "results": results,
        "n_targets": len(teacher_rows),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} exists and is non-empty; use --overwrite"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bundle = asyncio.run(run(args))
    results = bundle["results"]

    # write per-sample traces
    traces_path = out_dir / "teacher_traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    # per-target: was any sample kept?
    kept_by_idx: Dict[int, bool] = {}
    for r in results:
        kept_by_idx[r["idx"]] = (
            kept_by_idx.get(r["idx"], False) or bool(r["kept"])
        )

    kept_flags = [
        r["copy_flags"]
        for r in results
        if r["kept"]
    ]

    summary = {
        "model": args.model_name_or_path,
        "input_file": args.input_file,
        "hint_strategy": args.hint_strategy,
        "num_samples": args.num_samples,
        "n_targets": bundle["n_targets"],
        "n_samples_total": len(results),
        "n_verified_samples": sum(
            1 for r in results if r["verified"]
        ),
        "n_kept_samples": sum(
            1 for r in results if r["kept"]
        ),
        "n_leaked_samples": sum(
            1 for r in results if r["leak_reasons"]
        ),
        "targets_with_kept_trace": sum(
            1 for v in kept_by_idx.values() if v
        ),
        "target_coverage_rate": (
            sum(1 for v in kept_by_idx.values() if v)
            / max(1, bundle["n_targets"])
        ),
        "copy_rate_over_kept": summarize_copy_rate(kept_flags),
        "elapsed_s": round(time.time() - t0, 1),
    }

    (out_dir / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== teacher summary ===")
    print(json.dumps(summary, indent=2))
    print("traces ->", traces_path)


if __name__ == "__main__":
    main()