#!/usr/bin/env python3
"""Stage A2 / A2b / A3a — generate verified agentic traces.

One generator serves all three runs; only the flags differ.

    A2   greedy teacher   --hint_strategy full_sql --num_samples 1 --temperature 0.0
    A2b  sampled teacher  --hint_strategy full_sql --num_samples 8 --temperature 0.7
    A3a  student self     --hint_strategy none     --num_samples 2 --temperature 0.7

For each target example:

1. Load the hint-free system/user prompt.
2. For the teacher runs, build a *separate* privileged prompt containing the
   gold SQL inside <internal_reference_do_not_reveal>. The hint is used only for
   generation; the saved transcript always carries the hint-free prompt.
3. Run the agentic tool loop, capturing a structured multi-turn transcript
   (assistant turns and tool responses as distinct messages).
4. Verify the final SQL against gold with BIRD row-set equality.
5. Screen for hard/soft leaks and copy behaviour.

A sample is kept iff it is **verified** and has **no hard leak**. Soft leaks and
copy flags are recorded but do not reject.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_inference_bird import (  # noqa: E402
    _async_vllm_generate_text,
    build_assistant_tool_message,
    get_generation_messages,
    keep_first_tool_call_only,
    prepare_rows_for_generation,
    render_prompt,
    resolve_vllm_tokenizer_source,
)
from nl2sql_gspo.inference_tool_executor import (  # noqa: E402
    configure_tool_env,
    execute_tool_calls,
    extract_tool_calls,
)
from nl2sql_gspo.schema_utils import extract_columns_from_sql, extract_tables_from_sql  # noqa: E402
from nl2sql_gspo.sql_utils import (  # noqa: E402
    bird_execute_sql,
    bird_get_gold_rows,
    bird_result_match,
    extract_sql,
    is_safe_readonly_sql,
)
from teacher.teacher_hint import detect_copy, detect_leaks, inject_privileged_hint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default="google/gemma-4-31B-it")
    parser.add_argument("--input_file", default="outputs/train-6601-schema-bare-tool.jsonl")
    parser.add_argument("--database_dir", default="databases/train_databases")
    parser.add_argument("--target_idx_file", default=None, help="One source_idx per line. Omit to run all rows.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hint_strategy", choices=["full_sql", "none"], default="full_sql")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=8)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=43000)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard index. Use with --num_shards to split targets across processes.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Total shards. Targets are split by source_idx %% num_shards, and the "
            "output dir gets a shard-XXXXX-of-YYYYY suffix. Launch one process per "
            "shard with its own CUDA_VISIBLE_DEVICES, then merge with --merge_shard_dirs."
        ),
    )
    parser.add_argument(
        "--merge_shard_dirs",
        nargs="*",
        default=None,
        help="Merge finished shard dirs instead of generating. Requires --merge_output_dir.",
    )
    parser.add_argument("--merge_output_dir", default=None)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards")
    return args


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class GoldCache:
    """Gold rows are re-used across the samples of one example."""

    def __init__(self, database_dir: str, timeout_s: float) -> None:
        self.database_dir = database_dir
        self.timeout_s = timeout_s
        self._cache: Dict[Tuple[str, str], Tuple[bool, Any, str]] = {}

    def get(self, gold_sql: str, db_id: str) -> Tuple[bool, Any, str]:
        key = (db_id, gold_sql)
        if key not in self._cache:
            try:
                self._cache[key] = bird_get_gold_rows(
                    gold_sql, db_id, self.database_dir, timeout_s=self.timeout_s
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._cache[key] = (False, None, str(exc))
        return self._cache[key]


async def run_trace(
    engine,
    sampling_params_cls,
    tokenizer,
    row: Dict[str, Any],
    sample_id: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """One agentic rollout, returning a structured transcript.

    The generation prompt may carry the privileged hint; ``transcript`` always
    starts from the hint-free prompt so the record is safe to train on.
    """
    hint_free_messages = [dict(m) for m in get_generation_messages(row)]
    gold_sql = row.get("gold_sql", "") or ""
    tools = row.get("tools")

    working_messages = inject_privileged_hint(hint_free_messages, gold_sql, args.hint_strategy)
    transcript: List[Dict[str, Any]] = []

    completion_tokens = 0
    tool_rounds = 0
    tool_call_count = 0
    first_tool_sql = ""
    stop_reason = "finished"
    assistant_texts: List[str] = []
    final_text = ""

    for round_index in range(args.max_tool_rounds + 1):
        prompt_text = render_prompt(tokenizer, working_messages, tools)
        prompt_tokens = len(tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"])
        remaining = min(args.max_new_tokens - completion_tokens, args.vllm_max_model_len - prompt_tokens)
        if remaining <= 0:
            stop_reason = "context_length_exceeded"
            break

        generated_text, round_tokens = await _async_vllm_generate_text(
            engine=engine,
            sampling_params_cls=sampling_params_cls,
            prompt_text=prompt_text,
            max_tokens=remaining,
            temperature=args.temperature,
            top_p=args.top_p,
            request_prefix=f"idx{row.get('source_idx', -1)}-s{sample_id}",
        )
        completion_tokens += round_tokens
        tool_calls = extract_tool_calls(generated_text)

        if not tool_calls:
            assistant_texts.append(generated_text)
            transcript.append({"role": "assistant", "content": generated_text})
            final_text = generated_text
            stop_reason = "max_new_tokens" if round_tokens >= remaining else "finished"
            break

        if round_index >= args.max_tool_rounds:
            stop_reason = "max_tool_rounds"
            break

        generated_text, tool_calls = keep_first_tool_call_only(generated_text, tool_calls)
        assistant_texts.append(generated_text)
        if not first_tool_sql:
            for call in tool_calls:
                arguments = (call.get("function") or {}).get("arguments") or {}
                if isinstance(arguments, dict) and arguments.get("sql"):
                    first_tool_sql = str(arguments["sql"])
                    break

        tool_responses = await asyncio.to_thread(execute_tool_calls, tool_calls, args.eval_timeout)
        assistant_message = build_assistant_tool_message(generated_text, tool_calls, tool_responses)
        transcript.append(assistant_message)
        working_messages.append(assistant_message)
        tool_rounds += 1
        tool_call_count += len(tool_calls)

    final_sql = extract_sql(final_text) if final_text else ""
    return {
        "sample_id": sample_id,
        "transcript": transcript,
        "assistant_texts": assistant_texts,
        "final_sql": final_sql,
        "first_tool_sql": first_tool_sql,
        "tool_rounds": tool_rounds,
        "tool_call_count": tool_call_count,
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
    }


def verify_and_screen(
    result: Dict[str, Any],
    row: Dict[str, Any],
    gold_cache: GoldCache,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Attach verification, leak and copy findings to one rollout."""
    gold_sql = row.get("gold_sql", "") or ""
    db_id = row.get("db_id", "") or ""
    final_sql = result["final_sql"]

    verified = False
    verify_error = ""
    if not final_sql:
        verify_error = "no_final_sql"
    elif not is_safe_readonly_sql(final_sql):
        verify_error = "unsafe_sql"
    else:
        gold_ok, gold_rows, gold_err = gold_cache.get(extract_sql(gold_sql), db_id)
        if not gold_ok:
            verify_error = f"gold_execution_failed:{gold_err[:120]}"
        else:
            try:
                pred_ok, pred_rows, pred_err = bird_execute_sql(
                    final_sql, db_id, args.database_dir, timeout_s=args.eval_timeout
                )
            except Exception as exc:  # pragma: no cover - defensive
                pred_ok, pred_rows, pred_err = False, None, str(exc)
            if not pred_ok:
                verify_error = f"pred_execution_failed:{pred_err[:120]}"
            elif not bird_result_match(pred_rows, gold_rows):
                verify_error = "result_mismatch"
            else:
                verified = True

    leaks = detect_leaks(
        result["assistant_texts"],
        gold_sql,
        gold_tables=sorted(extract_tables_from_sql(extract_sql(gold_sql))),
        gold_columns=sorted(extract_columns_from_sql(extract_sql(gold_sql))),
    )
    copy_flags = detect_copy(
        gold_sql=extract_sql(gold_sql),
        final_sql=final_sql,
        first_tool_sql=result["first_tool_sql"],
        tool_rounds=result["tool_rounds"],
    )

    result = dict(result)
    result.update(
        {
            "verified": verified,
            "verify_error": verify_error,
            "hard_leaks": leaks["hard"],
            "soft_leaks": leaks["soft"],
            "copy_flags": copy_flags,
            "kept": bool(verified and not leaks["hard"]),
        }
    )
    return result


def build_summary(
    results: List[Dict[str, Any]],
    n_targets: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Summary statistics shared by single-shard runs and merges."""
    kept = [r for r in results if r["kept"]]
    verified = [r for r in results if r["verified"]]
    leaked = [r for r in results if r["hard_leaks"]]
    soft_only = [r for r in results if r["soft_leaks"] and not r["hard_leaks"]]
    kept_idxs = sorted({r["idx"] for r in kept})
    copies = [r for r in kept if r["copy_flags"]]
    return {
        "model": args.model_name_or_path,
        "hint_strategy": args.hint_strategy,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_targets": n_targets,
        "n_samples_total": len(results),
        "n_verified_samples": len(verified),
        "n_kept_samples": len(kept),
        "n_leaked_samples": len(leaked),
        "n_soft_leak_only_samples": len(soft_only),
        "n_kept_unique_idxs": len(kept_idxs),
        "target_coverage_rate": round(len(kept_idxs) / n_targets, 4) if n_targets else 0.0,
        "copy_rate_over_kept": round(len(copies) / len(kept), 4) if kept else 0.0,
    }


def merge_shards(args: argparse.Namespace) -> None:
    """Concatenate shard trace files and recompute the summary over the union."""
    if not args.merge_output_dir:
        raise SystemExit("--merge_shard_dirs requires --merge_output_dir")
    out_dir = Path(args.merge_output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    seen: set = set()
    duplicates = 0
    for shard_dir in args.merge_shard_dirs:
        path = Path(shard_dir) / "teacher_traces.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        rows = load_jsonl(path)
        print(f"[teacher] merging {len(rows)} samples from {shard_dir}")
        for row in rows:
            key = (int(row["idx"]), int(row["sample_id"]))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            results.append(row)

    results.sort(key=lambda r: (r["idx"], r["sample_id"]))
    write_jsonl(out_dir / "teacher_traces.jsonl", results)

    n_targets = len({r["idx"] for r in results})
    summary = build_summary(results, n_targets=n_targets, args=args)
    summary["merged_from"] = list(args.merge_shard_dirs)
    summary["duplicate_samples_dropped"] = duplicates
    (out_dir / "teacher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[teacher] wrote {out_dir}/teacher_traces.jsonl")


async def main_async(args: argparse.Namespace) -> None:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    from nl2sql_gspo.model_utils import load_tokenizer

    out_dir = Path(args.output_dir)
    if args.num_shards > 1:
        shard_name = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
        if out_dir.name != shard_name:
            out_dir = out_dir / shard_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_tool_env(args.database_dir)

    rows = load_jsonl(Path(args.input_file))
    for position, row in enumerate(rows):
        row["source_idx"] = position

    if args.target_idx_file:
        wanted = {int(line) for line in Path(args.target_idx_file).read_text().split() if line.strip()}
        rows = [row for row in rows if row["source_idx"] in wanted]
        print(f"[teacher] target file selected {len(rows)}/{len(wanted)} ids")
    if args.limit >= 0:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = [row for row in rows if row["source_idx"] % args.num_shards == args.shard_index]
        print(f"[teacher] shard {args.shard_index}/{args.num_shards} -> {len(rows)} targets")
        if not rows:
            raise SystemExit(f"shard {args.shard_index}/{args.num_shards} received no rows")

    tokenizer_source = resolve_vllm_tokenizer_source(args.model_name_or_path)
    tokenizer = load_tokenizer(tokenizer_source)
    rows, skipped_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)
    if skipped_rows:
        write_jsonl(out_dir / "skipped_prompts.jsonl", skipped_rows)

    print(
        f"[teacher] model={args.model_name_or_path} hint={args.hint_strategy} "
        f"samples={args.num_samples} temp={args.temperature} targets={len(rows)}"
    )

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=args.model_name_or_path,
            tokenizer=tokenizer_source,
            trust_remote_code=True,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            distributed_executor_backend="mp",
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            dtype="bfloat16",
            disable_log_stats=True,
        )
    )

    gold_cache = GoldCache(args.database_dir, args.eval_timeout)
    semaphore = asyncio.Semaphore(max(1, args.vllm_async_concurrency))
    total_jobs = len(rows) * args.num_samples
    done = 0
    start = time.time()

    async def one_job(row: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        nonlocal done
        async with semaphore:
            try:
                raw = await run_trace(engine, SamplingParams, tokenizer, row, sample_id, args)
                screened = await asyncio.to_thread(verify_and_screen, raw, row, gold_cache, args)
            except Exception as exc:  # pragma: no cover - defensive
                screened = {
                    "sample_id": sample_id,
                    "transcript": [],
                    "assistant_texts": [],
                    "final_sql": "",
                    "first_tool_sql": "",
                    "tool_rounds": 0,
                    "tool_call_count": 0,
                    "completion_tokens": 0,
                    "stop_reason": "generation_error",
                    "verified": False,
                    "verify_error": f"{type(exc).__name__}: {exc}",
                    "hard_leaks": [],
                    "soft_leaks": [],
                    "copy_flags": [],
                    "kept": False,
                }
            done += 1
            if done == 1 or done % 50 == 0 or done == total_jobs:
                print(f"[teacher] {done}/{total_jobs} samples", flush=True)

            screened.update(
                {
                    "idx": row["source_idx"],
                    "db_id": row.get("db_id", ""),
                    "question": row.get("question", ""),
                    "evidence": row.get("evidence", ""),
                    "gold_sql": row.get("gold_sql", ""),
                    "prompt": [dict(m) for m in get_generation_messages(row)],
                    "tools": row.get("tools") or [],
                    "hint_strategy": args.hint_strategy,
                }
            )
            return screened

    results = await asyncio.gather(
        *(one_job(row, sample_id) for row in rows for sample_id in range(args.num_samples))
    )
    engine.shutdown()

    results.sort(key=lambda r: (r["idx"], r["sample_id"]))
    write_jsonl(out_dir / "teacher_traces.jsonl", results)

    summary = build_summary(results, n_targets=len(rows), args=args)
    summary["skipped_prompts"] = len(skipped_rows)
    summary["elapsed_s"] = round(time.time() - start, 1)
    if args.num_shards > 1:
        summary["shard_index"] = args.shard_index
        summary["num_shards"] = args.num_shards
    (out_dir / "teacher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[teacher] wrote {out_dir}/teacher_traces.jsonl")


def main() -> None:
    args = parse_args()
    if args.merge_shard_dirs:
        merge_shards(args)
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
