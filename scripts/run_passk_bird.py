#!/usr/bin/env python3
"""Run async vLLM BIRD pass@k evaluation with the existing tool-call loop.

This script samples N independent completions per example once, evaluates every
candidate, and derives pass@k for all k <= N from the same candidate set.
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nl2sql_gspo.inference_tool_executor import configure_tool_env, extract_and_execute_tools, extract_tool_calls
from nl2sql_gspo.prompt_builder import PromptBuilder, add_prompt_args, prompt_config_from_args
from nl2sql_gspo.resume import (
    add_resume_args,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    build_manifest,
    checkpoint_map,
    incremental_enabled,
    prepare_manifest,
    validate_resume_args,
)
from nl2sql_gspo.sql_utils import bird_get_gold_rows, extract_sql

try:
    from run_inference_bird import (
        _async_vllm_generate_text,
        build_generation_detail,
        evaluate_one_bird,
        generate_one_with_vllm_async_tool_loop,
        load_diff_rows,
        load_rows,
        prepare_rows_for_generation,
        preview_text,
        render_prompt,
        resolve_vllm_tokenizer_source,
        get_generation_messages,
        should_use_agentic_tool_loop,
    )
except ModuleNotFoundError:
    from scripts.run_inference_bird import (
        _async_vllm_generate_text,
        build_generation_detail,
        evaluate_one_bird,
        generate_one_with_vllm_async_tool_loop,
        load_diff_rows,
        load_rows,
        prepare_rows_for_generation,
        preview_text,
        render_prompt,
        resolve_vllm_tokenizer_source,
        get_generation_messages,
        should_use_agentic_tool_loop,
    )


PASSK_MANIFEST_FIELDS = [
    "model_name_or_path",
    "input_file",
    "raw_input_file",
    "database_dir",
    "bird_mode",
    "build_prompts_at_runtime",
    "meanings_file",
    "include_column_comments",
    "include_fewshots",
    "include_stats",
    "include_nullability",
    "example_num",
    "tool_mode",
    "prompt_template",
    "skill_headers",
    "temperature",
    "top_p",
    "max_new_tokens",
    "max_prompt_length",
    "max_tool_rounds",
    "num_generations",
    "vllm_tensor_parallel_size",
    "vllm_max_model_len",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BIRD pass@k from one multi-sample async vLLM run.")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--input_file", type=str, default="outputs/old-dev-schema-bare-tool.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/bird_dev.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_tool_passk16_vllm_async_temp08")
    parser.add_argument("--limit", type=int, default=-1, help="Number of examples to run; -1 means all examples.")
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard index. Use with --num_shards to split examples across processes/GPUs.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of example shards. Each shard keeps original source_idx values.",
    )
    parser.add_argument(
        "--no_append_shard_to_output_dir",
        action="store_true",
        help=(
            "Do not append shard-XXXXX-of-YYYYY to output_dir when num_shards > 1. "
            "Only use this when each shard already has a unique output_dir."
        ),
    )
    parser.add_argument(
        "--merge_shard_dirs",
        nargs="*",
        default=None,
        help=(
            "Merge already-evaluated shard output directories instead of running generation. "
            "Each directory must contain passk_candidates.jsonl."
        ),
    )
    parser.add_argument(
        "--merge_output_dir",
        type=str,
        default=None,
        help="Directory for merged pass@k outputs. Defaults to --output_dir.",
    )
    parser.add_argument("--num_generations", type=int, default=16, help="Candidates sampled per example.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=30000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=8)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    add_resume_args(parser)
    add_prompt_args(parser)
    args = parser.parse_args()
    validate_resume_args(args)
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards")
    return args


def ensure_output_dir(path: Path, overwrite: bool, resume: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite and not resume:
        raise FileExistsError(f"Output directory already exists and is non-empty: {path}. Use --overwrite or --resume.")
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.num_shards > 1 and not args.no_append_shard_to_output_dir:
        shard_name = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
        if output_dir.name != shard_name:
            output_dir = output_dir / shard_name
    return output_dir


def shard_rows(rows: List[Dict[str, Any]], shard_index: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards == 1:
        return rows
    return [
        row
        for row in rows
        if int(row.get("source_idx", -1)) % num_shards == shard_index
    ]


def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def tool_order(prediction_text: str) -> List[str]:
    return [
        call.get("function", {}).get("name", "")
        for call in extract_tool_calls(prediction_text)
        if call.get("function", {}).get("name")
    ]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_difficulty_by_idx(diff_json_path: str, limit: int) -> Dict[int, str]:
    try:
        diff_rows = load_diff_rows(diff_json_path)
    except Exception as exc:
        print(f"[passk] warning: failed to load difficulty file: {exc}")
        return {}
    if limit >= 0:
        diff_rows = diff_rows[:limit]
    return {
        idx: str(row.get("difficulty", "unknown"))
        for idx, row in enumerate(diff_rows)
    }


def evaluate_gold_rows(
    rows: List[Dict[str, Any]],
    database_dir: str,
    timeout_s: float,
    workers: int,
) -> Dict[int, Dict[str, Any]]:
    def run(row: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(row.get("source_idx", -1))
        gold_sql = extract_sql(row.get("gold_sql", ""))
        try:
            executed, row_set, error = bird_get_gold_rows(
                gold_sql,
                row.get("db_id", ""),
                database_dir,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            executed, row_set, error = False, None, str(exc)
        return {
            "idx": idx,
            "gold_sql": gold_sql,
            "gold_executed": executed,
            "gold_row_set": row_set,
            "gold_error": error,
        }

    gold_by_idx: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run, row) for row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            gold_by_idx[result["idx"]] = result
            if completed == 1 or completed == len(futures) or completed % 100 == 0:
                print(f"[passk] prepared gold rows {completed}/{len(futures)}")
    return gold_by_idx


async def generate_candidates(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    prompt_builder: Optional[PromptBuilder] = None,
) -> List[Dict[str, Any]]:
    checkpoint_path = Path(getattr(args, "_candidate_checkpoint_path", Path(args.output_dir) / "passk_candidates_raw.incremental.jsonl"))
    if incremental_enabled(args) and not getattr(args, "resume", False) and not getattr(args, "_candidate_checkpoint_initialized", False):
        checkpoint_path.write_text("", encoding="utf-8")
        args._candidate_checkpoint_initialized = True

    existing_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    if getattr(args, "resume", False):
        existing_by_key = {
            (int(key[0]), int(key[1])): candidate
            for key, candidate in checkpoint_map(
                checkpoint_path,
                lambda row: (int(row["idx"]), int(row["sample_id"])),
            ).items()
        }
        if existing_by_key:
            print(f"[resume] loaded {len(existing_by_key)} completed pass@k candidates")

    raw_jobs: List[tuple[int, Dict[str, Any], int]] = []
    for row_index, row in enumerate(rows):
        source_idx = int(row.get("source_idx", row_index))
        for sample_id in range(args.num_generations):
            if (source_idx, sample_id) in existing_by_key:
                continue
            raw_jobs.append((row_index, row, sample_id))

    if not raw_jobs:
        print(f"[resume] no missing candidates; reusing {len(existing_by_key)} checkpointed candidates")
        return sorted(existing_by_key.values(), key=lambda item: (int(item["idx"]), int(item["sample_id"])))

    try:
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception as exc:
        raise RuntimeError("Async vLLM could not be imported.") from exc

    from nl2sql_gspo.model_utils import load_tokenizer

    tokenizer_source = resolve_vllm_tokenizer_source(args.model_name_or_path)
    tokenizer = load_tokenizer(tokenizer_source)
    prompt_builder = prompt_builder or PromptBuilder(prompt_config_from_args(args))
    prepared_cache: Dict[tuple[int, Optional[int]], Optional[Dict[str, Any]]] = {}
    skipped_rows: List[Dict[str, Any]] = []

    def prepare_sample_row(row_index: int, row: Dict[str, Any], sample_id: int) -> Optional[Dict[str, Any]]:
        source_idx = int(row.get("source_idx", row_index))
        built = prompt_builder.build_row(row, force_rebuild=False, sample_id=sample_id)
        skill_id = built.get("skill_id")
        cache_key = (source_idx, skill_id)
        if cache_key in prepared_cache:
            return prepared_cache[cache_key]

        prompt_messages = get_generation_messages(built)
        tools = built.get("tools")
        prompt_text = render_prompt(tokenizer, prompt_messages, tools)
        prompt_token_count = len(tokenizer(prompt_text, truncation=False)["input_ids"])
        prepared = dict(built)
        prepared["source_idx"] = source_idx
        prepared["prompt_text"] = prompt_text
        prepared["prompt_tokens"] = prompt_token_count
        if prompt_token_count > args.max_prompt_length:
            skipped_rows.append(
                {
                    "idx": source_idx,
                    "db_id": prepared.get("db_id", ""),
                    "sample_id": sample_id,
                    "skill_id": prepared.get("skill_id"),
                    "skill_name": prepared.get("skill_name"),
                    "prompt_tokens": prompt_token_count,
                    "max_prompt_length": args.max_prompt_length,
                    "prompt_preview": preview_text(prompt_text),
                }
            )
            prepared_cache[cache_key] = None
            return None
        prepared_cache[cache_key] = prepared
        return prepared

    prepared_jobs: List[tuple[int, Dict[str, Any], int, Optional[Dict[str, Any]]]] = []
    for row_index, row, sample_id in raw_jobs:
        prepared_jobs.append((row_index, row, sample_id, prepare_sample_row(row_index, row, sample_id)))

    if skipped_rows:
        write_jsonl(Path(args.output_dir) / "skipped_prompts.jsonl", skipped_rows)

    max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)
    print(
        "[passk] loading async vLLM engine "
        f"tp={args.vllm_tensor_parallel_size} concurrency={args.vllm_async_concurrency} "
        f"max_model_len={max_model_len}"
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
    total = len(raw_jobs)
    completed = 0

    async def generate_one(
        row_index: int,
        row: Dict[str, Any],
        sample_id: int,
        prepared_row: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nonlocal completed
        source_idx = int(row.get("source_idx", row_index))
        row_for_sample = prepared_row
        async with semaphore:
            generation_error = ""
            try:
                if row_for_sample is None:
                    raise ValueError("prompt exceeds max_prompt_length")
                if should_use_agentic_tool_loop(row_for_sample, args.max_tool_rounds):
                    generated = await generate_one_with_vllm_async_tool_loop(
                        engine=engine,
                        sampling_params_cls=SamplingParams,
                        tokenizer=tokenizer,
                        row=row_for_sample,
                        max_new_tokens=args.max_new_tokens,
                        max_model_len=max_model_len,
                        max_tool_rounds=args.max_tool_rounds,
                        eval_timeout=args.eval_timeout,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                else:
                    generated_text, completion_tokens = await _async_vllm_generate_text(
                        engine=engine,
                        sampling_params_cls=SamplingParams,
                        prompt_text=row_for_sample["prompt_text"],
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        request_prefix=f"idx{source_idx}-sample{sample_id}",
                    )
                    if "call:" in generated_text:
                        generated_text = await asyncio.to_thread(
                            extract_and_execute_tools,
                            generated_text,
                            args.eval_timeout,
                        )
                    generated = build_generation_detail(
                        row=row_for_sample,
                        prediction_text=generated_text,
                        prompt_token_count=int(row_for_sample["prompt_tokens"]),
                        completion_token_count=completion_tokens,
                        tool_rounds=0,
                        tool_call_count=len(extract_tool_calls(generated_text)),
                        stop_reason="finished",
                    )
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                generated = build_generation_detail(
                    row=row_for_sample or row,
                    prediction_text="",
                    prompt_token_count=int((row_for_sample or {}).get("prompt_tokens", 0)),
                    completion_token_count=0,
                    tool_rounds=0,
                    tool_call_count=0,
                    stop_reason="generation_error",
                )

            completed += 1
            if completed == 1 or completed == total or completed % 50 == 0:
                print(
                    f"[passk] generated {completed}/{total} candidates "
                    f"idx={source_idx} sample={sample_id} stop={generated.get('stop_reason')} "
                    f"sql={preview_text(generated.get('pred_sql', ''), max_chars=100)}"
                )
            candidate = {
                "idx": source_idx,
                "sample_id": sample_id,
                "skill_id": (row_for_sample or {}).get("skill_id"),
                "skill_name": (row_for_sample or {}).get("skill_name", "default"),
                "generation_error": generation_error,
                **generated,
            }
            if incremental_enabled(args):
                append_jsonl(checkpoint_path, candidate)
            return candidate

    try:
        tasks = [
            generate_one(row_index, row, sample_id, prepared_row)
            for row_index, row, sample_id, prepared_row in prepared_jobs
        ]
        new_candidates = await asyncio.gather(*tasks)
        merged_by_key = dict(existing_by_key)
        for candidate in new_candidates:
            merged_by_key[(int(candidate["idx"]), int(candidate["sample_id"]))] = candidate
        return sorted(merged_by_key.values(), key=lambda item: (int(item["idx"]), int(item["sample_id"])))
    finally:
        engine.shutdown()


def evaluate_candidates(
    candidates: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    gold_by_idx = evaluate_gold_rows(rows, args.database_dir, args.eval_timeout, args.eval_workers)

    def run(candidate: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(candidate["idx"])
        gold = gold_by_idx[idx]
        result = evaluate_one_bird(
            predicted_sql=candidate.get("pred_sql", ""),
            db_id=candidate.get("db_id", ""),
            database_dir=args.database_dir,
            timeout_s=args.eval_timeout,
            gold_executed=gold["gold_executed"],
            gold_row_set=gold["gold_row_set"],
            gold_error=gold["gold_error"],
        )
        return {
            **candidate,
            "gold_sql": gold["gold_sql"],
            "correct": int(result["res"]),
            "status": result["status"],
            "pred_executed": result["pred_executed"],
            "gold_executed": result["gold_executed"],
            "pred_error": result["pred_error"],
            "gold_error": result["gold_error"],
            "tool_order": tool_order(candidate.get("prediction_text", "")),
        }

    evaluated: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.eval_workers)) as pool:
        futures = [pool.submit(run, candidate) for candidate in candidates]
        for completed, future in enumerate(as_completed(futures), start=1):
            evaluated.append(future.result())
            if completed == 1 or completed == len(futures) or completed % 200 == 0:
                print(f"[passk] evaluated {completed}/{len(futures)} candidates")

    evaluated.sort(key=lambda row: (int(row["idx"]), int(row["sample_id"])))
    return evaluated


def build_passk_summary(
    evaluated: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    timing: Dict[str, float],
) -> Dict[str, Any]:
    by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for candidate in evaluated:
        by_idx.setdefault(int(candidate["idx"]), []).append(candidate)

    per_example: List[Dict[str, Any]] = []
    passk_sums = {k: 0.0 for k in range(1, args.num_generations + 1)}
    prefix_sums = {k: 0.0 for k in range(1, args.num_generations + 1)}
    candidates_per_example = Counter(len(candidates) for candidates in by_idx.values())

    for row in rows:
        idx = int(row.get("source_idx", -1))
        candidates = sorted(by_idx.get(idx, []), key=lambda item: int(item["sample_id"]))
        corrects = [int(candidate.get("correct", 0)) for candidate in candidates]
        n = len(corrects)
        c = sum(corrects)
        example_passk: Dict[str, float] = {}
        example_prefix: Dict[str, int] = {}
        for k in range(1, args.num_generations + 1):
            effective_k = min(k, n)
            estimated = pass_at_k(n, c, effective_k) if n else 0.0
            prefix = int(any(corrects[:effective_k])) if effective_k else 0
            example_passk[str(k)] = estimated
            example_prefix[str(k)] = prefix
            passk_sums[k] += estimated
            prefix_sums[k] += prefix

        per_example.append(
            {
                "idx": idx,
                "db_id": row.get("db_id", ""),
                "num_candidates": n,
                "num_correct": c,
                "first_correct_sample_id": next(
                    (candidate["sample_id"] for candidate in candidates if candidate.get("correct")),
                    None,
                ),
                "pass_at_k": example_passk,
                "prefix_pass_at_k": example_prefix,
                "stop_reasons": dict(Counter(candidate.get("stop_reason", "") for candidate in candidates)),
                "tool_call_count_total": sum(int(candidate.get("tool_call_count", 0)) for candidate in candidates),
            }
        )

    denominator = max(1, len(rows))
    return {
        "run_config": {
            "model_name_or_path": args.model_name_or_path,
            "input_file": args.input_file,
            "database_dir": args.database_dir,
            "diff_json_path": args.diff_json_path,
            "limit": args.limit,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_tool_rounds": args.max_tool_rounds,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_async_concurrency": args.vllm_async_concurrency,
        },
        "total_examples": len(rows),
        "total_candidates": len(evaluated),
        "candidates_per_example": dict(candidates_per_example),
        "pass_at_k_estimated": {
            str(k): 100.0 * passk_sums[k] / denominator
            for k in range(1, args.num_generations + 1)
        },
        "prefix_pass_at_k": {
            str(k): 100.0 * prefix_sums[k] / denominator
            for k in range(1, args.num_generations + 1)
        },
        "candidate_accuracy": {
            "correct": sum(int(candidate.get("correct", 0)) for candidate in evaluated),
            "count": len(evaluated),
            "accuracy": 100.0
            * sum(int(candidate.get("correct", 0)) for candidate in evaluated)
            / max(1, len(evaluated)),
        },
        "pred_execution": {
            "executed": sum(int(candidate.get("pred_executed", False)) for candidate in evaluated),
            "failed": sum(int(not candidate.get("pred_executed", False)) for candidate in evaluated),
        },
        "stop_reasons": dict(Counter(candidate.get("stop_reason", "") for candidate in evaluated)),
        "tool_calls": {
            "total": sum(int(candidate.get("tool_call_count", 0)) for candidate in evaluated),
            "avg_per_candidate": sum(int(candidate.get("tool_call_count", 0)) for candidate in evaluated)
            / max(1, len(evaluated)),
        },
        "skill_headers": {
            "mode": args.skill_headers,
            "counts": dict(Counter(candidate.get("skill_name", "default") for candidate in evaluated)),
        },
        "timing_seconds": timing,
        "per_example": per_example,
    }


def write_markdown_summary(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# BIRD pass@k Summary",
        "",
        "## Run",
        "",
    ]
    for key, value in summary["run_config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## pass@k",
            "",
            "| k | estimated pass@k | prefix pass@k |",
            "| ---: | ---: | ---: |",
        ]
    )
    for k in summary["pass_at_k_estimated"]:
        lines.append(
            f"| {k} | {summary['pass_at_k_estimated'][k]:.2f} | "
            f"{summary['prefix_pass_at_k'][k]:.2f} |"
        )
    candidate_accuracy = summary["candidate_accuracy"]
    lines.extend(
        [
            "",
            "## Candidate Accuracy",
            "",
            f"- correct: `{candidate_accuracy['correct']}/{candidate_accuracy['count']}`",
            f"- accuracy: `{candidate_accuracy['accuracy']:.2f}%`",
            "",
            "## Execution",
            "",
            f"- pred_sql_executed: `{summary['pred_execution']['executed']}`",
            f"- pred_sql_execution_failed: `{summary['pred_execution']['failed']}`",
            f"- total_tool_calls: `{summary['tool_calls']['total']}`",
            f"- avg_tool_calls_per_candidate: `{summary['tool_calls']['avg_per_candidate']:.2f}`",
            "",
            "## Stop Reasons",
            "",
        ]
    )
    for reason, count in sorted(summary["stop_reasons"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {reason or '<empty>'}: `{count}`")
    lines.extend(
        [
            "",
            "## Timing",
            "",
        ]
    )
    for key, value in summary["timing_seconds"].items():
        lines.append(f"- {key}: `{value:.2f}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_shard_outputs(args: argparse.Namespace) -> None:
    shard_dirs = [Path(path) for path in args.merge_shard_dirs or []]
    if not shard_dirs:
        raise ValueError("--merge_shard_dirs requires at least one shard output directory")

    output_dir = Path(args.merge_output_dir or args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    rows = load_rows(args.input_file, args.limit)
    difficulty_by_idx = load_difficulty_by_idx(args.diff_json_path, args.limit)
    for row in rows:
        row["difficulty"] = difficulty_by_idx.get(int(row.get("source_idx", -1)), "unknown")

    evaluated: List[Dict[str, Any]] = []
    seen = set()
    duplicate_keys = []
    for shard_dir in shard_dirs:
        candidates_path = shard_dir / "passk_candidates.jsonl"
        if not candidates_path.exists():
            raise FileNotFoundError(candidates_path)
        shard_rows = read_jsonl(candidates_path)
        print(f"[passk] merging {len(shard_rows)} candidates from {shard_dir}")
        for candidate in shard_rows:
            key = (int(candidate["idx"]), int(candidate["sample_id"]))
            if key in seen:
                duplicate_keys.append(key)
            seen.add(key)
            evaluated.append(candidate)

    if duplicate_keys:
        preview = ", ".join(f"{idx}:{sample}" for idx, sample in duplicate_keys[:10])
        raise ValueError(f"Duplicate shard candidates found for idx:sample keys: {preview}")

    evaluated.sort(key=lambda row: (int(row["idx"]), int(row["sample_id"])))
    timing = {
        "generation": sum(
            float(json.loads((shard_dir / "passk_summary.json").read_text()).get("timing_seconds", {}).get("generation", 0.0))
            for shard_dir in shard_dirs
            if (shard_dir / "passk_summary.json").exists()
        ),
        "evaluation": sum(
            float(json.loads((shard_dir / "passk_summary.json").read_text()).get("timing_seconds", {}).get("evaluation", 0.0))
            for shard_dir in shard_dirs
            if (shard_dir / "passk_summary.json").exists()
        ),
        "total": sum(
            float(json.loads((shard_dir / "passk_summary.json").read_text()).get("timing_seconds", {}).get("total", 0.0))
            for shard_dir in shard_dirs
            if (shard_dir / "passk_summary.json").exists()
        ),
    }

    summary = build_passk_summary(evaluated, rows, args, timing)
    summary["merged_shards"] = [str(path) for path in shard_dirs]

    write_jsonl(output_dir / "passk_candidates.jsonl", evaluated)
    write_jsonl(output_dir / "passk_per_example.jsonl", summary["per_example"])
    atomic_write_json(output_dir / "passk_summary.json", {key: value for key, value in summary.items() if key != "per_example"})
    write_markdown_summary(output_dir / "passk_summary.md", summary)
    print(f"[passk] merged {len(evaluated)} candidates into {output_dir}")
    print(f"[passk] wrote {output_dir / 'passk_summary.md'}")


def main() -> None:
    args = parse_args()
    if args.merge_shard_dirs is not None:
        merge_shard_outputs(args)
        return

    output_dir = resolve_output_dir(args)
    args.output_dir = str(output_dir)
    ensure_output_dir(output_dir, args.overwrite, args.resume)
    configure_tool_env(args.database_dir)
    args._candidate_checkpoint_path = str(output_dir / "passk_candidates_raw.incremental.jsonl")

    prompt_builder = PromptBuilder(prompt_config_from_args(args))
    effective_input_file = args.raw_input_file or args.input_file
    args.input_file = effective_input_file
    manifest = build_manifest(args, mode="passk", fields=PASSK_MANIFEST_FIELDS)
    prepare_manifest(output_dir, manifest, resume=args.resume)

    print("[passk] starting pass@k run")
    print(f"[passk] model={args.model_name_or_path}")
    print(f"[passk] input={args.input_file}")
    print(f"[passk] output_dir={output_dir}")
    print(f"[passk] limit={args.limit} num_generations={args.num_generations} temperature={args.temperature}")
    print(f"[passk] shard_index={args.shard_index} num_shards={args.num_shards}")
    print(f"[passk] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    start_time = time.time()
    force_prompt_rebuild = bool(args.build_prompts_at_runtime or args.raw_input_file)
    rows = load_rows(
        effective_input_file,
        args.limit,
        require_gold_sql=args.bird_mode == "dev",
        prompt_builder=prompt_builder if force_prompt_rebuild else None,
        force_prompt_rebuild=force_prompt_rebuild,
    )
    difficulty_by_idx = load_difficulty_by_idx(args.diff_json_path, args.limit)
    for row in rows:
        row["difficulty"] = difficulty_by_idx.get(int(row.get("source_idx", -1)), "unknown")
    original_row_count = len(rows)
    rows = shard_rows(rows, args.shard_index, args.num_shards)
    if not rows:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} received no rows from "
            f"{original_row_count} loaded examples."
        )
    print(
        f"[passk] shard rows={len(rows)}/{original_row_count} "
        f"first_idx={rows[0].get('source_idx')} last_idx={rows[-1].get('source_idx')}"
    )

    generation_start = time.time()
    candidates = asyncio.run(generate_candidates(args, rows, prompt_builder=prompt_builder))
    generation_seconds = time.time() - generation_start
    write_jsonl(output_dir / "passk_candidates_raw.jsonl", candidates)

    evaluation_start = time.time()
    evaluated = evaluate_candidates(candidates, rows, args)
    evaluation_seconds = time.time() - evaluation_start
    write_jsonl(output_dir / "passk_candidates.jsonl", evaluated)

    timing = {
        "generation": generation_seconds,
        "evaluation": evaluation_seconds,
        "total": time.time() - start_time,
    }
    summary = build_passk_summary(evaluated, rows, args, timing)
    write_jsonl(output_dir / "passk_per_example.jsonl", summary["per_example"])
    atomic_write_json(output_dir / "passk_summary.json", {key: value for key, value in summary.items() if key != "per_example"})
    write_markdown_summary(output_dir / "passk_summary.md", summary)

    print("[passk] complete")
    print(f"[passk] wrote {output_dir / 'passk_summary.md'}")
    for k, value in summary["pass_at_k_estimated"].items():
        print(f"[passk] pass@{k} estimated={value:.2f}% prefix={summary['prefix_pass_at_k'][k]:.2f}%")


if __name__ == "__main__":
    main()
