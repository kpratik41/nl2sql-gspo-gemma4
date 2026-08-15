#!/usr/bin/env python3
"""Run BIRD pass@k with Qwen-native tool calls on in-process async vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nl2sql_gspo.inference_tool_executor import configure_tool_env
from scripts.run_inference_bird import load_rows, preview_text
from scripts.run_inference_bird_qwen_async import run_one_async
from scripts.run_passk_bird import (
    build_passk_summary,
    evaluate_candidates,
    merge_shard_outputs,
    write_jsonl,
    write_markdown_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BIRD pass@k with Qwen async vLLM.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev_unpatched.json")
    parser.add_argument("--output_dir", default="outputs/passk/qwen_async_passk")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--no_append_shard_to_output_dir", action="store_true")
    parser.add_argument("--merge_shard_dirs", nargs="*", default=None)
    parser.add_argument("--merge_output_dir", default=None)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=43000)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--preserve_thinking", action="store_true")
    parser.add_argument("--no_prompt_rewrite", action="store_true")
    parser.add_argument("--no_force_finalize", action="store_true")
    parser.add_argument("--empty_tool_retries", type=int, default=1)
    parser.add_argument(
        "--tool_choice_policy",
        choices=["auto", "required_first", "required_until_tool", "required_always"],
        default="required_first",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards")
    args.model = args.model_name_or_path
    args.inference_backend = "qwen_vllm_async"
    args.vllm_data_parallel_size = 1
    return args


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already contains files; pass --overwrite to reuse it")
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir)
    if args.num_shards > 1 and not args.no_append_shard_to_output_dir:
        return base / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    return base


def shard_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [row for row in rows if int(row.get("source_idx", 0)) % args.num_shards == args.shard_index]


async def generate_candidates(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(
        "[qwen-async-passk] loading AsyncLLMEngine "
        f"tp={args.vllm_tensor_parallel_size} concurrency={args.vllm_async_concurrency} "
        f"max_model_len={args.vllm_max_model_len}"
    )
    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        enable_prefix_caching=True,
        dtype="bfloat16",
        disable_log_stats=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    semaphore = asyncio.Semaphore(max(1, args.vllm_async_concurrency))
    total = len(rows) * args.num_generations
    completed = 0

    async def run_candidate(row_pos: int, row: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        nonlocal completed
        source_idx = int(row.get("source_idx", row_pos))
        async with semaphore:
            generation_error = ""
            try:
                detail = await run_one_async(row, args, engine, SamplingParams, tokenizer)
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                detail = {
                    "idx": source_idx,
                    "source_idx": source_idx,
                    "db_id": row.get("db_id", ""),
                    "prediction_text": "",
                    "transcript_text": "",
                    "pred_sql": "",
                    "completion_token_count": 0,
                    "tool_rounds": 0,
                    "tool_call_count": 0,
                    "tool_names": [],
                    "tool_order": "",
                    "rejected_tool_names": [],
                    "rejected_tool_call_count": 0,
                    "stop_reason": "generation_error",
                    "error_message": generation_error,
                    "rounds": [],
                }
            completed += 1
            if completed == 1 or completed == total or completed % 50 == 0:
                print(
                    f"[qwen-async-passk] generated {completed}/{total} "
                    f"idx={source_idx} sample={sample_id} stop={detail.get('stop_reason')} "
                    f"sql={preview_text(detail.get('pred_sql', ''), 100)}"
                )
            return {
                "idx": source_idx,
                "sample_id": sample_id,
                "generation_error": generation_error,
                "candidate_request_id": f"{source_idx}-{sample_id}-{uuid.uuid4().hex[:8]}",
                **detail,
            }

    try:
        candidates = await asyncio.gather(
            *(
                run_candidate(row_pos, row, sample_id)
                for row_pos, row in enumerate(rows)
                for sample_id in range(args.num_generations)
            )
        )
    finally:
        engine.shutdown()
    candidates.sort(key=lambda item: (int(item["idx"]), int(item["sample_id"])))
    return candidates


def main() -> None:
    args = parse_args()
    if args.merge_shard_dirs is not None:
        merge_shard_outputs(args)
        return

    output_dir = resolve_output_dir(args)
    args.output_dir = str(output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    configure_tool_env(args.database_dir)

    rows = load_rows(args.input_file, args.limit)
    for idx, row in enumerate(rows):
        row.setdefault("source_idx", idx)
    rows = shard_rows(rows, args)

    print("[qwen-async-passk] starting pass@k run")
    print(f"[qwen-async-passk] model={args.model_name_or_path}")
    print(f"[qwen-async-passk] input={args.input_file}")
    print(
        f"[qwen-async-passk] rows={len(rows)} shard={args.shard_index}/{args.num_shards} "
        f"generations={args.num_generations} temperature={args.temperature}"
    )

    started = time.monotonic()
    gen_started = time.monotonic()
    candidates = asyncio.run(generate_candidates(args, rows))
    generation_seconds = time.monotonic() - gen_started

    eval_started = time.monotonic()
    evaluated = evaluate_candidates(candidates, rows, args)
    evaluation_seconds = time.monotonic() - eval_started
    timing = {
        "generation": generation_seconds,
        "evaluation": evaluation_seconds,
        "total": time.monotonic() - started,
    }
    summary = build_passk_summary(evaluated, rows, args, timing)
    write_jsonl(output_dir / "passk_candidates.jsonl", evaluated)
    write_jsonl(output_dir / "passk_per_example.jsonl", summary["per_example"])
    with (output_dir / "passk_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in summary.items() if key != "per_example"}, handle, indent=2)
    write_markdown_summary(output_dir / "passk_summary.md", summary)
    print(f"[qwen-async-passk] wrote {output_dir / 'passk_summary.md'}")


if __name__ == "__main__":
    main()
