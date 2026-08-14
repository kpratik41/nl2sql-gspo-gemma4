#!/usr/bin/env python3
"""Run BIRD pass@k through a Qwen OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nl2sql_gspo.inference_tool_executor import configure_tool_env
from scripts.run_inference_bird import load_rows, preview_text
from scripts.run_inference_bird_qwen_server import run_one, wait_for_server
from scripts.run_passk_bird import (
    build_passk_summary,
    evaluate_candidates,
    merge_shard_outputs,
    write_jsonl,
    write_markdown_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BIRD pass@k through a Qwen vLLM server.")
    parser.add_argument("--server_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3p8-27b")
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev_unpatched.json")
    parser.add_argument("--output_dir", default="outputs/passk/qwen3p8_27b_olddev_temp1p2")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--no_append_shard_to_output_dir", action="store_true")
    parser.add_argument("--merge_shard_dirs", nargs="*", default=None)
    parser.add_argument("--merge_output_dir", default=None)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--request_timeout", type=float, default=900.0)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=16)
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

    args.model_name_or_path = args.model
    args.vllm_tensor_parallel_size = "server"
    args.vllm_async_concurrency = args.concurrency
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
    selected = rows[args.shard_index :: args.num_shards]
    for row in selected:
        row["source_idx"] = int(row.get("source_idx", rows.index(row)))
    return selected


def generate_candidates(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(rows) * args.num_generations
    candidates: List[Dict[str, Any]] = []

    def generate_one(row_pos: int, row: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        source_idx = int(row.get("source_idx", row_pos))
        try:
            _, detail = run_one(row, args)
            generation_error = ""
        except Exception as exc:
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
                "error_message": str(exc),
                "rounds": [],
            }
            generation_error = f"{type(exc).__name__}: {exc}"
        return {
            "idx": source_idx,
            "sample_id": sample_id,
            "generation_error": generation_error,
            **detail,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(generate_one, row_pos, row, sample_id): (row, sample_id)
            for row_pos, row in enumerate(rows)
            for sample_id in range(args.num_generations)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            candidate = future.result()
            candidates.append(candidate)
            if completed == 1 or completed == total or completed % 50 == 0:
                print(
                    f"[qwen-passk] generated {completed}/{total} candidates "
                    f"idx={candidate.get('idx')} sample={candidate.get('sample_id')} "
                    f"stop={candidate.get('stop_reason')} "
                    f"sql={preview_text(candidate.get('pred_sql', ''), max_chars=100)}"
                )

    candidates.sort(key=lambda row: (int(row["idx"]), int(row["sample_id"])))
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
    wait_for_server(args.server_url)

    rows = load_rows(args.input_file, args.limit)
    for idx, row in enumerate(rows):
        row.setdefault("source_idx", idx)
    rows = shard_rows(rows, args)

    print("[qwen-passk] starting pass@k run")
    print(f"[qwen-passk] model={args.model}")
    print(f"[qwen-passk] input={args.input_file}")
    print(f"[qwen-passk] output_dir={output_dir}")
    print(
        f"[qwen-passk] rows={len(rows)} shard={args.shard_index}/{args.num_shards} "
        f"generations={args.num_generations} temperature={args.temperature} concurrency={args.concurrency}"
    )

    started = time.monotonic()
    generation_started = time.monotonic()
    candidates = generate_candidates(args, rows)
    generation_seconds = time.monotonic() - generation_started

    evaluation_started = time.monotonic()
    evaluated = evaluate_candidates(candidates, rows, args)
    evaluation_seconds = time.monotonic() - evaluation_started
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
    print(f"[qwen-passk] wrote {output_dir / 'passk_summary.md'}")


if __name__ == "__main__":
    main()
