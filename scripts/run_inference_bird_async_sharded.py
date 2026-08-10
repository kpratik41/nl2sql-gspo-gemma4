#!/usr/bin/env python3
"""Run BIRD inference with multiple async-vLLM shard workers.

The main async backend intentionally supports one vLLM data-parallel process.
This wrapper launches several independent async workers, each on its own GPU
group, while preserving the original source_idx values for merged evaluation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from nl2sql_gspo.data import normalize_record

from scripts.run_inference_bird import (
    build_generation_stats,
    build_per_example_report_rows,
    configure_tool_env,
    ensure_output_dir,
    evaluate_predictions,
    generate_predictions_with_vllm_async,
    load_diff_rows,
    load_rows,
    print_summary_tables,
    write_per_example_report_csv,
    write_run_report_markdown,
    write_summary_csv,
    write_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--gpu_groups", nargs="+", default=None)
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=43500)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard_file", default=None)
    parser.add_argument("--shard_output_dir", default=None)
    return parser.parse_args()


def worker_args_from(parent_args: argparse.Namespace, output_dir: str) -> argparse.Namespace:
    return SimpleNamespace(
        inference_backend="vllm_async",
        model_name_or_path=parent_args.model_name_or_path,
        input_file=parent_args.input_file,
        database_dir=parent_args.database_dir,
        diff_json_path=parent_args.diff_json_path,
        output_dir=output_dir,
        max_prompt_length=parent_args.max_prompt_length,
        max_new_tokens=parent_args.max_new_tokens,
        temperature=parent_args.temperature,
        top_p=parent_args.top_p,
        num_examples=parent_args.num_examples,
        eval_timeout=parent_args.eval_timeout,
        eval_workers=parent_args.eval_workers,
        vllm_tensor_parallel_size=parent_args.vllm_tensor_parallel_size,
        vllm_data_parallel_size=1,
        vllm_gpu_memory_utilization=parent_args.vllm_gpu_memory_utilization,
        vllm_max_model_len=parent_args.vllm_max_model_len,
        vllm_async_concurrency=parent_args.vllm_async_concurrency,
        max_tool_rounds=parent_args.max_tool_rounds,
        skip_generation=False,
        overwrite=True,
    )


def read_shard_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            source_idx = int(raw.get("source_idx", len(rows)))
            row = normalize_record(raw)
            row["source_idx"] = source_idx
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_worker(args: argparse.Namespace) -> None:
    if not args.shard_file or not args.shard_output_dir:
        raise ValueError("--worker requires --shard_file and --shard_output_dir")

    output_dir = Path(args.shard_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_shard_rows(Path(args.shard_file))
    print(f"[worker] loaded {len(rows)} rows from {args.shard_file}")
    print(f"[worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    configure_tool_env(args.database_dir)
    worker_args = worker_args_from(args, str(output_dir))
    rows, official_predictions, detailed_predictions, filtered_rows = generate_predictions_with_vllm_async(
        rows,
        worker_args,
    )

    with (output_dir / "predict_dev.json").open("w", encoding="utf-8") as handle:
        json.dump(official_predictions, handle, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "prediction_details.jsonl", detailed_predictions)
    write_jsonl(output_dir / "filtered_examples.jsonl", filtered_rows)
    print(f"[worker] wrote shard outputs to {output_dir}")


def shard_rows(rows: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for row in rows:
        source_idx = int(row.get("source_idx", 0))
        shards[source_idx % num_shards].append(row)
    return shards


def default_gpu_groups(num_shards: int, tensor_parallel_size: int) -> List[str]:
    groups: List[str] = []
    next_gpu = 0
    for _ in range(num_shards):
        gpu_ids = [str(gpu) for gpu in range(next_gpu, next_gpu + tensor_parallel_size)]
        groups.append(",".join(gpu_ids))
        next_gpu += tensor_parallel_size
    return groups


def merge_shard_outputs(shard_dirs: List[Path]) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    filtered_rows: List[Dict[str, Any]] = []

    for shard_dir in shard_dirs:
        with (shard_dir / "predict_dev.json").open("r", encoding="utf-8") as handle:
            official_predictions.update(json.load(handle))
        with (shard_dir / "prediction_details.jsonl").open("r", encoding="utf-8") as handle:
            detailed_predictions.extend(json.loads(line) for line in handle if line.strip())
        filtered_path = shard_dir / "filtered_examples.jsonl"
        if filtered_path.exists():
            with filtered_path.open("r", encoding="utf-8") as handle:
                filtered_rows.extend(json.loads(line) for line in handle if line.strip())

    detailed_predictions.sort(key=lambda item: int(item.get("idx", 0)))
    filtered_rows.sort(key=lambda item: int(item.get("source_idx", item.get("idx", 0))))
    return official_predictions, detailed_predictions, filtered_rows


def write_merged_outputs(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    official_predictions: Dict[str, str],
    detailed_predictions: List[Dict[str, Any]],
    filtered_rows: List[Dict[str, Any]],
    generation_seconds: float,
    run_started_at: float,
) -> None:
    output_dir = Path(args.output_dir)
    diff_rows = load_diff_rows(args.diff_json_path)
    print(f"[merge] loaded {len(diff_rows)} diff rows")

    with (output_dir / "predict_dev.json").open("w", encoding="utf-8") as handle:
        json.dump(official_predictions, handle, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "prediction_details.jsonl", detailed_predictions)
    write_jsonl(output_dir / "filtered_examples.jsonl", filtered_rows)

    evaluation_started_at = time.monotonic()
    per_example_results, summary = evaluate_predictions(
        rows=rows,
        predictions=official_predictions,
        database_dir=args.database_dir,
        diff_rows=diff_rows,
        timeout_s=args.eval_timeout,
        eval_workers=args.eval_workers,
    )
    evaluation_seconds = time.monotonic() - evaluation_started_at
    total_seconds = time.monotonic() - run_started_at
    summary["timing_seconds"] = {
        "generation": generation_seconds,
        "evaluation": evaluation_seconds,
        "total": total_seconds,
    }
    summary["generation_stats"] = build_generation_stats(detailed_predictions, filtered_rows)

    write_jsonl(output_dir / "eval_results.jsonl", per_example_results)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_rows = build_per_example_report_rows(detailed_predictions, per_example_results)
    write_per_example_report_csv(report_rows, output_dir / "per_example_report.csv")
    write_summary_markdown(summary, output_dir / "eval_summary.md", args, len(rows))
    write_run_report_markdown(summary, report_rows, output_dir / "run_report.md", args, len(rows))
    write_summary_csv(summary["by_difficulty"], output_dir / "eval_summary_by_difficulty.csv")
    write_summary_csv(summary["by_db"], output_dir / "eval_summary_by_db.csv")
    print_summary_tables(summary)
    print(
        "[merge] timing_seconds "
        f"generation={generation_seconds:.2f} "
        f"evaluation={evaluation_seconds:.2f} "
        f"total={total_seconds:.2f}"
    )


def run_parent(args: argparse.Namespace) -> None:
    run_started_at = time.monotonic()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    rows = load_rows(args.input_file, args.num_examples)
    print(f"[run] loaded {len(rows)} rows")

    gpu_groups = args.gpu_groups or default_gpu_groups(args.num_shards, args.vllm_tensor_parallel_size)
    if len(gpu_groups) != args.num_shards:
        raise ValueError("--gpu_groups must provide exactly --num_shards entries")

    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    row_shards = shard_rows(rows, args.num_shards)
    shard_dirs: List[Path] = []
    processes: List[subprocess.Popen] = []
    generation_started_at = time.monotonic()

    for shard_idx, shard in enumerate(row_shards):
        shard_dir = shard_root / f"shard-{shard_idx:05d}-of-{args.num_shards:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_file = shard_dir / "input_rows.jsonl"
        write_jsonl(shard_file, shard)
        log_path = shard_dir / "worker.log"
        shard_dirs.append(shard_dir)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model_name_or_path",
            args.model_name_or_path,
            "--input_file",
            args.input_file,
            "--database_dir",
            args.database_dir,
            "--diff_json_path",
            args.diff_json_path,
            "--output_dir",
            args.output_dir,
            "--shard_file",
            str(shard_file),
            "--shard_output_dir",
            str(shard_dir),
            "--num_examples",
            str(args.num_examples),
            "--max_prompt_length",
            str(args.max_prompt_length),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--eval_timeout",
            str(args.eval_timeout),
            "--eval_workers",
            str(args.eval_workers),
            "--vllm_tensor_parallel_size",
            str(args.vllm_tensor_parallel_size),
            "--vllm_gpu_memory_utilization",
            str(args.vllm_gpu_memory_utilization),
            "--vllm_max_model_len",
            str(args.vllm_max_model_len),
            "--vllm_async_concurrency",
            str(args.vllm_async_concurrency),
            "--max_tool_rounds",
            str(args.max_tool_rounds),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_groups[shard_idx]
        with log_path.open("w", encoding="utf-8") as log_handle:
            print(
                f"[run] starting shard {shard_idx + 1}/{args.num_shards} "
                f"rows={len(shard)} CUDA_VISIBLE_DEVICES={gpu_groups[shard_idx]} log={log_path}"
            )
            processes.append(
                subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(Path.cwd()),
                )
            )

    failures = []
    for shard_idx, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failures.append((shard_idx, return_code, shard_dirs[shard_idx] / "worker.log"))
    if failures:
        details = "; ".join(
            f"shard={idx} rc={rc} log={log}" for idx, rc, log in failures
        )
        raise RuntimeError(f"One or more shard workers failed: {details}")

    generation_seconds = time.monotonic() - generation_started_at
    official_predictions, detailed_predictions, filtered_rows = merge_shard_outputs(shard_dirs)
    write_merged_outputs(
        args=args,
        rows=rows,
        official_predictions=official_predictions,
        detailed_predictions=detailed_predictions,
        filtered_rows=filtered_rows,
        generation_seconds=generation_seconds,
        run_started_at=run_started_at,
    )


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        args.inference_backend = "vllm_async"
        args.vllm_data_parallel_size = args.num_shards
        args.skip_generation = False
        run_parent(args)


if __name__ == "__main__":
    main()
