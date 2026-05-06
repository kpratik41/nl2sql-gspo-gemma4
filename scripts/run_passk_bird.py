import argparse
import csv
import json
import multiprocessing as mp
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.run_inference_bird import (
    build_group_summary,
    extract_sql,
    load_diff_rows,
    load_rows,
    plan_vllm_device_groups,
    prepare_rows_for_generation,
    shard_rows_for_data_parallel,
    should_log_progress_tick,
)
from nl2sql_gspo.model_utils import load_tokenizer
from nl2sql_gspo.sql_utils import bird_execute_sql, bird_get_gold_rows, bird_result_match


def should_log_passk_progress_tick(current_index: int, total_count: int) -> bool:
    completed = current_index + 1
    if completed == 1 or completed == total_count:
        return True

    if total_count <= 100:
        return completed % 10 == 0

    return completed % 10 == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="outputs/dev-20251106-schema.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/dev_20251106.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_passk")
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=30000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--eval_timeout", type=float, default=120.0)
    parser.add_argument("--eval_workers", type=int, default=32)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    parser.add_argument("--vllm_data_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already contains files. Use --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def print_passk_configuration(args: argparse.Namespace, output_dir: Path) -> None:
    print("[run] starting pass@k evaluation")
    print(f"[run] model_name_or_path={args.model_name_or_path}")
    print(f"[run] input_file={args.input_file}")
    print(f"[run] database_dir={args.database_dir}")
    print(f"[run] diff_json_path={args.diff_json_path}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] num_examples={args.num_examples}")
    print(f"[run] max_prompt_length={args.max_prompt_length}")
    print(f"[run] max_new_tokens={args.max_new_tokens}")
    print(f"[run] temperature={args.temperature}")
    print(f"[run] top_p={args.top_p}")
    print(f"[run] num_generations={args.num_generations}")
    print(f"[run] eval_timeout={args.eval_timeout}")
    print(f"[run] eval_workers={args.eval_workers}")
    print(f"[run] vllm_tensor_parallel_size={args.vllm_tensor_parallel_size}")
    print(f"[run] vllm_data_parallel_size={args.vllm_data_parallel_size}")
    print(f"[run] vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
    print(f"[run] vllm_max_model_len={args.vllm_max_model_len}")


def _vllm_generate_passk_worker(
    queue,
    shard_id: int,
    device_group: List[str],
    rows: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_group)

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=llm_config["model_name_or_path"],
            tokenizer=llm_config["model_name_or_path"],
            trust_remote_code=True,
            tensor_parallel_size=llm_config["tensor_parallel_size"],
            distributed_executor_backend="mp",
            gpu_memory_utilization=llm_config["gpu_memory_utilization"],
            max_model_len=llm_config["max_model_len"],
            dtype="bfloat16",
        )

        sampling_params = SamplingParams(
            n=llm_config["num_generations"],
            temperature=llm_config["temperature"],
            top_p=llm_config["top_p"],
            max_tokens=llm_config["max_new_tokens"],
        )
        prompt_texts = [row["prompt_text"] for row in rows]
        outputs = llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)

        results: List[Dict[str, Any]] = []
        for row, request_output in zip(rows, outputs):
            generations: List[Dict[str, Any]] = []
            for sample_idx, output in enumerate(request_output.outputs[: llm_config["num_generations"]]):
                prediction_text = (output.text or "").strip()
                generations.append(
                    {
                        "sample_idx": sample_idx,
                        "prediction_text": prediction_text,
                        "pred_sql": extract_sql(prediction_text),
                        "completion_token_count": len(output.token_ids),
                    }
                )

            results.append(
                {
                    "idx": row.get("source_idx", -1),
                    "db_id": row.get("db_id", ""),
                    "gold_sql": extract_sql(row.get("gold_sql", "")),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "generations": generations,
                }
            )

        queue.put({"status": "ok", "shard_id": shard_id, "results": results})
    except Exception as exc:
        queue.put({"status": "error", "shard_id": shard_id, "error": str(exc)})


def generate_predictions_with_vllm_n(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tokenizer = load_tokenizer(args.model_name_or_path)
    rows, skipped_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)

    vllm_max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)
    tensor_parallel_size = args.vllm_tensor_parallel_size
    data_parallel_size = args.vllm_data_parallel_size

    # Pass@k uses a strict two-phase workflow: finish all sampled generations
    # first, then run execution-based scoring over the saved candidate SQLs.
    # That keeps pass@1..pass@16 comparable because every k is computed from
    # prefixes of the same 16 samples per example.
    if data_parallel_size > 1:
        device_groups = plan_vllm_device_groups(tensor_parallel_size, data_parallel_size)
        row_shards = shard_rows_for_data_parallel(rows, data_parallel_size)
        active_shards = [
            (shard_id, device_group, shard_rows)
            for shard_id, (device_group, shard_rows) in enumerate(zip(device_groups, row_shards))
            if shard_rows
        ]
        print(
            "[inference] loading vLLM engines in multi-process data-parallel mode "
            f"tensor_parallel_size={tensor_parallel_size} data_parallel_size={data_parallel_size} "
            f"device_groups={['+'.join(group) for group in device_groups]} max_model_len={vllm_max_model_len}"
        )

        llm_config = {
            "model_name_or_path": args.model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": vllm_max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.num_generations,
        }

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []
        for shard_id, device_group, shard_rows in active_shards:
            print(
                f"[inference] starting vLLM shard {shard_id + 1}/{len(active_shards)} "
                f"gpus={','.join(device_group)} prompts={len(shard_rows)}"
            )
            process = ctx.Process(
                target=_vllm_generate_passk_worker,
                args=(queue, shard_id, device_group, shard_rows, llm_config),
            )
            process.start()
            processes.append(process)

        collected_results: Dict[int, Dict[str, Any]] = {}
        try:
            for _ in processes:
                message = queue.get()
                if message.get("status") != "ok":
                    raise RuntimeError(
                        "vLLM pass@k worker failed"
                        + (f" (shard {message.get('shard_id')})" if "shard_id" in message else "")
                        + ": "
                        + message.get("error", "unknown error")
                    )

                for result in message["results"]:
                    collected_results[result["idx"]] = result
        finally:
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        prediction_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            source_idx = row.get("source_idx", idx)
            generated = collected_results.get(source_idx)
            if generated is None:
                raise RuntimeError(f"Missing vLLM pass@k result for idx={source_idx}")
            prediction_rows.append(generated)
            if should_log_passk_progress_tick(idx, len(rows)):
                print(f"[inference] generated {idx + 1}/{len(rows)} prompts")

        return prediction_rows, skipped_rows

    from vllm import LLM, SamplingParams

    print(
        "[inference] loading vLLM engine "
        f"tensor_parallel_size={tensor_parallel_size} data_parallel_size={data_parallel_size} "
        f"max_model_len={vllm_max_model_len}"
    )
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    prompt_texts = [row["prompt_text"] for row in rows]
    outputs = llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)

    prediction_rows: List[Dict[str, Any]] = []
    for idx, (row, request_output) in enumerate(zip(rows, outputs)):
        generations: List[Dict[str, Any]] = []
        for sample_idx, output in enumerate(request_output.outputs[: args.num_generations]):
            prediction_text = (output.text or "").strip()
            generations.append(
                {
                    "sample_idx": sample_idx,
                    "prediction_text": prediction_text,
                    "pred_sql": extract_sql(prediction_text),
                    "completion_token_count": len(output.token_ids),
                }
            )

        prediction_rows.append(
            {
                "idx": row.get("source_idx", idx),
                "db_id": row.get("db_id", ""),
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "prompt_tokens": int(row["prompt_tokens"]),
                "generations": generations,
            }
        )
        if should_log_passk_progress_tick(idx, len(rows)):
            print(f"[inference] generated {idx + 1}/{len(rows)} prompts")

    return prediction_rows, skipped_rows


def evaluate_generation(
    predicted_sql: str,
    gold_sql: str,
    db_id: str,
    database_dir: str,
    timeout_s: float,
) -> Dict[str, Any]:
    pred_executed, pred_rows, pred_error = bird_execute_sql(
        sql=predicted_sql,
        db_id=db_id,
        database_dir=database_dir,
        timeout_s=timeout_s,
    )
    gold_executed, gold_row_set, gold_error = bird_get_gold_rows(
        gold_sql=gold_sql,
        db_id=db_id,
        database_dir=database_dir,
        timeout_s=timeout_s,
    )
    matched = pred_executed and gold_executed and bird_result_match(pred_rows, gold_row_set)

    if pred_executed and gold_executed:
        status = "ok" if matched else "mismatch"
    else:
        parts = []
        if pred_error:
            parts.append(f"pred_error: {pred_error}")
        if gold_error:
            parts.append(f"gold_error: {gold_error}")
        status = "; ".join(parts) if parts else "error"

    return {
        "res": int(matched),
        "status": status,
        "pred_executed": bool(pred_executed),
        "gold_executed": bool(gold_executed),
        "pred_error": pred_error,
        "gold_error": gold_error,
        "pred_sql_extracted": bool(predicted_sql.strip()),
        "gold_sql_extracted": bool(gold_sql.strip()),
    }


def evaluate_passk(
    prediction_rows: List[Dict[str, Any]],
    diff_rows: List[Dict[str, Any]],
    database_dir: str,
    timeout_s: float,
    max_workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    sample_results: List[Dict[str, Any]] = []
    worker_count = max(1, min(max_workers, 32))

    # Flatten to one execution job per generated candidate. For 1534 prompts
    # with 16 samples each, this produces 24,544 execution checks before the
    # final pass@k prefix aggregation step.
    prepared_jobs: List[Tuple[int, int, Dict[str, Any], str]] = []
    for row_idx, row in enumerate(prediction_rows):
        difficulty = diff_rows[row["idx"]].get("difficulty", "unknown") if row["idx"] < len(diff_rows) else "unknown"
        for generation in row["generations"]:
            prepared_jobs.append((row_idx, row["idx"], row, difficulty, generation))

    def _run_job(job: Tuple[int, int, Dict[str, Any], str, Dict[str, Any]]) -> Dict[str, Any]:
        _, source_idx, row, difficulty, generation = job
        result = evaluate_generation(
            predicted_sql=generation["pred_sql"],
            gold_sql=row["gold_sql"],
            db_id=row["db_id"],
            database_dir=database_dir,
            timeout_s=timeout_s,
        )
        return {
            "idx": source_idx,
            "db_id": row["db_id"],
            "difficulty": difficulty,
            "sample_idx": generation["sample_idx"],
            "pred_sql": generation["pred_sql"],
            "gold_sql": row["gold_sql"],
            **result,
        }

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        ordered_results = []
        total_jobs = len(prepared_jobs)
        for job_index, result in enumerate(executor.map(_run_job, prepared_jobs)):
            ordered_results.append(result)
            if should_log_passk_progress_tick(job_index, total_jobs):
                print(f"[evaluation] scored {job_index + 1}/{total_jobs} candidate generations")

    sample_results.extend(ordered_results)

    sample_results_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for result in sample_results:
        sample_results_by_idx.setdefault(result["idx"], []).append(result)

    for result_list in sample_results_by_idx.values():
        result_list.sort(key=lambda item: item["sample_idx"])

    passk_rows: List[Dict[str, Any]] = []
    for k in range(1, 17):
        aggregated: List[Dict[str, Any]] = []
        for row in prediction_rows:
            # pass@k asks whether any of the first k samples solves the item,
            # so each row collapses from k candidate evaluations to one hit/miss.
            row_results = sample_results_by_idx[row["idx"]][:k]
            matched = any(item["res"] for item in row_results)
            pred_executed = any(item["pred_executed"] for item in row_results)
            gold_executed = any(item["gold_executed"] for item in row_results)
            aggregated.append(
                {
                    "idx": row["idx"],
                    "db_id": row["db_id"],
                    "difficulty": diff_rows[row["idx"]].get("difficulty", "unknown") if row["idx"] < len(diff_rows) else "unknown",
                    "res": int(matched),
                    "pred_sql_extracted": any(item["pred_sql_extracted"] for item in row_results),
                    "gold_sql_extracted": any(item["gold_sql_extracted"] for item in row_results),
                    "pred_executed": pred_executed,
                    "gold_executed": gold_executed,
                }
            )

        total_correct = sum(item["res"] for item in aggregated)
        total_count = len(aggregated)
        execution_stats = {
            "pred_sql_extracted": sum(int(item["pred_sql_extracted"]) for item in aggregated),
            "pred_sql_missing": total_count - sum(int(item["pred_sql_extracted"]) for item in aggregated),
            "gold_sql_extracted": sum(int(item["gold_sql_extracted"]) for item in aggregated),
            "gold_sql_missing": total_count - sum(int(item["gold_sql_extracted"]) for item in aggregated),
            "pred_sql_executed": sum(int(item["pred_executed"]) for item in aggregated),
            "pred_sql_execution_failed": total_count - sum(int(item["pred_executed"]) for item in aggregated),
            "gold_sql_executed": sum(int(item["gold_executed"]) for item in aggregated),
            "gold_sql_execution_failed": total_count - sum(int(item["gold_executed"]) for item in aggregated),
            "both_sql_executed": sum(int(item["pred_executed"] and item["gold_executed"]) for item in aggregated),
        }
        passk_rows.append(
            {
                "k": k,
                "correct": total_correct,
                "count": total_count,
                "accuracy": 100.0 * total_correct / max(1, total_count),
                "execution_stats": execution_stats,
                "by_difficulty": build_group_summary(aggregated, "difficulty", ["simple", "moderate", "challenging"]),
                "by_db": build_group_summary(aggregated, "db_id"),
            }
        )
        print(
            f"[summary] computed pass@{k}: {100.0 * total_correct / max(1, total_count):.2f}% "
            f"({total_correct}/{total_count})"
        )

    return sample_results, {"passk": passk_rows}


def write_passk_markdown(summary: Dict[str, Any], markdown_path: Path) -> None:
    lines = [
        "# BIRD Dev Pass@K Summary",
        "",
        "| K | Correct | Count | Pass@K | Pred Executed | Both Executed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["passk"]:
        execution_stats = row["execution_stats"]
        lines.append(
            f"| {row['k']} | {row['correct']} | {row['count']} | {row['accuracy']:.2f} | "
            f"{execution_stats['pred_sql_executed']} | {execution_stats['both_sql_executed']} |"
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_passk_csv(summary: Dict[str, Any], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "correct",
                "count",
                "pass_at_k",
                "pred_sql_executed",
                "pred_sql_execution_failed",
                "gold_sql_executed",
                "gold_sql_execution_failed",
                "both_sql_executed",
            ],
        )
        writer.writeheader()
        for row in summary["passk"]:
            execution_stats = row["execution_stats"]
            writer.writerow(
                {
                    "k": row["k"],
                    "correct": row["correct"],
                    "count": row["count"],
                    "pass_at_k": f"{row['accuracy']:.2f}",
                    "pred_sql_executed": execution_stats["pred_sql_executed"],
                    "pred_sql_execution_failed": execution_stats["pred_sql_execution_failed"],
                    "gold_sql_executed": execution_stats["gold_sql_executed"],
                    "gold_sql_execution_failed": execution_stats["gold_sql_execution_failed"],
                    "both_sql_executed": execution_stats["both_sql_executed"],
                }
            )


def print_passk_summary(summary: Dict[str, Any]) -> None:
    print("Pass@K Summary")
    print(f"{'k':>3} {'correct':>10} {'count':>10} {'pass@k':>10} {'pred_exec':>12} {'both_exec':>12}")
    for row in summary["passk"]:
        execution_stats = row["execution_stats"]
        print(
            f"{row['k']:>3} {row['correct']:>10} {row['count']:>10} {row['accuracy']:>9.2f} "
            f"{execution_stats['pred_sql_executed']:>12} {execution_stats['both_sql_executed']:>12}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    print_passk_configuration(args, output_dir)

    rows = load_rows(args.input_file, args.num_examples)
    print(f"[run] loaded {len(rows)} input rows")
    diff_rows = load_diff_rows(args.diff_json_path)
    print(f"[run] loaded {len(diff_rows)} diff rows")

    prediction_rows, skipped_rows = generate_predictions_with_vllm_n(rows, args)
    if skipped_rows:
        raise RuntimeError(
            f"Found {len(skipped_rows)} prompts exceeding max_prompt_length={args.max_prompt_length}; rerun with a larger context budget."
        )

    sample_results, summary = evaluate_passk(
        prediction_rows=prediction_rows,
        diff_rows=diff_rows,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        max_workers=args.eval_workers,
    )

    generations_path = output_dir / "prediction_samples.jsonl"
    filtered_path = output_dir / "filtered_examples.jsonl"
    sample_results_path = output_dir / "eval_results_samples.jsonl"
    summary_path = output_dir / "passk_summary.json"
    summary_markdown_path = output_dir / "passk_summary.md"
    summary_csv_path = output_dir / "passk_summary.csv"

    with generations_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with filtered_path.open("w", encoding="utf-8") as handle:
        for row in skipped_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with sample_results_path.open("w", encoding="utf-8") as handle:
        for row in sample_results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_passk_markdown(summary, summary_markdown_path)
    write_passk_csv(summary, summary_csv_path)
    print_passk_summary(summary)
    print(f"Saved generation samples to {generations_path}")
    print(f"Saved filtered-example report to {filtered_path}")
    print(f"Saved sample evaluation results to {sample_results_path}")
    print(f"Saved pass@k summary to {summary_path}")
    print(f"Saved pass@k markdown summary to {summary_markdown_path}")
    print(f"Saved pass@k CSV summary to {summary_csv_path}")


if __name__ == "__main__":
    main()