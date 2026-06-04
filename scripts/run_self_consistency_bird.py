import argparse
import asyncio
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.run_inference_bird import build_summary, write_summary_csv
from scripts.run_passk_bird import (
    PASSK_MANIFEST_FIELDS,
    ensure_output_dir,
    generate_candidates,
)
from scripts.run_inference_bird import load_diff_rows, load_rows
from nl2sql_gspo.inference_tool_executor import configure_tool_env
from nl2sql_gspo.prompt_builder import PromptBuilder, add_prompt_args, prompt_config_from_args
from nl2sql_gspo.resume import add_resume_args, atomic_write_json, atomic_write_jsonl, build_manifest, prepare_manifest, validate_resume_args
from nl2sql_gspo.sql_utils import bird_execute_sql, bird_get_gold_rows, bird_result_match

SELF_CONSISTENCY_MANIFEST_FIELDS = PASSK_MANIFEST_FIELDS + ["vllm_data_parallel_size"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="outputs/old-dev-schema.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/bird_dev.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_self_consistency")
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=30000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=8)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    parser.add_argument("--vllm_data_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    add_resume_args(parser)
    add_prompt_args(parser)
    args = parser.parse_args()
    validate_resume_args(args)
    return args


def print_configuration(args: argparse.Namespace, output_dir: Path) -> None:
    print("[run] starting self-consistency evaluation")
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
    print(f"[run] vllm_async_concurrency={args.vllm_async_concurrency}")
    print(f"[run] vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
    print(f"[run] vllm_max_model_len={args.vllm_max_model_len}")


def align_rows_with_raw_references(
    rows: List[Dict[str, Any]],
    diff_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_by_qid = {
        row.get("question_id"): row
        for row in diff_rows
        if row.get("question_id") is not None
    }
    aligned = []
    for row_index, row in enumerate(rows):
        updated = dict(row)
        raw = raw_by_qid.get(row.get("question_id"))
        if raw is None and row_index < len(diff_rows):
            raw = diff_rows[row_index]
        if raw:
            updated["difficulty"] = raw.get("difficulty", updated.get("difficulty", "unknown"))
        else:
            updated["difficulty"] = updated.get("difficulty", "unknown")
        aligned.append(updated)
    return aligned


def generate_predictions_with_vllm_n(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    prompt_builder: PromptBuilder,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = asyncio.run(generate_candidates(args, rows, prompt_builder=prompt_builder))
    grouped: Dict[int, Dict[str, Any]] = OrderedDict()
    for candidate in sorted(candidates, key=lambda item: (int(item["idx"]), int(item["sample_id"]))):
        idx = int(candidate["idx"])
        if idx not in grouped:
            grouped[idx] = {
                "idx": idx,
                "db_id": candidate.get("db_id", ""),
                "gold_sql": next(
                    (row.get("gold_sql", "") for row in rows if int(row.get("source_idx", -1)) == idx),
                    "",
                ),
                "difficulty": next(
                    (row.get("difficulty", "unknown") for row in rows if int(row.get("source_idx", -1)) == idx),
                    "unknown",
                ),
                "generations": [],
            }
        grouped[idx]["generations"].append(
            {
                "sample_idx": int(candidate["sample_id"]),
                "prediction_text": candidate.get("prediction_text", ""),
                "pred_sql": candidate.get("pred_sql", ""),
                "skill_id": candidate.get("skill_id"),
                "skill_name": candidate.get("skill_name", "default"),
                "prompt_tokens": candidate.get("prompt_tokens", 0),
                "completion_token_count": candidate.get("completion_token_count", 0),
                "tool_rounds": candidate.get("tool_rounds", 0),
                "tool_call_count": candidate.get("tool_call_count", 0),
                "stop_reason": candidate.get("stop_reason", ""),
                "generation_error": candidate.get("generation_error", ""),
            }
        )
    return list(grouped.values()), []


def rows_to_vote_signature(rows: Optional[List[Tuple[Any, ...]]]) -> frozenset:
    if not rows:
        return frozenset()

    hashable_rows: List[Tuple[Any, ...]] = []
    for row in rows:
        try:
            hashable_rows.append(tuple(row))
        except Exception:
            hashable_rows.append((repr(row),))

    try:
        return frozenset(hashable_rows)
    except TypeError:
        return frozenset(tuple(repr(cell) for cell in row) for row in hashable_rows)


def is_nonempty_execution_result(rows: Optional[List[Tuple[Any, ...]]]) -> bool:
    return bool(rows)


def choose_majority_vote_candidate(candidate_results: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    valid_candidates = [
        candidate
        for candidate in candidate_results
        if candidate["pred_executed"] and is_nonempty_execution_result(candidate.get("pred_rows"))
    ]
    ignored_empty_results = sum(
        int(candidate["pred_executed"] and not is_nonempty_execution_result(candidate.get("pred_rows")))
        for candidate in candidate_results
    )

    if not valid_candidates:
        return None, {
            "num_candidates": len(candidate_results),
            "num_valid_votes": 0,
            "ignored_empty_results": ignored_empty_results,
            "winning_vote_count": 0,
        }

    groups: Dict[frozenset, List[Dict[str, Any]]] = OrderedDict()
    for candidate in valid_candidates:
        signature = rows_to_vote_signature(candidate.get("pred_rows"))
        groups.setdefault(signature, []).append(candidate)

    winning_group = min(
        groups.values(),
        key=lambda items: (
            -len(items),
            min(item["sample_idx"] for item in items),
            min(len(item["pred_sql"]) for item in items),
        ),
    )
    winner = min(
        winning_group,
        key=lambda item: (item["sample_idx"], len(item["pred_sql"]), item["pred_sql"]),
    )
    return winner, {
        "num_candidates": len(candidate_results),
        "num_valid_votes": len(valid_candidates),
        "ignored_empty_results": ignored_empty_results,
        "winning_vote_count": len(winning_group),
    }


def should_log_progress_tick(completed: int, total: int) -> bool:
    if completed == 1 or completed == total:
        return True
    if total <= 100:
        return completed % 10 == 0
    return completed % 100 == 0


def evaluate_candidates(
    prediction_rows: List[Dict[str, Any]],
    database_dir: str,
    timeout_s: float,
    max_workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sample_results: List[Dict[str, Any]] = []
    selected_results: List[Dict[str, Any]] = []
    worker_count = max(1, min(max_workers, 32))

    prepared_jobs: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for row in prediction_rows:
        for generation in row["generations"]:
            prepared_jobs.append((row["idx"], row, generation))

    def _run_job(job: Tuple[int, Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        source_idx, row, generation = job
        pred_rows: Optional[List[Tuple[Any, ...]]] = None
        pred_executed = False
        pred_error = ""
        predicted_sql = generation["pred_sql"]
        if predicted_sql.strip():
            pred_executed, pred_rows, pred_error = bird_execute_sql(
                sql=predicted_sql,
                db_id=row["db_id"],
                database_dir=database_dir,
                timeout_s=timeout_s,
            )

        gold_executed, gold_row_set, gold_error = bird_get_gold_rows(
            gold_sql=row["gold_sql"],
            db_id=row["db_id"],
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
            "idx": source_idx,
            "db_id": row["db_id"],
            "difficulty": row.get("difficulty", "unknown"),
            "sample_idx": generation["sample_idx"],
            "skill_id": generation.get("skill_id"),
            "skill_name": generation.get("skill_name", "default"),
            "pred_sql": predicted_sql,
            "gold_sql": row["gold_sql"],
            "pred_rows": pred_rows,
            "pred_row_count": len(pred_rows) if pred_rows else 0,
            "res": int(matched),
            "status": status,
            "pred_executed": bool(pred_executed),
            "gold_executed": bool(gold_executed),
            "pred_error": pred_error,
            "gold_error": gold_error,
            "pred_sql_extracted": bool(predicted_sql.strip()),
            "gold_sql_extracted": bool(row["gold_sql"].strip()),
        }

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        total_jobs = len(prepared_jobs)
        for job_index, result in enumerate(executor.map(_run_job, prepared_jobs), start=1):
            sample_results.append(result)
            if should_log_progress_tick(job_index, total_jobs):
                print(f"[evaluation] scored {job_index}/{total_jobs} candidate generations")

    sample_results_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for result in sample_results:
        sample_results_by_idx.setdefault(result["idx"], []).append(result)

    voting_stats = {
        "examples_with_valid_vote": 0,
        "examples_without_valid_vote": 0,
        "ignored_empty_results": 0,
        "selected_vote_count_total": 0,
    }

    for row_index, row in enumerate(prediction_rows, start=1):
        candidate_results = sorted(sample_results_by_idx[row["idx"]], key=lambda item: item["sample_idx"])
        winner, vote_meta = choose_majority_vote_candidate(candidate_results)
        voting_stats["ignored_empty_results"] += vote_meta["ignored_empty_results"]
        voting_stats["selected_vote_count_total"] += vote_meta["winning_vote_count"]

        if winner is None:
            selected_results.append(
                {
                    "idx": row["idx"],
                    "db_id": row["db_id"],
                    "difficulty": row.get("difficulty", "unknown"),
                    "pred_sql": "",
                    "gold_sql": row["gold_sql"],
                    "res": 0,
                    "status": "no valid non-empty vote candidates",
                    "pred_executed": False,
                    "gold_executed": any(item["gold_executed"] for item in candidate_results),
                    "pred_error": "no valid non-empty vote candidates",
                    "gold_error": next((item["gold_error"] for item in candidate_results if item["gold_error"]), ""),
                    "pred_sql_extracted": False,
                    "gold_sql_extracted": bool(row["gold_sql"].strip()),
                    "selected_sample_idx": None,
                    "selected_skill_id": None,
                    "selected_skill_name": "",
                    "selected_vote_count": 0,
                    "valid_vote_candidates": 0,
                    "ignored_empty_results": vote_meta["ignored_empty_results"],
                }
            )
            voting_stats["examples_without_valid_vote"] += 1
        else:
            selected_results.append(
                {
                    "idx": winner["idx"],
                    "db_id": winner["db_id"],
                    "difficulty": winner["difficulty"],
                    "pred_sql": winner["pred_sql"],
                    "gold_sql": winner["gold_sql"],
                    "res": winner["res"],
                    "status": winner["status"],
                    "pred_executed": winner["pred_executed"],
                    "gold_executed": winner["gold_executed"],
                    "pred_error": winner["pred_error"],
                    "gold_error": winner["gold_error"],
                    "pred_sql_extracted": winner["pred_sql_extracted"],
                    "gold_sql_extracted": winner["gold_sql_extracted"],
                    "selected_sample_idx": winner["sample_idx"],
                    "selected_skill_id": winner.get("skill_id"),
                    "selected_skill_name": winner.get("skill_name", "default"),
                    "selected_vote_count": vote_meta["winning_vote_count"],
                    "valid_vote_candidates": vote_meta["num_valid_votes"],
                    "ignored_empty_results": vote_meta["ignored_empty_results"],
                }
            )
            voting_stats["examples_with_valid_vote"] += 1

        if should_log_progress_tick(row_index, len(prediction_rows)):
            print(f"[selection] selected {row_index}/{len(prediction_rows)} majority-vote predictions")

    summary = build_summary(selected_results)
    summary["self_consistency"] = {
        **voting_stats,
        "num_generations": prediction_rows[0]["generations"][-1]["sample_idx"] + 1 if prediction_rows else 0,
        "skill_header_counts": {
            name: sum(
                1
                for row in prediction_rows
                for generation in row["generations"]
                if generation.get("skill_name", "default") == name
            )
            for name in sorted(
                {
                    generation.get("skill_name", "default")
                    for row in prediction_rows
                    for generation in row["generations"]
                }
            )
        },
        "selection_rule": "majority vote over executable non-empty result sets; ties break by earliest sample idx then shorter SQL",
    }
    return sample_results, selected_results, summary


def write_self_consistency_markdown(summary: Dict[str, Any], markdown_path: Path) -> None:
    details = summary["self_consistency"]
    with markdown_path.open("w", encoding="utf-8") as handle:
        total = summary["total"]
        handle.write("# BIRD Self-Consistency Summary\n\n")
        handle.write(
            f"Overall EX Accuracy: {total['accuracy']:.2f}% "
            f"({total['correct']}/{total['count']})\n\n"
        )
        handle.write("## By Difficulty\n\n")
        handle.write("| Group | Correct | Count | Accuracy |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for group, values in summary["by_difficulty"].items():
            handle.write(
                f"| {group} | {values['correct']} | {values['count']} | "
                f"{values['accuracy']:.2f} |\n"
            )
        handle.write("\n## Self-Consistency Stats\n\n")
        handle.write(f"- examples_with_valid_vote: {details['examples_with_valid_vote']}\n")
        handle.write(f"- examples_without_valid_vote: {details['examples_without_valid_vote']}\n")
        handle.write(f"- ignored_empty_results: {details['ignored_empty_results']}\n")
        handle.write(f"- selected_vote_count_total: {details['selected_vote_count_total']}\n")
        handle.write(f"- num_generations: {details['num_generations']}\n")
        handle.write(f"- selection_rule: {details['selection_rule']}\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite, args.resume)

    prompt_builder = PromptBuilder(prompt_config_from_args(args))
    effective_input_file = args.raw_input_file or args.input_file
    args.input_file = effective_input_file
    args._candidate_checkpoint_path = str(output_dir / "passk_candidates_raw.incremental.jsonl")
    manifest = build_manifest(args, mode="self_consistency", fields=SELF_CONSISTENCY_MANIFEST_FIELDS)
    prepare_manifest(output_dir, manifest, resume=args.resume)
    print_configuration(args, output_dir)

    force_prompt_rebuild = bool(args.build_prompts_at_runtime or args.raw_input_file)
    rows = load_rows(
        effective_input_file,
        args.num_examples,
        require_gold_sql=args.bird_mode == "dev",
        prompt_builder=prompt_builder if force_prompt_rebuild else None,
        force_prompt_rebuild=force_prompt_rebuild,
    )
    print(f"[run] loaded {len(rows)} input rows")
    diff_rows = load_diff_rows(args.diff_json_path)
    print(f"[run] loaded {len(diff_rows)} diff rows")
    rows = align_rows_with_raw_references(rows, diff_rows)
    configure_tool_env(args.database_dir)

    prediction_rows, skipped_rows = generate_predictions_with_vllm_n(rows, args, prompt_builder)
    if skipped_rows:
        raise RuntimeError(
            f"Found {len(skipped_rows)} prompts exceeding max_prompt_length={args.max_prompt_length}; rerun with a larger context budget."
        )

    sample_results, selected_results, summary = evaluate_candidates(
        prediction_rows=prediction_rows,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        max_workers=args.eval_workers,
    )

    generations_path = output_dir / "prediction_samples.jsonl"
    filtered_path = output_dir / "filtered_examples.jsonl"
    sample_results_path = output_dir / "eval_results_samples.jsonl"
    selected_results_path = output_dir / "self_consistency_results.jsonl"
    summary_path = output_dir / "self_consistency_summary.json"
    summary_markdown_path = output_dir / "self_consistency_summary.md"
    difficulty_csv_path = output_dir / "self_consistency_summary_by_difficulty.csv"
    db_csv_path = output_dir / "self_consistency_summary_by_db.csv"

    atomic_write_jsonl(generations_path, prediction_rows)
    atomic_write_jsonl(filtered_path, skipped_rows)
    atomic_write_jsonl(
        sample_results_path,
        [{key: value for key, value in row.items() if key != "pred_rows"} for row in sample_results],
    )
    atomic_write_jsonl(selected_results_path, selected_results)
    atomic_write_json(summary_path, summary)

    write_self_consistency_markdown(summary, summary_markdown_path)
    write_summary_csv(summary["by_difficulty"], difficulty_csv_path)
    write_summary_csv(summary["by_db"], db_csv_path)

    total = summary["total"]
    print(
        f"[summary] self-consistency EX accuracy: {total['accuracy']:.2f}% "
        f"({total['correct']}/{total['count']})"
    )
    print(f"Saved generation samples to {generations_path}")
    print(f"Saved filtered-example report to {filtered_path}")
    print(f"Saved sample evaluation results to {sample_results_path}")
    print(f"Saved self-consistency results to {selected_results_path}")
    print(f"Saved self-consistency summary to {summary_path}")
    print(f"Saved self-consistency markdown summary to {summary_markdown_path}")
    print(f"Saved self-consistency difficulty CSV to {difficulty_csv_path}")
    print(f"Saved self-consistency DB CSV to {db_csv_path}")


if __name__ == "__main__":
    main()
