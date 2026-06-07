#!/usr/bin/env python3
"""Run self-consistency over pass@k candidates only.

This script consumes ``passk_candidates.jsonl`` from ``run_passk_bird.py``.
It does not load a model and does not use any temperature-0 inference output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nl2sql_gspo.sql_utils import bird_execute_sql, bird_get_gold_rows, bird_result_match, extract_sql
from scripts.run_inference_bird import build_summary, write_summary_csv, write_summary_markdown
from scripts.run_passk_bird import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passk_candidates_path",
        required=True,
        help="Path to passk_candidates.jsonl written by scripts/run_passk_bird.py.",
    )
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--output_dir", type=str, default="outputs/self_consistency")
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def choose_majority_vote_candidate(
    candidate_results: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
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
    for candidate in sorted(valid_candidates, key=lambda item: int(item["sample_idx"])):
        signature = rows_to_vote_signature(candidate.get("pred_rows"))
        groups.setdefault(signature, []).append(candidate)

    winning_group = min(
        groups.values(),
        key=lambda items: (
            -len(items),
            min(int(item["sample_idx"]) for item in items),
            min(len(item["pred_sql"]) for item in items),
        ),
    )
    winner = min(
        winning_group,
        key=lambda item: (int(item["sample_idx"]), len(item["pred_sql"]), item["pred_sql"]),
    )
    return winner, {
        "num_candidates": len(candidate_results),
        "num_valid_votes": len(valid_candidates),
        "ignored_empty_results": ignored_empty_results,
        "winning_vote_count": len(winning_group),
    }


def normalize_passk_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    sample_idx = row.get("sample_idx", row.get("sample_id", 0))
    return {
        "idx": int(row.get("idx", row.get("source_idx", -1))),
        "sample_idx": int(sample_idx),
        "db_id": row.get("db_id", ""),
        "difficulty": row.get("difficulty", "unknown"),
        "prediction_text": row.get("prediction_text", ""),
        "pred_sql": extract_sql(row.get("pred_sql") or row.get("prediction_text", "")),
        "gold_sql": extract_sql(row.get("gold_sql", "")),
        "generation_error": row.get("generation_error", ""),
    }


def load_passk_candidates(path: Path) -> List[Dict[str, Any]]:
    candidates = [normalize_passk_candidate(row) for row in read_jsonl(path)]
    missing = [
        f"idx={row['idx']} sample={row['sample_idx']}"
        for row in candidates
        if not row["db_id"] or not row["gold_sql"]
    ]
    if missing:
        raise ValueError(
            "pass@k candidates are missing required db_id or gold_sql fields: "
            + "; ".join(missing[:5])
        )
    return candidates


def execute_candidates(
    candidates: List[Dict[str, Any]],
    database_dir: str,
    timeout_s: float,
    max_workers: int,
) -> List[Dict[str, Any]]:
    worker_count = max(1, min(max_workers, 32))

    def run(candidate: Dict[str, Any]) -> Dict[str, Any]:
        pred_rows: Optional[List[Tuple[Any, ...]]] = None
        pred_executed = False
        pred_error = ""
        predicted_sql = candidate["pred_sql"]
        if predicted_sql.strip():
            pred_executed, pred_rows, pred_error = bird_execute_sql(
                sql=predicted_sql,
                db_id=candidate["db_id"],
                database_dir=database_dir,
                timeout_s=timeout_s,
            )
        else:
            pred_error = "empty sql"

        gold_executed, gold_row_set, gold_error = bird_get_gold_rows(
            gold_sql=candidate["gold_sql"],
            db_id=candidate["db_id"],
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
            **candidate,
            "pred_rows": pred_rows,
            "pred_row_count": len(pred_rows) if pred_rows else 0,
            "res": int(matched),
            "status": status,
            "pred_executed": bool(pred_executed),
            "gold_executed": bool(gold_executed),
            "pred_error": pred_error,
            "gold_error": gold_error,
            "pred_sql_extracted": bool(predicted_sql.strip()),
            "gold_sql_extracted": bool(candidate["gold_sql"].strip()),
        }

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        total = len(candidates)
        for completed, result in enumerate(executor.map(run, candidates), start=1):
            results.append(result)
            if completed == 1 or completed == total or completed % 100 == 0:
                print(f"[evaluation] scored {completed}/{total} pass@k candidates")

    results.sort(key=lambda row: (int(row["idx"]), int(row["sample_idx"])))
    return results


def select_self_consistency_results(
    sample_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sample_results_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for result in sample_results:
        sample_results_by_idx.setdefault(int(result["idx"]), []).append(result)

    selected_results: List[Dict[str, Any]] = []
    voting_stats = {
        "examples_with_valid_vote": 0,
        "examples_without_valid_vote": 0,
        "ignored_empty_results": 0,
        "selected_vote_count_total": 0,
    }

    for row_index, (idx, candidate_results) in enumerate(sorted(sample_results_by_idx.items()), start=1):
        candidate_results = sorted(candidate_results, key=lambda item: int(item["sample_idx"]))
        base = candidate_results[0]
        winner, vote_meta = choose_majority_vote_candidate(candidate_results)
        voting_stats["ignored_empty_results"] += vote_meta["ignored_empty_results"]
        voting_stats["selected_vote_count_total"] += vote_meta["winning_vote_count"]

        if winner is None:
            selected_results.append(
                {
                    "idx": idx,
                    "db_id": base["db_id"],
                    "difficulty": base.get("difficulty", "unknown"),
                    "pred_sql": "",
                    "gold_sql": base["gold_sql"],
                    "res": 0,
                    "status": "no valid non-empty vote candidates",
                    "pred_executed": False,
                    "gold_executed": any(item["gold_executed"] for item in candidate_results),
                    "pred_error": "no valid non-empty vote candidates",
                    "gold_error": next((item["gold_error"] for item in candidate_results if item["gold_error"]), ""),
                    "pred_sql_extracted": False,
                    "gold_sql_extracted": bool(base["gold_sql"].strip()),
                    "selected_sample_idx": None,
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
                    "difficulty": winner.get("difficulty", "unknown"),
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
                    "selected_vote_count": vote_meta["winning_vote_count"],
                    "valid_vote_candidates": vote_meta["num_valid_votes"],
                    "ignored_empty_results": vote_meta["ignored_empty_results"],
                }
            )
            voting_stats["examples_with_valid_vote"] += 1

        if row_index == 1 or row_index == len(sample_results_by_idx) or row_index % 100 == 0:
            print(f"[selection] selected {row_index}/{len(sample_results_by_idx)} majority-vote predictions")

    return selected_results, voting_stats


def write_self_consistency_markdown(summary: Dict[str, Any], markdown_path: Path) -> None:
    write_summary_markdown(summary, markdown_path)
    details = summary["self_consistency"]
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Self-Consistency Stats\n\n")
        handle.write(f"- examples_with_valid_vote: {details['examples_with_valid_vote']}\n")
        handle.write(f"- examples_without_valid_vote: {details['examples_without_valid_vote']}\n")
        handle.write(f"- ignored_empty_results: {details['ignored_empty_results']}\n")
        handle.write(f"- selected_vote_count_total: {details['selected_vote_count_total']}\n")
        handle.write(f"- selection_rule: {details['selection_rule']}\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    candidates_path = Path(args.passk_candidates_path)
    candidates = load_passk_candidates(candidates_path)
    print(f"[run] loaded {len(candidates)} pass@k candidates from {candidates_path}")

    sample_results = execute_candidates(
        candidates=candidates,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        max_workers=args.eval_workers,
    )
    selected_results, voting_stats = select_self_consistency_results(sample_results)
    summary = build_summary(selected_results)
    summary["self_consistency"] = {
        **voting_stats,
        "source": str(candidates_path),
        "selection_rule": (
            "majority vote over executable non-empty pass@k result sets; "
            "ties break by earliest sample idx then shorter SQL"
        ),
    }

    sample_results_path = output_dir / "eval_results_samples.jsonl"
    selected_results_path = output_dir / "self_consistency_results.jsonl"
    summary_path = output_dir / "self_consistency_summary.json"
    summary_markdown_path = output_dir / "self_consistency_summary.md"
    difficulty_csv_path = output_dir / "self_consistency_summary_by_difficulty.csv"
    db_csv_path = output_dir / "self_consistency_summary_by_db.csv"

    serializable_samples = []
    for row in sample_results:
        serializable = dict(row)
        serializable.pop("pred_rows", None)
        serializable_samples.append(serializable)
    write_jsonl(sample_results_path, serializable_samples)
    write_jsonl(selected_results_path, selected_results)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_self_consistency_markdown(summary, summary_markdown_path)
    write_summary_csv(summary["by_difficulty"], difficulty_csv_path)
    write_summary_csv(summary["by_db"], db_csv_path)

    total = summary["total"]
    print(
        f"[summary] self-consistency EX accuracy: {total['accuracy']:.2f}% "
        f"({total['correct']}/{total['count']})"
    )
    print(f"Saved sample evaluation results to {sample_results_path}")
    print(f"Saved self-consistency results to {selected_results_path}")
    print(f"Saved self-consistency summary to {summary_path}")
    print(f"Saved self-consistency markdown summary to {summary_markdown_path}")
    print(f"Saved self-consistency difficulty CSV to {difficulty_csv_path}")
    print(f"Saved self-consistency DB CSV to {db_csv_path}")


if __name__ == "__main__":
    main()
