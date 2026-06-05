#!/usr/bin/env python3
"""Build self-consistency results from pass@k shard candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nl2sql_gspo.resume import atomic_write_json, atomic_write_jsonl
from scripts.run_inference_bird import write_summary_csv
from scripts.run_self_consistency_bird import (
    evaluate_candidates,
    write_self_consistency_markdown,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_candidates(shard_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for shard_dir in shard_dirs:
        candidates_path = shard_dir / "passk_candidates_raw.jsonl"
        if not candidates_path.exists():
            candidates_path = shard_dir / "passk_candidates_raw.incremental.jsonl"
        if not candidates_path.exists():
            raise FileNotFoundError(f"No pass@k candidate JSONL found in {shard_dir}")
        shard_rows = read_jsonl(candidates_path)
        print(f"[merge-sc] loaded {len(shard_rows)} candidates from {candidates_path}")
        rows.extend(shard_rows)
    return rows


def group_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = OrderedDict()
    seen: set[tuple[int, int]] = set()
    for candidate in sorted(candidates, key=lambda item: (int(item["idx"]), int(item["sample_id"]))):
        idx = int(candidate["idx"])
        sample_id = int(candidate["sample_id"])
        key = (idx, sample_id)
        if key in seen:
            raise ValueError(f"Duplicate candidate for idx={idx} sample_id={sample_id}")
        seen.add(key)
        if idx not in grouped:
            grouped[idx] = {
                "idx": idx,
                "db_id": candidate.get("db_id", ""),
                "gold_sql": candidate.get("gold_sql", ""),
                "difficulty": candidate.get("difficulty", "unknown"),
                "generations": [],
            }
        grouped[idx]["generations"].append(
            {
                "sample_idx": sample_id,
                "prediction_text": candidate.get("prediction_text", ""),
                "pred_sql": candidate.get("pred_sql", ""),
                "sample_plan_id": candidate.get("sample_plan_id"),
                "skill_id": candidate.get("skill_id"),
                "skill_name": candidate.get("skill_name", "default"),
                "temperature": candidate.get("temperature"),
                "top_p": candidate.get("top_p"),
                "replica_label": candidate.get("replica_label", ""),
                "prompt_tokens": candidate.get("prompt_tokens", 0),
                "completion_token_count": candidate.get("completion_token_count", 0),
                "tool_rounds": candidate.get("tool_rounds", 0),
                "tool_call_count": candidate.get("tool_call_count", 0),
                "stop_reason": candidate.get("stop_reason", ""),
                "generation_error": candidate.get("generation_error", ""),
            }
        )
    return list(grouped.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates([Path(path) for path in args.shard_dirs])
    prediction_rows = group_candidates(candidates)
    print(f"[merge-sc] grouped {len(candidates)} candidates into {len(prediction_rows)} examples")

    sample_results, selected_results, summary = evaluate_candidates(
        prediction_rows=prediction_rows,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        max_workers=args.eval_workers,
    )

    atomic_write_jsonl(output_dir / "prediction_samples.jsonl", prediction_rows)
    atomic_write_jsonl(
        output_dir / "eval_results_samples.jsonl",
        [{key: value for key, value in row.items() if key != "pred_rows"} for row in sample_results],
    )
    atomic_write_jsonl(output_dir / "self_consistency_results.jsonl", selected_results)
    atomic_write_json(output_dir / "self_consistency_summary.json", summary)
    write_self_consistency_markdown(summary, output_dir / "self_consistency_summary.md")
    write_summary_csv(summary["by_difficulty"], output_dir / "self_consistency_summary_by_difficulty.csv")
    write_summary_csv(summary["by_db"], output_dir / "self_consistency_summary_by_db.csv")

    total = summary["total"]
    print(
        f"[merge-sc] self-consistency EX accuracy: {total['accuracy']:.2f}% "
        f"({total['correct']}/{total['count']})"
    )


if __name__ == "__main__":
    main()
