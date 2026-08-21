#!/usr/bin/env python3
"""Standalone BIRD execution-accuracy (EX) scorer.

Scores a predictions file against gold SQL by executing both and comparing
result sets, using the same helpers the pipeline itself uses
(`nl2sql_gspo.sql_utils.bird_execute_sql` / `bird_result_match`), so a number
produced here is directly comparable to the pipeline's own eval output.

Semantics match the official BIRD evaluator:
  - set equality over RAW rows: `set(pred) == set(gold)` -- order- and
    duplicate-insensitive, no value normalization
  - one fresh read-only connection per query
  - per-query wallclock cap, default 30s, matching BIRD's `--meta_time_out`
  - a query that times out, errors, or is empty scores as wrong

Note the two different timeouts in this repository. This scorer's
`--meta_time_out` (30s) applies to the GRADED query only. Tool calls made by
the model DURING generation use a separate, more generous 60s budget
(`TOOL_TIMEOUT` in scripts/run_bird_test_pipeline.sh) -- that is the model
exploring the database, not the query being scored. Do not conflate them.

Usage:
  python scripts/eval_bird_ex.py \
    --predictions outputs/<run>/self_consistency/predict_test.json \
    --gold data/bird_dev_data/raw/bird_dev.json \
    --database_dir databases/dev_databases

Accepted prediction formats (auto-detected):
  BIRD submission : {"0": "SELECT ...\t----- bird -----\tdb_id", ...}
  plain SQL map   : {"0": "SELECT ...", ...}
  JSONL           : one {"idx"/"question_id", "pred_sql"/"predicted_sql"} per line
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from nl2sql_gspo.sql_utils import (  # noqa: E402
    bird_execute_sql,
    bird_get_gold_rows,
    bird_result_match,
)

BIRD_SEPARATOR = "\t----- bird -----\t"


def load_predictions(path: Path) -> Dict[int, str]:
    """Return {question_id: predicted_sql}, tolerating the three known layouts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"[eval] predictions file is empty: {path}")

    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        blob = json.loads(text)
        if isinstance(blob, list):
            return {
                int(row.get("question_id", position)): str(
                    row.get("pred_sql") or row.get("predicted_sql") or ""
                )
                for position, row in enumerate(blob)
            }
        out: Dict[int, str] = {}
        for key, value in blob.items():
            sql = str(value)
            # BIRD submission format packs the db_id after a separator.
            if BIRD_SEPARATOR in sql:
                sql = sql.split(BIRD_SEPARATOR)[0]
            out[int(key)] = sql
        return out

    out = {}
    for position, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        idx = int(row.get("idx", row.get("question_id", position)))
        out[idx] = str(row.get("pred_sql") or row.get("predicted_sql") or "")
    return out


def load_gold(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"[eval] gold file must be a JSON list: {path}")
    missing = sum(1 for row in rows if not str(row.get("SQL", "")).strip())
    if missing:
        raise SystemExit(
            f"[eval] {missing}/{len(rows)} gold rows have no SQL. This looks like a "
            f"test-split file with labels stripped; EX cannot be computed from it."
        )
    return rows


def score_one(
    task: Tuple[int, Dict[str, Any], str, str, float]
) -> Dict[str, Any]:
    idx, gold_row, predicted_sql, database_dir, timeout_s = task
    db_id = gold_row["db_id"]

    gold_executed, gold_row_set, gold_error = bird_get_gold_rows(
        gold_sql=gold_row["SQL"],
        db_id=db_id,
        database_dir=database_dir,
        timeout_s=timeout_s,
    )

    pred_rows: Optional[List[Tuple[Any, ...]]] = None
    pred_executed = False
    pred_error = ""
    if predicted_sql.strip():
        pred_executed, pred_rows, pred_error = bird_execute_sql(
            sql=predicted_sql,
            db_id=db_id,
            database_dir=database_dir,
            timeout_s=timeout_s,
        )
    else:
        pred_error = "empty sql"

    correct = bool(
        pred_executed and gold_executed and bird_result_match(pred_rows, gold_row_set)
    )
    return {
        "idx": idx,
        "db_id": db_id,
        "difficulty": gold_row.get("difficulty", "unknown"),
        "correct": int(correct),
        "pred_executed": bool(pred_executed),
        "gold_executed": bool(gold_executed),
        "pred_error": pred_error,
        "gold_error": gold_error,
    }


def breakdown(results: List[Dict[str, Any]], field: str) -> List[Tuple[str, int, int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for row in results:
        buckets[str(row[field])].append(row["correct"])
    ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    return [(name, sum(vals), len(vals)) for name, vals in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="Predictions file to score.")
    parser.add_argument("--gold", required=True, help="Gold JSON with a populated 'SQL' field per row.")
    parser.add_argument("--database_dir", required=True, help="Directory of SQLite databases.")
    parser.add_argument(
        "--meta_time_out",
        type=float,
        default=30.0,
        help="Per-query wallclock cap in seconds for the GRADED query (BIRD default 30).",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel scoring threads.")
    parser.add_argument("--output_json", default="", help="Optional path to write per-question results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_rows = load_gold(Path(args.gold))
    predictions = load_predictions(Path(args.predictions))

    # Score every gold row. A question with no prediction is wrong, not skipped:
    # silently dropping it would inflate accuracy on a short predictions file.
    missing = [i for i in range(len(gold_rows)) if i not in predictions]
    tasks = [
        (i, gold_rows[i], predictions.get(i, ""), args.database_dir, args.meta_time_out)
        for i in range(len(gold_rows))
    ]

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 32))) as pool:
        results = list(pool.map(score_one, tasks))

    total = len(results)
    correct = sum(row["correct"] for row in results)
    gold_failed = sum(1 for row in results if not row["gold_executed"])
    pred_failed = sum(1 for row in results if not row["pred_executed"])

    print(f"\npredictions : {args.predictions}")
    print(f"gold        : {args.gold}")
    print(f"databases   : {args.database_dir}")
    print(f"meta_time_out: {args.meta_time_out}s (graded query only; tool calls during")
    print(f"               generation use the separate 60s TOOL_TIMEOUT budget)")
    if missing:
        print(f"\nWARNING: {len(missing)} gold question(s) had no prediction and scored "
              f"as wrong; first={missing[:10]}")
    if gold_failed:
        print(f"WARNING: {gold_failed} gold quer(ies) failed to execute.")

    for label, field in (("By Difficulty", "difficulty"), ("By Database", "db_id")):
        rows = breakdown(results, field)
        if len(rows) <= 1 and rows and rows[0][0] == "unknown":
            continue
        print(f"\n## {label}")
        print(f"{'group':<28} {'correct':>8} {'count':>7} {'acc':>8}")
        for name, ok, n in rows:
            print(f"{name:<28} {ok:>8} {n:>7} {100.0 * ok / n:>7.2f}%")

    print(f"\npredictions that failed to execute: {pred_failed}/{total}")
    print(f"\nOverall EX Accuracy: {100.0 * correct / total:.2f}% ({correct}/{total})")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "overall": {"correct": correct, "total": total,
                                "accuracy": round(100.0 * correct / total, 4)},
                    "meta_time_out": args.meta_time_out,
                    "missing_predictions": missing,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote per-question results to {out_path}")


if __name__ == "__main__":
    main()
