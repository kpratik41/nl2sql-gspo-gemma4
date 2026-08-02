#!/usr/bin/env python3
"""Stage A1.3 — build the Stage A2 target id lists from a pass@16 run.

Splits the training set into the three pass@k bands and writes two id files:

* ``target_idx_all_wrong.txt``  — the teacher band. Examples where all
  ``num_generations`` candidates failed, minus examples whose *gold* SQL does
  not execute (verification against gold could never pass, so they are not
  teachable). Stage A2 runs these with ``--hint_strategy full_sql``.

* ``target_idx_selftrace.txt``  — the anchor band. All "sometimes" examples
  plus a per-database capped sample of the "all-correct" examples, so that
  easy over-represented databases do not dominate. Stage A2 runs these with
  ``--hint_strategy none``.

Both files are one ``source_idx`` per line, ascending. A JSON summary records
the counts and exactly what the cap removed.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--passk-dir",
        required=True,
        help="Merged pass@k directory containing passk_per_example.jsonl.",
    )
    parser.add_argument(
        "--analysis-jsonl",
        default=None,
        help=(
            "all_wrong_analysis.jsonl from scripts/analyze_passk_all_wrong.py. "
            "Defaults to <passk-dir>/all_wrong_analysis.jsonl. Used to drop "
            "gold_sql_execution_failed ids from the teacher band."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/teacher",
        help="Directory for the generated id files and summary.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Candidates per example in the pass@k run; defines the all-correct band.",
    )
    parser.add_argument(
        "--all-correct-cap-per-db",
        type=int,
        default=12,
        help="Max all-correct ids kept per database. Use -1 to disable the cap.",
    )
    parser.add_argument(
        "--keep-gold-failures",
        action="store_true",
        help="Keep gold_sql_execution_failed ids in the teacher band (default drops them).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the per-db all-correct sample.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_idx_file(path: Path, idxs: List[int]) -> None:
    path.write_text("".join(f"{i}\n" for i in sorted(idxs)), encoding="utf-8")


def main() -> None:
    args = parse_args()
    passk_dir = Path(args.passk_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_example = load_jsonl(passk_dir / "passk_per_example.jsonl")
    n_gen = args.num_generations

    all_wrong: List[int] = []
    sometimes: List[int] = []
    all_correct: List[int] = []
    db_by_idx: Dict[int, str] = {}

    for row in per_example:
        idx = int(row["idx"])
        correct = int(row.get("num_correct", 0))
        db_by_idx[idx] = row.get("db_id", "")
        if correct == 0:
            all_wrong.append(idx)
        elif correct >= n_gen:
            all_correct.append(idx)
        else:
            sometimes.append(idx)

    # Teacher band: drop examples whose gold SQL does not execute.
    analysis_path = Path(args.analysis_jsonl) if args.analysis_jsonl else passk_dir / "all_wrong_analysis.jsonl"
    gold_failed: List[int] = []
    if analysis_path.exists():
        for row in load_jsonl(analysis_path):
            if "gold_sql_execution_failed" in (row.get("failure_labels") or []):
                gold_failed.append(int(row["idx"]))
    else:
        print(f"[warn] {analysis_path} not found; teacher band keeps all all-wrong ids")

    gold_failed_set = set(gold_failed)
    teacher_idxs = (
        list(all_wrong) if args.keep_gold_failures else [i for i in all_wrong if i not in gold_failed_set]
    )

    # Anchor band: all "sometimes" ids, plus per-db capped all-correct ids.
    rng = random.Random(args.seed)
    by_db: Dict[str, List[int]] = collections.defaultdict(list)
    for idx in all_correct:
        by_db[db_by_idx.get(idx, "")].append(idx)

    kept_all_correct: List[int] = []
    cap = args.all_correct_cap_per_db
    for db in sorted(by_db):
        ids = sorted(by_db[db])
        if cap is not None and cap >= 0 and len(ids) > cap:
            ids = sorted(rng.sample(ids, cap))
        kept_all_correct.extend(ids)

    selftrace_idxs = sorted(set(sometimes) | set(kept_all_correct))

    teacher_path = out_dir / "target_idx_all_wrong.txt"
    selftrace_path = out_dir / "target_idx_selftrace.txt"
    write_idx_file(teacher_path, teacher_idxs)
    write_idx_file(selftrace_path, selftrace_idxs)

    summary = {
        "passk_dir": str(passk_dir),
        "num_generations": n_gen,
        "total_examples": len(per_example),
        "bands": {
            "all_wrong": len(all_wrong),
            "sometimes": len(sometimes),
            "all_correct": len(all_correct),
        },
        "teacher_band": {
            "file": str(teacher_path),
            "count": len(teacher_idxs),
            "hint_strategy": "full_sql",
            "gold_sql_execution_failed_dropped": 0 if args.keep_gold_failures else len(gold_failed),
            "gold_sql_execution_failed_idxs": sorted(gold_failed),
        },
        "selftrace_band": {
            "file": str(selftrace_path),
            "count": len(selftrace_idxs),
            "hint_strategy": "none",
            "sometimes_kept": len(sometimes),
            "all_correct_kept": len(kept_all_correct),
            "all_correct_dropped_by_cap": len(all_correct) - len(kept_all_correct),
            "all_correct_cap_per_db": cap,
            "databases": len(by_db),
            "seed": args.seed,
        },
    }
    summary_path = out_dir / "target_idx_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"total examples      : {len(per_example)}")
    print(f"  all-wrong         : {len(all_wrong)}")
    print(f"  sometimes         : {len(sometimes)}")
    print(f"  all-correct       : {len(all_correct)}")
    print()
    print(f"teacher band (full_sql) : {len(teacher_idxs)}  -> {teacher_path}")
    if not args.keep_gold_failures:
        print(f"  dropped gold-failures : {len(gold_failed)}")
    print(f"selftrace band (none)   : {len(selftrace_idxs)}  -> {selftrace_path}")
    print(f"  sometimes             : {len(sometimes)}")
    print(f"  all-correct kept      : {len(kept_all_correct)} (cap {cap}/db over {len(by_db)} dbs)")
    print(f"  all-correct dropped   : {len(all_correct) - len(kept_all_correct)}")
    print()
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
