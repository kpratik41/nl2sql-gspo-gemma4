#!/usr/bin/env python3
"""Sample a database-stratified subset of the RL training set.

Why not train on all 6573
-------------------------
Rows are distributed very unevenly across the 69 BIRD train databases -- 383 for
works_cycles down to 5 for craftbeer. Sampling uniformly at random would let a
handful of large schemas dominate the gradient, so this allocates proportionally
to sqrt(n) instead: large databases stay largest, but their share is compressed
and small ones keep enough rows to contribute signal.

What this does NOT do
---------------------
It does not prune trivially-solved prompts. That is the higher-value filter --
SIRIUS-SQL concentrates training compute on prompts where the policy is still
uncertain, and on BIRD dev 26.1% of examples were solved by all 16 samples,
contributing zero advantage under a group-relative objective. Doing it properly
needs a pass@k run over the *train* split to measure per-example difficulty,
which this repo does not have yet. Pass --difficulty-json once it does.

Usage
-----
    python scripts/data_generation/build_train_subset.py \\
        --input outputs/qwen-train-6573-schema-bare-tool.jsonl \\
        --output outputs/qwen-train-3000-schema-bare-tool.jsonl \\
        --size 3000
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-per-db",
        type=int,
        default=5,
        help="Floor per database, so small schemas are not sampled out entirely.",
    )
    parser.add_argument(
        "--difficulty-json",
        default=None,
        help=(
            "Optional {idx: num_correct_out_of_k} map. When given, rows solved by "
            "every sample are dropped first -- they carry no group-relative advantage."
        ),
    )
    parser.add_argument(
        "--drop-if-solved-at-least",
        type=int,
        default=None,
        help="With --difficulty-json, drop rows whose num_correct is >= this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = [
        json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()
    ]
    print(f"[subset] input {len(rows)} rows")

    if args.difficulty_json and args.drop_if_solved_at_least is not None:
        difficulty = {int(k): int(v) for k, v in json.load(open(args.difficulty_json)).items()}
        before = len(rows)
        rows = [
            row
            for index, row in enumerate(rows)
            if difficulty.get(index, 0) < args.drop_if_solved_at_least
        ]
        print(f"[subset] difficulty filter dropped {before - len(rows)} trivially-solved rows")

    by_db: Dict[str, List[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_db[row.get("db_id", "")].append(index)

    # sqrt allocation: compresses the spread between the biggest and smallest
    # schemas without flattening it entirely.
    weights = {db: math.sqrt(len(idxs)) for db, idxs in by_db.items()}
    total_weight = sum(weights.values())
    quota = {
        db: max(args.min_per_db, round(args.size * weights[db] / total_weight))
        for db in by_db
    }
    for db, idxs in by_db.items():
        quota[db] = min(quota[db], len(idxs))

    # Trim or top up to land on --size exactly, largest databases absorbing it.
    order = sorted(by_db, key=lambda d: -len(by_db[d]))
    while sum(quota.values()) > args.size:
        for db in order:
            if sum(quota.values()) <= args.size:
                break
            if quota[db] > args.min_per_db:
                quota[db] -= 1
    while sum(quota.values()) < args.size:
        for db in order:
            if sum(quota.values()) >= args.size:
                break
            if quota[db] < len(by_db[db]):
                quota[db] += 1

    rng = random.Random(args.seed)
    keep: List[int] = []
    for db, idxs in by_db.items():
        keep.extend(rng.sample(idxs, quota[db]))
    keep.sort()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index in keep:
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")

    counts = collections.Counter(rows[i]["db_id"] for i in keep)
    print(f"[subset] wrote {len(keep)} rows across {len(counts)} databases -> {out_path}")
    print(f"[subset] per-db min={min(counts.values())} median={sorted(counts.values())[len(counts)//2]} max={max(counts.values())}")
    print(f"[subset] largest: {counts.most_common(5)}")


if __name__ == "__main__":
    main()
