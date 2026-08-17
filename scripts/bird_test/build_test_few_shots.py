#!/usr/bin/env python3
"""Attach BM25-retrieved few-shot examples to BIRD's test.json.

Retrieval pool is the *train* split (data/bird_train_data/raw/train-6601.jsonl),
which is the same pool the dev files use -- verified: every few-shot example in
the dev artifacts comes from a train database, with zero dev-database leakage.
Test databases are likewise never retrieved from, so no test question can be
used as a demonstration for another test question.

test.json rows carry "SQL": "" (BIRD withholds the ground truth). That field is
copied through untouched; nothing here reads it, and nothing downstream may
depend on it.

Usage:
    python scripts/bird_test/build_test_few_shots.py \\
        --test-input  data/bird_test_data/raw/test.json \\
        --train-input data/bird_train_data/raw/train-6601.jsonl \\
        --output      data/bird_test_data/raw/test-few-shot.json \\
        --top-n 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def example_text(row: Dict[str, Any]) -> str:
    return " ".join(str(row.get(k, "")) for k in ("question", "evidence"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-input", required=True, type=Path)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=REPO_ROOT / "data/bird_train_data/raw/train-6601.jsonl",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be >= 1")

    from rank_bm25 import BM25Okapi

    train = [json.loads(line) for line in args.train_input.open(encoding="utf-8") if line.strip()]
    test = json.loads(args.test_input.read_text(encoding="utf-8"))
    if not isinstance(test, list):
        raise ValueError(f"Expected a JSON array in {args.test_input}")
    print(f"[few-shot] train pool {len(train)} rows, test {len(test)} rows, top_n={args.top_n}")

    bm25 = BM25Okapi([tokenize(example_text(r)) for r in train])

    out: List[Dict[str, Any]] = []
    for row in test:
        scores = bm25.get_scores(tokenize(example_text(row)))
        ranked = sorted(range(len(train)), key=lambda i: -scores[i])[: args.top_n]
        row = dict(row)
        row["few_shot_examples"] = [
            {
                "rank": rank,
                "bm25_score": float(scores[i]),
                "db_id": train[i].get("db_id", ""),
                "question": train[i].get("question", ""),
                "evidence": train[i].get("evidence", ""),
                "SQL": train[i].get("SQL", ""),
            }
            for rank, i in enumerate(ranked, start=1)
        ]
        out.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[few-shot] wrote {len(out)} rows -> {args.output}")

    leaked = {e["db_id"] for r in out for e in r["few_shot_examples"]} & {r.get("db_id") for r in test}
    if leaked:
        raise SystemExit(f"[few-shot] ABORT: test databases appeared in the retrieval pool: {sorted(leaked)}")
    print("[few-shot] verified: no test database was retrieved as a demonstration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
