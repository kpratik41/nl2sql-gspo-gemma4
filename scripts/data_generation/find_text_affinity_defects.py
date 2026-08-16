#!/usr/bin/env python3
"""Find (and optionally drop) training rows whose gold SQL is wrong because of
SQLite TEXT affinity.

Background
----------
SQLite is dynamically typed. A column declared TEXT that holds '0.25', '-1.0',
'nan' stores those as *text*, not numbers -- `typeof()` says so. BIRD has 178
such columns across the train databases.

A gold SQL that writes `T2.Sentiment_Polarity > 0.5` against one of those
columns is not doing a numeric comparison. Under SQLite's type ordering, every
text value sorts above every numeric value, so `text_col > 0.5` is true for
essentially every non-NULL row. The query runs, returns a plausible-looking
answer, and is silently wrong -- the intended query needs
`CAST(Sentiment_Polarity AS REAL) > 0.5`.

We do not want to rewrite gold SQL (that would be inventing ground truth), and
we do not want to train on it either. So this script identifies the affected
rows by *executing* both readings and keeping only the rows where the two
disagree -- i.e. where the affinity bug demonstrably changes the answer.

Method
------
1. For every train database, use typeof() to find columns whose storage class
   is text but whose values parse as numbers ("text-affinity numeric").
   Date-looking columns are excluded: CASTing a date to REAL is meaningless,
   so a disagreement there says nothing about the gold's intent.
2. For each row's gold SQL, find comparisons of such a column against a numeric
   literal (`col <op> 0.5`), and build a variant with the column wrapped in
   CAST(... AS REAL).
3. Execute both. If the result sets differ, the gold is defective.

Usage
-----
    python scripts/data_generation/find_text_affinity_defects.py \\
        --input outputs/qwen-train-6601-schema-bare-tool.jsonl \\
        --report outputs/train-text-affinity-defects.json

    # then, to write the filtered file
    python scripts/data_generation/find_text_affinity_defects.py \\
        --input outputs/qwen-train-6601-schema-bare-tool.jsonl \\
        --report outputs/train-text-affinity-defects.json \\
        --output outputs/qwen-train-schema-bare-tool-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_generation.schema_build import _looks_like_date, _is_numeric  # noqa: E402


# A column reference may be bare (`col`) or alias-qualified (`T2.col`). Matching
# the qualifier and re-emitting it is not optional: rewriting `T2.col > 0.5` to
# `T2.CAST(col AS REAL) > 0.5` is a syntax error, which earlier showed up as 14
# spurious "defects" that were really just broken rewrites.
QUALIFIER = r"((?:[A-Za-z_][A-Za-z0-9_]*\.)?)"
COMPARISON_OPS = r"(<=|>=|<>|!=|<|>|=)"
NUMERIC_LITERAL = r"(-?\d+(?:\.\d+)?)"


def database_path(db_id: str, split: str = "train") -> Path:
    base = REPO_ROOT / "databases" / f"{split}_databases" / db_id
    for candidate in (base / f"{db_id}.sqlite", base / f"{db_id}.db"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no sqlite file for db_id={db_id} under {base}")


def text_affinity_numeric_columns(conn: sqlite3.Connection) -> Set[str]:
    """Columns stored as text whose values are numbers (lowercased names)."""

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    suspects: Set[str] = set()
    for table in tables:
        try:
            cur.execute(f"PRAGMA table_info(`{table}`);")
            columns = [row[1] for row in cur.fetchall()]
        except sqlite3.Error:
            continue

        for col in columns:
            try:
                cur.execute(
                    f"SELECT typeof(`{col}`) FROM `{table}` WHERE `{col}` IS NOT NULL "
                    f"GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1;"
                )
                row = cur.fetchone()
                if not row or row[0] != "text":
                    continue
                cur.execute(
                    f"SELECT `{col}` FROM `{table}` WHERE `{col}` IS NOT NULL LIMIT 20;"
                )
                samples = [r[0] for r in cur.fetchall()]
            except sqlite3.Error:
                continue

            if not samples:
                continue
            # Dates are text by necessity; a CAST-to-REAL comparison on them is
            # meaningless, so they can never evidence an affinity defect.
            if all(_looks_like_date(v) for v in samples[:10]):
                continue
            if all(_is_numeric(v) for v in samples):
                suspects.add(col.lower())

    return suspects


def cast_rewrite(sql: str, columns: Set[str]) -> Tuple[str, List[str]]:
    """Wrap suspect columns in CAST(... AS REAL) in numeric contexts.

    Three contexts make a text column behave numerically-wrong, and all three
    occur in BIRD golds:

      comparison   `T2.Sentiment_Polarity > 0.5`   -- text sorts above numbers
      aggregate    `MIN(T2.Sentiment_Polarity)`    -- lexicographic min
      ordering     `ORDER BY T1.grad_150 DESC`     -- lexicographic order

    Restricting the rewrite to these contexts (rather than wrapping every
    occurrence) keeps projected columns alone, so a row is never flagged just
    because '1.0' renders differently from 1.0.
    """

    touched: List[str] = []

    def wrap(qualifier: str, name: str) -> str:
        touched.append(name)
        return f"CAST({qualifier}{name} AS REAL)"

    def replace_comparison(match: re.Match[str]) -> str:
        qualifier, name, op, literal = match.groups()
        if name.lower() not in columns:
            return match.group(0)
        return f"{wrap(qualifier, name)} {op} {literal}"

    def replace_aggregate(match: re.Match[str]) -> str:
        agg, qualifier, name = match.groups()
        if name.lower() not in columns:
            return match.group(0)
        return f"{agg}({wrap(qualifier, name)}"

    def replace_order_by(match: re.Match[str]) -> str:
        lead, qualifier, name = match.groups()
        if name.lower() not in columns:
            return match.group(0)
        return f"{lead}{wrap(qualifier, name)}"

    sql = re.compile(
        rf"\b{QUALIFIER}([A-Za-z_][A-Za-z0-9_]*)\s*{COMPARISON_OPS}\s*{NUMERIC_LITERAL}"
    ).sub(replace_comparison, sql)
    sql = re.compile(
        rf"\b(MIN|MAX|AVG|SUM)\s*\(\s*{QUALIFIER}([A-Za-z_][A-Za-z0-9_]*)\s*(?=\))",
        re.IGNORECASE,
    ).sub(replace_aggregate, sql)
    sql = re.compile(
        rf"(\bORDER\s+BY\s+){QUALIFIER}([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE
    ).sub(replace_order_by, sql)

    return sql, touched


def normalize_cell(value: Any) -> Any:
    """'1.0' and 1.0 are the same answer written two ways.

    Without this, every CAST rewrite of a projected or aggregated column looks
    like a behaviour change purely because the storage class changed.
    """

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, int):
        return float(value)
    return value


def run(conn: sqlite3.Connection, sql: str) -> Tuple[bool, Any]:
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(normalize_cell(cell) for cell in row) for row in cur.fetchall()]
        return True, sorted(map(repr, rows))
    except sqlite3.Error as exc:
        return False, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tool-format train JSONL.")
    parser.add_argument("--report", required=True, help="Where to write the defect JSON.")
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write the input minus the defective rows here.",
    )
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    print(f"[scan] {len(rows)} rows from {args.input}")

    connections: Dict[str, sqlite3.Connection] = {}
    suspects_by_db: Dict[str, Set[str]] = {}
    defects: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        if args.log_every and index and index % args.log_every == 0:
            print(f"[scan] {index}/{len(rows)} rows, {len(defects)} defects so far")

        db_id = row.get("db_id")
        gold = (row.get("gold_sql") or "").strip()
        if not db_id or not gold:
            continue

        if db_id not in connections:
            connections[db_id] = sqlite3.connect(str(database_path(db_id, args.split)))
            suspects_by_db[db_id] = text_affinity_numeric_columns(connections[db_id])
        conn = connections[db_id]

        rewritten, touched = cast_rewrite(gold, suspects_by_db[db_id])
        if not touched:
            continue

        gold_ok, gold_result = run(conn, gold)
        cast_ok, cast_result = run(conn, rewritten)
        if not (gold_ok and cast_ok):
            # A rewrite that fails to parse is a bug in this script, not a
            # defective gold. Surface it rather than silently dropping the row.
            print(f"[warn] row {index} ({db_id}) failed to execute: gold_ok={gold_ok} cast_ok={cast_ok}")
            continue
        if gold_result == cast_result:
            continue

        defects.append(
            {
                "index": index,
                "db_id": db_id,
                "columns": sorted(set(touched)),
                "gold_sql": gold,
                "cast_sql": rewritten,
                "gold_rows": len(gold_result),
                "cast_rows": len(cast_result),
            }
        )

    for conn in connections.values():
        conn.close()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(defects, indent=2), encoding="utf-8")
    print(f"[scan] {len(defects)} defective rows -> {args.report}")
    for defect in defects:
        print(
            f"  idx={defect['index']:>5} {defect['db_id']:<22} "
            f"{','.join(defect['columns']):<28} rows {defect['gold_rows']} -> {defect['cast_rows']}"
        )

    if args.output:
        drop = {d["index"] for d in defects}
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                if index in drop:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {len(rows) - len(drop)} rows -> {out_path}")


if __name__ == "__main__":
    main()
