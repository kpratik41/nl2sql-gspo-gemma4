#!/usr/bin/env python3
"""Analyze pass@k examples where every sampled candidate was wrong.

The script is intentionally heuristic: it cannot prove a question is ambiguous
or a gold SQL is wrong, but it surfaces evidence that helps a human decide.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nl2sql_gspo.sql_utils import get_database_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all-wrong pass@k examples and export the source rows.")
    parser.add_argument(
        "--passk-dir",
        default="outputs/bird_dev_schema_tool_passk16_vllm_async_temp08_limit1534-0514",
        help="Directory containing passk_candidates.jsonl and passk_per_example.jsonl.",
    )
    parser.add_argument(
        "--source-jsonl",
        default="outputs/old-dev-schema.jsonl",
        help="Original JSONL to filter into the all-wrong subset.",
    )
    parser.add_argument(
        "--tool-jsonl",
        default="outputs/old-dev-schema-tool.jsonl",
        help="Tool JSONL containing top-level question/evidence/db_id/gold_sql.",
    )
    parser.add_argument("--database-dir", default="databases/dev_databases")
    parser.add_argument("--output-jsonl", default="outputs/old-dev-schema-all-wrong.jsonl")
    parser.add_argument(
        "--analysis-jsonl",
        default=None,
        help="Per-example analysis JSONL. Defaults to <passk-dir>/all_wrong_analysis.jsonl.",
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Markdown summary. Defaults to <passk-dir>/all_wrong_summary.md.",
    )
    parser.add_argument("--max-sql-variants", type=int, default=5)
    parser.add_argument(
        "--execute-sql",
        action="store_true",
        help="Execute gold/top predicted SQL variants for row-count/shape diagnostics. Slower.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_db_path(db_id: str, database_dir: str) -> str:
    db_path = get_database_path(db_id, database_dir)
    if not db_path:
        raise FileNotFoundError(f"No database found for db_id={db_id} under {database_dir}")
    return db_path


def execute_with_columns(
    sql: str,
    db_id: str,
    database_dir: str,
    max_sample_rows: int = 5,
    max_rows: int = 5000,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    if not sql:
        return {"executed": False, "error": "empty sql", "columns": [], "row_count": None, "sample_rows": []}

    db_path = get_db_path(db_id, database_dir)
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False, timeout=30)
        timer = threading.Timer(timeout_s, conn.interrupt)
        timer.daemon = True
        try:
            timer.start()
            cur = conn.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            columns = [desc[0] for desc in cur.description or []]
            truncated = len(rows) > max_rows
            if truncated:
                rows = rows[:max_rows]
            return {
                "executed": True,
                "error": "",
                "columns": columns,
                "column_count": len(columns),
                "row_count": len(rows),
                "row_count_truncated": truncated,
                "sample_rows": rows[:max_sample_rows],
                "result_set_size": len(set(rows)),
            }
        finally:
            timer.cancel()
            if conn is not None:
                conn.close()
    except Exception as exc:
        return {"executed": False, "error": str(exc), "columns": [], "row_count": None, "sample_rows": []}


def result_signature(result: Dict[str, Any]) -> str:
    if not result.get("executed"):
        return "ERROR:" + str(result.get("error", ""))
    return json.dumps(
        {
            "columns": result.get("columns", []),
            "row_count": result.get("row_count"),
            "result_set_size": result.get("result_set_size"),
            "sample_rows": result.get("sample_rows", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def truncate(text: Any, max_chars: int = 500) -> str:
    text = "" if text is None else str(text)
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def classify_failure(
    candidates: List[Dict[str, Any]],
    gold_result: Dict[str, Any],
    top_results: List[Dict[str, Any]],
    unique_sql_count: int,
) -> List[str]:
    labels: List[str] = []
    total = len(candidates)
    pred_exec_count = sum(bool(c.get("pred_executed")) for c in candidates)
    stop_reasons = Counter(c.get("stop_reason", "") for c in candidates)
    pred_errors = Counter((c.get("pred_error") or "").split("\n")[0] for c in candidates if not c.get("pred_executed"))
    top_result = top_results[0]["result"] if top_results else {}

    if not gold_result.get("executed"):
        labels.append("gold_sql_execution_failed")
    if pred_exec_count == 0:
        labels.append("all_pred_sql_execution_failed")
    elif pred_exec_count < total:
        labels.append("some_pred_sql_execution_failed")
    else:
        labels.append("all_pred_sql_executed_but_wrong")

    if stop_reasons.get("max_tool_rounds"):
        labels.append("hit_max_tool_rounds")
    if stop_reasons.get("max_new_tokens"):
        labels.append("hit_max_new_tokens")
    if stop_reasons.get("generation_error"):
        labels.append("generation_error")

    if pred_errors:
        top_error, _ = pred_errors.most_common(1)[0]
        if "no such column" in top_error.lower():
            labels.append("schema_column_error")
        elif "syntax error" in top_error.lower():
            labels.append("sql_syntax_or_transcript_extraction_error")
        elif "unsafe" in top_error.lower():
            labels.append("unsafe_sql_error")

    if unique_sql_count <= 2:
        labels.append("low_diversity_repeated_wrong_sql")
    elif unique_sql_count >= 8:
        labels.append("high_diversity_no_correct_sql")

    if gold_result.get("executed") and top_result.get("executed"):
        if (
            top_result.get("column_count") is not None
            and gold_result.get("column_count") is not None
            and top_result.get("column_count") != gold_result.get("column_count")
        ):
            labels.append("output_column_count_mismatch")
        if (
            top_result.get("row_count") is not None
            and gold_result.get("row_count") is not None
            and top_result.get("row_count") != gold_result.get("row_count")
        ):
            labels.append("row_count_mismatch")
        if (
            top_result.get("row_count") == 0
            and gold_result.get("row_count") is not None
            and gold_result.get("row_count", 0) > 0
        ):
            labels.append("pred_empty_gold_nonempty")
        if (
            top_result.get("row_count") is not None
            and top_result.get("row_count", 0) > 0
            and gold_result.get("row_count") == 0
        ):
            labels.append("pred_nonempty_gold_empty")
        if (
            top_result.get("column_count") is not None
            and gold_result.get("column_count") is not None
            and top_result.get("row_count") is not None
            and gold_result.get("row_count") is not None
            and top_result.get("column_count") == gold_result.get("column_count")
            and top_result.get("row_count") == gold_result.get("row_count")
        ):
            labels.append("same_shape_but_wrong_values")

    result_sig_counts = Counter(item["signature"] for item in top_results if item["result"].get("executed"))
    if result_sig_counts and result_sig_counts.most_common(1)[0][1] >= 8:
        labels.append("strong_consensus_wrong_result")

    return labels


def analyze_one(
    idx: int,
    candidates: List[Dict[str, Any]],
    tool_row: Dict[str, Any],
    database_dir: str,
    max_sql_variants: int,
    execute_sql_diagnostics: bool,
) -> Dict[str, Any]:
    sql_counter = Counter(c.get("pred_sql", "") for c in candidates)
    top_sqls = sql_counter.most_common(max_sql_variants)
    db_id = tool_row.get("db_id") or candidates[0].get("db_id", "")
    gold_sql = tool_row.get("gold_sql") or candidates[0].get("gold_sql", "")
    gold_result = (
        execute_with_columns(gold_sql, db_id, database_dir)
        if execute_sql_diagnostics
        else {
            "executed": all(bool(c.get("gold_executed")) for c in candidates),
            "error": next((c.get("gold_error", "") for c in candidates if c.get("gold_error")), ""),
            "columns": [],
            "column_count": None,
            "row_count": None,
            "sample_rows": [],
        }
    )

    top_result_rows: List[Dict[str, Any]] = []
    for sql, count in top_sqls:
        result = (
            execute_with_columns(sql, db_id, database_dir)
            if execute_sql_diagnostics
            else {
                "executed": any(c.get("pred_sql") == sql and c.get("pred_executed") for c in candidates),
                "error": next(
                    (
                        c.get("pred_error", "")
                        for c in candidates
                        if c.get("pred_sql") == sql and c.get("pred_error")
                    ),
                    "",
                ),
                "columns": [],
                "column_count": None,
                "row_count": None,
                "sample_rows": [],
            }
        )
        top_result_rows.append(
            {
                "count": count,
                "sql": sql,
                "result": result,
                "signature": result_signature(result),
            }
        )

    stop_reasons = Counter(c.get("stop_reason", "") for c in candidates)
    pred_errors = Counter((c.get("pred_error") or "").split("\n")[0] for c in candidates if not c.get("pred_executed"))
    tool_orders = Counter(" -> ".join(c.get("tool_order") or []) or "<none>" for c in candidates)
    tool_call_counts = Counter(int(c.get("tool_call_count", 0)) for c in candidates)

    labels = classify_failure(candidates, gold_result, top_result_rows, len(sql_counter))

    return {
        "idx": idx,
        "db_id": db_id,
        "question": tool_row.get("question", ""),
        "evidence": tool_row.get("evidence", ""),
        "gold_sql": gold_sql,
        "gold_result": gold_result,
        "num_candidates": len(candidates),
        "num_correct": sum(int(c.get("correct", 0)) for c in candidates),
        "pred_executed": sum(bool(c.get("pred_executed")) for c in candidates),
        "pred_execution_failed": sum(not bool(c.get("pred_executed")) for c in candidates),
        "unique_pred_sql": len(sql_counter),
        "stop_reasons": dict(stop_reasons),
        "pred_errors": dict(pred_errors.most_common(10)),
        "tool_call_count_distribution": dict(sorted(tool_call_counts.items())),
        "total_tool_calls": sum(int(c.get("tool_call_count", 0)) for c in candidates),
        "tool_orders": dict(tool_orders.most_common(10)),
        "failure_labels": labels,
        "top_pred_sqls": [
            {
                "count": item["count"],
                "sql": item["sql"],
                "executed": item["result"].get("executed"),
                "error": item["result"].get("error", ""),
                "columns": item["result"].get("columns", []),
                "row_count": item["result"].get("row_count"),
                "result_set_size": item["result"].get("result_set_size"),
                "sample_rows": item["result"].get("sample_rows", []),
            }
            for item in top_result_rows
        ],
    }


def render_summary(analysis_rows: List[Dict[str, Any]], output_jsonl: Path, source_jsonl: Path) -> str:
    labels = Counter(label for row in analysis_rows for label in row["failure_labels"])
    db_counts = Counter(row["db_id"] for row in analysis_rows)
    stop_reasons = Counter()
    for row in analysis_rows:
        stop_reasons.update(row["stop_reasons"])

    lines = [
        "# All-Wrong pass@16 Analysis",
        "",
        f"- all_wrong_examples: `{len(analysis_rows)}`",
        f"- filtered_source_jsonl: `{output_jsonl}`",
        f"- source_jsonl: `{source_jsonl}`",
        "",
        "## Failure Labels",
        "",
    ]
    for label, count in labels.most_common():
        lines.append(f"- {label}: `{count}`")

    lines += ["", "## Stop Reasons Across All-Wrong Candidates", ""]
    for reason, count in stop_reasons.most_common():
        lines.append(f"- {reason or '<empty>'}: `{count}`")

    lines += ["", "## DB Counts", ""]
    for db_id, count in db_counts.most_common():
        lines.append(f"- {db_id}: `{count}`")

    lines += [
        "",
        "## Examples To Inspect First",
        "",
        "| idx | db | labels | pred_exec | unique_sql | question |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    priority = sorted(
        analysis_rows,
        key=lambda row: (
            "gold_sql_execution_failed" not in row["failure_labels"],
            "strong_consensus_wrong_result" not in row["failure_labels"],
            -row["pred_executed"],
            row["unique_pred_sql"],
        ),
    )
    for row in priority[:50]:
        labels_text = ", ".join(row["failure_labels"][:4])
        lines.append(
            f"| {row['idx']} | {row['db_id']} | {labels_text} | "
            f"{row['pred_executed']}/16 | {row['unique_pred_sql']} | {truncate(row['question'], 120)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    passk_dir = Path(args.passk_dir)
    source_jsonl = Path(args.source_jsonl)
    tool_jsonl = Path(args.tool_jsonl)
    output_jsonl = Path(args.output_jsonl)
    analysis_jsonl = Path(args.analysis_jsonl) if args.analysis_jsonl else passk_dir / "all_wrong_analysis.jsonl"
    summary_md = Path(args.summary_md) if args.summary_md else passk_dir / "all_wrong_summary.md"

    per_examples = load_jsonl(passk_dir / "passk_per_example.jsonl")
    all_wrong_idxs = sorted(int(row["idx"]) for row in per_examples if int(row.get("num_correct", 0)) == 0)
    all_wrong_set = set(all_wrong_idxs)

    source_rows = load_jsonl(source_jsonl)
    tool_rows = load_jsonl(tool_jsonl)
    write_jsonl(output_jsonl, (source_rows[idx] for idx in all_wrong_idxs))

    candidates_by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in iter_jsonl(passk_dir / "passk_candidates.jsonl"):
        idx = int(candidate["idx"])
        if idx in all_wrong_set:
            candidates_by_idx[idx].append(candidate)

    analysis_rows = [
        analyze_one(
            idx=idx,
            candidates=sorted(candidates_by_idx[idx], key=lambda row: int(row.get("sample_id", 0))),
            tool_row=tool_rows[idx],
            database_dir=args.database_dir,
            max_sql_variants=args.max_sql_variants,
            execute_sql_diagnostics=args.execute_sql,
        )
        for idx in all_wrong_idxs
    ]
    write_jsonl(analysis_jsonl, analysis_rows)
    summary_md.write_text(render_summary(analysis_rows, output_jsonl, source_jsonl), encoding="utf-8")

    print(f"all_wrong_examples={len(all_wrong_idxs)}")
    print(f"wrote filtered source rows to {output_jsonl}")
    print(f"wrote analysis to {analysis_jsonl}")
    print(f"wrote summary to {summary_md}")


if __name__ == "__main__":
    main()
