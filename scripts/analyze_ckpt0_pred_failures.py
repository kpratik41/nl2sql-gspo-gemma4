#!/usr/bin/env python3
"""Detailed pred-failed analysis for ckpt-0 temp0 inference runs."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nl2sql_gspo.inference_tool_executor import extract_tool_calls


OUT_DIR = Path("outputs/analysis/ckpt0_26b_vs_31b_failure_analysis")
DATA_FILE = Path("outputs/old-dev-schema-tool.jsonl")

MODELS = {
    "26B-A4B": Path(
        "outputs/training/0530_beta_schedule_gemma-4-26b-A4b-it/training/"
        "train-6601-schema-bare-tool/gemma-4-26B-A4B-it/"
        "grpo_deepspeed_p15500_c8000_g16_t1p2_bs3_ga16_lr1e-6_gemma4_26b_a4b_dapo10_beta30_20260529_190617/"
        "checkpoint-0/temp0_olddev_schema_tool_tp2_ctx43k"
    ),
    "31B": Path(
        "outputs/training/0530_beta_schedule_gemma-4-31b-it/training/"
        "train-6601-schema-bare-tool/gemma-4-31B-it/"
        "grpo_deepspeed_p15500_c8000_g16_t1p2_bs2_ga16_lr1e-6_inprocess_beta0p005_s0-40_beta0p001_s40-80_beta0_s80plus_olddev32_refinitfix_nods_dapo10_20260529_062557/"
        "checkpoint-0/temp0_olddev_schema_tool_vllm_async_tp4_ctx43k"
    ),
}

EXPECTED_RE = re.compile(r"ExpectedOutputColumns\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
SQLITE_RESPONSE_RE = re.compile(
    r"<\|tool_response\>response:sqlite_query\{value:(?P<json>\{.*?\})\}<tool_response\|>",
    re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def by_idx(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["idx"]): row for row in rows}


def load_source_rows() -> dict[int, dict[str, Any]]:
    rows = []
    with DATA_FILE.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["idx"] = idx
            rows.append(row)
    return by_idx(rows)


def parse_expected_columns(text: str) -> list[str]:
    matches = list(EXPECTED_RE.finditer(text or ""))
    if not matches:
        return []
    raw = matches[-1].group(1)
    cols = []
    for part in raw.split(","):
        col = part.strip().strip("`\"'")
        if col:
            cols.append(col)
    return cols


def parse_sqlite_responses(text: str) -> list[dict[str, Any]]:
    # Tool responses can contain nested JSON, so use a lightweight balanced parser.
    marker = "<|tool_response>response:sqlite_query{value:"
    out = []
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start < 0:
            break
        i = start + len(marker)
        depth = 0
        in_string = False
        escaped = False
        end = None
        for offset, char in enumerate(text[i:], start=i):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    end = offset
                    break
                depth -= 1
        if end is None:
            break
        try:
            out.append(json.loads(text[i:end]))
        except Exception:
            pass
        pos = end + 1
    return out


def norm_cols(cols: list[Any]) -> list[str]:
    return [str(col).strip().strip("`\"'").lower() for col in cols or []]


def extract_failure_class(eval_row: dict[str, Any], detail: dict[str, Any]) -> str:
    stop = detail.get("stop_reason", "")
    pred_sql = str(eval_row.get("pred_sql") or "")
    pred_error = str(eval_row.get("pred_error") or "")
    text = str(detail.get("prediction_text") or "")
    tool_calls = extract_tool_calls(text)

    if not pred_sql.strip():
        if stop == "max_tool_rounds":
            return "unfinished_tool_loop_empty_sql"
        if stop == "max_new_tokens":
            return "length_exhausted_empty_sql"
        return "missing_final_sql_empty"
    if stop == "max_new_tokens":
        return "length_exhausted_malformed_extraction"
    if pred_error and ("syntax error" in pred_error.lower() or "incomplete input" in pred_error.lower()):
        return "malformed_sql_or_reasoning_extracted"
    if tool_calls and stop == "max_tool_rounds":
        return "tool_loop_no_final_answer"
    return "sql_execution_error"


def infer_help(eval_row: dict[str, Any], detail: dict[str, Any], source: dict[str, Any]) -> list[str]:
    text = str(detail.get("prediction_text") or "")
    stop = str(detail.get("stop_reason") or "")
    pred_sql = str(eval_row.get("pred_sql") or "")
    pred_error = str(eval_row.get("pred_error") or "")
    calls = extract_tool_calls(text)
    sqlite_calls = [c for c in calls if c.get("function", {}).get("name") == "sqlite_query"]
    responses = parse_sqlite_responses(text)
    expected = parse_expected_columns(text)
    last_cols = responses[-1].get("columns") if responses else []
    notes = []

    if stop == "max_tool_rounds":
        notes.append("Add/strengthen stop rule: after a successful sqlite_query with expected columns and no unresolved uncertainty, finalize instead of probing.")
    if stop == "max_new_tokens":
        notes.append("Reduce rambling: require final_answer once a plausible SQL is formed; avoid continuing scratch-pad after max_new_tokens risk.")
    if not pred_sql.strip():
        notes.append("Model never emitted extractable final SQL; prompt should prioritize final_answer after verification or when tool budget is nearly exhausted.")
    if pred_sql.strip() and pred_error:
        notes.append("Malformed extracted SQL; stricter final_answer/sql_code formatting or disabling raw-text fallback for max_new_tokens cases may help.")
    if sqlite_calls and responses:
        if expected and norm_cols(expected) == norm_cols(last_cols):
            notes.append("Last sqlite_query returned expected columns; a conservative fallback could use it, but only with extra checks because column match alone is often wrong.")
        if responses[-1].get("warning") and "0 rows" in str(responses[-1].get("warning")):
            notes.append("Last query returned zero rows; test alternate plausible predicate/join columns before broad exploration.")
    if "near" in pred_error.lower() or "syntax" in pred_error.lower():
        notes.append("Syntax repair should simplify the SQL and retry with sqlite_query before finalizing.")
    if source.get("hint"):
        notes.append("Use the hint as authoritative for predicate mapping and aggregation grain.")
    return notes


def summarize_tool_trace(detail: dict[str, Any]) -> dict[str, Any]:
    text = str(detail.get("prediction_text") or "")
    calls = extract_tool_calls(text)
    order = [c.get("function", {}).get("name", "") for c in calls]
    sqlite_responses = parse_sqlite_responses(text)
    return {
        "tool_order": order,
        "sqlite_query_calls": sum(1 for name in order if name == "sqlite_query"),
        "sqlite_responses": len(sqlite_responses),
        "last_sqlite_columns": sqlite_responses[-1].get("columns") if sqlite_responses else [],
        "last_sqlite_warning": sqlite_responses[-1].get("warning", "") if sqlite_responses else "",
        "expected_output_columns": parse_expected_columns(text),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = load_source_rows()
    all_rows = []
    summary: dict[str, Any] = {}

    for model_name, temp_dir in MODELS.items():
        eval_rows = by_idx(read_jsonl(temp_dir / "eval_results.jsonl"))
        details = by_idx(read_jsonl(temp_dir / "prediction_details.jsonl"))
        pred_failed = [row for row in eval_rows.values() if not row.get("pred_executed")]
        class_counts = Counter()
        stop_counts = Counter()
        db_counts = Counter()
        tool_counts = Counter()

        for row in pred_failed:
            idx = int(row["idx"])
            detail = details[idx]
            source = source_rows.get(idx, {})
            cls = extract_failure_class(row, detail)
            class_counts[cls] += 1
            stop_counts[str(detail.get("stop_reason") or "")] += 1
            db_counts[str(row.get("db_id") or "")] += 1
            trace = summarize_tool_trace(detail)
            tool_counts.update(trace["tool_order"])
            notes = infer_help(row, detail, source)
            record = {
                "model": model_name,
                "idx": idx,
                "db_id": row.get("db_id", ""),
                "difficulty": row.get("difficulty", ""),
                "question": source.get("question", ""),
                "evidence": source.get("evidence", ""),
                "failure_class": cls,
                "stop_reason": detail.get("stop_reason", ""),
                "tool_rounds": detail.get("tool_rounds", 0),
                "tool_call_count": detail.get("tool_call_count", 0),
                "completion_token_count": detail.get("completion_token_count", 0),
                "prompt_tokens": detail.get("prompt_tokens", 0),
                "pred_sql": row.get("pred_sql", ""),
                "pred_error": row.get("pred_error", ""),
                "gold_sql": row.get("gold_sql", ""),
                "trace": trace,
                "what_could_help": notes,
                "prediction_text_prefix": str(detail.get("prediction_text") or "")[:2000],
            }
            all_rows.append(record)

        summary[model_name] = {
            "pred_failed_count": len(pred_failed),
            "failure_class_counts": dict(class_counts),
            "stop_reason_counts": dict(stop_counts),
            "db_counts": dict(db_counts),
            "tool_name_counts_in_failed": dict(tool_counts),
            "temp_dir": str(temp_dir),
        }

    with (OUT_DIR / "pred_failed_detailed.jsonl").open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with (OUT_DIR / "pred_failed_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    write_markdown(summary, all_rows)
    print(f"Wrote pred-failed analysis to {OUT_DIR}")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Detailed Pred-Failed Analysis",
        "",
        "Pred failed means the extracted prediction did not execute. This includes empty SQL, malformed SQL extracted from reasoning, max-tool-round rollouts without final SQL, and SQL execution errors.",
        "",
        "## Summary",
        "",
    ]
    lines.append(
        md_table(
            ["Model", "Pred Failed", "Top Classes", "Top DBs", "Stop Reasons"],
            [
                [
                    model,
                    info["pred_failed_count"],
                    ", ".join(f"{k}: {v}" for k, v in Counter(info["failure_class_counts"]).most_common(4)),
                    ", ".join(f"{k}: {v}" for k, v in Counter(info["db_counts"]).most_common(4)),
                    ", ".join(f"{k}: {v}" for k, v in Counter(info["stop_reason_counts"]).most_common()),
                ]
                for model, info in summary.items()
            ],
        )
    )
    lines += [
        "",
        "## What Would Have Helped",
        "",
        "- Most max-tool-round failures need a stronger finalization rule: after an executable `sqlite_query` with expected columns, finalize unless a concrete mismatch remains.",
        "- Max-token failures often contain long scratch-pad reasoning and no valid final answer. These need stricter formatting and earlier finalization.",
        "- Column match alone is not enough for rescue fallback: it can turn empty SQL into executable but wrong SQL. Use fallback only with additional checks.",
        "- Zero-row tool results should trigger targeted alternate predicate-column checks, especially when joined tables contain similar `id`, name, status, or code columns.",
        "",
        "## Representative Failures",
        "",
    ]
    for model in summary:
        lines += [f"### {model}", ""]
        model_rows = [row for row in rows if row["model"] == model]
        seen_classes = set()
        for row in model_rows:
            if row["failure_class"] in seen_classes:
                continue
            seen_classes.add(row["failure_class"])
            lines += [
                f"#### {row['failure_class']}",
                "",
                f"- idx `{row['idx']}`, db `{row['db_id']}`, difficulty `{row['difficulty']}`",
                f"- question: {row['question']}",
                f"- stop: `{row['stop_reason']}`, tools: `{row['trace']['tool_order']}`",
                f"- expected columns: `{row['trace']['expected_output_columns']}`; last sqlite columns: `{row['trace']['last_sqlite_columns']}`",
                f"- pred error: `{row['pred_error']}`",
                f"- pred SQL: `{str(row['pred_sql'])[:700]}`",
                f"- gold SQL: `{str(row['gold_sql'])[:700]}`",
                "- likely fixes:",
            ]
            lines.extend(f"  - {note}" for note in row["what_could_help"])
            lines.append("")
    (OUT_DIR / "pred_failed_analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
