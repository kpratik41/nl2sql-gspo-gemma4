#!/usr/bin/env python3
"""Compare ckpt-0 26B-A4B and 31B inference/pass@k failure modes."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_26B = Path(
    "outputs/training/0530_beta_schedule_gemma-4-26b-A4b-it/training/"
    "train-6601-schema-bare-tool/gemma-4-26B-A4B-it/"
    "grpo_deepspeed_p15500_c8000_g16_t1p2_bs3_ga16_lr1e-6_gemma4_26b_a4b_dapo10_beta30_20260529_190617/"
    "checkpoint-0"
)
ROOT_31B = Path(
    "outputs/training/0530_beta_schedule_gemma-4-31b-it/training/"
    "train-6601-schema-bare-tool/gemma-4-31B-it/"
    "grpo_deepspeed_p15500_c8000_g16_t1p2_bs2_ga16_lr1e-6_inprocess_beta0p005_s0-40_beta0p001_s40-80_beta0_s80plus_olddev32_refinitfix_nods_dapo10_20260529_062557/"
    "checkpoint-0"
)

MODELS = {
    "26B-A4B": {
        "root": ROOT_26B,
        "temp": "temp0_olddev_schema_tool_tp2_ctx43k",
        "passk": "passk16_olddev_schema_tool_temp1p2_tp2_ctx43p5k",
        "sc": "self_consistency_passk16_olddev_schema_tool_temp1p2_tp2_ctx43p5k",
    },
    "31B": {
        "root": ROOT_31B,
        "temp": "temp0_olddev_schema_tool_vllm_async_tp4_ctx43k",
        "passk": "passk16_olddev_schema_tool_temp1p2_tp4_ctx45k",
        "sc": "passk16_olddev_schema_tool_temp1p2_tp4_ctx45k/self_consistency",
    },
}

OUT_DIR = Path("outputs/analysis/ckpt0_26b_vs_31b_failure_analysis")

TOOL_RE = re.compile(
    r"(?:<\|tool_call\>\s*)?call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{",
    re.DOTALL,
)
KNOWN_TOOLS = {"sqlite_peek", "sqlite_query", "bm25_search_sqlite", "consensus_at_1"}


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_idx(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["idx"]): row for row in rows}


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def bool_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    return bool(row.get("res"))


def extract_tool_order(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("tool_order"), list):
        return [str(x) for x in row["tool_order"]]
    text = row.get("prediction_text") or ""
    return [m.group("name") for m in TOOL_RE.finditer(text)]


def tool_bucket(n: int, stop_reason: str = "") -> str:
    if stop_reason == "max_tool_rounds":
        return "max_tool_rounds"
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def pred_failed(eval_row: dict[str, Any], detail_row: dict[str, Any] | None = None) -> bool:
    return not bool(eval_row.get("pred_executed"))


def classify_temp_failure(
    eval_row: dict[str, Any],
    detail_row: dict[str, Any],
    passk_rows: list[dict[str, Any]],
    sc_row: dict[str, Any] | None,
) -> list[str]:
    labels: list[str] = []
    correct = bool_correct(eval_row)
    pred_extracted = bool(eval_row.get("pred_sql_extracted"))
    pred_exec = bool(eval_row.get("pred_executed"))
    stop = str(detail_row.get("stop_reason", ""))
    tool_count = int(detail_row.get("tool_call_count") or 0)
    text = str(detail_row.get("prediction_text") or "")
    passk_correct = any(bool(r.get("correct")) for r in passk_rows)

    if not correct:
        if not pred_extracted or not str(eval_row.get("pred_sql") or "").strip():
            labels.append("missing_final_sql")
        if stop == "max_tool_rounds":
            labels.append("tool_loop_exhausted")
        if stop in {"max_new_tokens", "context_length_exceeded"}:
            labels.append("length_exhausted")
        if pred_extracted and not pred_exec:
            labels.append("sql_execution_error")
        if pred_exec:
            labels.append("semantic_wrong_executed")
        if tool_count == 0:
            labels.append("tool_underuse")
        if tool_count >= 4 or stop == "max_tool_rounds":
            labels.append("tool_overuse_or_loop")
        if ("\"error\"" in text or "'error'" in text or "error:" in text.lower()) and (
            not pred_exec or not correct or stop == "max_tool_rounds"
        ):
            labels.append("tool_error_unrecovered")
        if passk_correct and sc_row and not bool(sc_row.get("option2_correct")):
            labels.append("candidate_selection_failure")
        if not passk_correct:
            labels.append("search_space_failure")
    return labels


def summarize_rate_by_bucket(rows: list[dict[str, Any]], correct_key: str = "correct") -> dict[str, Any]:
    buckets: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        order = extract_tool_order(row)
        n = int(row.get("tool_call_count") or len(order))
        bucket = tool_bucket(n, str(row.get("stop_reason", "")))
        ok = bool(row.get(correct_key)) if correct_key in row else bool_correct(row)
        buckets[bucket]["count"] += 1
        buckets[bucket]["correct"] += int(ok)
        buckets[bucket]["failed"] += int(not bool(row.get("pred_executed", True)))
    return {
        k: {
            "count": v["count"],
            "correct": v["correct"],
            "accuracy": pct(v["correct"], v["count"]),
            "pred_failed": v["failed"],
            "pred_failed_rate": pct(v["failed"], v["count"]),
        }
        for k, v in sorted(buckets.items())
    }


def top_counter(counter: Counter, n: int = 10) -> list[dict[str, Any]]:
    return [{"value": k, "count": v} for k, v in counter.most_common(n)]


def load_model(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["root"])
    temp_dir = root / cfg["temp"]
    passk_dir = root / cfg["passk"]
    sc_dir = root / cfg["sc"]

    temp_summary = read_json(temp_dir / "eval_summary.json")
    passk_summary = read_json(passk_dir / "passk_summary.json")
    sc_summary = read_json(sc_dir / "self_consistency_summary.json")[0]
    temp_eval = by_idx(read_jsonl(temp_dir / "eval_results.jsonl"))
    temp_details = by_idx(read_jsonl(temp_dir / "prediction_details.jsonl"))
    passk_candidates = read_jsonl(passk_dir / "passk_candidates.jsonl")
    passk_per_example = by_idx(read_jsonl(passk_dir / "passk_per_example.jsonl"))
    sc_per_example = by_idx(read_jsonl(sc_dir / "self_consistency_per_example.jsonl"))

    passk_by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in passk_candidates:
        passk_by_idx[int(row["idx"])].append(row)

    for row in temp_details.values():
        row["tool_order"] = extract_tool_order(row)
    for row in passk_candidates:
        row["tool_order"] = extract_tool_order(row)

    failure_labels_by_idx = {}
    for idx, eval_row in temp_eval.items():
        labels = classify_temp_failure(
            eval_row,
            temp_details.get(idx, {}),
            passk_by_idx.get(idx, []),
            sc_per_example.get(idx),
        )
        failure_labels_by_idx[idx] = labels

    return {
        "name": name,
        "root": str(root),
        "temp_dir": str(temp_dir),
        "passk_dir": str(passk_dir),
        "sc_dir": str(sc_dir),
        "temp_summary": temp_summary,
        "passk_summary": passk_summary,
        "sc_summary": sc_summary,
        "temp_eval": temp_eval,
        "temp_details": temp_details,
        "passk_candidates": passk_candidates,
        "passk_by_idx": passk_by_idx,
        "passk_per_example": passk_per_example,
        "sc_per_example": sc_per_example,
        "failure_labels_by_idx": failure_labels_by_idx,
    }


def summarize_model(model: dict[str, Any]) -> dict[str, Any]:
    temp_summary = model["temp_summary"]
    passk_summary = model["passk_summary"]
    sc_summary = model["sc_summary"]
    temp_eval = model["temp_eval"]
    temp_details = model["temp_details"]
    passk_by_idx = model["passk_by_idx"]
    sc_per = model["sc_per_example"]

    labels = Counter()
    for vals in model["failure_labels_by_idx"].values():
        labels.update(vals)

    temp_rows_joined = []
    for idx, eval_row in temp_eval.items():
        detail = temp_details.get(idx, {})
        merged = dict(eval_row)
        merged.update(
            {
                "tool_call_count": detail.get("tool_call_count", 0),
                "stop_reason": detail.get("stop_reason", ""),
                "tool_order": detail.get("tool_order", []),
            }
        )
        temp_rows_joined.append(merged)

    all_passk = model["passk_candidates"]
    seq_counter = Counter()
    first_tool_counter = Counter()
    for row in all_passk:
        order = row.get("tool_order") or []
        seq_counter[" -> ".join(order[:4]) if order else "<none>"] += 1
        first_tool_counter[order[0] if order else "<none>"] += 1

    temp_seq_counter = Counter()
    temp_first_tool_counter = Counter()
    for row in temp_rows_joined:
        order = row.get("tool_order") or []
        temp_seq_counter[" -> ".join(order[:4]) if order else "<none>"] += 1
        temp_first_tool_counter[order[0] if order else "<none>"] += 1

    rescue = Counter()
    candidate_selection = Counter()
    passk_pred_failed_examples = Counter()
    for idx, eval_row in temp_eval.items():
        temp_ok = bool_correct(eval_row)
        temp_failed = pred_failed(eval_row, temp_details.get(idx))
        cands = passk_by_idx.get(idx, [])
        any_correct = any(bool(c.get("correct")) for c in cands)
        any_executed = any(bool(c.get("pred_executed")) for c in cands)
        all_failed = bool(cands) and all(not bool(c.get("pred_executed")) for c in cands)
        mostly_failed = bool(cands) and sum(not bool(c.get("pred_executed")) for c in cands) >= math.ceil(len(cands) / 2)
        sc = sc_per.get(idx, {})
        if not temp_ok and any_correct:
            rescue["temp0_wrong_passk_has_correct"] += 1
        if temp_failed and any_executed:
            rescue["temp0_pred_failed_passk_has_executable"] += 1
        if temp_failed and any_correct:
            rescue["temp0_pred_failed_passk_has_correct"] += 1
        if any_correct and not bool(sc.get("option2_correct")):
            candidate_selection["passk_has_correct_sc_option2_wrong"] += 1
        if any_correct and not bool(sc.get("option1_correct")):
            candidate_selection["passk_has_correct_sc_option1_wrong"] += 1
        if any_correct and all_failed:
            passk_pred_failed_examples["passk_has_correct_but_all_failed"] += 1
        if any_correct and mostly_failed:
            passk_pred_failed_examples["passk_has_correct_but_mostly_failed"] += 1

    by_db = {}
    for db, stats in temp_summary.get("by_db", {}).items():
        by_db[db] = {
            "accuracy": stats["accuracy"],
            "correct": stats["correct"],
            "count": stats["count"],
        }

    by_difficulty = {}
    for diff, stats in temp_summary.get("by_difficulty", {}).items():
        by_difficulty[diff] = {
            "accuracy": stats["accuracy"],
            "correct": stats["correct"],
            "count": stats["count"],
        }

    return {
        "name": model["name"],
        "paths": {
            "root": model["root"],
            "temp_dir": model["temp_dir"],
            "passk_dir": model["passk_dir"],
            "sc_dir": model["sc_dir"],
        },
        "temp0": {
            "accuracy": temp_summary["total"]["accuracy"],
            "correct": temp_summary["total"]["correct"],
            "count": temp_summary["total"]["count"],
            "execution_stats": temp_summary["execution_stats"],
            "stop_reasons": temp_summary.get("generation_stats", {}).get("stop_reason_counts")
            or Counter(str(r.get("stop_reason", "")) for r in temp_details.values()),
            "avg_tool_calls": temp_summary.get("generation_stats", {}).get(
                "avg_tool_calls_per_example",
                sum(int(r.get("tool_call_count") or 0) for r in temp_details.values()) / max(1, len(temp_details)),
            ),
            "tool_name_counts": temp_summary.get("generation_stats", {}).get(
                "tool_name_counts",
                dict(Counter(t for r in temp_details.values() for t in r.get("tool_order", []))),
            ),
            "by_tool_bucket": summarize_rate_by_bucket(temp_rows_joined),
            "top_first_tools": top_counter(temp_first_tool_counter, 8),
            "top_tool_sequences": top_counter(temp_seq_counter, 12),
        },
        "passk": {
            "pass_at_16": passk_summary["pass_at_k_estimated"]["16"],
            "candidate_accuracy": passk_summary["candidate_accuracy"]["accuracy"],
            "candidate_correct": passk_summary["candidate_accuracy"]["correct"],
            "candidate_count": passk_summary["candidate_accuracy"]["count"],
            "pred_execution": passk_summary["pred_execution"],
            "stop_reasons": passk_summary["stop_reasons"],
            "avg_tool_calls": passk_summary["tool_calls"]["avg_per_candidate"],
            "tool_calls_total": passk_summary["tool_calls"]["total"],
            "by_tool_bucket": summarize_rate_by_bucket(all_passk),
            "top_first_tools": top_counter(first_tool_counter, 8),
            "top_tool_sequences": top_counter(seq_counter, 12),
        },
        "self_consistency": {
            "option1_accuracy": sc_summary["option1_accuracy"],
            "option1_correct": sc_summary["option1_correct"],
            "option2_accuracy": sc_summary["option2_accuracy"],
            "option2_correct": sc_summary["option2_correct"],
            "option1_delta_vs_temp0": sc_summary["option1_accuracy"] - temp_summary["total"]["accuracy"],
            "option2_delta_vs_temp0": sc_summary["option2_accuracy"] - temp_summary["total"]["accuracy"],
            "option1_ties": sc_summary["option1_ties"],
            "option2_ties_after_adding_temp0": sc_summary["option2_ties_after_adding_temp0"],
        },
        "failure_taxonomy": dict(labels),
        "retry_rescue": dict(rescue),
        "candidate_selection": dict(candidate_selection),
        "passk_pred_failed_examples": dict(passk_pred_failed_examples),
        "by_db": by_db,
        "by_difficulty": by_difficulty,
    }


def compact_example(model: dict[str, Any], idx: int) -> dict[str, Any]:
    eval_row = model["temp_eval"].get(idx, {})
    detail = model["temp_details"].get(idx, {})
    cands = model["passk_by_idx"].get(idx, [])
    sc = model["sc_per_example"].get(idx, {})
    return {
        "temp0_correct": bool_correct(eval_row) if eval_row else False,
        "temp0_pred_executed": bool(eval_row.get("pred_executed")),
        "temp0_pred_sql_extracted": bool(eval_row.get("pred_sql_extracted")),
        "temp0_pred_error": eval_row.get("pred_error", ""),
        "temp0_stop_reason": detail.get("stop_reason", ""),
        "temp0_tool_calls": int(detail.get("tool_call_count") or 0),
        "temp0_tool_order": detail.get("tool_order", []),
        "passk_num_correct": sum(int(bool(c.get("correct"))) for c in cands),
        "passk_any_correct": any(bool(c.get("correct")) for c in cands),
        "passk_any_executed": any(bool(c.get("pred_executed")) for c in cands),
        "passk_pred_failed": sum(int(not bool(c.get("pred_executed"))) for c in cands),
        "passk_stop_reasons": dict(Counter(str(c.get("stop_reason", "")) for c in cands)),
        "passk_tool_calls_avg": (
            sum(int(c.get("tool_call_count") or 0) for c in cands) / len(cands) if cands else 0.0
        ),
        "sc_option1_correct": bool(sc.get("option1_correct")),
        "sc_option2_correct": bool(sc.get("option2_correct")),
        "sc_option1_source": sc.get("option1_source", ""),
        "sc_option2_source": sc.get("option2_source", ""),
        "failure_labels": model["failure_labels_by_idx"].get(idx, []),
        "pred_sql": eval_row.get("pred_sql", ""),
        "gold_sql": eval_row.get("gold_sql", ""),
    }


def make_per_example(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    idxs = sorted(set(models["26B-A4B"]["temp_eval"]) | set(models["31B"]["temp_eval"]))
    rows = []
    for idx in idxs:
        base = models["26B-A4B"]["temp_eval"].get(idx) or models["31B"]["temp_eval"].get(idx) or {}
        row = {
            "idx": idx,
            "db_id": base.get("db_id", ""),
            "difficulty": base.get("difficulty", ""),
            "26B-A4B": compact_example(models["26B-A4B"], idx),
            "31B": compact_example(models["31B"], idx),
        }
        row["comparison"] = {
            "both_temp0_correct": row["26B-A4B"]["temp0_correct"] and row["31B"]["temp0_correct"],
            "both_temp0_wrong": (not row["26B-A4B"]["temp0_correct"]) and (not row["31B"]["temp0_correct"]),
            "only_26B_temp0_correct": row["26B-A4B"]["temp0_correct"] and not row["31B"]["temp0_correct"],
            "only_31B_temp0_correct": row["31B"]["temp0_correct"] and not row["26B-A4B"]["temp0_correct"],
        }
        rows.append(row)
    return rows


def compare_models(summaries: dict[str, Any], per_example: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = Counter()
    for row in per_example:
        comp = row["comparison"]
        for k, v in comp.items():
            overlap[k] += int(v)

    db_deltas = []
    for db, s26 in summaries["26B-A4B"]["by_db"].items():
        s31 = summaries["31B"]["by_db"].get(db)
        if not s31:
            continue
        db_deltas.append(
            {
                "db_id": db,
                "count": s26["count"],
                "accuracy_26B_A4B": s26["accuracy"],
                "accuracy_31B": s31["accuracy"],
                "delta_31B_minus_26B": s31["accuracy"] - s26["accuracy"],
            }
        )
    db_deltas.sort(key=lambda x: x["delta_31B_minus_26B"])

    diff_deltas = []
    for diff, s26 in summaries["26B-A4B"]["by_difficulty"].items():
        s31 = summaries["31B"]["by_difficulty"].get(diff)
        if not s31:
            continue
        diff_deltas.append(
            {
                "difficulty": diff,
                "count": s26["count"],
                "accuracy_26B_A4B": s26["accuracy"],
                "accuracy_31B": s31["accuracy"],
                "delta_31B_minus_26B": s31["accuracy"] - s26["accuracy"],
            }
        )
    return {
        "temp0_overlap": dict(overlap),
        "db_deltas_sorted": db_deltas,
        "difficulty_deltas": diff_deltas,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def select_examples(per_example: list[dict[str, Any]], label: str, model_name: str, limit: int = 5) -> list[dict[str, Any]]:
    found = []
    for row in per_example:
        if label in row[model_name]["failure_labels"]:
            found.append(row)
        if len(found) >= limit:
            break
    return found


def write_report(summaries: dict[str, Any], comparison: dict[str, Any], per_example: list[dict[str, Any]]) -> None:
    lines = [
        "# Ckpt-0 26B-A4B vs 31B Failure Analysis",
        "",
        "This analysis uses only existing checkpoint-0 artifacts: temp0 inference, pass@16 generations, and self-consistency outputs.",
        "",
        "## Headline Metrics",
        "",
    ]
    rows = []
    for name, s in summaries.items():
        rows.append(
            [
                name,
                fmt_pct(s["temp0"]["accuracy"]),
                s["temp0"]["execution_stats"]["pred_sql_execution_failed"],
                f'{fmt_pct(s["passk"]["pass_at_16"])}',
                fmt_pct(s["passk"]["candidate_accuracy"]),
                f'{s["passk"]["pred_execution"]["failed"]} / {s["passk"]["candidate_count"]}',
                f'{s["temp0"]["avg_tool_calls"]:.2f}',
                f'{s["passk"]["avg_tool_calls"]:.2f}',
                fmt_pct(s["self_consistency"]["option1_accuracy"]),
                fmt_pct(s["self_consistency"]["option2_accuracy"]),
            ]
        )
    lines.append(
        md_table(
            [
                "Model",
                "Temp0 Acc",
                "Temp0 Pred Failed",
                "Pass@16",
                "Candidate Acc",
                "Pass@k Pred Failed",
                "Temp0 Tool Avg",
                "Pass@k Tool Avg",
                "SC Opt1",
                "SC Opt2",
            ],
            rows,
        )
    )
    lines += [
        "",
        "## Main Findings",
        "",
        "- 31B is stronger at temp0 and has far fewer temp0 prediction failures, but 26B-A4B has the stronger pass@16 oracle and stronger SC.",
        "- 26B-A4B benefits more from retries: its pass@16 is 8.54 points above temp0, versus 3.78 points for 31B.",
        "- 31B uses more pass@k tools on average and executes more reliably, so its bottleneck looks less like raw tool access and more like candidate diversity or candidate selection.",
        "- SC underperforms the pass@16 oracle for both models because the majority cluster can select a repeated wrong answer even when a correct candidate exists.",
        "",
        "## Temp0 Overlap",
        "",
    ]
    overlap = comparison["temp0_overlap"]
    lines.append(
        md_table(
            ["Set", "Count"],
            [
                ["both correct", overlap.get("both_temp0_correct", 0)],
                ["both wrong", overlap.get("both_temp0_wrong", 0)],
                ["26B only correct", overlap.get("only_26B_temp0_correct", 0)],
                ["31B only correct", overlap.get("only_31B_temp0_correct", 0)],
            ],
        )
    )

    lines += ["", "## Failure Taxonomy", ""]
    for name, s in summaries.items():
        total_wrong = s["temp0"]["count"] - s["temp0"]["correct"] if "count" in s["temp0"] else 1534 - int(round(s["temp0"]["accuracy"] * 1534 / 100))
        rows = []
        for label, count in sorted(s["failure_taxonomy"].items(), key=lambda x: (-x[1], x[0])):
            rows.append([label, count, fmt_pct(pct(count, total_wrong))])
        lines += [f"### {name}", "", md_table(["Bucket", "Count", "Share of Temp0 Wrong"], rows), ""]

    lines += ["## Retry and Self-Consistency", ""]
    rows = []
    for name, s in summaries.items():
        rows.append(
            [
                name,
                s["retry_rescue"].get("temp0_wrong_passk_has_correct", 0),
                s["retry_rescue"].get("temp0_pred_failed_passk_has_executable", 0),
                s["retry_rescue"].get("temp0_pred_failed_passk_has_correct", 0),
                s["candidate_selection"].get("passk_has_correct_sc_option1_wrong", 0),
                s["candidate_selection"].get("passk_has_correct_sc_option2_wrong", 0),
            ]
        )
    lines.append(
        md_table(
            [
                "Model",
                "Temp0 Wrong Rescued By Pass@16",
                "Temp0 Pred Failed -> Any Executable Candidate",
                "Temp0 Pred Failed -> Correct Candidate",
                "Pass@16 Has Correct But SC1 Wrong",
                "Pass@16 Has Correct But SC2 Wrong",
            ],
            rows,
        )
    )

    lines += ["", "## Tool Use Signals", ""]
    for name, s in summaries.items():
        lines += [f"### {name}", ""]
        lines.append("Temp0 by tool-call bucket:")
        lines.append(
            md_table(
                ["Bucket", "Count", "Accuracy", "Pred Failed"],
                [
                    [b, v["count"], fmt_pct(v["accuracy"]), v["pred_failed"]]
                    for b, v in s["temp0"]["by_tool_bucket"].items()
                ],
            )
        )
        lines.append("")
        lines.append("Pass@k candidates by tool-call bucket:")
        lines.append(
            md_table(
                ["Bucket", "Count", "Accuracy", "Pred Failed"],
                [
                    [b, v["count"], fmt_pct(v["accuracy"]), v["pred_failed"]]
                    for b, v in s["passk"]["by_tool_bucket"].items()
                ],
            )
        )
        lines.append("")
        lines.append("Most common pass@k tool sequences:")
        lines.append(
            md_table(
                ["Sequence", "Count"],
                [[x["value"], x["count"]] for x in s["passk"]["top_tool_sequences"][:8]],
            )
        )
        lines.append("")

    lines += [
        "## Per-Database Deltas",
        "",
        "Positive delta means 31B beats 26B-A4B on temp0.",
        "",
    ]
    lines.append(
        md_table(
            ["DB", "Count", "26B-A4B", "31B", "31B - 26B"],
            [
                [
                    r["db_id"],
                    r["count"],
                    fmt_pct(r["accuracy_26B_A4B"]),
                    fmt_pct(r["accuracy_31B"]),
                    f'{r["delta_31B_minus_26B"]:+.2f}',
                ]
                for r in comparison["db_deltas_sorted"]
            ],
        )
    )

    lines += [
        "",
        "## Recommendations",
        "",
        "- For a single-model leaderboard submission, use a retry/selection strategy rather than raw temp0: the pass@16 oracle shows meaningful recoverable headroom.",
        "- Add a prompt rule that after any failed `sqlite_query`, the model should inspect the error, repair the SQL, and call the tool again before finalizing.",
        "- Add a prompt rule to avoid finalizing after zero tool calls on queries involving ambiguous values, date filters, unusual columns, or aggregations.",
        "- Add a prompt rule to stop tool loops: after two equivalent `sqlite_query` failures or after a successful query with the expected columns, finalize instead of continuing.",
        "- Consider a selector stronger than pure majority SC: prefer executable clusters, penalize max-tool-round/empty SQL candidates, and optionally use temp0 as fallback when SC has no valid executed cluster.",
        "- 31B specifically needs more diversity or better selection; it is already executable, but its pass@16 and SC lag 26B-A4B despite fewer pred failures.",
        "",
        "## Why pass@k/SC Still Has Pred Failures",
        "",
        "Pass@k samples are independent generations. Some candidates still exhaust tool rounds, hit length limits, or fail to emit extractable final SQL. SC can reduce this only if the selected result cluster is executable; it cannot fix candidates outside the selected cluster, and majority voting can amplify repeated wrong-but-executable SQL.",
    ]

    (OUT_DIR / "failure_analysis_report.md").write_text("\n".join(lines) + "\n")


def write_examples(summaries: dict[str, Any], models: dict[str, Any], per_example: list[dict[str, Any]]) -> None:
    labels = [
        "missing_final_sql",
        "tool_loop_exhausted",
        "length_exhausted",
        "sql_execution_error",
        "semantic_wrong_executed",
        "tool_underuse",
        "tool_overuse_or_loop",
        "tool_error_unrecovered",
        "candidate_selection_failure",
        "search_space_failure",
    ]
    lines = ["# Representative Failure Examples", ""]
    for model_name in ["26B-A4B", "31B"]:
        lines += [f"## {model_name}", ""]
        for label in labels:
            examples = select_examples(per_example, label, model_name, 3)
            if not examples:
                continue
            lines += [f"### {label}", ""]
            for row in examples:
                m = row[model_name]
                lines += [
                    f"- idx `{row['idx']}`, db `{row['db_id']}`, difficulty `{row['difficulty']}`",
                    f"  - temp0 correct: `{m['temp0_correct']}`, pred executed: `{m['temp0_pred_executed']}`, stop: `{m['temp0_stop_reason']}`, tools: `{m['temp0_tool_order']}`",
                    f"  - pass@k correct candidates: `{m['passk_num_correct']}/16`, pass@k pred failed: `{m['passk_pred_failed']}/16`, SC2 correct: `{m['sc_option2_correct']}`",
                    f"  - pred SQL: `{str(m['pred_sql'])[:500]}`",
                    f"  - gold SQL: `{str(m['gold_sql'])[:500]}`",
                    "",
                ]
    (OUT_DIR / "failure_examples.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(name, cfg) for name, cfg in MODELS.items()}
    summaries = {name: summarize_model(model) for name, model in models.items()}
    # Preserve total/count in the temp0 subsection for report math.
    for name, model in models.items():
        summaries[name]["temp0"]["count"] = model["temp_summary"]["total"]["count"]
        summaries[name]["temp0"]["correct"] = model["temp_summary"]["total"]["correct"]

    per_example = make_per_example(models)
    comparison = compare_models(summaries, per_example)
    summary = {
        "models": summaries,
        "comparison": comparison,
        "known_tools": sorted(KNOWN_TOOLS),
    }

    with (OUT_DIR / "failure_analysis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    with (OUT_DIR / "per_example_comparison.jsonl").open("w") as f:
        for row in per_example:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    write_report(summaries, comparison, per_example)
    write_examples(summaries, models, per_example)

    print(f"Wrote analysis to {OUT_DIR}")
    print("Report:", OUT_DIR / "failure_analysis_report.md")


if __name__ == "__main__":
    main()
