#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tool-call latency study runs.")
    parser.add_argument("--study_dir", default="outputs/latency_study/gemma4_31b_tool_rl_latency")
    parser.add_argument("--output_md", default=None)
    return parser.parse_args()


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_generation_seconds(summary: Dict[str, Any]) -> str:
    generation_seconds = nested(summary, "timing_seconds", "generation")
    base = fmt_float(generation_seconds)
    try:
        loaded = int(summary.get("loaded_examples") or 0)
        generation_value = float(generation_seconds)
    except (TypeError, ValueError):
        return base
    if loaded > 0 and loaded != 200:
        normalized = generation_value * 200.0 / loaded
        return f"{base} ({fmt_float(normalized)}/200eq)"
    return base


def fmt_serving_seconds(summary: Dict[str, Any]) -> str:
    serving_seconds = nested(summary, "timing_seconds", "serving_generation")
    if serving_seconds is None:
        serving_seconds = nested(summary, "timing_seconds", "generation")
    base = fmt_float(serving_seconds)
    try:
        loaded = int(summary.get("loaded_examples") or 0)
        serving_value = float(serving_seconds)
    except (TypeError, ValueError):
        return base
    if loaded > 0 and loaded != 200:
        normalized = serving_value * 200.0 / loaded
        return f"{base} ({fmt_float(normalized)}/200eq)"
    return base


def nested(summary: Dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_summaries(study_dir: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in sorted((study_dir / "runs").glob("*/latency_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["_path"] = str(path.parent)
        summaries.append(summary)
    return summaries


def summary_sort_key(summary: Dict[str, Any]) -> tuple:
    label = str(summary.get("run_label") or Path(summary.get("_path", "")).name)
    is_mtp = int("mtp" in label.lower() or nested(summary, "speculative_config", "model") is not None)
    try:
        tp = int(summary.get("vllm_tensor_parallel_size") or 0)
    except (TypeError, ValueError):
        tp = 0
    try:
        concurrency = int(summary.get("vllm_async_concurrency") or 0)
    except (TypeError, ValueError):
        concurrency = 0
    try:
        loaded = int(summary.get("loaded_examples") or 0)
    except (TypeError, ValueError):
        loaded = 0
    return (is_mtp, tp, concurrency, loaded, label)


def is_mtp_summary(summary: Dict[str, Any]) -> bool:
    label = str(summary.get("run_label") or Path(summary.get("_path", "")).name).lower()
    return "mtp" in label or bool(summary.get("spec_model")) or nested(summary, "speculative_config", "model") is not None


def is_special_batched_compare(summary: Dict[str, Any]) -> bool:
    return summary.get("vllm_max_num_batched_tokens") is not None


def average_tool_calls(summary: Dict[str, Any]) -> float:
    loaded = int(summary.get("loaded_examples") or 0)
    tool_calls = nested(summary, "tool_behavior", "tool_calls_per_example") or {}
    if not loaded:
        return 0.0
    return sum(int(k) * int(v) for k, v in tool_calls.items()) / loaded


def spec_decode_stats(summary: Dict[str, Any]) -> Dict[str, Any]:
    stats = summary.get("spec_decode_stats")
    return stats if isinstance(stats, dict) else {}


def append_results_table(
    lines: List[str],
    summaries: List[Dict[str, Any]],
    include_path: bool = True,
    drop_all_na_columns: bool = False,
) -> None:
    columns = [
        ("run", "---"),
        ("TP", "---:"),
        ("concurrency", "---:"),
        ("max batched tok", "---:"),
        ("spec tok", "---:"),
        ("accept rate", "---:"),
        ("mean accept len", "---:"),
        ("accepted/drafted tok", "---:"),
        ("EX acc", "---:"),
        ("correct/total", "---:"),
        ("total gen sec", "---:"),
        ("serving gen sec", "---:"),
        ("serving ex/s", "---:"),
        ("avg e2e sec", "---:"),
        ("TTFT p50", "---:"),
        ("TTFT p95", "---:"),
        ("e2e p50", "---:"),
        ("e2e p95", "---:"),
        ("avg decode sec", "---:"),
        ("out tok/s serving", "---:"),
        ("avg tool calls", "---:"),
        ("pred executed", "---:"),
    ]
    if include_path:
        columns.append(("path", "---"))

    rendered_rows: List[Dict[str, str]] = []
    for summary in sorted(summaries, key=summary_sort_key):
        accuracy = nested(summary, "accuracy") or {}
        execution = nested(summary, "execution_stats") or {}
        loaded = int(summary.get("loaded_examples") or 0)
        serving_out_tps = nested(summary, "throughput", "completion_tokens_per_second_serving")
        if serving_out_tps in (None, 0):
            serving_out_tps = nested(summary, "throughput", "completion_tokens_per_second_generation")
        serving_exps = nested(summary, "throughput", "examples_per_second_serving")
        if serving_exps in (None, 0):
            serving_exps = nested(summary, "throughput", "examples_per_second_generation")
        spec_stats = spec_decode_stats(summary)
        row = {
            "run": str(summary.get("run_label", Path(summary["_path"]).name)),
            "TP": str(summary.get("vllm_tensor_parallel_size")),
            "concurrency": str(summary.get("vllm_async_concurrency")),
            "max batched tok": str(summary.get("vllm_max_num_batched_tokens") or "default"),
            "spec tok": str(summary.get("spec_tokens") or "n/a"),
            "accept rate": fmt_percent(spec_stats.get("acceptance_rate")),
            "mean accept len": fmt_float(spec_stats.get("mean_acceptance_length"), 3),
            "accepted/drafted tok": f"{spec_stats.get('num_accepted_tokens', 'n/a')}/{spec_stats.get('num_draft_tokens', 'n/a')}",
            "EX acc": f"{fmt_float(accuracy.get('accuracy'))}%",
            "correct/total": f"{accuracy.get('correct', 'n/a')}/{accuracy.get('count', 'n/a')}",
            "total gen sec": fmt_generation_seconds(summary),
            "serving gen sec": fmt_serving_seconds(summary),
            "serving ex/s": fmt_float(serving_exps, 3),
            "avg e2e sec": fmt_float(nested(summary, "latency_seconds", "example_total", "average"), 3),
            "TTFT p50": fmt_float(nested(summary, "latency_seconds", "ttft", "p50"), 3),
            "TTFT p95": fmt_float(nested(summary, "latency_seconds", "ttft", "p95"), 3),
            "e2e p50": fmt_float(nested(summary, "latency_seconds", "example_total", "p50"), 3),
            "e2e p95": fmt_float(nested(summary, "latency_seconds", "example_total", "p95"), 3),
            "avg decode sec": fmt_float(nested(summary, "latency_seconds", "example_decode_after_ttft", "average"), 3),
            "out tok/s serving": fmt_float(serving_out_tps, 2),
            "avg tool calls": f"{average_tool_calls(summary):.3f}",
            "pred executed": f"{execution.get('pred_sql_executed', 'n/a')}/{loaded}",
        }
        if include_path:
            row["path"] = f"`{summary['_path']}`"
        rendered_rows.append(row)

    if drop_all_na_columns:
        columns = [
            column
            for column in columns
            if not all(row.get(column[0]) in {"n/a", "n/a/n/a"} for row in rendered_rows)
        ]

    lines.append("| " + " | ".join(column[0] for column in columns) + " |")
    lines.append("| " + " | ".join(column[1] for column in columns) + " |")
    for row in rendered_rows:
        lines.append("| " + " | ".join(row.get(column[0], "n/a") for column in columns) + " |")


def render_markdown(study_dir: Path, summaries: List[Dict[str, Any]]) -> str:
    lines = [
        "# Gemma 4 31B Tool-Call Latency Study",
        "",
        "## Purpose",
        "",
        "Measure async vLLM latency, throughput, token usage, tool behavior, and BIRD execution accuracy on `outputs/old-dev-schema-tool-unpatched.jsonl`. Most runs use the first 200 examples; the concurrency=128 run uses 400 examples and reports a 200-example-equivalent generation time.",
        "",
        "## Method",
        "",
        "- Backend: async vLLM",
        "- vLLM env: `nl2sql_vllm024` / vLLM `0.24.0`",
        "- Model: `google/gemma-4-31B-it`",
        "- Temperature: `0.0`",
        "- Async concurrency: `16` for the initial TP sweep",
        "- Sampler: `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid FlashInfer sampler JIT/header mismatch on this host",
        "- Max prompt length: `34000`",
        "- Max output tokens: `8000`",
        "- vLLM max model length: `43500`",
        "- Max tool rounds: `8`",
        "- TTFT is measured from `engine.generate(...)` submission to the first streamed request output with text/token content.",
        "- End-to-end example latency includes all tool-call rounds and synchronous tool execution.",
        "- `serving gen sec` falls back to total generation time for older runs that were recorded before engine-load timing was split out.",
        "",
        "## Concurrency Note",
        "",
        "vLLM has an internal scheduler and request queue, but this study still sets `--vllm_async_concurrency` because the repo's async caller uses it as the application-level limit for in-flight examples. Holding it fixed at `16` makes the initial TP sweep comparable; the TP=8 concurrency sweep then measures loaded-serving behavior.",
        "",
        "## Results",
        "",
    ]
    if not summaries:
        lines.extend(["No completed runs found yet.", ""])
        return "\n".join(lines)

    main_summaries = [
        summary
        for summary in summaries
        if not is_mtp_summary(summary) and not is_special_batched_compare(summary)
    ]
    append_results_table(lines, main_summaries, drop_all_na_columns=True)

    mtp_compare = [
        summary
        for summary in summaries
        if int(summary.get("vllm_tensor_parallel_size") or 0) == 8
        and int(summary.get("vllm_async_concurrency") or 0) == 32
        and int(summary.get("loaded_examples") or 0) == 200
        and is_special_batched_compare(summary)
    ]
    if mtp_compare:
        lines.extend(
            [
                "",
                "## MTP Comparison",
                "",
                "Matched TP=8/concurrency=32 runs using the same first 200 examples. `serving gen sec` excludes vLLM engine load, compilation, and CUDA graph capture; `total gen sec` includes them.",
                "",
            ]
        )
        append_results_table(lines, mtp_compare)

    lines.extend(
        [
            "",
            "## Follow-Up",
            "",
            "- Compare TP=8 concurrency settings using both throughput and user-visible end-to-end latency; higher concurrency can improve aggregate throughput while worsening per-request latency.",
            "- Compare the MTP assistant row against the non-MTP TP=8/concurrency=32 row.",
            "- Add one non-async vLLM baseline only if we need to quantify the benefit of async scheduling.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    study_dir = Path(args.study_dir)
    output_md = Path(args.output_md) if args.output_md else study_dir / "latency_study.md"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(study_dir)
    output_md.write_text(render_markdown(study_dir, summaries) + "\n", encoding="utf-8")
    print(f"[latency-summary] wrote {output_md} with {len(summaries)} run(s)")


if __name__ == "__main__":
    main()
