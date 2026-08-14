#!/usr/bin/env python3
"""Run BIRD inference through a Muse-Glimmer OpenAI-compatible vLLM server.

Muse-Glimmer uses channel-scoped ATEM tool calls. The vLLM server converts those
into standard OpenAI ``tool_calls`` when launched with the Muse parsers, so this
runner keeps the repo-side loop at the OpenAI message level instead of parsing
Gemma's compact ``call:tool{...}`` text format.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nl2sql_gspo.inference_tool_executor import configure_tool_env, execute_tool_calls
from nl2sql_gspo.sql_utils import extract_sql
from scripts.run_inference_bird import (
    BIRD_SPLIT_MARKER,
    build_per_example_report_rows,
    build_summary,
    evaluate_predictions,
    load_diff_rows,
    load_rows,
    print_summary_tables,
    write_per_example_report_csv,
    write_run_report_markdown,
    write_summary_csv,
    write_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="muse-glimmer")
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev_unpatched.json")
    parser.add_argument("--output_dir", default="outputs/inference/dev/old-dev-schema-tool-unpatched/Muse-Glimmer-30B/temp0_openai_tool")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--reasoning_strength", default="high", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--no_prompt_rewrite", action="store_true")
    parser.add_argument("--no_force_finalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Fields consumed by the shared markdown renderer.
    args.inference_backend = "muse_openai_server"
    args.model_name_or_path = args.model
    args.max_prompt_length = "server"
    args.vllm_tensor_parallel_size = "server"
    args.vllm_data_parallel_size = "server"
    args.vllm_async_concurrency = "server"
    args.vllm_gpu_memory_utilization = "server"
    args.vllm_max_model_len = "server"
    return args


def slice_rows(rows: List[Dict[str, Any]], start_index: int, end_index: int) -> List[Dict[str, Any]]:
    start = max(0, start_index)
    end = len(rows) if end_index < 0 else min(len(rows), end_index)
    if start > end:
        raise ValueError(f"start_index={start_index} must be <= end_index={end_index}")
    sliced = rows[start:end]
    for offset, row in enumerate(sliced):
        row.setdefault("source_idx", start + offset)
    return sliced


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already contains files; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)


def post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc


def get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {body}") from exc


def wait_for_server(server_url: str, timeout_s: float = 5.0) -> None:
    models_url = server_url.rstrip("/") + "/models"
    loaded = get_json(models_url, timeout_s)
    model_ids = [item.get("id", "") for item in loaded.get("data", [])]
    print(f"[muse] server reachable; models={model_ids}")


def rewrite_system_prompt_for_muse(content: str, reasoning_strength: str) -> str:
    text = content

    replacements = {
        "Native tool-call syntax is mandatory:": "Runtime tool calls are mandatory:",
        "Emit tool calls exactly as `call:tool_name{arg1:value1,arg2:value2}`.": (
            "When you need a tool, use the tool-calling interface provided by the runtime."
        ),
        "After emitting one `call:...{...}`, stop immediately and wait for the tool response before writing any more scratch-pad text, another tool call, or a final answer.": (
            "After requesting one tool call, stop immediately and wait for the tool response before writing more scratch-pad text, another tool call, or a final answer."
        ),
        "Do not wrap tool calls in <tool_code>, XML tags, markdown fences, or JSON-only blocks.": (
            "Do not write tool calls manually in prose, markdown fences, XML tags, or JSON-only blocks."
        ),
        "Do not write \"Then call ...\" followed by raw JSON. The actual assistant output must be the native `call:...{...}` line.": (
            "Do not write \"Then call ...\" followed by raw JSON. Actually request the runtime tool call."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if "Reasoning strength:" not in text:
        text = text.rstrip() + f"\n\nReasoning strength: {reasoning_strength}."
    return text


def generation_messages(row: Dict[str, Any], rewrite: bool, reasoning_strength: str) -> List[Dict[str, Any]]:
    messages = row.get("prompt") or [message for message in row.get("messages", []) if message.get("role") != "assistant"]
    rendered: List[Dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        if rewrite and copied.get("role") == "system" and isinstance(copied.get("content"), str):
            copied["content"] = rewrite_system_prompt_for_muse(copied["content"], reasoning_strength)
        rendered.append(copied)
    return rendered


def normalize_tool_call(call: Dict[str, Any], index: int) -> Dict[str, Any]:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw_arguments": arguments}
    return {
        "id": call.get("id") or f"call_{index}",
        "type": "function",
        "function": {
            "name": function.get("name", ""),
            "arguments": arguments,
        },
    }


def openai_assistant_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call.get("id", ""),
        "type": "function",
        "function": {
            "name": function.get("name", ""),
            "arguments": arguments,
        },
    }


def assistant_text(message: Dict[str, Any]) -> str:
    parts = []
    for key in ("reasoning_content", "reasoning", "content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def run_one(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    messages = generation_messages(
        row,
        rewrite=not args.no_prompt_rewrite,
        reasoning_strength=args.reasoning_strength,
    )
    tools = row.get("tools") or []
    transcript: List[Dict[str, Any]] = []
    tool_names: List[str] = []
    completion_tokens = 0
    prompt_tokens = 0
    stop_reason = "finished"
    final_text = ""

    for round_index in range(args.max_tool_rounds + 1):
        payload = {
            "model": args.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_new_tokens,
        }
        response = post_json(
            args.server_url.rstrip("/") + "/chat/completions",
            payload,
            timeout_s=args.request_timeout,
        )
        usage = response.get("usage") or {}
        prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
        completion_tokens += int(usage.get("completion_tokens") or 0)

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_tool_calls = message.get("tool_calls") or []
        text = assistant_text(message)
        if text:
            final_text = text

        transcript.append({"assistant": message})

        if not raw_tool_calls:
            stop_reason = choice.get("finish_reason") or "finished"
            break

        if round_index >= args.max_tool_rounds:
            stop_reason = "max_tool_rounds"
            if not args.no_force_finalize:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The tool budget is exhausted. Do not call another tool. "
                            "Using the prior tool responses, provide the required final answer now. "
                            "Return the SQL only inside <final_answer><sql_code>...</sql_code></final_answer>."
                        ),
                    }
                )
                final_payload = {
                    "model": args.model,
                    "messages": messages,
                    "tool_choice": "none",
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_new_tokens,
                }
                final_response = post_json(
                    args.server_url.rstrip("/") + "/chat/completions",
                    final_payload,
                    timeout_s=args.request_timeout,
                )
                usage = final_response.get("usage") or {}
                prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
                completion_tokens += int(usage.get("completion_tokens") or 0)
                final_choice = (final_response.get("choices") or [{}])[0]
                final_message = final_choice.get("message") or {}
                transcript.append({"assistant_forced_final": final_message})
                final_text = assistant_text(final_message) or final_text
                if not (final_message.get("tool_calls") or []):
                    stop_reason = "forced_final"
            break

        tool_calls = [normalize_tool_call(call, index) for index, call in enumerate(raw_tool_calls)]
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [openai_assistant_tool_call(call) for call in tool_calls],
            }
        )

        tool_responses = execute_tool_calls(tool_calls, timeout_s=args.eval_timeout)
        for tool_call, tool_response in zip(tool_calls, tool_responses):
            name = tool_response.get("name") or tool_call.get("function", {}).get("name", "")
            tool_names.append(name)
            content = json.dumps(tool_response.get("raw_response"), ensure_ascii=False, default=str)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": content,
                }
            )
            transcript.append({"tool": {"name": name, "content": content}})

    else:
        stop_reason = "max_tool_rounds"

    detail = {
        "idx": row.get("source_idx", -1),
        "source_idx": row.get("source_idx", -1),
        "db_id": row.get("db_id", ""),
        "prompt_tokens": prompt_tokens,
        "prediction_text": final_text,
        "pred_sql": extract_sql(final_text),
        "completion_token_count": completion_tokens,
        "tool_rounds": len(tool_names),
        "tool_call_count": len(tool_names),
        "tool_names": tool_names,
        "tool_order": " -> ".join(tool_names),
        "stop_reason": stop_reason,
        "error_message": "",
        "transcript": transcript,
    }
    return final_text, detail


def generate_predictions(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    predictions: Dict[str, str] = {}
    details: List[Dict[str, Any]] = [None] * len(rows)

    def generate_index(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        source_idx = row.get("source_idx", index)
        print(f"[muse] generating {index + 1}/{len(rows)} idx={source_idx} db_id={row.get('db_id', '')}")
        try:
            _, detail = run_one(row, args)
        except Exception as exc:
            detail = {
                "idx": source_idx,
                "source_idx": source_idx,
                "db_id": row.get("db_id", ""),
                "prompt_tokens": 0,
                "prediction_text": "",
                "pred_sql": "",
                "completion_token_count": 0,
                "tool_rounds": 0,
                "tool_call_count": 0,
                "tool_names": [],
                "tool_order": "",
                "stop_reason": "error",
                "error_message": str(exc),
                "transcript": [],
            }
            print(f"[muse] ERROR idx={source_idx}: {exc}")
        return detail

    worker_count = max(1, int(args.concurrency))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(generate_index, index, row): (index, row)
            for index, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            index, row = futures[future]
            detail = future.result()
            completed += 1
            source_idx = row.get("source_idx", index)
            print(
                f"[muse] completed {completed}/{len(rows)} idx={source_idx} "
                f"stop={detail.get('stop_reason')} calls={detail.get('tool_call_count')}"
            )
            details[index] = detail

    for index, row in enumerate(rows):
        detail = details[index]
        source_idx = row.get("source_idx", index)
        pred_sql = detail.get("pred_sql", "")
        predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{row.get('db_id', '')}"
    return predictions, details


def build_muse_generation_stats(details: List[Dict[str, Any]], filtered_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stop_counts = Counter(str(detail.get("stop_reason") or "unknown") for detail in details)
    tool_counts = Counter()
    for detail in details:
        tool_counts.update(detail.get("tool_names") or [])
    prompt_tokens = [int(detail.get("prompt_tokens") or 0) for detail in details]
    completion_total = sum(int(detail.get("completion_token_count") or 0) for detail in details)
    tool_call_total = sum(int(detail.get("tool_call_count") or 0) for detail in details)
    tool_round_total = sum(int(detail.get("tool_rounds") or 0) for detail in details)
    total = len(details)
    return {
        "generated_examples": total,
        "filtered_examples": len(filtered_rows),
        "stop_reason_counts": dict(stop_counts),
        "tool_call_count_total": tool_call_total,
        "tool_round_count_total": tool_round_total,
        "avg_tool_calls_per_example": tool_call_total / max(1, total),
        "avg_tool_rounds_per_example": tool_round_total / max(1, total),
        "tool_name_counts": dict(tool_counts),
        "completion_token_total": completion_total,
        "avg_completion_tokens": completion_total / max(1, total),
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
    }


def main() -> None:
    run_started_at = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    configure_tool_env(args.database_dir)
    wait_for_server(args.server_url)

    rows = load_rows(args.input_file, -1)
    rows = slice_rows(rows, args.start_index, args.end_index)
    if args.num_examples >= 0:
        rows = rows[: args.num_examples]
    diff_rows = load_diff_rows(args.diff_json_path)
    print(
        f"[muse] loaded rows={len(rows)} diff_rows={len(diff_rows)} "
        f"start_index={args.start_index} end_index={args.end_index}"
    )

    generation_started_at = time.monotonic()
    predictions, details = generate_predictions(rows, args)
    generation_seconds = time.monotonic() - generation_started_at

    predictions_path = output_dir / "predict_dev.json"
    details_path = output_dir / "prediction_details.jsonl"
    filtered_path = output_dir / "filtered_examples.jsonl"
    per_example_eval_path = output_dir / "eval_results.jsonl"
    summary_path = output_dir / "eval_summary.json"
    summary_markdown_path = output_dir / "eval_summary.md"
    run_report_path = output_dir / "run_report.md"
    per_example_report_csv_path = output_dir / "per_example_report.csv"
    difficulty_csv_path = output_dir / "eval_summary_by_difficulty.csv"
    db_csv_path = output_dir / "eval_summary_by_db.csv"

    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    with details_path.open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    filtered_rows: List[Dict[str, Any]] = []
    filtered_path.write_text("", encoding="utf-8")

    evaluation_started_at = time.monotonic()
    per_example_results, summary = evaluate_predictions(
        rows=rows,
        predictions=predictions,
        database_dir=args.database_dir,
        diff_rows=diff_rows,
        timeout_s=args.eval_timeout,
        eval_workers=args.eval_workers,
    )
    evaluation_seconds = time.monotonic() - evaluation_started_at
    total_seconds = time.monotonic() - run_started_at
    summary["timing_seconds"] = {
        "generation": generation_seconds,
        "evaluation": evaluation_seconds,
        "total": total_seconds,
    }
    summary["generation_stats"] = build_muse_generation_stats(details, filtered_rows)

    with per_example_eval_path.open("w", encoding="utf-8") as handle:
        for record in per_example_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_rows = build_per_example_report_rows(details, per_example_results)
    for row in report_rows:
        detail = next((item for item in details if item.get("source_idx") == row.get("idx")), None)
        if detail:
            row["tool_order"] = detail.get("tool_order", "")
    write_per_example_report_csv(report_rows, per_example_report_csv_path)
    write_summary_markdown(summary, summary_markdown_path, args, len(rows))
    write_run_report_markdown(summary, report_rows, run_report_path, args, len(rows))
    write_summary_csv(summary["by_difficulty"], difficulty_csv_path)
    write_summary_csv(summary["by_db"], db_csv_path)
    print_summary_tables(summary)
    print(f"[muse] saved summary={summary_path}")
    print(f"[muse] saved run_report={run_report_path}")


if __name__ == "__main__":
    main()
