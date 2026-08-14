#!/usr/bin/env python3
"""Run BIRD inference through a Qwen OpenAI-compatible vLLM server.

Qwen3.6 tool use is handled through OpenAI-compatible structured tool calls.
The BIRD input rows were originally built for Gemma, so this runner rewrites
only the system prompt contract while keeping the same database tools.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nl2sql_gspo.inference_tool_executor import configure_tool_env, execute_tool_calls
from nl2sql_gspo.sql_utils import extract_final_answer_sql
from scripts.run_inference_bird import (
    BIRD_SPLIT_MARKER,
    build_per_example_report_rows,
    evaluate_predictions,
    load_diff_rows,
    load_rows,
    print_summary_tables,
    write_per_example_report_csv,
    write_run_report_markdown,
    write_summary_csv,
    write_summary_markdown,
)


GEMMA_NATIVE_TOOL_BLOCK_RE = re.compile(
    r"Native tool-call syntax is mandatory:\n"
    r"- Emit tool calls exactly as `call:tool_name\{arg1:value1,arg2:value2\}`\.\n"
    r"- A tool call ends the assistant turn\. After emitting one `call:\.\.\.\{\.\.\.\}`, stop immediately and wait for the tool response before writing any more scratch-pad text, another tool call, or a final answer\.\n"
    r"- Do not wrap tool calls in <tool_code>, XML tags, markdown fences, or JSON-only blocks\.\n"
    r"- Do not write \"Then call \.\.\.\" followed by raw JSON\. The actual assistant output must be the native `call:\.\.\.\{\.\.\.\}` line\.\n"
    r"- Never invent or write tool responses\. Only use tool results that appear in the conversation after your tool call\.\n"
    r"- For SQL arguments, pass the SQL directly after `sql:`; quoting the entire SQL string is optional, but the whole SQL must be present\.\n*",
    re.DOTALL,
)

GEMMA_EXAMPLE_CALL_RE = re.compile(
    r"^call:(sqlite_query|bm25_search_sqlite|sqlite_peek)\{.*?\}\n?",
    re.MULTILINE,
)

OUTPUT_FORMAT_EXAMPLES_RE = re.compile(r"OUTPUT FORMAT EXAMPLES\n\n.*\Z", re.DOTALL)

QWEN_TOOL_BLOCK = (
    "Qwen structured tool calling is mandatory when you need database feedback:\n"
    "- The available functions are provided separately in the OpenAI-compatible tools field.\n"
    "- To use a tool, emit one structured tool call/function_call. Do not print tool-call JSON, XML, markdown, prose, or legacy inline call syntax in assistant text.\n"
    "- The visible assistant content for a tool turn should contain only <scratch_pad>...</scratch_pad>; the selected function name and arguments must be in the structured tool call field.\n"
    "- Use at most one tool call per assistant turn, then wait for the tool response.\n"
    "- Never invent or write tool responses. Only use tool results that appear in the conversation after your tool call.\n\n"
)

QWEN_OUTPUT_FORMAT_EXAMPLES = """OUTPUT FORMAT EXAMPLES

Tool turn visible assistant content:
<scratch_pad>
ExpectedOutputColumns=[customer_id, order_count]
CandidateSQL=SELECT c.customer_id, COUNT(DISTINCT o.order_id) AS order_count FROM customers AS c JOIN orders AS o ON c.customer_id=o.customer_id WHERE o.status='shipped' GROUP BY c.customer_id
I will execute this candidate before finalizing.
</scratch_pad>

For that tool turn, do not print the function call. Put the function name sqlite_query and its arguments in the structured tool_calls/function_call field supplied by the runtime.

Final answer, only after successful execution:
<scratch_pad>The last sqlite_query executed successfully. Returned columns match ExpectedOutputColumns, literals and numeric/date scales are verified or unambiguous.</scratch_pad>
<relevant_tables>table1, table2</relevant_tables>
<relevant_columns>table1.col1, table1.id, table2.ref_id, table2.metric</relevant_columns>
<final_answer>
<sql_code>SELECT ...</sql_code>
</final_answer>
"""

EMPTY_TOOL_RETRY_PROMPT = (
    "Your previous assistant message did not contain a final answer and did not "
    "contain a structured tool call. Continue now with exactly one structured "
    "runtime function call using the tools field. If your scratch pad contains "
    "CandidateSQL, call sqlite_query with that SQL."
)

FINALIZE_PROMPT = (
    "Do not call another tool. Using the prior reasoning and any tool responses, "
    "provide the required final answer now. Return the SQL only inside "
    "<final_answer><sql_code>...</sql_code></final_answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3p6-35b-a3b")
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev_unpatched.json")
    parser.add_argument(
        "--output_dir",
        default="outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-35B-A3B/smoke_temp0_openai_tool",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--no_prompt_rewrite", action="store_true")
    parser.add_argument("--no_force_finalize", action="store_true")
    parser.add_argument("--empty_tool_retries", type=int, default=1)
    parser.add_argument(
        "--tool_choice_policy",
        choices=["auto", "required_first", "required_until_tool", "required_always"],
        default="required_first",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.inference_backend = "qwen_openai_server"
    args.model_name_or_path = args.model
    args.max_prompt_length = "server"
    args.vllm_tensor_parallel_size = "server"
    args.vllm_data_parallel_size = "server"
    args.vllm_async_concurrency = "server"
    args.vllm_gpu_memory_utilization = "server"
    args.vllm_max_model_len = "server"
    return args


def get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - local vLLM server
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - local vLLM server
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def wait_for_server(server_url: str, timeout_s: float = 5.0) -> None:
    models_url = server_url.rstrip("/") + "/models"
    loaded = get_json(models_url, timeout_s)
    model_ids = [item.get("id", "") for item in loaded.get("data", [])]
    print(f"[qwen] server reachable; models={model_ids}")


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


def rewrite_system_prompt_for_qwen(content: str) -> str:
    text = GEMMA_NATIVE_TOOL_BLOCK_RE.sub(QWEN_TOOL_BLOCK, content)
    replacements = {
        "Use at most one native tool call in any single assistant turn.": (
            "Use at most one structured runtime tool call in any single assistant turn."
        ),
        "one native tool call to bm25_search_sqlite, sqlite_peek, or sqlite_query": (
            "one structured runtime tool call to bm25_search_sqlite, sqlite_peek, or sqlite_query"
        ),
        "Native tool-call syntax is mandatory:": "Qwen structured tool calling is mandatory:",
        "Emit tool calls exactly as `call:tool_name{arg1:value1,arg2:value2}`.": (
            "Use the OpenAI-compatible structured tool_calls/function_call field."
        ),
        "Do not wrap tool calls in <tool_code>, XML tags, markdown fences, or JSON-only blocks.": (
            "Do not print tool calls in assistant text."
        ),
        "Do not write \"Then call ...\" followed by raw JSON. The actual assistant output must be the native `call:...{...}` line.": (
            "Do not describe a future tool call in prose; emit the structured tool call instead."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = OUTPUT_FORMAT_EXAMPLES_RE.sub(QWEN_OUTPUT_FORMAT_EXAMPLES, text)
    text = GEMMA_EXAMPLE_CALL_RE.sub("", text)
    return text


def generation_messages(row: Dict[str, Any], rewrite: bool) -> List[Dict[str, Any]]:
    messages = row.get("prompt") or [
        message for message in row.get("messages", []) if message.get("role") != "assistant"
    ]
    rendered: List[Dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        if rewrite and copied.get("role") == "system" and isinstance(copied.get("content"), str):
            copied["content"] = rewrite_system_prompt_for_qwen(copied["content"])
        rendered.append(copied)
    return rendered


def message_text_fields(message: Dict[str, Any]) -> Dict[str, str]:
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "reasoning": message.get("reasoning") or "",
    }


def assistant_text(message: Dict[str, Any]) -> str:
    fields = message_text_fields(message)
    parts = [value.strip() for value in fields.values() if isinstance(value, str) and value.strip()]
    return "\n".join(parts)


def allowed_tool_names(tools: List[Dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


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


def normalize_valid_tool_calls(
    raw_tool_calls: List[Dict[str, Any]],
    valid_names: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for index, raw_call in enumerate(raw_tool_calls):
        call = normalize_tool_call(raw_call, index)
        name = call.get("function", {}).get("name", "")
        if name in valid_names:
            valid.append(call)
        else:
            rejected.append(call)
    return valid, rejected


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


def tool_response_message(tool_call: Dict[str, Any], tool_response: Dict[str, Any]) -> Dict[str, Any]:
    name = tool_response.get("name") or tool_call.get("function", {}).get("name", "")
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": name,
        "content": json.dumps(tool_response.get("raw_response"), ensure_ascii=False, default=str),
    }


def render_transcript_text(rounds: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for item in rounds:
        assistant_parts = []
        for key in ("assistant_reasoning_content", "assistant_reasoning", "assistant_content"):
            if item.get(key):
                assistant_parts.append(item[key])
        if assistant_parts:
            chunks.append("\n".join(part.strip() for part in assistant_parts if part.strip()))
        for call in item.get("tool_calls") or []:
            chunks.append(json.dumps({"tool_call": call}, ensure_ascii=False, default=str))
        for response in item.get("tool_responses") or []:
            chunks.append(json.dumps({"tool_response": response}, ensure_ascii=False, default=str))
    return "\n".join(chunk for chunk in chunks if chunk)


def request_payload(
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_choice: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "tool_choice": tool_choice,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 20,
        "max_tokens": args.max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)},
    }
    if tools and tool_choice != "none":
        payload["tools"] = tools
        payload["parallel_tool_calls"] = False
    return payload


def choose_tool_choice(args: argparse.Namespace, round_index: int, tool_names: List[str]) -> str:
    if args.tool_choice_policy == "required_always":
        return "required"
    if args.tool_choice_policy == "required_first" and round_index == 0:
        return "required"
    if args.tool_choice_policy == "required_until_tool" and not tool_names:
        return "required"
    return "auto"


def force_finalize(
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    rounds: List[Dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    reason: str,
) -> Tuple[str, int, int]:
    messages.append({"role": "user", "content": FINALIZE_PROMPT})
    response = post_json(
        args.server_url.rstrip("/") + "/chat/completions",
        request_payload(args, messages, [], "none"),
        timeout_s=args.request_timeout,
    )
    usage = response.get("usage") or {}
    prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
    completion_tokens += int(usage.get("completion_tokens") or 0)
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    rounds.append(
        {
            "round_index": len(rounds),
            "finish_reason": choice.get("finish_reason"),
            "assistant_content": message.get("content") or "",
            "assistant_reasoning_content": message.get("reasoning_content") or "",
            "assistant_reasoning": message.get("reasoning") or "",
            "raw_assistant_message": message,
            "tool_calls": message.get("tool_calls") or [],
            "tool_responses": [],
            "forced_final": True,
            "forced_final_reason": reason,
        }
    )
    return assistant_text(message), prompt_tokens, completion_tokens


def run_one(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    messages = generation_messages(row, rewrite=not args.no_prompt_rewrite)
    tools = row.get("tools") or []
    valid_tool_names = allowed_tool_names(tools)
    rounds: List[Dict[str, Any]] = []
    tool_names: List[str] = []
    rejected_tool_names: List[str] = []
    completion_tokens = 0
    prompt_tokens = 0
    stop_reason = "finished"
    final_text = ""
    empty_retries = 0
    force_required_next = False

    for round_index in range(args.max_tool_rounds + 1):
        if tools and force_required_next:
            tool_choice = "required"
            force_required_next = False
        else:
            tool_choice = choose_tool_choice(args, round_index, tool_names) if tools else "none"
        response = post_json(
            args.server_url.rstrip("/") + "/chat/completions",
            request_payload(args, messages, tools, tool_choice),
            timeout_s=args.request_timeout,
        )
        usage = response.get("usage") or {}
        prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
        completion_tokens += int(usage.get("completion_tokens") or 0)

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text_fields = message_text_fields(message)
        normalized_tool_calls, rejected_tool_calls = normalize_valid_tool_calls(
            message.get("tool_calls") or [],
            valid_tool_names,
        )
        rejected_tool_names.extend(
            call.get("function", {}).get("name", "") for call in rejected_tool_calls
        )

        round_record: Dict[str, Any] = {
            "round_index": round_index,
            "request_tool_choice": tool_choice,
            "finish_reason": choice.get("finish_reason"),
            "assistant_content": text_fields["content"],
            "assistant_reasoning_content": text_fields["reasoning_content"],
            "assistant_reasoning": text_fields["reasoning"],
            "raw_assistant_message": message,
            "tool_calls": normalized_tool_calls,
            "rejected_tool_calls": rejected_tool_calls,
            "tool_call_source": "openai_structured",
            "tool_responses": [],
        }

        if not normalized_tool_calls:
            text = assistant_text(message)
            final_sql = extract_final_answer_sql(text)
            if final_sql:
                final_text = text
                stop_reason = choice.get("finish_reason") or "finished"
                rounds.append(round_record)
                break

            rounds.append(round_record)
            if tools and empty_retries < max(0, args.empty_tool_retries):
                empty_retries += 1
                force_required_next = True
                messages.append({"role": "assistant", "content": message.get("content") or ""})
                messages.append({"role": "user", "content": EMPTY_TOOL_RETRY_PROMPT})
                continue

            if tool_names and not args.no_force_finalize:
                final_text, prompt_tokens, completion_tokens = force_finalize(
                    args,
                    messages,
                    rounds,
                    prompt_tokens,
                    completion_tokens,
                    "no_tool_no_final_after_tool",
                )
                stop_reason = "forced_final"
            else:
                final_text = text
                stop_reason = "no_tool_no_final"
            break

        rounds.append(round_record)
        if round_index >= args.max_tool_rounds:
            if not args.no_force_finalize:
                final_text, prompt_tokens, completion_tokens = force_finalize(
                    args,
                    messages,
                    rounds,
                    prompt_tokens,
                    completion_tokens,
                    "max_tool_rounds",
                )
                stop_reason = "forced_final"
            else:
                stop_reason = "max_tool_rounds"
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [openai_assistant_tool_call(call) for call in normalized_tool_calls],
            }
        )

        tool_responses = execute_tool_calls(normalized_tool_calls, timeout_s=args.eval_timeout)
        round_record["tool_responses"] = tool_responses
        for tool_call, tool_response in zip(normalized_tool_calls, tool_responses):
            name = tool_response.get("name") or tool_call.get("function", {}).get("name", "")
            tool_names.append(name)
            messages.append(tool_response_message(tool_call, tool_response))

    transcript_text = render_transcript_text(rounds)
    if not final_text and transcript_text:
        final_text = transcript_text

    detail = {
        "idx": row.get("source_idx", -1),
        "source_idx": row.get("source_idx", -1),
        "db_id": row.get("db_id", ""),
        "prompt_tokens": prompt_tokens,
        "prediction_text": final_text,
        "transcript_text": transcript_text,
        "pred_sql": extract_final_answer_sql(final_text),
        "completion_token_count": completion_tokens,
        "tool_rounds": len(tool_names),
        "tool_call_count": len(tool_names),
        "tool_names": tool_names,
        "tool_order": " -> ".join(tool_names),
        "rejected_tool_names": rejected_tool_names,
        "rejected_tool_call_count": len(rejected_tool_names),
        "stop_reason": stop_reason,
        "error_message": "",
        "qwen_enable_thinking": bool(args.enable_thinking),
        "prompt_rewritten": not args.no_prompt_rewrite,
        "tool_choice_policy": args.tool_choice_policy,
        "empty_tool_retries": empty_retries,
        "rounds": rounds,
    }
    return final_text, detail


def generate_predictions(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    predictions: Dict[str, str] = {}
    details: List[Dict[str, Any]] = [None] * len(rows)  # type: ignore[list-item]

    def generate_index(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        source_idx = row.get("source_idx", index)
        print(f"[qwen] generating {index + 1}/{len(rows)} idx={source_idx} db_id={row.get('db_id', '')}")
        try:
            _, detail = run_one(row, args)
        except Exception as exc:
            detail = {
                "idx": source_idx,
                "source_idx": source_idx,
                "db_id": row.get("db_id", ""),
                "prompt_tokens": 0,
                "prediction_text": "",
                "transcript_text": "",
                "pred_sql": "",
                "completion_token_count": 0,
                "tool_rounds": 0,
                "tool_call_count": 0,
                "tool_names": [],
                "tool_order": "",
                "rejected_tool_names": [],
                "rejected_tool_call_count": 0,
                "stop_reason": "error",
                "error_message": str(exc),
                "rounds": [],
            }
            print(f"[qwen] ERROR idx={source_idx}: {exc}")
        return detail

    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
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
                f"[qwen] completed {completed}/{len(rows)} idx={source_idx} "
                f"stop={detail.get('stop_reason')} calls={detail.get('tool_call_count')}"
            )
            details[index] = detail

    for index, row in enumerate(rows):
        detail = details[index]
        source_idx = row.get("source_idx", index)
        pred_sql = detail.get("pred_sql", "")
        predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{row.get('db_id', '')}"
    return predictions, details


def build_qwen_generation_stats(details: List[Dict[str, Any]], filtered_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stop_counts = Counter(str(detail.get("stop_reason") or "unknown") for detail in details)
    tool_counts = Counter()
    rejected_counts = Counter()
    for detail in details:
        tool_counts.update(detail.get("tool_names") or [])
        rejected_counts.update(detail.get("rejected_tool_names") or [])
    prompt_tokens = [int(detail.get("prompt_tokens") or 0) for detail in details]
    completion_total = sum(int(detail.get("completion_token_count") or 0) for detail in details)
    tool_call_total = sum(int(detail.get("tool_call_count") or 0) for detail in details)
    tool_round_total = sum(int(detail.get("tool_rounds") or 0) for detail in details)
    rejected_total = sum(int(detail.get("rejected_tool_call_count") or 0) for detail in details)
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
        "rejected_tool_call_count_total": rejected_total,
        "rejected_tool_name_counts": dict(rejected_counts),
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
        f"[qwen] loaded rows={len(rows)} diff_rows={len(diff_rows)} "
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
    summary["generation_stats"] = build_qwen_generation_stats(details, filtered_rows)

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
    print(f"[qwen] saved summary={summary_path}")
    print(f"[qwen] saved details={details_path}")


if __name__ == "__main__":
    main()
