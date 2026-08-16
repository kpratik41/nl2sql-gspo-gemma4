#!/usr/bin/env python3
"""Run BIRD inference with Qwen-native tool calls on in-process async vLLM.

This avoids the OpenAI HTTP server loop while keeping Qwen's own chat-template
tool format:

<tool_call>
<function=sqlite_query>
<parameter=sql>
SELECT ...
</parameter>
</function>
</tool_call>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo import tool_loop_guard
from nl2sql_gspo.inference_tool_executor import configure_tool_env, execute_tool_calls
from nl2sql_gspo.sql_utils import extract_final_answer_sql
from scripts.run_inference_bird import (
    BIRD_SPLIT_MARKER,
    build_per_example_report_rows,
    ensure_output_dir,
    evaluate_predictions,
    load_diff_rows,
    load_rows,
    preview_text,
    print_summary_tables,
    should_log_each_example,
    should_log_progress_tick,
    write_per_example_report_csv,
    write_run_report_markdown,
    write_summary_csv,
    write_summary_markdown,
)
from scripts.run_inference_bird_qwen_server import (
    EMPTY_TOOL_RETRY_PROMPT,
    FINALIZE_PROMPT,
    allowed_tool_names,
    generation_messages,
    normalize_valid_tool_calls,
)


REQUIRED_TOOL_PROMPT = (
    "Before giving a final answer, continue with exactly one Qwen-native tool call "
    "using the available function-call format. Put any scratch-pad reasoning before "
    "the tool call, then end the assistant turn immediately after </tool_call>."
)

QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*"
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<body>.*?)"
    r"</function>\s*"
    r"</tool_call>",
    re.DOTALL,
)
QWEN_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<value>.*?)"
    r"\s*</parameter>",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev_unpatched.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--gpu_groups", nargs="+", default=None)
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=43000)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--preserve_thinking", action="store_true")
    parser.add_argument("--no_prompt_rewrite", action="store_true")
    parser.add_argument("--no_force_finalize", action="store_true")
    parser.add_argument("--empty_tool_retries", type=int, default=1)
    parser.add_argument(
        "--tool_choice_policy",
        choices=["auto", "required_first", "required_until_tool", "required_always"],
        default="required_first",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard_file", default=None)
    parser.add_argument("--shard_output_dir", default=None)
    args = parser.parse_args()
    args.inference_backend = "qwen_vllm_async"
    args.vllm_data_parallel_size = args.num_shards
    args.skip_generation = False
    return args


def slice_rows(rows: List[Dict[str, Any]], start_index: int, end_index: int) -> List[Dict[str, Any]]:
    start = max(0, start_index)
    end = len(rows) if end_index < 0 else min(len(rows), end_index)
    if start > end:
        raise ValueError(f"start_index={start_index} must be <= end_index={end_index}")
    sliced = rows[start:end]
    for offset, row in enumerate(sliced):
        row["source_idx"] = int(row.get("source_idx", start + offset))
    return sliced


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            source_idx = int(raw.get("source_idx", len(rows)))
            row = normalize_record(raw)
            row["source_idx"] = source_idx
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shard_rows(rows: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for row in rows:
        shards[int(row.get("source_idx", 0)) % num_shards].append(row)
    return shards


def default_gpu_groups(num_shards: int, tensor_parallel_size: int) -> List[str]:
    groups: List[str] = []
    next_gpu = 0
    for _ in range(num_shards):
        groups.append(",".join(str(gpu) for gpu in range(next_gpu, next_gpu + tensor_parallel_size)))
        next_gpu += tensor_parallel_size
    return groups


def token_id_for_text(tokenizer: Any, text: str) -> Optional[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return int(token_ids[0]) if len(token_ids) == 1 else None


def qwen_stop_token_ids(tokenizer: Any) -> List[int]:
    token_ids: List[int] = []
    for text in ("<|im_end|>",):
        token_id = token_id_for_text(tokenizer, text)
        if token_id is not None and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def render_qwen_prompt(tokenizer: Any, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools if tools else None,
        enable_thinking=bool(args.enable_thinking),
        preserve_thinking=bool(args.preserve_thinking),
    )


def parse_parameter_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped in {"true", "false"}:
        return stripped == "true"
    if stripped == "null":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def extract_qwen_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for index, match in enumerate(QWEN_TOOL_CALL_RE.finditer(text or "")):
        arguments: Dict[str, Any] = {}
        for param in QWEN_PARAMETER_RE.finditer(match.group("body") or ""):
            arguments[param.group("name")] = parse_parameter_value(param.group("value"))
        calls.append(
            {
                "id": f"call_{index}_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": arguments,
                },
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return calls


def split_qwen_text(text: str) -> Tuple[str, str]:
    stripped = (text or "").strip()
    if stripped.startswith("<think>") and "</think>" in stripped:
        before, after = stripped.split("</think>", 1)
        return before[len("<think>") :].strip(), after.strip()
    return "", stripped


def assistant_message_for_history(
    assistant_text: str,
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reasoning_content, content = split_qwen_text(assistant_text)
    if tool_calls:
        first_start = min(int(call.get("start", 0)) for call in tool_calls)
        content_before_call = assistant_text[:first_start].strip()
        reasoning_content, content = split_qwen_text(content_before_call)
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": [
            {
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": call.get("function", {}).get("name", ""),
                    "arguments": call.get("function", {}).get("arguments") or {},
                },
            }
            for call in tool_calls
        ],
    }


def tool_message_for_history(tool_call: Dict[str, Any], tool_response: Dict[str, Any]) -> Dict[str, Any]:
    name = tool_response.get("name") or tool_call.get("function", {}).get("name", "")
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id", ""),
        "name": name,
        "content": json.dumps(tool_response.get("raw_response"), ensure_ascii=False, default=str),
    }


def render_transcript_text(rounds: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for item in rounds:
        if item.get("assistant_text"):
            chunks.append(item["assistant_text"])
        for call in item.get("tool_calls") or []:
            chunks.append(json.dumps({"tool_call": call}, ensure_ascii=False, default=str))
        for response in item.get("tool_responses") or []:
            chunks.append(json.dumps({"tool_response": response}, ensure_ascii=False, default=str))
    return "\n".join(chunk for chunk in chunks if chunk)


def choose_tool_required(args: argparse.Namespace, round_index: int, tool_names: List[str]) -> bool:
    if args.tool_choice_policy == "required_always":
        return True
    if args.tool_choice_policy == "required_first" and round_index == 0:
        return True
    if args.tool_choice_policy == "required_until_tool" and not tool_names:
        return True
    return False


def append_required_tool_instruction(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    is_retry: bool,
) -> None:
    if not tools:
        return
    messages.append({"role": "user", "content": EMPTY_TOOL_RETRY_PROMPT if is_retry else REQUIRED_TOOL_PROMPT})


async def async_generate_text(
    engine: Any,
    sampling_params_cls: Any,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    request_prefix: str,
    stop: Optional[List[str]] = None,
) -> Tuple[str, int]:
    sampling_params = sampling_params_cls(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=stop,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=True,
    )
    final_output = None
    request_id = f"{request_prefix}-{uuid.uuid4().hex}"
    async for request_output in engine.generate(prompt_text, sampling_params, request_id=request_id):
        final_output = request_output
    first = final_output.outputs[0] if final_output and final_output.outputs else None
    return ((first.text or "").strip() if first else "", len(first.token_ids) if first else 0)


async def run_one_async(row: Dict[str, Any], args: argparse.Namespace, engine: Any, sampling_params_cls: Any, tokenizer: Any) -> Dict[str, Any]:
    # One deduplication scope per rollout. Both this runner and the pass@k
    # runner enter through here, so scoping it once covers both. The scope is
    # a no-op unless NL2SQL_TOOL_LOOP_GUARD is set, and its store dies with the
    # scope, so concurrent rollouts never share state.
    with tool_loop_guard.rollout_scope(f"{row.get('source_idx', -1)}"):
        return await _run_one_async_inner(row, args, engine, sampling_params_cls, tokenizer)


async def _run_one_async_inner(row: Dict[str, Any], args: argparse.Namespace, engine: Any, sampling_params_cls: Any, tokenizer: Any) -> Dict[str, Any]:
    messages = generation_messages(row, rewrite=not args.no_prompt_rewrite)
    tools = row.get("tools") or []
    valid_tool_names = allowed_tool_names(tools)
    prompt_tokens = 0
    completion_tokens = 0
    tool_names: List[str] = []
    rejected_tool_names: List[str] = []
    rounds: List[Dict[str, Any]] = []
    final_text = ""
    stop_reason = "finished"
    empty_retries = 0
    force_required_next = False
    stop_token_ids = qwen_stop_token_ids(tokenizer)

    for round_index in range(args.max_tool_rounds + 1):
        retry_required = force_required_next
        required = retry_required or choose_tool_required(args, round_index, tool_names)
        force_required_next = False
        if required and tools:
            append_required_tool_instruction(messages, tools, is_retry=retry_required)

        prompt_text = render_qwen_prompt(tokenizer, messages, tools, args)
        current_prompt_tokens = len(tokenizer(prompt_text, truncation=False)["input_ids"])
        prompt_tokens = max(prompt_tokens, current_prompt_tokens)
        available = args.vllm_max_model_len - current_prompt_tokens
        remaining = args.max_new_tokens - completion_tokens
        if available <= 0 or remaining <= 0:
            stop_reason = "context_length_exceeded" if available <= 0 else "max_new_tokens"
            break

        text, round_tokens = await async_generate_text(
            engine=engine,
            sampling_params_cls=sampling_params_cls,
            prompt_text=prompt_text,
            max_tokens=min(available, remaining),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_token_ids=stop_token_ids,
            stop=["</tool_call>"] if tools else None,
            request_prefix=f"idx{row.get('source_idx', -1)}-r{round_index}",
        )
        completion_tokens += round_tokens
        raw_tool_calls = extract_qwen_tool_calls(text)
        tool_calls, rejected_tool_calls = normalize_valid_tool_calls(raw_tool_calls, valid_tool_names)
        rejected_tool_names.extend(call.get("function", {}).get("name", "") for call in rejected_tool_calls)

        round_record: Dict[str, Any] = {
            "round_index": round_index,
            "request_tool_choice": "required" if required and tools else "auto",
            "assistant_text": text,
            "tool_calls": tool_calls,
            "rejected_tool_calls": rejected_tool_calls,
            "tool_call_source": "qwen_native_async",
            "tool_responses": [],
        }

        if not tool_calls:
            final_sql = extract_final_answer_sql(text)
            rounds.append(round_record)
            if final_sql:
                final_text = text
                stop_reason = "finished"
                break
            if tools and empty_retries < max(0, args.empty_tool_retries):
                empty_retries += 1
                messages.append({"role": "assistant", "content": text})
                force_required_next = True
                continue
            if tool_names and not args.no_force_finalize:
                messages.append({"role": "user", "content": FINALIZE_PROMPT})
                final_text, final_tokens = await async_generate_text(
                    engine=engine,
                    sampling_params_cls=sampling_params_cls,
                    prompt_text=render_qwen_prompt(tokenizer, messages, [], args),
                    max_tokens=min(args.max_new_tokens - completion_tokens, args.vllm_max_model_len),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stop_token_ids=stop_token_ids,
                    request_prefix=f"idx{row.get('source_idx', -1)}-final",
                )
                completion_tokens += final_tokens
                rounds.append(
                    {
                        "round_index": len(rounds),
                        "assistant_text": final_text,
                        "tool_calls": [],
                        "rejected_tool_calls": [],
                        "tool_responses": [],
                        "forced_final": True,
                    }
                )
                stop_reason = "forced_final"
            else:
                final_text = text
                stop_reason = "no_tool_no_final"
            break

        rounds.append(round_record)
        if round_index >= args.max_tool_rounds:
            # The round budget is spent but the model is still asking for tools.
            # Without this branch the rollout was simply cut off mid-loop and
            # returned no SQL at all -- an automatic zero. It never saw
            # FINALIZE_PROMPT, because the forced-final path above only fires
            # when the model stops calling tools of its own accord, so
            # `forced_final` never appeared in any run's stop_reason_counts.
            #
            # These rollouts are not lost causes: every truncated rollout in the
            # last full eval had already run a successful query, and 65 of 76 had
            # written out a CandidateSQL. Give them one non-tool turn to commit
            # to an answer using what they already have.
            #
            # The pending tool call is deliberately NOT executed -- doing so
            # would spend a round beyond max_tool_rounds -- and the assistant
            # turn that requested it is dropped, so the transcript never carries
            # a tool_call with no matching tool response.
            if tool_names and not args.no_force_finalize:
                messages.append({"role": "user", "content": FINALIZE_PROMPT})
                final_text, final_tokens = await async_generate_text(
                    engine=engine,
                    sampling_params_cls=sampling_params_cls,
                    prompt_text=render_qwen_prompt(tokenizer, messages, [], args),
                    max_tokens=max(1, min(args.max_new_tokens - completion_tokens, args.vllm_max_model_len)),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stop_token_ids=stop_token_ids,
                    request_prefix=f"idx{row.get('source_idx', -1)}-capfinal",
                )
                completion_tokens += final_tokens
                rounds.append(
                    {
                        "round_index": len(rounds),
                        "assistant_text": final_text,
                        "tool_calls": [],
                        "rejected_tool_calls": [],
                        "tool_responses": [],
                        "forced_final": True,
                    }
                )
                stop_reason = "forced_final_at_cap"
            else:
                stop_reason = "max_tool_rounds"
            break

        messages.append(assistant_message_for_history(text, tool_calls[:1]))
        tool_responses = await asyncio.to_thread(execute_tool_calls, tool_calls[:1], args.eval_timeout)
        round_record["tool_responses"] = tool_responses
        for tool_call, tool_response in zip(tool_calls[:1], tool_responses):
            name = tool_response.get("name") or tool_call.get("function", {}).get("name", "")
            tool_names.append(name)
            messages.append(tool_message_for_history(tool_call, tool_response))

    transcript_text = render_transcript_text(rounds)
    if not final_text and transcript_text:
        final_text = transcript_text

    return {
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
        "qwen_preserve_thinking": bool(args.preserve_thinking),
        "prompt_rewritten": not args.no_prompt_rewrite,
        "tool_choice_policy": args.tool_choice_policy,
        "empty_tool_retries": empty_retries,
        "rounds": rounds,
    }


async def generate_details_async(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    kept_rows: List[Dict[str, Any]] = []
    for row in rows:
        prompt = render_qwen_prompt(
            tokenizer,
            generation_messages(row, rewrite=not args.no_prompt_rewrite),
            row.get("tools") or [],
            args,
        )
        prompt_tokens = len(tokenizer(prompt, truncation=False)["input_ids"])
        if prompt_tokens <= args.max_prompt_length:
            kept = dict(row)
            kept["prompt_tokens"] = prompt_tokens
            kept_rows.append(kept)
        else:
            print(
                "[qwen-async] skipping over-length prompt "
                f"idx={row.get('source_idx')} db={row.get('db_id')} tokens={prompt_tokens}"
            )

    print(
        "[qwen-async] loading AsyncLLMEngine "
        f"tp={args.vllm_tensor_parallel_size} concurrency={args.vllm_async_concurrency} "
        f"max_model_len={args.vllm_max_model_len}"
    )
    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        enable_prefix_caching=True,
        dtype="bfloat16",
        disable_log_stats=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    semaphore = asyncio.Semaphore(max(1, args.vllm_async_concurrency))
    log_each = should_log_each_example(len(kept_rows))
    completed = 0

    async def run_index(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed
        async with semaphore:
            try:
                detail = await run_one_async(row, args, engine, SamplingParams, tokenizer)
            except Exception as exc:
                detail = {
                    "idx": row.get("source_idx", index),
                    "source_idx": row.get("source_idx", index),
                    "db_id": row.get("db_id", ""),
                    "prompt_tokens": row.get("prompt_tokens", 0),
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
                    "stop_reason": "generation_error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "rounds": [],
                }
            completed += 1
            if log_each or should_log_progress_tick(completed - 1, len(kept_rows)):
                print(
                    f"[qwen-async] generated {completed}/{len(kept_rows)} "
                    f"idx={detail.get('idx')} stop={detail.get('stop_reason')} "
                    f"calls={detail.get('tool_call_count')} sql={preview_text(detail.get('pred_sql', ''), 100)}"
                )
            return detail

    try:
        details = await asyncio.gather(*(run_index(index, row) for index, row in enumerate(kept_rows)))
    finally:
        engine.shutdown()
    details.sort(key=lambda item: int(item.get("idx", 0)))
    return details


def build_generation_stats(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(details)
    stop_counts = Counter(str(detail.get("stop_reason") or "unknown") for detail in details)
    tool_counts = Counter()
    rejected_counts = Counter()
    for detail in details:
        tool_counts.update(detail.get("tool_names") or [])
        rejected_counts.update(detail.get("rejected_tool_names") or [])
    completion_total = sum(int(detail.get("completion_token_count") or 0) for detail in details)
    tool_call_total = sum(int(detail.get("tool_call_count") or 0) for detail in details)
    return {
        "generated_examples": total,
        "filtered_examples": 0,
        "stop_reason_counts": dict(stop_counts),
        "tool_call_count_total": tool_call_total,
        "tool_round_count_total": tool_call_total,
        "avg_tool_calls_per_example": tool_call_total / max(1, total),
        "avg_tool_rounds_per_example": tool_call_total / max(1, total),
        "tool_name_counts": dict(tool_counts),
        "rejected_tool_call_count_total": sum(int(detail.get("rejected_tool_call_count") or 0) for detail in details),
        "rejected_tool_name_counts": dict(rejected_counts),
        "completion_token_total": completion_total,
        "avg_completion_tokens": completion_total / max(1, total),
        "max_prompt_tokens": max((int(detail.get("prompt_tokens") or 0) for detail in details), default=0),
    }


def predictions_from_details(details: List[Dict[str, Any]]) -> Dict[str, str]:
    return {
        str(detail.get("idx")): f"{detail.get('pred_sql', '')}{BIRD_SPLIT_MARKER}{detail.get('db_id', '')}"
        for detail in details
    }


def run_worker(args: argparse.Namespace) -> None:
    if not args.shard_file or not args.shard_output_dir:
        raise ValueError("--worker requires --shard_file and --shard_output_dir")
    configure_tool_env(args.database_dir)
    output_dir = Path(args.shard_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl_rows(Path(args.shard_file))
    print(f"[worker] loaded {len(rows)} rows CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    details = asyncio.run(generate_details_async(rows, args))
    with (output_dir / "predict_dev.json").open("w", encoding="utf-8") as handle:
        json.dump(predictions_from_details(details), handle, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "prediction_details.jsonl", details)
    print(f"[worker] wrote {output_dir}")


def merge_shards(shard_dirs: List[Path]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    predictions: Dict[str, str] = {}
    details: List[Dict[str, Any]] = []
    for shard_dir in shard_dirs:
        with (shard_dir / "predict_dev.json").open("r", encoding="utf-8") as handle:
            predictions.update(json.load(handle))
        with (shard_dir / "prediction_details.jsonl").open("r", encoding="utf-8") as handle:
            details.extend(json.loads(line) for line in handle if line.strip())
    details.sort(key=lambda item: int(item.get("idx", 0)))
    return predictions, details


def write_outputs(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    predictions: Dict[str, str],
    details: List[Dict[str, Any]],
    generation_seconds: float,
    started_at: float,
) -> None:
    output_dir = Path(args.output_dir)
    diff_rows = load_diff_rows(args.diff_json_path)
    with (output_dir / "predict_dev.json").open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "prediction_details.jsonl", details)
    (output_dir / "filtered_examples.jsonl").write_text("", encoding="utf-8")

    eval_started = time.monotonic()
    per_example_results, summary = evaluate_predictions(
        rows=rows,
        predictions=predictions,
        database_dir=args.database_dir,
        diff_rows=diff_rows,
        timeout_s=args.eval_timeout,
        eval_workers=args.eval_workers,
    )
    evaluation_seconds = time.monotonic() - eval_started
    summary["timing_seconds"] = {
        "generation": generation_seconds,
        "evaluation": evaluation_seconds,
        "total": time.monotonic() - started_at,
    }
    summary["generation_stats"] = build_generation_stats(details)
    write_jsonl(output_dir / "eval_results.jsonl", per_example_results)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    report_rows = build_per_example_report_rows(details, per_example_results)
    write_per_example_report_csv(report_rows, output_dir / "per_example_report.csv")
    write_summary_markdown(summary, output_dir / "eval_summary.md", args, len(rows))
    write_run_report_markdown(summary, report_rows, output_dir / "run_report.md", args, len(rows))
    write_summary_csv(summary["by_difficulty"], output_dir / "eval_summary_by_difficulty.csv")
    write_summary_csv(summary["by_db"], output_dir / "eval_summary_by_db.csv")
    print_summary_tables(summary)
    print(f"[qwen-async] saved summary={output_dir / 'eval_summary.json'}")


def run_parent(args: argparse.Namespace) -> None:
    started_at = time.monotonic()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    configure_tool_env(args.database_dir)
    rows = load_rows(args.input_file, -1)
    rows = slice_rows(rows, args.start_index, args.end_index)
    if args.num_examples >= 0:
        rows = rows[: args.num_examples]
    print(f"[qwen-async] loaded rows={len(rows)}")

    generation_started = time.monotonic()
    if args.num_shards <= 1:
        details = asyncio.run(generate_details_async(rows, args))
        generation_seconds = time.monotonic() - generation_started
        write_outputs(args, rows, predictions_from_details(details), details, generation_seconds, started_at)
        return

    gpu_groups = args.gpu_groups or default_gpu_groups(args.num_shards, args.vllm_tensor_parallel_size)
    if len(gpu_groups) != args.num_shards:
        raise ValueError("--gpu_groups must provide exactly --num_shards entries")

    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes: List[subprocess.Popen] = []
    shard_dirs: List[Path] = []
    for shard_idx, shard in enumerate(shard_rows(rows, args.num_shards)):
        shard_dir = shard_root / f"shard-{shard_idx:05d}-of-{args.num_shards:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_file = shard_dir / "input_rows.jsonl"
        write_jsonl(shard_file, shard)
        shard_dirs.append(shard_dir)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model_name_or_path",
            args.model_name_or_path,
            "--input_file",
            args.input_file,
            "--database_dir",
            args.database_dir,
            "--diff_json_path",
            args.diff_json_path,
            "--output_dir",
            args.output_dir,
            "--shard_file",
            str(shard_file),
            "--shard_output_dir",
            str(shard_dir),
            "--max_prompt_length",
            str(args.max_prompt_length),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--max_tool_rounds",
            str(args.max_tool_rounds),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--top_k",
            str(args.top_k),
            "--eval_timeout",
            str(args.eval_timeout),
            "--eval_workers",
            str(args.eval_workers),
            "--vllm_tensor_parallel_size",
            str(args.vllm_tensor_parallel_size),
            "--vllm_gpu_memory_utilization",
            str(args.vllm_gpu_memory_utilization),
            "--vllm_max_model_len",
            str(args.vllm_max_model_len),
            "--vllm_async_concurrency",
            str(args.vllm_async_concurrency),
            "--tool_choice_policy",
            args.tool_choice_policy,
            "--empty_tool_retries",
            str(args.empty_tool_retries),
        ]
        if args.enable_thinking:
            cmd.append("--enable_thinking")
        if args.preserve_thinking:
            cmd.append("--preserve_thinking")
        if args.no_prompt_rewrite:
            cmd.append("--no_prompt_rewrite")
        if args.no_force_finalize:
            cmd.append("--no_force_finalize")
        log_path = shard_dir / "worker.log"
        print(f"[qwen-async] starting shard={shard_idx} rows={len(shard)} gpus={gpu_groups[shard_idx]}")
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_groups[shard_idx]
        env["VLLM_CACHE_ROOT"] = str(shard_dir / "vllm_cache")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(shard_dir / "torchinductor_cache")
        processes.append(subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env))

    failed = False
    for shard_idx, process in enumerate(processes):
        return_code = process.wait()
        print(f"[qwen-async] shard {shard_idx} exited code={return_code}")
        failed = failed or return_code != 0
    if failed:
        raise RuntimeError(f"one or more Qwen async shards failed; see {shard_root}")

    generation_seconds = time.monotonic() - generation_started
    predictions, details = merge_shards(shard_dirs)
    write_outputs(args, rows, predictions, details, generation_seconds, started_at)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
