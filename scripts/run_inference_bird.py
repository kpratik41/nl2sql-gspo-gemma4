import argparse
import asyncio
import csv
import json
import multiprocessing as mp
import os
import sqlite3
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.sql_utils import bird_execute_sql, bird_get_gold_rows, bird_result_match, extract_sql, get_database_path
from nl2sql_gspo.inference_tool_executor import (
    configure_tool_env,
    execute_tool_calls,
    extract_and_execute_tools,
    extract_tool_calls,
    text_before_first_tool_call,
)

BIRD_SPLIT_MARKER = "\t----- bird -----\t"


def should_log_each_example(total_count: int) -> bool:
    return total_count <= 25


def should_log_progress_tick(current_index: int, total_count: int) -> bool:
    completed = current_index + 1
    if completed == 1 or completed == total_count:
        return True

    if total_count <= 50:
        return completed % 5 == 0

    return completed % 50 == 0


def has_sql_content(sql: str) -> bool:
    return bool(sql and sql.strip())


def print_run_configuration(args: argparse.Namespace, output_dir: Path) -> None:
    print("[run] starting standalone inference")
    print(f"[run] inference_backend={args.inference_backend}")
    print(f"[run] model_name_or_path={args.model_name_or_path}")
    print(f"[run] input_file={args.input_file}")
    print(f"[run] database_dir={args.database_dir}")
    print(f"[run] diff_json_path={args.diff_json_path}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] max_prompt_length={args.max_prompt_length}")
    print(f"[run] max_new_tokens={args.max_new_tokens}")
    print(f"[run] num_examples={args.num_examples}")
    print(f"[run] eval_timeout={args.eval_timeout}")
    print(f"[run] eval_workers={args.eval_workers}")
    print(f"[run] skip_generation={args.skip_generation}")
    print(f"[run] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if args.inference_backend in {"vllm", "vllm_async"}:
        print(f"[run] vllm_tensor_parallel_size={args.vllm_tensor_parallel_size}")
        print(f"[run] vllm_data_parallel_size={args.vllm_data_parallel_size}")
        print(f"[run] vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
        print(f"[run] vllm_max_model_len={args.vllm_max_model_len}")
    if args.inference_backend == "vllm_async":
        print(f"[run] vllm_async_concurrency={args.vllm_async_concurrency}")
    print(f"[run] max_tool_rounds={args.max_tool_rounds}")


def resolve_vllm_tokenizer_source(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return model_name_or_path

    if (model_path / "processor_config.json").exists() or (model_path / "preprocessor_config.json").exists():
        return model_name_or_path

    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with tokenizer_config_path.open("r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)

        if tokenizer_config.get("processor_class") == "Gemma4Processor":
            fallback_source = "google/gemma-4-31B-it"
            print(
                f"[inference] local checkpoint {model_name_or_path} is missing Gemma 4 processor files; "
                f"loading processor/tokenizer from {fallback_source} instead."
            )
            return fallback_source

    return model_name_or_path


def load_diff_rows(diff_json_path: str) -> List[Dict[str, Any]]:
    with open(diff_json_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    return loaded if isinstance(loaded, list) else [loaded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_backend", type=str, choices=["vllm", "vllm_async"], default="vllm")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/dev_20251106.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_inference")
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--vllm_data_parallel_size", type=int, default=None)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=43000)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.inference_backend == "vllm_async":
        if args.vllm_tensor_parallel_size is None:
            args.vllm_tensor_parallel_size = 8
        if args.vllm_data_parallel_size is None:
            args.vllm_data_parallel_size = 1
    else:
        if args.vllm_tensor_parallel_size is None:
            args.vllm_tensor_parallel_size = 2
        if args.vllm_data_parallel_size is None:
            args.vllm_data_parallel_size = 4
    return args


def load_rows(input_file: str, num_examples: int) -> List[Dict[str, Any]]:
    input_path = Path(input_file)

    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            raw_rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with input_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            raw_rows = loaded if isinstance(loaded, list) else [loaded]

    rows = [normalize_record(row) for row in raw_rows]

    if num_examples >= 0:
        rows = rows[:num_examples]

    for idx, row in enumerate(rows):
        row["source_idx"] = idx

    missing_examples: List[str] = []
    for row in rows:
        missing_fields = []
        if not row.get("db_id"):
            missing_fields.append("db_id")
        if not row.get("gold_sql"):
            missing_fields.append("gold_sql")

        if missing_fields:
            missing_examples.append(
                f"idx={row.get('source_idx', -1)} missing={','.join(missing_fields)}"
            )

        if len(missing_examples) >= 5:
            break

    if missing_examples:
        raise ValueError(
            "Input rows are missing required fields after normalization: "
            + "; ".join(missing_examples)
        )

    return rows


# Chat-template kwargs applied to every rendered prompt. Empty by default, which
# leaves gemma-4's template at its own default of enable_thinking=false -- the
# setting every result recorded before this existed was produced under.
#
# Worth knowing when porting this: templates do not agree on the default.
# gemma-4 sets `enable_thinking | default(false)`, while Qwen3.8 uses
# `enable_thinking is undefined or enable_thinking is true`, i.e. on. Passing
# nothing therefore means thinking-off for Gemma and thinking-ON for Qwen, with
# nothing to flag it. Prefer configuring this explicitly over relying on it.
_CHAT_TEMPLATE_KWARGS: Dict[str, Any] = {}


def configure_chat_template_kwargs(enable_thinking: bool = False, preserve_thinking: bool = False) -> None:
    """Set the thinking kwargs passed to apply_chat_template for this process."""
    global _CHAT_TEMPLATE_KWARGS
    _CHAT_TEMPLATE_KWARGS = {
        "enable_thinking": bool(enable_thinking),
        "preserve_thinking": bool(preserve_thinking),
    }
    print(
        f"[prompt] chat template kwargs: enable_thinking={bool(enable_thinking)} "
        f"preserve_thinking={bool(preserve_thinking)}"
    )


def render_prompt(tokenizer, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
                **_CHAT_TEMPLATE_KWARGS,
            )
        except ValueError as exc:
            if "tokenizer.chat_template is not set" not in str(exc):
                raise

            print("[prompt] tokenizer chat template is not set; using plain text fallback prompt formatting")

    lines = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def get_generation_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt_messages = row.get("prompt") or []
    if prompt_messages:
        return prompt_messages

    messages = row.get("messages") or []
    return [message for message in messages if message.get("role") != "assistant"]


def preview_text(text: str, max_chars: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact

    return f"{compact[:max_chars - 3]}..."


def filter_rows_by_prompt_length(rows: List[Dict[str, Any]], tokenizer, max_prompt_length: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for row in rows:
        prompt_messages = get_generation_messages(row)
        tools = row.get("tools")
        prompt_text = render_prompt(tokenizer, prompt_messages, tools)
        prompt_token_count = len(tokenizer(prompt_text, truncation=False)["input_ids"])

        prepared_row = dict(row)
        prepared_row["prompt_text"] = prompt_text
        prepared_row["prompt_tokens"] = prompt_token_count

        if prompt_token_count > max_prompt_length:
            skipped_row = {
                "idx": row.get("source_idx", -1),
                "db_id": row.get("db_id", ""),
                "prompt_tokens": prompt_token_count,
                "max_prompt_length": max_prompt_length,
                "prompt_preview": preview_text(prompt_text),
            }
            skipped_rows.append(skipped_row)
            print(
                "[filter] skipping "
                f"idx={skipped_row['idx']} db_id={skipped_row['db_id']} "
                f"prompt_tokens={prompt_token_count} max_prompt_length={max_prompt_length} "
                f"prompt={skipped_row['prompt_preview']}"
            )
            continue

        kept_rows.append(prepared_row)

    if skipped_rows:
        print(
            f"[filter] skipped {len(skipped_rows)} over-length prompts; "
            f"continuing with {len(kept_rows)} prompts"
        )
    else:
        print(f"[filter] no prompts exceeded max_prompt_length={max_prompt_length}")

    return kept_rows, skipped_rows


def infer_visible_gpu_count() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return 1

    return max(1, len([device for device in visible_devices.split(",") if device.strip()]))


def get_visible_devices() -> List[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return ["0"]

    devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    return devices or ["0"]


def plan_vllm_device_groups(tensor_parallel_size: int, data_parallel_size: int) -> List[List[str]]:
    visible_devices = get_visible_devices()
    required_devices = tensor_parallel_size * data_parallel_size

    if len(visible_devices) < required_devices:
        raise ValueError(
            "Not enough visible GPUs for the requested vLLM parallelism: "
            f"need {required_devices} GPUs for tensor_parallel_size={tensor_parallel_size} "
            f"and data_parallel_size={data_parallel_size}, but CUDA_VISIBLE_DEVICES exposes "
            f"{len(visible_devices)} ({','.join(visible_devices)})."
        )

    return [
        visible_devices[offset: offset + tensor_parallel_size]
        for offset in range(0, required_devices, tensor_parallel_size)
    ]


def shard_rows_for_data_parallel(rows: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for row_index, row in enumerate(rows):
        shards[row_index % num_shards].append(row)
    return shards


def prepare_rows_for_generation(rows: List[Dict[str, Any]], tokenizer, max_prompt_length: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows, skipped_rows = filter_rows_by_prompt_length(rows, tokenizer, max_prompt_length)
    print(f"[inference] running generation for {len(rows)} prompts")

    return rows, skipped_rows


def build_assistant_tool_message(
    generated_text: str,
    tool_calls: List[Dict[str, Any]],
    tool_responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_calls = []
    for index, call in enumerate(tool_calls):
        function = call.get("function") or {}
        normalized_calls.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments") or {},
                },
            }
        )

    return {
        "role": "assistant",
        "content": "",
        "reasoning": text_before_first_tool_call(generated_text),
        "tool_calls": normalized_calls,
        "tool_responses": [
            {
                "name": response.get("name", "unknown"),
                "response": response.get("response", {"value": response.get("raw_response")}),
            }
            for response in tool_responses
        ],
    }


def build_generation_detail(
    row: Dict[str, Any],
    prediction_text: str,
    prompt_token_count: int,
    completion_token_count: int,
    tool_rounds: int,
    tool_call_count: int,
    stop_reason: str,
    error_message: str = "",
) -> Dict[str, Any]:
    return {
        "source_idx": row.get("source_idx", -1),
        "db_id": row.get("db_id", ""),
        "prompt_tokens": prompt_token_count,
        "prediction_text": prediction_text,
        "pred_sql": extract_sql(prediction_text),
        "completion_token_count": completion_token_count,
        "tool_rounds": tool_rounds,
        "tool_call_count": tool_call_count,
        "stop_reason": stop_reason,
        "error_message": error_message,
    }


def should_use_agentic_tool_loop(row: Dict[str, Any], max_tool_rounds: int) -> bool:
    return bool(row.get("tools")) and max_tool_rounds > 0


def token_id_for_text(tokenizer, text: str) -> Optional[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def gemma_tool_loop_stop_token_ids(tokenizer) -> List[int]:
    """Return Gemma-native stop tokens for one assistant/tool turn."""

    stop_texts = [
        "<tool_call|>",      # complete tool call; let Python execute it
        "<|tool_response>",  # prevent the model from fabricating tool output
        "<turn|>",           # normal assistant turn end
        "<eos>",
    ]
    token_ids: List[int] = []
    for text in stop_texts:
        token_id = token_id_for_text(tokenizer, text)
        if token_id is not None and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def keep_first_tool_call_only(generated_text: str, tool_calls: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Keep one tool call per assistant turn and drop speculative text after it."""

    if not tool_calls:
        return generated_text, []
    first_call = dict(tool_calls[0])
    end = int(first_call.get("end") or len(generated_text or ""))
    return (generated_text or "")[:end].strip(), [first_call]


def generate_one_with_vllm_tool_loop(
    llm,
    sampling_params_cls,
    tokenizer,
    row: Dict[str, Any],
    max_new_tokens: int,
    max_tool_rounds: int,
    eval_timeout: float,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    messages = [dict(message) for message in get_generation_messages(row)]
    tools = row.get("tools")
    generated_parts: List[str] = []
    prompt_token_count = int(row["prompt_tokens"])
    completion_token_count = 0
    tool_rounds = 0
    tool_call_count = 0
    stop_reason = "finished"

    for round_index in range(max_tool_rounds + 1):
        remaining_tokens = max_new_tokens - completion_token_count
        if remaining_tokens <= 0:
            stop_reason = "max_new_tokens"
            break

        prompt_text = render_prompt(tokenizer, messages, tools)
        current_prompt_tokens = len(tokenizer(prompt_text, truncation=False)["input_ids"])
        if round_index == 0:
            prompt_token_count = current_prompt_tokens

        sampling_params = sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=remaining_tokens,
            stop_token_ids=gemma_tool_loop_stop_token_ids(tokenizer),
        )
        request_output = llm.generate([prompt_text], sampling_params=sampling_params, use_tqdm=False)[0]
        first_output = request_output.outputs[0] if request_output.outputs else None
        generated_text = (first_output.text or "").strip() if first_output else ""
        round_tokens = len(first_output.token_ids) if first_output else 0
        tool_calls = extract_tool_calls(generated_text)
        if not tool_calls:
            completion_token_count += round_tokens
            if generated_text:
                generated_parts.append(generated_text)
            stop_reason = "max_new_tokens" if round_tokens >= remaining_tokens else "finished"
            break

        if round_index >= max_tool_rounds:
            stop_reason = "max_tool_rounds"
            break

        generated_text, tool_calls = keep_first_tool_call_only(generated_text, tool_calls)
        completion_token_count += len(
            tokenizer(generated_text, truncation=False, add_special_tokens=False)["input_ids"]
        )
        if generated_text:
            generated_parts.append(generated_text)

        tool_responses = execute_tool_calls(tool_calls, timeout_s=eval_timeout)
        for response in tool_responses:
            generated_parts.append(response["rendered"])
        messages.append(build_assistant_tool_message(generated_text, tool_calls, tool_responses))
        tool_rounds += 1
        tool_call_count += len(tool_calls)

    prediction_text = "\n".join(part for part in generated_parts if part).strip()
    return build_generation_detail(
        row=row,
        prediction_text=prediction_text,
        prompt_token_count=prompt_token_count,
        completion_token_count=completion_token_count,
        tool_rounds=tool_rounds,
        tool_call_count=tool_call_count,
        stop_reason=stop_reason,
    )


async def _async_vllm_generate_text(
    engine,
    sampling_params_cls,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_prefix: str,
    stop_token_ids: Optional[List[int]] = None,
) -> Tuple[str, int]:
    sampling_params = sampling_params_cls(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    request_id = f"{request_prefix}-{uuid.uuid4().hex}"
    final_output = None
    async for request_output in engine.generate(prompt_text, sampling_params, request_id=request_id):
        final_output = request_output

    first_output = final_output.outputs[0] if final_output and final_output.outputs else None
    generated_text = (first_output.text or "").strip() if first_output else ""
    completion_tokens = len(first_output.token_ids) if first_output else 0
    return generated_text, completion_tokens


async def generate_one_with_vllm_async_tool_loop(
    engine,
    sampling_params_cls,
    tokenizer,
    row: Dict[str, Any],
    max_new_tokens: int,
    max_model_len: int,
    max_tool_rounds: int,
    eval_timeout: float,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    messages = [dict(message) for message in get_generation_messages(row)]
    tools = row.get("tools")
    generated_parts: List[str] = []
    prompt_token_count = int(row["prompt_tokens"])
    completion_token_count = 0
    tool_rounds = 0
    tool_call_count = 0
    stop_reason = "finished"
    request_prefix = f"idx{row.get('source_idx', -1)}"

    for round_index in range(max_tool_rounds + 1):
        remaining_tokens = max_new_tokens - completion_token_count
        if remaining_tokens <= 0:
            stop_reason = "max_new_tokens"
            break

        prompt_text = render_prompt(tokenizer, messages, tools)
        max_prompt_chars = max_model_len * 31
        if len(prompt_text) > max_prompt_chars:
            stop_reason = "context_length_exceeded"
            print(
                "[inference] skipping over-context tool-loop sample "
                f"idx={row.get('source_idx', -1)} db_id={row.get('db_id', '')} "
                f"round={round_index} prompt_chars={len(prompt_text)} "
                f"max_prompt_chars={max_prompt_chars} max_model_len={max_model_len}"
            )
            break

        current_prompt_tokens = len(tokenizer(prompt_text, truncation=False)["input_ids"])
        if round_index == 0:
            prompt_token_count = current_prompt_tokens

        available_context_tokens = max_model_len - current_prompt_tokens
        if available_context_tokens <= 0:
            stop_reason = "context_length_exceeded"
            print(
                "[inference] skipping over-context tool-loop sample "
                f"idx={row.get('source_idx', -1)} db_id={row.get('db_id', '')} "
                f"round={round_index} prompt_tokens={current_prompt_tokens} "
                f"max_model_len={max_model_len}"
            )
            break

        request_max_tokens = min(remaining_tokens, available_context_tokens)
        if request_max_tokens < remaining_tokens:
            print(
                "[inference] limiting generation budget to remaining context "
                f"idx={row.get('source_idx', -1)} db_id={row.get('db_id', '')} "
                f"round={round_index} prompt_tokens={current_prompt_tokens} "
                f"requested_max_tokens={remaining_tokens} "
                f"request_max_tokens={request_max_tokens} max_model_len={max_model_len}"
            )

        generated_text, round_tokens = await _async_vllm_generate_text(
            engine=engine,
            sampling_params_cls=sampling_params_cls,
            prompt_text=prompt_text,
            max_tokens=request_max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_prefix=request_prefix,
            stop_token_ids=gemma_tool_loop_stop_token_ids(tokenizer),
        )
        tool_calls = extract_tool_calls(generated_text)
        if not tool_calls:
            completion_token_count += round_tokens
            if generated_text:
                generated_parts.append(generated_text)
            if round_tokens >= remaining_tokens:
                stop_reason = "max_new_tokens"
            elif round_tokens >= request_max_tokens and request_max_tokens < remaining_tokens:
                stop_reason = "context_window_limited"
            else:
                stop_reason = "finished"
            break

        if round_index >= max_tool_rounds:
            stop_reason = "max_tool_rounds"
            break

        generated_text, tool_calls = keep_first_tool_call_only(generated_text, tool_calls)
        completion_token_count += len(
            tokenizer(generated_text, truncation=False, add_special_tokens=False)["input_ids"]
        )
        if generated_text:
            generated_parts.append(generated_text)

        tool_responses = await asyncio.to_thread(execute_tool_calls, tool_calls, eval_timeout)
        for response in tool_responses:
            generated_parts.append(response["rendered"])
        messages.append(build_assistant_tool_message(generated_text, tool_calls, tool_responses))
        tool_rounds += 1
        tool_call_count += len(tool_calls)

    prediction_text = "\n".join(part for part in generated_parts if part).strip()
    return build_generation_detail(
        row=row,
        prediction_text=prediction_text,
        prompt_token_count=prompt_token_count,
        completion_token_count=completion_token_count,
        tool_rounds=tool_rounds,
        tool_call_count=tool_call_count,
        stop_reason=stop_reason,
    )


def _vllm_generate_worker(
    queue,
    shard_id: int,
    device_group: List[str],
    rows: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_group)

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=llm_config["model_name_or_path"],
            tokenizer=llm_config["tokenizer_name_or_path"],
            trust_remote_code=True,
            tensor_parallel_size=llm_config["tensor_parallel_size"],
            distributed_executor_backend="mp",
            gpu_memory_utilization=llm_config["gpu_memory_utilization"],
            max_model_len=llm_config["max_model_len"],
            dtype="bfloat16",
        )
        from nl2sql_gspo.model_utils import load_tokenizer

        tokenizer = load_tokenizer(llm_config["tokenizer_name_or_path"])

        results = []
        for row in rows:
            if should_use_agentic_tool_loop(row, llm_config["max_tool_rounds"]):
                configure_tool_env(llm_config.get("database_dir", "databases"))
                results.append(
                    generate_one_with_vllm_tool_loop(
                        llm=llm,
                        sampling_params_cls=SamplingParams,
                        tokenizer=tokenizer,
                        row=row,
                        max_new_tokens=llm_config["max_new_tokens"],
                        max_tool_rounds=llm_config["max_tool_rounds"],
                        eval_timeout=llm_config.get("eval_timeout", 60.0),
                        temperature=llm_config["temperature"],
                        top_p=llm_config["top_p"],
                    )
                )
                continue

            sampling_params = SamplingParams(
                temperature=llm_config["temperature"],
                top_p=llm_config["top_p"],
                max_tokens=llm_config["max_new_tokens"],
            )
            request_output = llm.generate([row["prompt_text"]], sampling_params=sampling_params, use_tqdm=False)[0]
            first_output = request_output.outputs[0] if request_output.outputs else None
            prediction_text = (first_output.text or "").strip() if first_output else ""
            completion_token_count = len(first_output.token_ids) if first_output else 0

            if "call:" in prediction_text:
                configure_tool_env(llm_config.get("database_dir", "databases"))
                prediction_text = extract_and_execute_tools(prediction_text, timeout_s=llm_config.get("eval_timeout", 60.0))

            results.append(
                build_generation_detail(
                    row=row,
                    prediction_text=prediction_text,
                    prompt_token_count=int(row["prompt_tokens"]),
                    completion_token_count=completion_token_count,
                    tool_rounds=0,
                    tool_call_count=len(extract_tool_calls(prediction_text)),
                    stop_reason="finished",
                )
            )

        queue.put({"status": "ok", "shard_id": shard_id, "results": results})
    except Exception:
        queue.put({"status": "error", "shard_id": shard_id, "error": traceback.format_exc()})


def generate_predictions_with_vllm_data_parallel(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    tensor_parallel_size: int,
    data_parallel_size: int,
    vllm_max_model_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    device_groups = plan_vllm_device_groups(tensor_parallel_size, data_parallel_size)
    row_shards = shard_rows_for_data_parallel(rows, data_parallel_size)
    active_shards = [
        (shard_id, device_group, shard_rows)
        for shard_id, (device_group, shard_rows) in enumerate(zip(device_groups, row_shards))
        if shard_rows
    ]

    print(
        "[inference] loading vLLM engines in multi-process data-parallel mode "
        f"tensor_parallel_size={tensor_parallel_size} data_parallel_size={data_parallel_size} "
        f"device_groups={['+'.join(group) for group in device_groups]} max_model_len={vllm_max_model_len}"
    )

    llm_config = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": resolve_vllm_tokenizer_source(args.model_name_or_path),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": vllm_max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_tool_rounds": args.max_tool_rounds,
        "database_dir": args.database_dir,
        "eval_timeout": args.eval_timeout,
    }

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    collect_error: Optional[BaseException] = None

    for shard_id, device_group, shard_rows in active_shards:
        print(
            f"[inference] starting vLLM shard {shard_id + 1}/{len(active_shards)} "
            f"gpus={','.join(device_group)} prompts={len(shard_rows)}"
        )
        process = ctx.Process(
            target=_vllm_generate_worker,
            args=(queue, shard_id, device_group, shard_rows, llm_config),
        )
        process.start()
        processes.append(process)

    collected_results: Dict[int, Dict[str, Any]] = {}
    try:
        for _ in processes:
            message = queue.get()
            if message.get("status") != "ok":
                collect_error = RuntimeError(
                    "vLLM data-parallel worker failed"
                    + (f" (shard {message.get('shard_id')})" if "shard_id" in message else "")
                    + ":\n"
                    + message.get("error", "unknown error")
                )
                raise collect_error

            for result in message["results"]:
                collected_results[result["source_idx"]] = result
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive() and collect_error is not None:
                process.terminate()
                process.join(timeout=5)

    for process in processes:
        if process.is_alive():
            print(
                f"[inference] warning: vLLM worker pid={process.pid} was still shutting down after results were collected; "
                "continuing without waiting for a clean exit"
            )
            process.terminate()
            process.join(timeout=5)

    for process in processes:
        if collect_error is None and process.exitcode in (0, None, -15):
            continue
        if process.exitcode not in (0, None):
            raise RuntimeError(f"vLLM data-parallel worker exited with code {process.exitcode}")

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    log_each_example = should_log_each_example(len(rows))

    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        generated = collected_results.get(source_idx)
        if generated is None:
            raise RuntimeError(f"Missing vLLM generation result for idx={source_idx}")

        db_id = generated["db_id"]
        pred_sql = generated["pred_sql"]
        prediction_text = generated["prediction_text"]
        prompt_token_count = generated["prompt_tokens"]
        completion_token_count = generated["completion_token_count"]

        official_predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{db_id}"
        detailed_predictions.append(
            {
                "idx": source_idx,
                "db_id": db_id,
                "prediction_text": prediction_text,
                "pred_sql": pred_sql,
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "prompt_tokens": prompt_token_count,
                "completion_token_count": completion_token_count,
                "tool_rounds": generated.get("tool_rounds", 0),
                "tool_call_count": generated.get("tool_call_count", 0),
                "stop_reason": generated.get("stop_reason", ""),
            }
        )

        if log_each_example:
            print(
                f"[inference] finished sample {idx + 1}/{len(rows)} "
                f"idx={source_idx} completion_tokens={completion_token_count} "
                f"pred_sql={preview_text(pred_sql, max_chars=120)}"
            )

        if should_log_progress_tick(idx, len(rows)):
            print(f"[inference] generated {idx + 1}/{len(rows)} prompts")

    return rows, official_predictions, detailed_predictions, []


def generate_predictions_with_vllm(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "vLLM backend requested, but vllm could not be imported in the current environment."
        ) from exc

    from nl2sql_gspo.model_utils import load_tokenizer

    tokenizer = load_tokenizer(args.model_name_or_path)
    rows, skipped_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)

    tensor_parallel_size = args.vllm_tensor_parallel_size
    data_parallel_size = args.vllm_data_parallel_size
    vllm_max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)

    print(
        "[inference] loading vLLM engine "
        f"tensor_parallel_size={tensor_parallel_size} data_parallel_size={data_parallel_size} "
        f"max_model_len={vllm_max_model_len}"
    )

    if data_parallel_size > 1:
        return generate_predictions_with_vllm_data_parallel(
            rows=rows,
            args=args,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            vllm_max_model_len=vllm_max_model_len,
        )

    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=resolve_vllm_tokenizer_source(args.model_name_or_path),
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        dtype="bfloat16",
    )

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    log_each_example = should_log_each_example(len(rows))

    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        db_id = row.get("db_id", "")
        prompt_token_count = int(row["prompt_tokens"])

        if log_each_example:
            print(
                f"[inference] generating sample {idx + 1}/{len(rows)} "
                f"idx={source_idx} db_id={db_id} prompt_tokens={prompt_token_count}"
            )

        if should_use_agentic_tool_loop(row, args.max_tool_rounds):
            generated = generate_one_with_vllm_tool_loop(
                llm=llm,
                sampling_params_cls=SamplingParams,
                tokenizer=tokenizer,
                row=row,
                max_new_tokens=args.max_new_tokens,
                max_tool_rounds=args.max_tool_rounds,
                eval_timeout=args.eval_timeout,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            prediction_text = generated["prediction_text"]
            pred_sql = generated["pred_sql"]
            prompt_token_count = generated["prompt_tokens"]
            completion_token_count = generated["completion_token_count"]
        else:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            )
            request_output = llm.generate([row["prompt_text"]], sampling_params=sampling_params, use_tqdm=False)[0]
            first_output = request_output.outputs[0] if request_output.outputs else None
            prediction_text = (first_output.text or "").strip() if first_output else ""
            if "call:" in prediction_text:
                prediction_text = extract_and_execute_tools(prediction_text, timeout_s=args.eval_timeout)
            pred_sql = extract_sql(prediction_text)
            completion_token_count = len(first_output.token_ids) if first_output else 0
            generated = build_generation_detail(
                row=row,
                prediction_text=prediction_text,
                prompt_token_count=prompt_token_count,
                completion_token_count=completion_token_count,
                tool_rounds=0,
                tool_call_count=len(extract_tool_calls(prediction_text)),
                stop_reason="finished",
            )

        official_predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{db_id}"
        detailed_predictions.append(
            {
                "idx": source_idx,
                "db_id": db_id,
                "prediction_text": prediction_text,
                "pred_sql": pred_sql,
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "prompt_tokens": prompt_token_count,
                "completion_token_count": completion_token_count,
                "tool_rounds": generated.get("tool_rounds", 0),
                "tool_call_count": generated.get("tool_call_count", 0),
                "stop_reason": generated.get("stop_reason", ""),
            }
        )

        if log_each_example:
            print(
                f"[inference] finished sample {idx + 1}/{len(rows)} "
                f"idx={source_idx} completion_tokens={completion_token_count} "
                f"pred_sql={preview_text(pred_sql, max_chars=120)}"
            )

        if should_log_progress_tick(idx, len(rows)):
            print(f"[inference] generated {idx + 1}/{len(rows)} prompts")

    return rows, official_predictions, detailed_predictions, skipped_rows


async def _generate_predictions_with_vllm_async_impl(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception as exc:
        raise RuntimeError(
            "Async vLLM backend requested, but vLLM async engine could not be imported."
        ) from exc

    from nl2sql_gspo.model_utils import load_tokenizer

    if args.vllm_data_parallel_size != 1:
        # Invariant, not a user-facing limit: data_parallel_size > 1 is handled
        # by generate_predictions_with_vllm_async_data_parallel, which spawns one
        # process per shard and hands each child data_parallel_size=1.
        raise ValueError(
            "_generate_predictions_with_vllm_async_impl drives a single engine and "
            "requires vllm_data_parallel_size=1; shard via "
            "generate_predictions_with_vllm_async_data_parallel instead."
        )

    tokenizer_source = resolve_vllm_tokenizer_source(args.model_name_or_path)
    tokenizer = load_tokenizer(tokenizer_source)
    rows, skipped_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)

    tensor_parallel_size = args.vllm_tensor_parallel_size
    vllm_max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)
    concurrency = max(1, args.vllm_async_concurrency)

    print(
        "[inference] loading async vLLM engine "
        f"tensor_parallel_size={tensor_parallel_size} concurrency={concurrency} "
        f"max_model_len={vllm_max_model_len}"
    )

    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        tokenizer=tokenizer_source,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    semaphore = asyncio.Semaphore(concurrency)
    log_each_example = should_log_each_example(len(rows))
    completed = 0

    async def generate_row(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed
        source_idx = row.get("source_idx", idx)
        db_id = row.get("db_id", "")
        prompt_token_count = int(row["prompt_tokens"])
        async with semaphore:
            if log_each_example:
                print(
                    f"[inference] async generating sample {idx + 1}/{len(rows)} "
                    f"idx={source_idx} db_id={db_id} prompt_tokens={prompt_token_count}"
                )

            try:
                if should_use_agentic_tool_loop(row, args.max_tool_rounds):
                    generated = await generate_one_with_vllm_async_tool_loop(
                        engine=engine,
                        sampling_params_cls=SamplingParams,
                        tokenizer=tokenizer,
                        row=row,
                        max_new_tokens=args.max_new_tokens,
                        max_model_len=vllm_max_model_len,
                        max_tool_rounds=args.max_tool_rounds,
                        eval_timeout=args.eval_timeout,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                else:
                    available_context_tokens = vllm_max_model_len - prompt_token_count
                    if available_context_tokens <= 0:
                        print(
                            "[inference] skipping over-context sample "
                            f"idx={source_idx} db_id={db_id} "
                            f"prompt_tokens={prompt_token_count} max_model_len={vllm_max_model_len}"
                        )
                        generated = build_generation_detail(
                            row=row,
                            prediction_text="",
                            prompt_token_count=prompt_token_count,
                            completion_token_count=0,
                            tool_rounds=0,
                            tool_call_count=0,
                            stop_reason="context_length_exceeded",
                        )
                    else:
                        request_max_tokens = min(args.max_new_tokens, available_context_tokens)
                        generated_text, completion_tokens = await _async_vllm_generate_text(
                            engine=engine,
                            sampling_params_cls=SamplingParams,
                            prompt_text=row["prompt_text"],
                            max_tokens=request_max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            request_prefix=f"idx{source_idx}",
                        )
                        if "call:" in generated_text:
                            generated_text = await asyncio.to_thread(
                                extract_and_execute_tools,
                                generated_text,
                                args.eval_timeout,
                            )
                        generated = build_generation_detail(
                            row=row,
                            prediction_text=generated_text,
                            prompt_token_count=prompt_token_count,
                            completion_token_count=completion_tokens,
                            tool_rounds=0,
                            tool_call_count=len(extract_tool_calls(generated_text)),
                            stop_reason=(
                                "context_window_limited"
                                if completion_tokens >= request_max_tokens
                                and request_max_tokens < args.max_new_tokens
                                else "finished"
                            ),
                        )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                print(
                    "[inference] generation failed for sample; marking failed and continuing "
                    f"idx={source_idx} db_id={db_id} error={preview_text(error_message, max_chars=240)}"
                )
                generated = build_generation_detail(
                    row=row,
                    prediction_text="",
                    prompt_token_count=prompt_token_count,
                    completion_token_count=0,
                    tool_rounds=0,
                    tool_call_count=0,
                    stop_reason="generation_error",
                    error_message=error_message,
                )

            completed += 1
            if log_each_example:
                print(
                    f"[inference] async finished sample {idx + 1}/{len(rows)} "
                    f"idx={source_idx} completion_tokens={generated['completion_token_count']} "
                    f"pred_sql={preview_text(generated['pred_sql'], max_chars=120)}"
                )
            if should_log_progress_tick(completed - 1, len(rows)):
                print(f"[inference] async generated {completed}/{len(rows)} prompts")
            return generated

    try:
        detailed_results = await asyncio.gather(
            *(generate_row(idx, row) for idx, row in enumerate(rows))
        )
    finally:
        engine.shutdown()

    results_by_idx = {result["source_idx"]: result for result in detailed_results}
    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        generated = results_by_idx[source_idx]
        db_id = generated["db_id"]
        pred_sql = generated["pred_sql"]
        official_predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{db_id}"
        detailed_predictions.append(
            {
                "idx": source_idx,
                "db_id": db_id,
                "prediction_text": generated["prediction_text"],
                "pred_sql": pred_sql,
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "prompt_tokens": generated["prompt_tokens"],
                "completion_token_count": generated["completion_token_count"],
                "tool_rounds": generated.get("tool_rounds", 0),
                "tool_call_count": generated.get("tool_call_count", 0),
                "stop_reason": generated.get("stop_reason", ""),
                "error_message": generated.get("error_message", ""),
            }
        )

    return rows, official_predictions, detailed_predictions, skipped_rows


def _vllm_async_generate_worker(
    queue,
    shard_id: int,
    device_group: List[str],
    rows: List[Dict[str, Any]],
    worker_args: argparse.Namespace,
) -> None:
    """Drive one async engine over a single data-parallel shard.

    Spawned via ``mp.get_context("spawn")``, so this process re-imports the module
    and inherits nothing from the parent: ``CUDA_VISIBLE_DEVICES`` must be set
    before vLLM/torch touch CUDA, and the tool executor must be reconfigured
    (``main`` only does that in the parent).
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_group)
        configure_tool_env(worker_args.database_dir)
        _, _, detailed_predictions, _ = asyncio.run(
            _generate_predictions_with_vllm_async_impl(rows, worker_args)
        )
        queue.put({"status": "ok", "shard_id": shard_id, "results": detailed_predictions})
    except Exception:
        queue.put({"status": "error", "shard_id": shard_id, "error": traceback.format_exc()})


def generate_predictions_with_vllm_async_data_parallel(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Async equivalent of generate_predictions_with_vllm_data_parallel.

    Runs ``data_parallel_size`` worker processes, each owning a
    ``tensor_parallel_size``-wide async engine on its own GPU group and serving
    its shard with ``vllm_async_concurrency`` in-flight requests.
    """
    from nl2sql_gspo.model_utils import load_tokenizer

    tensor_parallel_size = args.vllm_tensor_parallel_size
    data_parallel_size = args.vllm_data_parallel_size
    vllm_max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)

    device_groups = plan_vllm_device_groups(tensor_parallel_size, data_parallel_size)

    # Filter in the parent so filtered_examples.jsonl covers the whole run rather
    # than whatever a single shard happened to drop. The child's own filter pass
    # is then a no-op.
    tokenizer = load_tokenizer(resolve_vllm_tokenizer_source(args.model_name_or_path))
    rows, skipped_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)

    row_shards = shard_rows_for_data_parallel(rows, data_parallel_size)
    active_shards = [
        (shard_id, device_group, shard_rows)
        for shard_id, (device_group, shard_rows) in enumerate(zip(device_groups, row_shards))
        if shard_rows
    ]

    worker_args = argparse.Namespace(**vars(args))
    worker_args.vllm_data_parallel_size = 1

    print(
        "[inference] loading async vLLM engines in multi-process data-parallel mode "
        f"tensor_parallel_size={tensor_parallel_size} data_parallel_size={data_parallel_size} "
        f"concurrency_per_shard={args.vllm_async_concurrency} "
        f"device_groups={['+'.join(group) for group in device_groups]} max_model_len={vllm_max_model_len}"
    )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    collect_error: Optional[BaseException] = None

    for shard_id, device_group, shard_rows in active_shards:
        print(
            f"[inference] starting async vLLM shard {shard_id + 1}/{len(active_shards)} "
            f"gpus={','.join(device_group)} prompts={len(shard_rows)}"
        )
        process = ctx.Process(
            target=_vllm_async_generate_worker,
            args=(queue, shard_id, device_group, shard_rows, worker_args),
        )
        process.start()
        processes.append(process)

    collected_results: Dict[int, Dict[str, Any]] = {}
    try:
        for _ in processes:
            message = queue.get()
            if message.get("status") != "ok":
                collect_error = RuntimeError(
                    "async vLLM data-parallel worker failed"
                    + (f" (shard {message.get('shard_id')})" if "shard_id" in message else "")
                    + ":\n"
                    + message.get("error", "unknown error")
                )
                raise collect_error

            for result in message["results"]:
                collected_results[result["idx"]] = result
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    for process in processes:
        if collect_error is None and process.exitcode in (0, None, -15):
            continue
        if process.exitcode not in (0, None):
            raise RuntimeError(f"async vLLM data-parallel worker exited with code {process.exitcode}")

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        generated = collected_results.get(source_idx)
        if generated is None:
            raise RuntimeError(f"Missing async vLLM generation result for idx={source_idx}")

        official_predictions[str(source_idx)] = (
            f"{generated['pred_sql']}{BIRD_SPLIT_MARKER}{generated['db_id']}"
        )
        detailed_predictions.append(generated)

    return rows, official_predictions, detailed_predictions, skipped_rows


def generate_predictions_with_vllm_async(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if (args.vllm_data_parallel_size or 1) > 1:
        return generate_predictions_with_vllm_async_data_parallel(rows, args)
    return asyncio.run(_generate_predictions_with_vllm_async_impl(rows, args))


def generate_predictions(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --skip_generation is set")

    if args.inference_backend == "vllm":
        return generate_predictions_with_vllm(rows, args)

    if args.inference_backend == "vllm_async":
        return generate_predictions_with_vllm_async(rows, args)

    raise ValueError(f"Unsupported inference backend: {args.inference_backend}")


def load_predictions(predictions_path: Path) -> Dict[str, str]:
    with predictions_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _execute_query_pair(queue, predicted_sql: str, ground_sql: str, db_path: str) -> None:
    conn = None
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        cursor = conn.cursor()

        pred_rows = []
        gold_rows = []
        pred_executed = False
        gold_executed = False
        pred_error = ""
        gold_error = ""

        if has_sql_content(predicted_sql):
            try:
                cursor.execute(predicted_sql)
                pred_rows = cursor.fetchall()
                pred_executed = True
            except Exception as exc:
                pred_error = str(exc)
        else:
            pred_error = "empty sql"

        if has_sql_content(ground_sql):
            try:
                cursor.execute(ground_sql)
                gold_rows = cursor.fetchall()
                gold_executed = True
            except Exception as exc:
                gold_error = str(exc)
        else:
            gold_error = "empty sql"

        if pred_executed and gold_executed:
            status = "ok"
            result = int(set(pred_rows) == set(gold_rows))
        else:
            parts = []
            if pred_error:
                parts.append(f"pred_error: {pred_error}")
            if gold_error:
                parts.append(f"gold_error: {gold_error}")
            status = "; ".join(parts) if parts else "error: execution failed"
            result = 0

        queue.put(
            {
                "res": result,
                "status": status,
                "pred_executed": pred_executed,
                "gold_executed": gold_executed,
                "pred_error": pred_error,
                "gold_error": gold_error,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "res": 0,
                "status": f"error: {exc}",
                "pred_executed": False,
                "gold_executed": False,
                "pred_error": str(exc),
                "gold_error": "",
            }
        )
    finally:
        if conn is not None:
            conn.close()


def evaluate_one(predicted_sql: str, ground_sql: str, db_path: str, timeout_s: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_execute_query_pair, args=(queue, predicted_sql, ground_sql, db_path))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "res": 0,
            "status": "timeout",
            "pred_executed": False,
            "gold_executed": False,
            "pred_error": "timeout",
            "gold_error": "timeout",
        }

    if not queue.empty():
        return queue.get()

    return {
        "res": 0,
        "status": "error: no result",
        "pred_executed": False,
        "gold_executed": False,
        "pred_error": "no result",
        "gold_error": "no result",
    }


def evaluate_one_bird(
    predicted_sql: str,
    db_id: str,
    database_dir: str,
    timeout_s: float,
    gold_executed: bool,
    gold_row_set: Optional[frozenset],
    gold_error: str,
) -> Dict[str, Any]:
    pred_rows = None
    pred_executed = False
    pred_error = ""

    if has_sql_content(predicted_sql):
        pred_executed, pred_rows, pred_error = bird_execute_sql(
            sql=predicted_sql,
            db_id=db_id,
            database_dir=database_dir,
            timeout_s=timeout_s,
        )
    else:
        pred_error = "empty sql"

    if pred_executed and gold_executed:
        status = "ok"
        result = int(bird_result_match(pred_rows, gold_row_set))
    else:
        parts = []
        if pred_error:
            parts.append(f"pred_error: {pred_error}")
        if gold_error:
            parts.append(f"gold_error: {gold_error}")
        status = "; ".join(parts) if parts else "error: execution failed"
        result = 0

    return {
        "res": result,
        "status": status,
        "pred_executed": pred_executed,
        "gold_executed": gold_executed,
        "pred_error": pred_error,
        "gold_error": gold_error,
    }

def build_group_summary(
    results: List[Dict[str, Any]],
    group_key: str,
    group_order: Optional[List[str]] = None,
) -> OrderedDict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    for result in results:
        group_value = str(result.get(group_key) or "unknown")
        if group_value not in summary:
            summary[group_value] = {
                "correct": 0,
                "count": 0,
            }

        summary[group_value]["correct"] += int(result["res"])
        summary[group_value]["count"] += 1

    ordered_summary: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    if group_order:
        for group_value in group_order:
            values = summary.pop(group_value, {"correct": 0, "count": 0})
            count = values["count"]
            values["accuracy"] = 100.0 * values["correct"] / max(1, count)
            ordered_summary[group_value] = values

    for group_value, values in sorted(summary.items(), key=lambda item: (-item[1]["count"], item[0])):
        count = values["count"]
        values["accuracy"] = 100.0 * values["correct"] / max(1, count)
        ordered_summary[group_value] = values

    return ordered_summary


def build_execution_stats(results: List[Dict[str, Any]]) -> Dict[str, int]:
    total_count = len(results)
    pred_sql_extracted = sum(int(result["pred_sql_extracted"]) for result in results)
    gold_sql_extracted = sum(int(result["gold_sql_extracted"]) for result in results)
    pred_sql_executed = sum(int(result["pred_executed"]) for result in results)
    gold_sql_executed = sum(int(result["gold_executed"]) for result in results)
    both_sql_executed = sum(int(result["pred_executed"] and result["gold_executed"]) for result in results)

    return {
        "pred_sql_extracted": pred_sql_extracted,
        "pred_sql_missing": total_count - pred_sql_extracted,
        "gold_sql_extracted": gold_sql_extracted,
        "gold_sql_missing": total_count - gold_sql_extracted,
        "pred_sql_executed": pred_sql_executed,
        "pred_sql_execution_failed": total_count - pred_sql_executed,
        "gold_sql_executed": gold_sql_executed,
        "gold_sql_execution_failed": total_count - gold_sql_executed,
        "both_sql_executed": both_sql_executed,
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_correct = sum(int(result["res"]) for result in results)
    total_count = len(results)

    return {
        "by_difficulty": build_group_summary(
            results,
            group_key="difficulty",
            group_order=["simple", "moderate", "challenging"],
        ),
        "by_db": build_group_summary(results, group_key="db_id"),
        "total": {
            "correct": total_correct,
            "count": total_count,
            "accuracy": 100.0 * total_correct / max(1, total_count),
        },
        "execution_stats": build_execution_stats(results),
    }


def build_generation_stats(
    detailed_predictions: List[Dict[str, Any]],
    filtered_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stop_reason_counts = Counter(
        str(detail.get("stop_reason") or "unknown") for detail in detailed_predictions
    )
    tool_call_count_total = sum(int(detail.get("tool_call_count") or 0) for detail in detailed_predictions)
    tool_round_count_total = sum(int(detail.get("tool_rounds") or 0) for detail in detailed_predictions)
    completion_token_total = sum(
        int(detail.get("completion_token_count") or 0) for detail in detailed_predictions
    )
    prompt_token_values = [
        int(detail.get("prompt_tokens") or 0)
        for detail in detailed_predictions
        if detail.get("prompt_tokens") not in {"", None}
    ]
    tool_name_counts: Counter[str] = Counter()
    for detail in detailed_predictions:
        for call in extract_tool_calls(detail.get("prediction_text", "")):
            name = call.get("function", {}).get("name", "")
            if name:
                tool_name_counts[name] += 1

    total = len(detailed_predictions)
    return {
        "generated_examples": total,
        "filtered_examples": len(filtered_rows),
        "stop_reason_counts": dict(stop_reason_counts),
        "tool_call_count_total": tool_call_count_total,
        "tool_round_count_total": tool_round_count_total,
        "avg_tool_calls_per_example": tool_call_count_total / max(1, total),
        "avg_tool_rounds_per_example": tool_round_count_total / max(1, total),
        "tool_name_counts": dict(tool_name_counts),
        "completion_token_total": completion_token_total,
        "avg_completion_tokens": completion_token_total / max(1, total),
        "max_prompt_tokens": max(prompt_token_values) if prompt_token_values else 0,
    }


def render_markdown_table(title: str, rows: OrderedDict[str, Dict[str, Any]]) -> str:
    lines = [f"## {title}", "", "| Group | Correct | Count | Accuracy |", "| --- | ---: | ---: | ---: |"]

    for group_name, values in rows.items():
        lines.append(
            f"| {group_name} | {values['correct']} | {values['count']} | {values['accuracy']:.2f} |"
        )

    return "\n".join(lines)


def print_summary_tables(summary: Dict[str, Any]) -> None:
    def print_group(title: str, rows: OrderedDict[str, Dict[str, Any]]) -> None:
        print(title)
        print(f"{'group':20} {'correct':>10} {'count':>10} {'accuracy':>10}")
        for group_name, values in rows.items():
            print(
                f"{group_name:20} {values['correct']:>10} {values['count']:>10} {values['accuracy']:>9.2f}"
            )
        print()

    print_group("Difficulty Summary", summary["by_difficulty"])
    print_group("DB Summary", summary["by_db"])

    total = summary["total"]
    print(
        f"Total EX Accuracy: {total['accuracy']:.2f}% ({total['correct']}/{total['count']})"
    )

    execution_stats = summary["execution_stats"]
    print("Execution Stats")
    print(f"{'metric':30} {'count':>10}")
    for metric_name in [
        "pred_sql_extracted",
        "pred_sql_missing",
        "gold_sql_extracted",
        "gold_sql_missing",
        "pred_sql_executed",
        "pred_sql_execution_failed",
        "gold_sql_executed",
        "gold_sql_execution_failed",
        "both_sql_executed",
    ]:
        print(f"{metric_name:30} {execution_stats[metric_name]:>10}")

    generation_stats = summary.get("generation_stats") or {}
    if generation_stats:
        print("Generation Stats")
        print(f"{'metric':30} {'value':>10}")
        for metric_name in [
            "generated_examples",
            "filtered_examples",
            "tool_call_count_total",
            "tool_round_count_total",
            "avg_tool_calls_per_example",
            "avg_tool_rounds_per_example",
            "avg_completion_tokens",
            "max_prompt_tokens",
        ]:
            value = generation_stats.get(metric_name, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            print(f"{metric_name:30} {value:>10}")
        print(f"stop_reason_counts: {generation_stats.get('stop_reason_counts', {})}")
        print(f"tool_name_counts: {generation_stats.get('tool_name_counts', {})}")


def _format_seconds(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _markdown_cell(value: Any, max_chars: int = 140) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _tool_order_from_detail(detail: Dict[str, Any]) -> str:
    calls = extract_tool_calls(detail.get("prediction_text", ""))
    return " -> ".join(
        call.get("function", {}).get("name", "")
        for call in calls
        if call.get("function", {}).get("name")
    )


def build_per_example_report_rows(
    detailed_predictions: List[Dict[str, Any]],
    per_example_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    details_by_idx = {detail.get("idx"): detail for detail in detailed_predictions}
    rows: List[Dict[str, Any]] = []

    for result in per_example_results:
        idx = result.get("idx")
        detail = details_by_idx.get(idx, {})
        rows.append(
            {
                "idx": idx,
                "db_id": result.get("db_id", detail.get("db_id", "")),
                "difficulty": result.get("difficulty", ""),
                "correct": int(result.get("res", 0)),
                "status": result.get("status", ""),
                "stop_reason": detail.get("stop_reason", ""),
                "prompt_tokens": detail.get("prompt_tokens", ""),
                "completion_tokens": detail.get("completion_token_count", ""),
                "tool_rounds": detail.get("tool_rounds", ""),
                "tool_call_count": detail.get("tool_call_count", ""),
                "tool_order": _tool_order_from_detail(detail) if detail else "",
                "generation_error": detail.get("error_message", ""),
                "pred_sql_extracted": result.get("pred_sql_extracted", ""),
                "pred_executed": result.get("pred_executed", ""),
                "gold_sql_extracted": result.get("gold_sql_extracted", ""),
                "gold_executed": result.get("gold_executed", ""),
                "pred_error": result.get("pred_error", ""),
                "gold_error": result.get("gold_error", ""),
                "pred_sql": result.get("pred_sql", ""),
                "gold_sql": result.get("gold_sql", ""),
            }
        )

    return rows


def write_per_example_report_csv(report_rows: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "idx",
        "db_id",
        "difficulty",
        "correct",
        "status",
        "stop_reason",
        "prompt_tokens",
        "completion_tokens",
        "tool_rounds",
        "tool_call_count",
        "tool_order",
        "generation_error",
        "pred_sql_extracted",
        "pred_executed",
        "gold_sql_extracted",
        "gold_executed",
        "pred_error",
        "gold_error",
        "pred_sql",
        "gold_sql",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def render_run_config(args: argparse.Namespace, row_count: int) -> List[str]:
    return [
        "## Run Configuration",
        "",
        f"- inference_backend: `{args.inference_backend}`",
        f"- model_name_or_path: `{args.model_name_or_path}`",
        f"- input_file: `{args.input_file}`",
        f"- database_dir: `{args.database_dir}`",
        f"- diff_json_path: `{args.diff_json_path}`",
        f"- num_examples: `{args.num_examples}`",
        f"- loaded_rows: `{row_count}`",
        f"- max_prompt_length: `{args.max_prompt_length}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        f"- max_tool_rounds: `{args.max_tool_rounds}`",
        f"- eval_timeout: `{args.eval_timeout}`",
        f"- eval_workers: `{args.eval_workers}`",
        f"- vllm_tensor_parallel_size: `{args.vllm_tensor_parallel_size}`",
        f"- vllm_data_parallel_size: `{args.vllm_data_parallel_size}`",
        f"- vllm_async_concurrency: `{args.vllm_async_concurrency}`",
        f"- vllm_gpu_memory_utilization: `{args.vllm_gpu_memory_utilization}`",
        f"- vllm_max_model_len: `{args.vllm_max_model_len}`",
        "",
    ]


def render_timing(summary: Dict[str, Any]) -> List[str]:
    timing = summary.get("timing_seconds") or {}
    return [
        "## Timing",
        "",
        f"- generation_seconds: `{_format_seconds(timing.get('generation'))}`",
        f"- evaluation_seconds: `{_format_seconds(timing.get('evaluation'))}`",
        f"- total_seconds: `{_format_seconds(timing.get('total'))}`",
        "",
    ]


def render_generation_stats(summary: Dict[str, Any]) -> List[str]:
    stats = summary.get("generation_stats") or {}
    if not stats:
        return []
    return [
        "## Generation Stats",
        "",
        f"- generated_examples: `{stats.get('generated_examples', 0)}`",
        f"- filtered_examples: `{stats.get('filtered_examples', 0)}`",
        f"- stop_reason_counts: `{stats.get('stop_reason_counts', {})}`",
        f"- tool_call_count_total: `{stats.get('tool_call_count_total', 0)}`",
        f"- tool_round_count_total: `{stats.get('tool_round_count_total', 0)}`",
        f"- avg_tool_calls_per_example: `{float(stats.get('avg_tool_calls_per_example', 0.0)):.3f}`",
        f"- avg_tool_rounds_per_example: `{float(stats.get('avg_tool_rounds_per_example', 0.0)):.3f}`",
        f"- tool_name_counts: `{stats.get('tool_name_counts', {})}`",
        f"- completion_token_total: `{stats.get('completion_token_total', 0)}`",
        f"- avg_completion_tokens: `{float(stats.get('avg_completion_tokens', 0.0)):.3f}`",
        f"- max_prompt_tokens: `{stats.get('max_prompt_tokens', 0)}`",
        "",
    ]


def render_per_example_markdown(report_rows: List[Dict[str, Any]], limit: int = 500) -> List[str]:
    rows = report_rows[:limit]
    lines = [
        "## Per-Example Report",
        "",
        "| idx | correct | status | stop | pred_exec | gold_exec | comp_tokens | rounds | calls | tool_order |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(row.get("idx"), 20),
                    _markdown_cell(row.get("correct"), 20),
                    _markdown_cell(row.get("status"), 80),
                    _markdown_cell(row.get("stop_reason"), 40),
                    _markdown_cell(row.get("pred_executed"), 20),
                    _markdown_cell(row.get("gold_executed"), 20),
                    _markdown_cell(row.get("completion_tokens"), 20),
                    _markdown_cell(row.get("tool_rounds"), 20),
                    _markdown_cell(row.get("tool_call_count"), 20),
                    _markdown_cell(row.get("tool_order"), 160),
                ]
            )
            + " |"
        )

    if len(report_rows) > limit:
        lines.extend(["", f"Showing first {limit} of {len(report_rows)} examples. See `per_example_report.csv` for all rows."])
    lines.append("")
    return lines


def render_report_stats(report_rows: List[Dict[str, Any]]) -> List[str]:
    stop_counts = Counter(str(row.get("stop_reason") or "unknown") for row in report_rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in report_rows)
    correct_counts = Counter(int(row.get("correct", 0)) for row in report_rows)
    return [
        "## Report Counts",
        "",
        f"- correct: `{correct_counts.get(1, 0)}`",
        f"- incorrect: `{correct_counts.get(0, 0)}`",
        f"- stop_reasons: `{dict(stop_counts)}`",
        f"- eval_statuses: `{dict(status_counts)}`",
        "",
    ]


def write_summary_markdown(
    summary: Dict[str, Any],
    markdown_path: Path,
    args: argparse.Namespace,
    row_count: int,
) -> None:
    execution_stats = summary["execution_stats"]
    content = [
        "# BIRD Dev Execution Accuracy Summary",
        "",
        *render_run_config(args, row_count),
        *render_timing(summary),
        *render_generation_stats(summary),
        render_markdown_table("By Difficulty", summary["by_difficulty"]),
        "",
        render_markdown_table("By Database", summary["by_db"]),
        "",
        (
            f"Overall EX Accuracy: {summary['total']['accuracy']:.2f}% "
            f"({summary['total']['correct']}/{summary['total']['count']})"
        ),
        "",
        "## Execution Stats",
        "",
        f"- pred_sql_extracted: {execution_stats['pred_sql_extracted']}",
        f"- pred_sql_missing: {execution_stats['pred_sql_missing']}",
        f"- gold_sql_extracted: {execution_stats['gold_sql_extracted']}",
        f"- gold_sql_missing: {execution_stats['gold_sql_missing']}",
        f"- pred_sql_executed: {execution_stats['pred_sql_executed']}",
        f"- pred_sql_execution_failed: {execution_stats['pred_sql_execution_failed']}",
        f"- gold_sql_executed: {execution_stats['gold_sql_executed']}",
        f"- gold_sql_execution_failed: {execution_stats['gold_sql_execution_failed']}",
        f"- both_sql_executed: {execution_stats['both_sql_executed']}",
        "",
    ]

    markdown_path.write_text("\n".join(content), encoding="utf-8")


def write_run_report_markdown(
    summary: Dict[str, Any],
    report_rows: List[Dict[str, Any]],
    report_path: Path,
    args: argparse.Namespace,
    row_count: int,
) -> None:
    content = [
        "# Inference Run Report",
        "",
        *render_run_config(args, row_count),
        *render_timing(summary),
        *render_generation_stats(summary),
        render_markdown_table("By Difficulty", summary["by_difficulty"]),
        "",
        render_markdown_table("By Database", summary["by_db"]),
        "",
        (
            f"Overall EX Accuracy: {summary['total']['accuracy']:.2f}% "
            f"({summary['total']['correct']}/{summary['total']['count']})"
        ),
        "",
        *render_report_stats(report_rows),
        *render_per_example_markdown(report_rows),
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")


def write_summary_csv(rows: OrderedDict[str, Dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "correct", "count", "accuracy"])
        writer.writeheader()
        for group_name, values in rows.items():
            writer.writerow(
                {
                    "group": group_name,
                    "correct": values["correct"],
                    "count": values["count"],
                    "accuracy": f"{values['accuracy']:.2f}",
                }
            )


def evaluate_predictions(
    rows: List[Dict[str, Any]],
    predictions: Dict[str, str],
    database_dir: str,
    diff_rows: List[Dict[str, Any]],
    timeout_s: float,
    eval_workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_example_results: List[Dict[str, Any]] = []
    log_each_example = should_log_each_example(len(rows))

    prepared_examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        packed_prediction = predictions.get(str(source_idx), "")
        if BIRD_SPLIT_MARKER in packed_prediction:
            predicted_sql, predicted_db_id = packed_prediction.split(BIRD_SPLIT_MARKER, 1)
        else:
            predicted_sql = packed_prediction
            predicted_db_id = row.get("db_id", "")

        db_id = predicted_db_id or row.get("db_id", "")
        difficulty = diff_rows[source_idx].get("difficulty", "unknown") if source_idx < len(diff_rows) else "unknown"
        db_path = get_database_path(db_id=db_id, database_dir=database_dir)
        gold_sql = extract_sql(row.get("gold_sql", ""))
        pred_sql_extracted = has_sql_content(predicted_sql)
        gold_sql_extracted = has_sql_content(gold_sql)

        prepared_examples.append(
            {
                "idx": idx,
                "source_idx": source_idx,
                "db_id": db_id,
                "difficulty": difficulty,
                "db_path": db_path,
                "predicted_sql": predicted_sql,
                "gold_sql": gold_sql,
                "pred_sql_extracted": pred_sql_extracted,
                "gold_sql_extracted": gold_sql_extracted,
            }
        )

    worker_count = max(1, eval_workers)
    eval_results: List[Dict[str, Any]] = [None] * len(prepared_examples)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        gold_eval_results = list(
            executor.map(
                lambda example: bird_get_gold_rows(
                    example["gold_sql"],
                    example["db_id"],
                    database_dir,
                    timeout_s,
                )
                if example["gold_sql_extracted"]
                else (False, None, "empty sql"),
                prepared_examples,
            )
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        ordered_results = executor.map(
            lambda item: evaluate_one_bird(
                item[0]["predicted_sql"],
                item[0]["db_id"],
                database_dir,
                timeout_s,
                item[1][0],
                item[1][1],
                item[1][2],
            ),
            zip(prepared_examples, gold_eval_results),
        )

        for example, gold_eval_result, eval_result in zip(prepared_examples, gold_eval_results, ordered_results):
            idx = example["idx"]
            source_idx = example["source_idx"]
            db_id = example["db_id"]
            difficulty = example["difficulty"]
            gold_executed = bool(gold_eval_result[0])
            gold_error = gold_eval_result[2]

            if log_each_example:
                print(
                    f"[evaluation] scoring sample {idx + 1}/{len(rows)} "
                    f"idx={source_idx} db_id={db_id} difficulty={difficulty}"
                )

            eval_results[idx] = {
                "idx": source_idx,
                "db_id": db_id,
                "difficulty": difficulty,
                "pred_sql": example["predicted_sql"],
                "gold_sql": example["gold_sql"],
                "pred_sql_extracted": example["pred_sql_extracted"],
                "gold_sql_extracted": example["gold_sql_extracted"],
                "res": int(eval_result["res"]),
                "status": eval_result["status"],
                "pred_executed": bool(eval_result["pred_executed"]),
                "gold_executed": gold_executed,
                "pred_error": eval_result["pred_error"],
                "gold_error": gold_error,
            }

            if log_each_example:
                print(
                    f"[evaluation] finished sample {idx + 1}/{len(rows)} "
                    f"idx={source_idx} status={eval_result['status']} correct={int(eval_result['res'])}"
                )

            if should_log_progress_tick(idx, len(rows)):
                print(f"[evaluation] scored {idx + 1}/{len(rows)} predictions")

    per_example_results = eval_results

    summary = build_summary(per_example_results)
    return per_example_results, summary


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already contains files. Use --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    run_started_at = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    print_run_configuration(args, output_dir)

    rows = load_rows(args.input_file, args.num_examples)
    print(f"[run] loaded {len(rows)} input rows")

    # Configure tool environment for database access
    configure_tool_env(args.database_dir)
    
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
    diff_rows = load_diff_rows(args.diff_json_path)
    print(f"[run] loaded {len(diff_rows)} diff rows")

    detailed_predictions: List[Dict[str, Any]] = []
    if args.skip_generation:
        generation_started_at = time.monotonic()
        official_predictions = load_predictions(predictions_path)
        filtered_rows: List[Dict[str, Any]] = []
        if details_path.exists():
            with details_path.open("r", encoding="utf-8") as handle:
                detailed_predictions = [json.loads(line) for line in handle if line.strip()]
        generation_seconds = time.monotonic() - generation_started_at
    else:
        generation_started_at = time.monotonic()
        rows, official_predictions, detailed_predictions, filtered_rows = generate_predictions(rows, args)
        generation_seconds = time.monotonic() - generation_started_at
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(official_predictions, handle, ensure_ascii=False, indent=2)

        with details_path.open("w", encoding="utf-8") as handle:
            for record in detailed_predictions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        with filtered_path.open("w", encoding="utf-8") as handle:
            for record in filtered_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    evaluation_started_at = time.monotonic()
    per_example_results, summary = evaluate_predictions(
        rows=rows,
        predictions=official_predictions,
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
    summary["generation_stats"] = build_generation_stats(detailed_predictions, filtered_rows)

    with per_example_eval_path.open("w", encoding="utf-8") as handle:
        for record in per_example_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_rows = build_per_example_report_rows(detailed_predictions, per_example_results)
    write_per_example_report_csv(report_rows, per_example_report_csv_path)
    write_summary_markdown(summary, summary_markdown_path, args, len(rows))
    write_run_report_markdown(summary, report_rows, run_report_path, args, len(rows))
    write_summary_csv(summary["by_difficulty"], difficulty_csv_path)
    write_summary_csv(summary["by_db"], db_csv_path)
    print_summary_tables(summary)
    print(
        "Timing Seconds "
        f"generation={generation_seconds:.2f} "
        f"evaluation={evaluation_seconds:.2f} "
        f"total={total_seconds:.2f}"
    )
    print(f"Saved official BIRD predictions to {predictions_path}")
    print(f"Saved filtered-example report to {filtered_path}")
    print(f"Saved per-example evaluation to {per_example_eval_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved markdown summary to {summary_markdown_path}")
    print(f"Saved run report to {run_report_path}")
    print(f"Saved per-example CSV report to {per_example_report_csv_path}")
    print(f"Saved difficulty CSV summary to {difficulty_csv_path}")
    print(f"Saved DB CSV summary to {db_csv_path}")


if __name__ == "__main__":
    main()
