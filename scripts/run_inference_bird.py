import argparse
import asyncio
import csv
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    print("[run] inference_backend=vllm_async")
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
    print(f"[run] shard_index={args.shard_index}")
    print(f"[run] num_shards={args.num_shards}")
    print(f"[run] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[run] vllm_tensor_parallel_size={args.vllm_tensor_parallel_size}")
    print(f"[run] vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
    print(f"[run] vllm_max_model_len={args.vllm_max_model_len}")
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
    if not diff_json_path:
        return []
    path = Path(diff_json_path)
    if not path.exists():
        print(f"[run] metadata path not found: {diff_json_path}; using unknown difficulty labels")
        return []

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    return loaded if isinstance(loaded, list) else [loaded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="outputs/bird_dev-schema.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/bird_dev.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_inference")
    parser.add_argument(
        "--predictions_filename",
        type=str,
        default="predict_dev.json",
        help="Official BIRD-format predictions filename to read/write.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=43000)
    parser.add_argument("--vllm_async_concurrency", type=int, default=8)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard index. Use with --num_shards to split examples across processes/GPUs.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of example shards. Each shard keeps original source_idx values.",
    )
    parser.add_argument(
        "--no_append_shard_to_output_dir",
        action="store_true",
        help=(
            "Do not append shard-XXXXX-of-YYYYY to output_dir when num_shards > 1. "
            "Only use this when each shard already has a unique output_dir."
        ),
    )
    parser.add_argument(
        "--merge_shard_dirs",
        nargs="*",
        default=None,
        help=(
            "Merge already-generated shard output directories instead of running generation. "
            "Each directory must contain the predictions file and may contain eval_results.jsonl."
        ),
    )
    parser.add_argument(
        "--merge_output_dir",
        type=str,
        default=None,
        help="Directory for merged inference outputs. Defaults to --output_dir.",
    )
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.vllm_tensor_parallel_size is None:
        args.vllm_tensor_parallel_size = 8
    return args


def predictions_path_for(output_dir: Path, args: argparse.Namespace) -> Path:
    return output_dir / args.predictions_filename


def row_has_gold_sql(row: Dict[str, Any]) -> bool:
    return has_sql_content(extract_sql(row.get("gold_sql", "")))


def rows_have_gold_sql(rows: List[Dict[str, Any]]) -> bool:
    return bool(rows) and all(row_has_gold_sql(row) for row in rows)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.num_shards > 1 and not args.no_append_shard_to_output_dir:
        shard_name = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
        if output_dir.name != shard_name:
            output_dir = output_dir / shard_name
    return output_dir


def shard_rows(rows: List[Dict[str, Any]], shard_index: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards == 1:
        return rows
    return [
        row
        for row in rows
        if int(row.get("source_idx", -1)) % num_shards == shard_index
    ]


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


def render_prompt(tokenizer, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
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


def generate_predictions_with_vllm_async(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    return asyncio.run(_generate_predictions_with_vllm_async_impl(rows, args))


def generate_predictions(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --skip_generation is set")

    return generate_predictions_with_vllm_async(rows, args)


def load_predictions(predictions_path: Path) -> Dict[str, str]:
    with predictions_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def evaluate_prediction_only(
    predicted_sql: str,
    db_id: str,
    database_dir: str,
    timeout_s: float,
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

    return {
        "res": None,
        "status": "pred_ok" if pred_executed else f"pred_error: {pred_error}",
        "pred_executed": pred_executed,
        "gold_executed": False,
        "pred_error": pred_error,
        "gold_error": "",
        "pred_row_count": len(pred_rows) if pred_rows is not None else None,
    }

def build_group_summary(
    results: List[Dict[str, Any]],
    group_key: str,
    group_order: Optional[List[str]] = None,
    has_gold_labels: bool = True,
) -> OrderedDict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    for result in results:
        group_value = str(result.get(group_key) or "unknown")
        if group_value not in summary:
            summary[group_value] = {
                "correct": 0,
                "count": 0,
            }

        summary[group_value]["correct"] += int(result.get("res") or 0)
        summary[group_value]["count"] += 1

    ordered_summary: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    if group_order:
        for group_value in group_order:
            values = summary.pop(group_value, {"correct": 0, "count": 0})
            count = values["count"]
            values["accuracy"] = 100.0 * values["correct"] / max(1, count) if has_gold_labels else None
            ordered_summary[group_value] = values

    for group_value, values in sorted(summary.items(), key=lambda item: (-item[1]["count"], item[0])):
        count = values["count"]
        values["accuracy"] = 100.0 * values["correct"] / max(1, count) if has_gold_labels else None
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
    has_gold_labels = bool(results) and all(bool(result.get("gold_sql_extracted")) for result in results)
    total_correct = sum(int(result.get("res") or 0) for result in results)
    total_count = len(results)

    return {
        "by_difficulty": build_group_summary(
            results,
            group_key="difficulty",
            group_order=["simple", "moderate", "challenging"],
            has_gold_labels=has_gold_labels,
        ),
        "by_db": build_group_summary(results, group_key="db_id", has_gold_labels=has_gold_labels),
        "total": {
            "correct": total_correct,
            "count": total_count,
            "accuracy": 100.0 * total_correct / max(1, total_count) if has_gold_labels else None,
        },
        "execution_stats": build_execution_stats(results),
        "has_gold_sql": has_gold_labels,
        "evaluation_mode": "execution_accuracy" if has_gold_labels else "prediction_execution_only",
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
        accuracy = values.get("accuracy")
        accuracy_text = f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else "n/a"
        lines.append(
            f"| {group_name} | {values['correct']} | {values['count']} | {accuracy_text} |"
        )

    return "\n".join(lines)


def print_summary_tables(summary: Dict[str, Any]) -> None:
    def print_group(title: str, rows: OrderedDict[str, Dict[str, Any]]) -> None:
        print(title)
        print(f"{'group':20} {'correct':>10} {'count':>10} {'accuracy':>10}")
        for group_name, values in rows.items():
            accuracy = values.get("accuracy")
            accuracy_text = f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else "n/a"
            print(
                f"{group_name:20} {values['correct']:>10} {values['count']:>10} {accuracy_text:>10}"
            )
        print()

    print_group("Difficulty Summary", summary["by_difficulty"])
    print_group("DB Summary", summary["by_db"])

    total = summary["total"]
    if isinstance(total.get("accuracy"), (int, float)):
        print(
            f"Total EX Accuracy: {total['accuracy']:.2f}% ({total['correct']}/{total['count']})"
        )
    else:
        print(f"Total EX Accuracy: n/a (no gold SQL; checked {total['count']} predictions)")

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
                "correct": "" if result.get("res") is None else int(result.get("res", 0)),
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
                "pred_row_count": result.get("pred_row_count", ""),
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
        "pred_row_count",
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
        "- inference_backend: `vllm_async`",
        f"- model_name_or_path: `{args.model_name_or_path}`",
        f"- input_file: `{args.input_file}`",
        f"- database_dir: `{args.database_dir}`",
        f"- diff_json_path: `{args.diff_json_path}`",
        f"- predictions_filename: `{args.predictions_filename}`",
        f"- temperature: `{args.temperature}`",
        f"- top_p: `{args.top_p}`",
        f"- num_examples: `{args.num_examples}`",
        f"- loaded_rows: `{row_count}`",
        f"- shard_index: `{args.shard_index}`",
        f"- num_shards: `{args.num_shards}`",
        f"- max_prompt_length: `{args.max_prompt_length}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        f"- max_tool_rounds: `{args.max_tool_rounds}`",
        f"- eval_timeout: `{args.eval_timeout}`",
        f"- eval_workers: `{args.eval_workers}`",
        f"- vllm_tensor_parallel_size: `{args.vllm_tensor_parallel_size}`",
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
    correct_counts = Counter(row.get("correct", "") for row in report_rows)
    return [
        "## Report Counts",
        "",
        f"- correct: `{correct_counts.get(1, 0)}`",
        f"- incorrect: `{correct_counts.get(0, 0)}`",
        f"- unlabeled: `{correct_counts.get('', 0)}`",
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
    has_gold_sql = bool(summary.get("has_gold_sql"))
    accuracy_line = (
        f"Overall EX Accuracy: {summary['total']['accuracy']:.2f}% "
        f"({summary['total']['correct']}/{summary['total']['count']})"
        if has_gold_sql
        else f"Overall EX Accuracy: n/a; no gold SQL labels found for {summary['total']['count']} rows."
    )
    content = [
        "# BIRD Inference Summary",
        "",
        *render_run_config(args, row_count),
        *render_timing(summary),
        *render_generation_stats(summary),
        render_markdown_table("By Difficulty", summary["by_difficulty"]),
        "",
        render_markdown_table("By Database", summary["by_db"]),
        "",
        accuracy_line,
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
    has_gold_sql = bool(summary.get("has_gold_sql"))
    accuracy_line = (
        f"Overall EX Accuracy: {summary['total']['accuracy']:.2f}% "
        f"({summary['total']['correct']}/{summary['total']['count']})"
        if has_gold_sql
        else f"Overall EX Accuracy: n/a; no gold SQL labels found for {summary['total']['count']} rows."
    )
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
        accuracy_line,
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
                    "accuracy": (
                        f"{values['accuracy']:.2f}"
                        if isinstance(values.get("accuracy"), (int, float))
                        else ""
                    ),
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

    has_gold_labels = bool(prepared_examples) and all(
        example["gold_sql_extracted"] for example in prepared_examples
    )
    worker_count = max(1, eval_workers)
    eval_results: List[Dict[str, Any]] = [None] * len(prepared_examples)
    if not has_gold_labels:
        print("[evaluation] no complete gold SQL labels found; running prediction execution checks only")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            ordered_results = executor.map(
                lambda example: evaluate_prediction_only(
                    example["predicted_sql"],
                    example["db_id"],
                    database_dir,
                    timeout_s,
                ),
                prepared_examples,
            )

            for example, eval_result in zip(prepared_examples, ordered_results):
                idx = example["idx"]
                source_idx = example["source_idx"]
                eval_results[idx] = {
                    "idx": source_idx,
                    "db_id": example["db_id"],
                    "difficulty": example["difficulty"],
                    "pred_sql": example["predicted_sql"],
                    "gold_sql": example["gold_sql"],
                    "pred_sql_extracted": example["pred_sql_extracted"],
                    "gold_sql_extracted": example["gold_sql_extracted"],
                    "res": None,
                    "status": eval_result["status"],
                    "pred_executed": bool(eval_result["pred_executed"]),
                    "gold_executed": False,
                    "pred_error": eval_result["pred_error"],
                    "gold_error": "",
                    "pred_row_count": eval_result.get("pred_row_count"),
                }

                if should_log_progress_tick(idx, len(rows)):
                    print(f"[evaluation] checked {idx + 1}/{len(rows)} predictions")

        per_example_results = eval_results
        summary = build_summary(per_example_results)
        return per_example_results, summary

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


def load_shard_timing(shard_dir: Path) -> Dict[str, float]:
    summary_path = shard_dir / "eval_summary.json"
    if not summary_path.exists():
        return {"generation": 0.0, "evaluation": 0.0, "total": 0.0}

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    timing = summary.get("timing_seconds") or {}
    return {
        "generation": float(timing.get("generation") or 0.0),
        "evaluation": float(timing.get("evaluation") or 0.0),
        "total": float(timing.get("total") or 0.0),
    }


def _preview_duplicates(values: List[Any]) -> str:
    return ", ".join(str(value) for value in values[:10])


def merge_shard_outputs(args: argparse.Namespace) -> None:
    shard_dirs = [Path(path) for path in args.merge_shard_dirs or []]
    if not shard_dirs:
        raise ValueError("--merge_shard_dirs requires at least one shard output directory")

    output_dir = Path(args.merge_output_dir or args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    print(f"[merge] output_dir={output_dir}")

    rows = load_rows(args.input_file, args.num_examples)
    diff_rows = load_diff_rows(args.diff_json_path)

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    filtered_rows: List[Dict[str, Any]] = []
    per_example_results: List[Dict[str, Any]] = []

    seen_prediction_idxs = set()
    seen_detail_idxs = set()
    seen_filtered_idxs = set()
    seen_eval_idxs = set()
    duplicate_prediction_idxs: List[int] = []
    duplicate_detail_idxs: List[int] = []
    duplicate_filtered_idxs: List[int] = []
    duplicate_eval_idxs: List[int] = []
    timing = {"generation": 0.0, "evaluation": 0.0, "total": 0.0}

    for shard_dir in shard_dirs:
        predictions_path = predictions_path_for(shard_dir, args)
        if not predictions_path.exists() and args.predictions_filename != "predict_dev.json":
            legacy_predictions_path = shard_dir / "predict_dev.json"
            if legacy_predictions_path.exists():
                predictions_path = legacy_predictions_path
        details_path = shard_dir / "prediction_details.jsonl"
        filtered_path = shard_dir / "filtered_examples.jsonl"
        eval_path = shard_dir / "eval_results.jsonl"

        if not predictions_path.exists():
            raise FileNotFoundError(predictions_path)

        shard_predictions = load_predictions(predictions_path)
        shard_details = read_jsonl(details_path)
        shard_filtered = read_jsonl(filtered_path)
        shard_eval = read_jsonl(eval_path)
        print(
            f"[merge] reading {shard_dir} "
            f"predictions={len(shard_predictions)} eval_rows={len(shard_eval)} "
            f"filtered={len(shard_filtered)}"
        )

        for key, value in shard_predictions.items():
            idx = int(key)
            if idx in seen_prediction_idxs:
                duplicate_prediction_idxs.append(idx)
            seen_prediction_idxs.add(idx)
            official_predictions[str(idx)] = value

        for detail in shard_details:
            idx = int(detail.get("idx", detail.get("source_idx", -1)))
            if idx in seen_detail_idxs:
                duplicate_detail_idxs.append(idx)
            seen_detail_idxs.add(idx)
            normalized_detail = dict(detail)
            normalized_detail["idx"] = idx
            detailed_predictions.append(normalized_detail)

        for filtered in shard_filtered:
            idx = int(filtered.get("idx", filtered.get("source_idx", -1)))
            if idx in seen_filtered_idxs:
                duplicate_filtered_idxs.append(idx)
            seen_filtered_idxs.add(idx)
            normalized_filtered = dict(filtered)
            normalized_filtered["idx"] = idx
            filtered_rows.append(normalized_filtered)

        for result in shard_eval:
            idx = int(result["idx"])
            if idx in seen_eval_idxs:
                duplicate_eval_idxs.append(idx)
            seen_eval_idxs.add(idx)
            normalized_result = dict(result)
            normalized_result["idx"] = idx
            if idx < len(diff_rows):
                normalized_result["difficulty"] = diff_rows[idx].get(
                    "difficulty",
                    normalized_result.get("difficulty", "unknown"),
                )
            per_example_results.append(normalized_result)

        shard_timing = load_shard_timing(shard_dir)
        for key in timing:
            timing[key] += shard_timing.get(key, 0.0)

    duplicate_messages = []
    if duplicate_prediction_idxs:
        duplicate_messages.append(f"predictions={_preview_duplicates(duplicate_prediction_idxs)}")
    if duplicate_detail_idxs:
        duplicate_messages.append(f"details={_preview_duplicates(duplicate_detail_idxs)}")
    if duplicate_filtered_idxs:
        duplicate_messages.append(f"filtered={_preview_duplicates(duplicate_filtered_idxs)}")
    if duplicate_eval_idxs:
        duplicate_messages.append(f"eval={_preview_duplicates(duplicate_eval_idxs)}")
    if duplicate_messages:
        raise ValueError("Duplicate shard rows found: " + "; ".join(duplicate_messages))

    official_predictions = {
        str(idx): official_predictions[str(idx)]
        for idx in sorted(int(key) for key in official_predictions)
    }
    detailed_predictions.sort(key=lambda row: int(row.get("idx", -1)))
    filtered_rows.sort(key=lambda row: int(row.get("idx", -1)))
    per_example_results.sort(key=lambda row: int(row.get("idx", -1)))

    evaluated_idxs = {int(row["idx"]) for row in per_example_results}
    filtered_idxs = {int(row["idx"]) for row in filtered_rows}
    expected_idxs = {int(row.get("source_idx", idx)) for idx, row in enumerate(rows)}
    missing_idxs = sorted(expected_idxs - evaluated_idxs - filtered_idxs)
    if missing_idxs:
        print(
            "[merge] warning: merged shards are missing "
            f"{len(missing_idxs)} loaded rows; first_missing={missing_idxs[:10]}"
        )

    if not per_example_results:
        print("[merge] no shard eval rows found; rebuilding evaluation/prediction checks from merged predictions")
        evaluation_started_at = time.monotonic()
        per_example_results, summary = evaluate_predictions(
            rows=rows,
            predictions=official_predictions,
            database_dir=args.database_dir,
            diff_rows=diff_rows,
            timeout_s=args.eval_timeout,
            eval_workers=args.eval_workers,
        )
        timing["evaluation"] += time.monotonic() - evaluation_started_at
        timing["total"] += timing["evaluation"]
    else:
        summary = build_summary(per_example_results)
    summary["timing_seconds"] = timing
    summary["generation_stats"] = build_generation_stats(detailed_predictions, filtered_rows)
    summary["merged_shards"] = [str(path) for path in shard_dirs]
    summary["merge_coverage"] = {
        "loaded_rows": len(rows),
        "evaluated_rows": len(per_example_results),
        "filtered_rows": len(filtered_rows),
        "missing_rows": len(missing_idxs),
    }

    predictions_path = predictions_path_for(output_dir, args)
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
        json.dump(official_predictions, handle, ensure_ascii=False, indent=2)
    write_jsonl(details_path, detailed_predictions)
    write_jsonl(filtered_path, filtered_rows)
    write_jsonl(per_example_eval_path, per_example_results)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_rows = build_per_example_report_rows(detailed_predictions, per_example_results)
    write_per_example_report_csv(report_rows, per_example_report_csv_path)
    write_summary_markdown(summary, summary_markdown_path, args, len(rows))
    write_run_report_markdown(summary, report_rows, run_report_path, args, len(rows))
    write_summary_csv(summary["by_difficulty"], difficulty_csv_path)
    write_summary_csv(summary["by_db"], db_csv_path)
    print_summary_tables(summary)
    print(f"[merge] merged {len(shard_dirs)} shards into {output_dir}")
    print(f"[merge] wrote {run_report_path}")


def main() -> None:
    run_started_at = time.monotonic()
    args = parse_args()
    if args.merge_shard_dirs is not None:
        merge_shard_outputs(args)
        return

    output_dir = resolve_output_dir(args)
    args.output_dir = str(output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    print_run_configuration(args, output_dir)

    rows = load_rows(args.input_file, args.num_examples)
    original_row_count = len(rows)
    print(f"[run] loaded {original_row_count} input rows")
    if rows_have_gold_sql(rows):
        print("[run] gold SQL labels detected; local EX evaluation will run")
    else:
        print("[run] no complete gold SQL labels detected; writing predictions and execution-check reports only")
    rows = shard_rows(rows, args.shard_index, args.num_shards)
    if not rows:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} received no rows from "
            f"{original_row_count} loaded examples."
        )
    if args.num_shards > 1:
        print(
            f"[run] shard rows={len(rows)}/{original_row_count} "
            f"first_idx={rows[0].get('source_idx')} last_idx={rows[-1].get('source_idx')}"
        )

    # Configure tool environment for database access
    configure_tool_env(args.database_dir)
    
    predictions_path = predictions_path_for(output_dir, args)
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
