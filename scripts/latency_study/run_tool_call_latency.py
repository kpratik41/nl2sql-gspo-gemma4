#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import math
import time
import uuid
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.run_inference_bird import (
    BIRD_SPLIT_MARKER,
    build_assistant_tool_message,
    build_generation_stats,
    build_per_example_report_rows,
    configure_tool_env,
    ensure_output_dir,
    evaluate_predictions,
    extract_tool_calls,
    gemma_tool_loop_stop_token_ids,
    get_generation_messages,
    keep_first_tool_call_only,
    load_diff_rows,
    load_rows,
    prepare_rows_for_generation,
    preview_text,
    render_prompt,
    resolve_vllm_tokenizer_source,
    should_log_each_example,
    should_log_progress_tick,
    should_use_agentic_tool_loop,
    write_run_report_markdown,
    write_summary_csv,
    write_summary_markdown,
)
from nl2sql_gspo.inference_tool_executor import execute_tool_calls
from nl2sql_gspo.sql_utils import extract_sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async vLLM tool-call latency study runner.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_file", default="outputs/old-dev-schema-tool-unpatched.jsonl")
    parser.add_argument("--database_dir", default="databases/dev_databases")
    parser.add_argument("--diff_json_path", default="data/bird_dev_data/raw/bird_dev_unpatched.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--max_prompt_length", type=int, default=34000)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--max_tool_rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, required=True)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--vllm_max_model_len", type=int, default=43500)
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--vllm_async_concurrency", type=int, default=16)
    parser.add_argument("--vllm_disable_log_stats", action="store_true")
    parser.add_argument("--speculative_config_json", default=None)
    parser.add_argument("--spec_method", default=None)
    parser.add_argument("--spec_model", default=None)
    parser.add_argument("--spec_tokens", type=int, default=None)
    parser.add_argument("--capture_spec_decode_stats", action="store_true")
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.inference_backend = "vllm_async_latency"
    args.vllm_data_parallel_size = 1
    args.skip_generation = False
    return args


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: List[float], q: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[int(position)]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def percentile_block(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def int_distribution(values: Iterable[int]) -> Dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def empty_spec_decode_stats() -> Dict[str, Any]:
    return {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "num_accepted_tokens_per_pos": [],
        "num_draft_tokens_per_pos": [],
    }


def add_position_counts(target: List[int], values: Iterable[int]) -> None:
    for index, value in enumerate(values):
        if index >= len(target):
            target.append(0)
        target[index] += int(value)


def finalize_spec_decode_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    num_drafts = int(stats.get("num_drafts") or 0)
    num_draft_tokens = int(stats.get("num_draft_tokens") or 0)
    num_accepted_tokens = int(stats.get("num_accepted_tokens") or 0)
    accepted_per_pos = [int(value) for value in stats.get("num_accepted_tokens_per_pos") or []]
    draft_per_pos = [int(value) for value in stats.get("num_draft_tokens_per_pos") or []]
    position_rates = []
    for index, accepted in enumerate(accepted_per_pos):
        denominator = draft_per_pos[index] if index < len(draft_per_pos) else num_drafts
        position_rates.append((accepted / denominator) if denominator else None)
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "num_accepted_tokens_per_pos": accepted_per_pos,
        "num_draft_tokens_per_pos": draft_per_pos,
        "acceptance_rate": (num_accepted_tokens / num_draft_tokens) if num_draft_tokens else None,
        "mean_acceptance_length": (1.0 + num_accepted_tokens / num_drafts) if num_drafts else None,
        "per_position_acceptance_rate": position_rates,
    }


async def generate_text_with_metrics(
    engine: Any,
    sampling_params_cls: Any,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_prefix: str,
    round_index: int,
    prompt_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
) -> Tuple[str, int, Dict[str, Any]]:
    sampling_params = sampling_params_cls(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    request_id = f"{request_prefix}-r{round_index}-{uuid.uuid4().hex}"
    started_at = time.perf_counter()
    first_token_seconds: Optional[float] = None
    final_output = None

    async for request_output in engine.generate(prompt_text, sampling_params, request_id=request_id):
        if first_token_seconds is None:
            output = request_output.outputs[0] if request_output.outputs else None
            if output is not None and (output.token_ids or output.text):
                first_token_seconds = time.perf_counter() - started_at
        final_output = request_output

    total_seconds = time.perf_counter() - started_at
    first_output = final_output.outputs[0] if final_output and final_output.outputs else None
    generated_text = (first_output.text or "").strip() if first_output else ""
    completion_tokens = len(first_output.token_ids) if first_output else 0
    metric = {
        "request_id": request_id,
        "round_index": round_index,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_token_seconds,
        "generation_seconds": total_seconds,
    }
    return generated_text, completion_tokens, metric


async def generate_one_with_latency(
    engine: Any,
    sampling_params_cls: Any,
    tokenizer: Any,
    row: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    example_started_at = time.perf_counter()
    messages = [dict(message) for message in get_generation_messages(row)]
    tools = row.get("tools")
    generated_parts: List[str] = []
    request_metrics: List[Dict[str, Any]] = []
    prompt_token_count = int(row["prompt_tokens"])
    completion_token_count = 0
    tool_rounds = 0
    tool_call_count = 0
    tool_execution_seconds = 0.0
    stop_reason = "finished"
    error_message = ""
    request_prefix = f"idx{row.get('source_idx', -1)}"

    try:
        for round_index in range(args.max_tool_rounds + 1):
            remaining_tokens = args.max_new_tokens - completion_token_count
            if remaining_tokens <= 0:
                stop_reason = "max_new_tokens"
                break

            prompt_text = render_prompt(tokenizer, messages, tools)
            current_prompt_tokens = len(tokenizer(prompt_text, truncation=False)["input_ids"])
            if round_index == 0:
                prompt_token_count = current_prompt_tokens

            available_context_tokens = args.vllm_max_model_len - current_prompt_tokens
            if available_context_tokens <= 0:
                stop_reason = "context_length_exceeded"
                break

            request_max_tokens = min(remaining_tokens, available_context_tokens)
            generated_text, round_tokens, request_metric = await generate_text_with_metrics(
                engine=engine,
                sampling_params_cls=sampling_params_cls,
                prompt_text=prompt_text,
                max_tokens=request_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_prefix=request_prefix,
                round_index=round_index,
                prompt_tokens=current_prompt_tokens,
                stop_token_ids=gemma_tool_loop_stop_token_ids(tokenizer),
            )
            request_metrics.append(request_metric)
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

            if round_index >= args.max_tool_rounds:
                stop_reason = "max_tool_rounds"
                break

            generated_text, tool_calls = keep_first_tool_call_only(generated_text, tool_calls)
            completion_token_count += len(tokenizer(generated_text, truncation=False, add_special_tokens=False)["input_ids"])
            if generated_text:
                generated_parts.append(generated_text)

            tool_started_at = time.perf_counter()
            tool_responses = await asyncio.to_thread(execute_tool_calls, tool_calls, args.eval_timeout)
            tool_execution_seconds += time.perf_counter() - tool_started_at
            for response in tool_responses:
                generated_parts.append(response["rendered"])
            messages.append(build_assistant_tool_message(generated_text, tool_calls, tool_responses))
            tool_rounds += 1
            tool_call_count += len(tool_calls)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        stop_reason = "generation_error"

    prediction_text = "\n".join(part for part in generated_parts if part).strip()
    llm_generation_seconds = sum(float(metric["generation_seconds"]) for metric in request_metrics)
    first_ttft = next((metric.get("ttft_seconds") for metric in request_metrics if metric.get("ttft_seconds") is not None), None)
    total_latency_seconds = time.perf_counter() - example_started_at
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
        "latency": {
            "first_ttft_seconds": first_ttft,
            "total_latency_seconds": total_latency_seconds,
            "llm_generation_seconds": llm_generation_seconds,
            "tool_execution_seconds": tool_execution_seconds,
            "request_count": len(request_metrics),
            "requests": request_metrics,
        },
    }


async def generate_predictions_with_latency(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, float],
]:
    from nl2sql_gspo.model_utils import load_tokenizer
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    spec_decode_stats = empty_spec_decode_stats()

    def spec_decode_stat_logger_factory(vllm_config: Any, engine_index: int = 0) -> Any:
        class SpecDecodeCaptureLogger:
            def __init__(self, vllm_config: Any, engine_index: int = 0):
                self.engine_index = engine_index

            def record(
                self,
                scheduler_stats: Any,
                iteration_stats: Any,
                mm_cache_stats: Any = None,
                engine_idx: int = 0,
            ) -> None:
                spec_stats = getattr(scheduler_stats, "spec_decoding_stats", None)
                if spec_stats is None:
                    return
                spec_decode_stats["num_drafts"] += int(getattr(spec_stats, "num_drafts", 0) or 0)
                spec_decode_stats["num_draft_tokens"] += int(getattr(spec_stats, "num_draft_tokens", 0) or 0)
                spec_decode_stats["num_accepted_tokens"] += int(getattr(spec_stats, "num_accepted_tokens", 0) or 0)
                add_position_counts(
                    spec_decode_stats["num_accepted_tokens_per_pos"],
                    getattr(spec_stats, "num_accepted_tokens_per_pos", None) or [],
                )
                add_position_counts(
                    spec_decode_stats["num_draft_tokens_per_pos"],
                    getattr(spec_stats, "num_draft_tokens_per_pos", None) or [],
                )

            def log_engine_initialized(self) -> None:
                return None

            def log(self) -> None:
                return None

            def record_sleep_state(self, is_awake: int, level: int) -> None:
                return None

        return SpecDecodeCaptureLogger(vllm_config, engine_index)

    tokenizer_source = resolve_vllm_tokenizer_source(args.model_name_or_path)
    tokenizer = load_tokenizer(tokenizer_source)
    rows, filtered_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)
    if any(not should_use_agentic_tool_loop(row, args.max_tool_rounds) for row in rows):
        print("[latency] warning: some rows do not have tools; they will still use the tool-loop renderer")

    print(
        "[latency] loading async vLLM engine "
        f"tp={args.vllm_tensor_parallel_size} concurrency={args.vllm_async_concurrency} "
        f"max_model_len={args.vllm_max_model_len} "
        f"max_num_batched_tokens={args.vllm_max_num_batched_tokens}"
    )
    engine_kwargs = {
        "model": args.model_name_or_path,
        "tokenizer": tokenizer_source,
        "trust_remote_code": True,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "distributed_executor_backend": "mp",
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "dtype": "bfloat16",
        "disable_log_stats": args.vllm_disable_log_stats,
    }
    if args.vllm_max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.vllm_max_num_batched_tokens
    if args.speculative_config_json:
        engine_kwargs["speculative_config"] = json.loads(args.speculative_config_json)
    if args.spec_method:
        engine_kwargs["spec_method"] = args.spec_method
    if args.spec_model:
        engine_kwargs["spec_model"] = args.spec_model
    if args.spec_tokens is not None:
        engine_kwargs["spec_tokens"] = args.spec_tokens

    engine_args = AsyncEngineArgs(
        **engine_kwargs,
    )
    engine_load_started_at = time.monotonic()
    stat_loggers = [spec_decode_stat_logger_factory] if args.capture_spec_decode_stats else None
    engine = AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)
    engine_load_seconds = time.monotonic() - engine_load_started_at
    semaphore = asyncio.Semaphore(max(1, args.vllm_async_concurrency))
    log_each_example = should_log_each_example(len(rows))
    completed = 0

    async def generate_row(row_index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed
        async with semaphore:
            source_idx = row.get("source_idx", row_index)
            if log_each_example:
                print(
                    f"[latency] generating sample {row_index + 1}/{len(rows)} "
                    f"idx={source_idx} db_id={row.get('db_id', '')}"
                )
            generated = await generate_one_with_latency(
                engine=engine,
                sampling_params_cls=SamplingParams,
                tokenizer=tokenizer,
                row=row,
                args=args,
            )
            completed += 1
            if should_log_progress_tick(completed - 1, len(rows)):
                print(f"[latency] generated {completed}/{len(rows)} examples")
            return generated

    try:
        serving_started_at = time.monotonic()
        generated_rows = await asyncio.gather(*(generate_row(idx, row) for idx, row in enumerate(rows)))
        serving_generation_seconds = time.monotonic() - serving_started_at
    finally:
        shutdown_started_at = time.monotonic()
        engine.shutdown()
        shutdown_seconds = time.monotonic() - shutdown_started_at

    results_by_idx = {row["source_idx"]: row for row in generated_rows}
    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        source_idx = row.get("source_idx", row_index)
        generated = results_by_idx[source_idx]
        official_predictions[str(source_idx)] = f"{generated['pred_sql']}{BIRD_SPLIT_MARKER}{generated['db_id']}"
        detail = {
            "idx": source_idx,
            "db_id": generated["db_id"],
            "prediction_text": generated["prediction_text"],
            "pred_sql": generated["pred_sql"],
            "gold_sql": extract_sql(row.get("gold_sql", "")),
            "prompt_tokens": generated["prompt_tokens"],
            "completion_token_count": generated["completion_token_count"],
            "tool_rounds": generated.get("tool_rounds", 0),
            "tool_call_count": generated.get("tool_call_count", 0),
            "stop_reason": generated.get("stop_reason", ""),
            "error_message": generated.get("error_message", ""),
        }
        detailed_predictions.append(detail)
        latency = generated["latency"]
        latency_rows.append(
            {
                "idx": source_idx,
                "db_id": generated["db_id"],
                "prompt_tokens": generated["prompt_tokens"],
                "completion_token_count": generated["completion_token_count"],
                "total_tokens": generated["prompt_tokens"] + generated["completion_token_count"],
                "tool_rounds": generated.get("tool_rounds", 0),
                "tool_call_count": generated.get("tool_call_count", 0),
                "stop_reason": generated.get("stop_reason", ""),
                "first_ttft_seconds": latency.get("first_ttft_seconds"),
                "total_latency_seconds": latency.get("total_latency_seconds"),
                "llm_generation_seconds": latency.get("llm_generation_seconds"),
                "tool_execution_seconds": latency.get("tool_execution_seconds"),
                "request_count": latency.get("request_count"),
                "requests": latency.get("requests"),
            }
        )

    timing_metadata = {
        "engine_load_seconds": engine_load_seconds,
        "serving_generation_seconds": serving_generation_seconds,
        "engine_shutdown_seconds": shutdown_seconds,
        "spec_decode_stats": finalize_spec_decode_stats(spec_decode_stats)
        if args.capture_spec_decode_stats
        else None,
    }
    return rows, official_predictions, detailed_predictions, filtered_rows, latency_rows, timing_metadata


def build_latency_summary(
    args: argparse.Namespace,
    latency_rows: List[Dict[str, Any]],
    detailed_predictions: List[Dict[str, Any]],
    eval_summary: Dict[str, Any],
    generation_seconds: float,
    timing_metadata: Dict[str, float],
    evaluation_seconds: float,
    total_seconds: float,
) -> Dict[str, Any]:
    request_rows = [
        request
        for row in latency_rows
        for request in (row.get("requests") or [])
    ]
    request_decode_seconds = [
        max(0.0, float(request["generation_seconds"]) - float(request["ttft_seconds"]))
        for request in request_rows
        if request.get("generation_seconds") is not None and request.get("ttft_seconds") is not None
    ]
    example_decode_seconds = [
        sum(
            max(0.0, float(request["generation_seconds"]) - float(request["ttft_seconds"]))
            for request in (row.get("requests") or [])
            if request.get("generation_seconds") is not None and request.get("ttft_seconds") is not None
        )
        for row in latency_rows
    ]
    completion_tokens = [int(row.get("completion_token_count") or 0) for row in latency_rows]
    prompt_tokens = [int(row.get("prompt_tokens") or 0) for row in latency_rows]
    total_tokens = [int(row.get("total_tokens") or 0) for row in latency_rows]
    generated_token_total = sum(completion_tokens)
    total_token_count = sum(total_tokens)
    loaded_count = len(latency_rows)
    example_total_seconds = [
        float(row["total_latency_seconds"])
        for row in latency_rows
        if row.get("total_latency_seconds") is not None
    ]
    llm_generation_seconds = [
        float(row["llm_generation_seconds"])
        for row in latency_rows
        if row.get("llm_generation_seconds") is not None
    ]
    tool_execution_seconds = [
        float(row["tool_execution_seconds"])
        for row in latency_rows
        if row.get("tool_execution_seconds") is not None
    ]
    return {
        "run_label": args.run_label or Path(args.output_dir).name,
        "model_name_or_path": args.model_name_or_path,
        "input_file": args.input_file,
        "num_examples": args.num_examples,
        "loaded_examples": len(latency_rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_async_concurrency": args.vllm_async_concurrency,
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "vllm_disable_log_stats": args.vllm_disable_log_stats,
        "speculative_config_json": args.speculative_config_json,
        "spec_method": args.spec_method,
        "spec_model": args.spec_model,
        "spec_tokens": args.spec_tokens,
        "capture_spec_decode_stats": args.capture_spec_decode_stats,
        "spec_decode_stats": timing_metadata.get("spec_decode_stats"),
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "max_tool_rounds": args.max_tool_rounds,
        "accuracy": eval_summary["total"],
        "execution_stats": eval_summary["execution_stats"],
        "timing_seconds": {
            "generation": generation_seconds,
            "engine_load": timing_metadata.get("engine_load_seconds"),
            "serving_generation": timing_metadata.get("serving_generation_seconds"),
            "engine_shutdown": timing_metadata.get("engine_shutdown_seconds"),
            "evaluation": evaluation_seconds,
            "total": total_seconds,
        },
        "throughput": {
            "examples_per_second_generation": loaded_count / generation_seconds if generation_seconds else 0.0,
            "completion_tokens_per_second_generation": generated_token_total / generation_seconds if generation_seconds else 0.0,
            "total_tokens_per_second_generation": total_token_count / generation_seconds if generation_seconds else 0.0,
            "examples_per_second_serving": (
                loaded_count / timing_metadata["serving_generation_seconds"]
                if timing_metadata.get("serving_generation_seconds")
                else 0.0
            ),
            "completion_tokens_per_second_serving": (
                generated_token_total / timing_metadata["serving_generation_seconds"]
                if timing_metadata.get("serving_generation_seconds")
                else 0.0
            ),
            "total_tokens_per_second_serving": (
                total_token_count / timing_metadata["serving_generation_seconds"]
                if timing_metadata.get("serving_generation_seconds")
                else 0.0
            ),
        },
        "latency_seconds": {
            "ttft": percentile_block([float(row["first_ttft_seconds"]) for row in latency_rows if row.get("first_ttft_seconds") is not None]),
            "example_total": {
                **percentile_block(example_total_seconds),
                "average": sum(example_total_seconds) / len(example_total_seconds) if example_total_seconds else 0.0,
            },
            "llm_generation": {
                **percentile_block(llm_generation_seconds),
                "average": sum(llm_generation_seconds) / len(llm_generation_seconds) if llm_generation_seconds else 0.0,
            },
            "tool_execution": {
                **percentile_block(tool_execution_seconds),
                "average": sum(tool_execution_seconds) / len(tool_execution_seconds) if tool_execution_seconds else 0.0,
            },
            "request_ttft": percentile_block([float(row["ttft_seconds"]) for row in request_rows if row.get("ttft_seconds") is not None]),
            "request_generation": percentile_block([float(row["generation_seconds"]) for row in request_rows if row.get("generation_seconds") is not None]),
            "request_decode_after_ttft": {
                **percentile_block(request_decode_seconds),
                "average": sum(request_decode_seconds) / len(request_decode_seconds) if request_decode_seconds else 0.0,
            },
            "example_decode_after_ttft": {
                **percentile_block(example_decode_seconds),
                "average": sum(example_decode_seconds) / len(example_decode_seconds) if example_decode_seconds else 0.0,
            },
        },
        "tokens": {
            "prompt": percentile_block([float(value) for value in prompt_tokens]),
            "completion": percentile_block([float(value) for value in completion_tokens]),
            "total": percentile_block([float(value) for value in total_tokens]),
            "prompt_average": sum(prompt_tokens) / loaded_count if loaded_count else 0.0,
            "completion_average": generated_token_total / loaded_count if loaded_count else 0.0,
            "total_average": total_token_count / loaded_count if loaded_count else 0.0,
            "completion_total": generated_token_total,
            "total_token_count": total_token_count,
        },
        "tool_behavior": {
            "tool_calls_per_example": int_distribution(int(row.get("tool_call_count") or 0) for row in detailed_predictions),
            "tool_rounds_per_example": int_distribution(int(row.get("tool_rounds") or 0) for row in detailed_predictions),
            "stop_reason_counts": dict(Counter(str(row.get("stop_reason") or "unknown") for row in detailed_predictions)),
        },
    }


def write_latency_csv(path: Path, latency_rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "idx",
        "db_id",
        "prompt_tokens",
        "completion_token_count",
        "total_tokens",
        "tool_rounds",
        "tool_call_count",
        "stop_reason",
        "first_ttft_seconds",
        "total_latency_seconds",
        "llm_generation_seconds",
        "tool_execution_seconds",
        "request_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in latency_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    run_started_at = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    configure_tool_env(args.database_dir)
    rows = load_rows(args.input_file, args.num_examples)
    diff_rows = load_diff_rows(args.diff_json_path)
    print(f"[latency] loaded rows={len(rows)} diff_rows={len(diff_rows)} output_dir={output_dir}")

    generation_started_at = time.monotonic()
    rows, official_predictions, detailed_predictions, filtered_rows, latency_rows, timing_metadata = asyncio.run(
        generate_predictions_with_latency(rows, args)
    )
    generation_seconds = time.monotonic() - generation_started_at

    with (output_dir / "predict_dev.json").open("w", encoding="utf-8") as handle:
        json.dump(official_predictions, handle, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "prediction_details.jsonl", detailed_predictions)
    write_jsonl(output_dir / "filtered_examples.jsonl", filtered_rows)
    write_jsonl(output_dir / "latency_per_example.jsonl", latency_rows)
    write_latency_csv(output_dir / "latency_per_example.csv", latency_rows)

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
    latency_summary = build_latency_summary(
        args=args,
        latency_rows=latency_rows,
        detailed_predictions=detailed_predictions,
        eval_summary=summary,
        generation_seconds=generation_seconds,
        timing_metadata=timing_metadata,
        evaluation_seconds=evaluation_seconds,
        total_seconds=total_seconds,
    )
    summary["latency_summary"] = latency_summary

    write_jsonl(output_dir / "eval_results.jsonl", per_example_results)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "latency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(latency_summary, handle, ensure_ascii=False, indent=2)
    if latency_summary.get("spec_decode_stats") is not None:
        with (output_dir / "spec_decode_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(latency_summary["spec_decode_stats"], handle, ensure_ascii=False, indent=2)

    report_rows = build_per_example_report_rows(detailed_predictions, per_example_results)
    write_summary_markdown(summary, output_dir / "eval_summary.md", args, len(rows))
    write_run_report_markdown(summary, report_rows, output_dir / "run_report.md", args, len(rows))
    write_summary_csv(summary["by_difficulty"], output_dir / "eval_summary_by_difficulty.csv")
    write_summary_csv(summary["by_db"], output_dir / "eval_summary_by_db.csv")

    total = summary["total"]
    print(
        "[latency] complete "
        f"accuracy={total['accuracy']:.2f}% ({total['correct']}/{total['count']}) "
        f"generation_seconds={generation_seconds:.2f} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
