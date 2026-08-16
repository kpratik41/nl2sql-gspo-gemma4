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

from scripts.latency_study.run_tool_call_latency import (  # noqa: E402
    build_latency_summary,
    int_distribution,
    percentile_block,
    write_jsonl,
)
from scripts.run_inference_bird import (  # noqa: E402
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
    render_prompt,
    resolve_vllm_tokenizer_source,
    should_log_each_example,
    should_log_progress_tick,
    should_use_agentic_tool_loop,
    write_run_report_markdown,
    write_summary_csv,
    write_summary_markdown,
)
from nl2sql_gspo.inference_tool_executor import execute_tool_calls  # noqa: E402
from nl2sql_gspo.sql_utils import extract_sql  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async SGLang tool-call latency study runner.")
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
    parser.add_argument("--sglang_tp_size", type=int, required=True)
    parser.add_argument("--sglang_concurrency", type=int, default=32)
    parser.add_argument("--sglang_mem_fraction_static", type=float, default=0.90)
    parser.add_argument("--sglang_context_length", type=int, default=43500)
    parser.add_argument("--sglang_max_running_requests", type=int, default=None)
    parser.add_argument("--sglang_max_total_tokens", type=int, default=None)
    parser.add_argument("--disable_radix_cache", action="store_true")
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Keep field names compatible with the existing vLLM latency summarizer.
    args.inference_backend = "sglang_async_latency"
    args.vllm_tensor_parallel_size = args.sglang_tp_size
    args.vllm_async_concurrency = args.sglang_concurrency
    args.vllm_gpu_memory_utilization = args.sglang_mem_fraction_static
    args.vllm_max_model_len = args.sglang_context_length
    args.vllm_max_num_batched_tokens = None
    args.vllm_disable_log_stats = False
    args.speculative_config_json = None
    args.spec_method = None
    args.spec_model = None
    args.spec_tokens = None
    args.capture_spec_decode_stats = False
    args.vllm_data_parallel_size = 1
    args.skip_generation = False
    return args


def trim_overlap(previous: str, current: str) -> str:
    if not previous or not current:
        return current
    max_overlap = min(len(previous), len(current))
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == current[:size]:
            return current[size:]
    return current


def completion_token_count_from_response(response: Optional[Dict[str, Any]], tokenizer: Any, text: str) -> int:
    meta = response.get("meta_info") if isinstance(response, dict) else None
    if isinstance(meta, dict):
        for key in ("completion_tokens", "output_tokens", "completion_token_count"):
            value = meta.get(key)
            if isinstance(value, int):
                return value
    return len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])


async def generate_text_with_sglang_metrics(
    engine: Any,
    tokenizer: Any,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_prefix: str,
    round_index: int,
    prompt_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
) -> Tuple[str, int, Dict[str, Any]]:
    sampling_params: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
    }
    if stop_token_ids:
        sampling_params["stop_token_ids"] = set(int(token_id) for token_id in stop_token_ids)

    request_id = f"{request_prefix}-r{round_index}-{uuid.uuid4().hex}"
    started_at = time.perf_counter()
    first_token_seconds: Optional[float] = None
    final_response: Optional[Dict[str, Any]] = None
    generated_text = ""

    generator = await engine.async_generate(
        prompt=prompt_text,
        sampling_params=sampling_params,
        stream=True,
        rid=request_id,
    )
    async for chunk in generator:
        final_response = chunk
        chunk_text = str(chunk.get("text") or "")
        cleaned_chunk = trim_overlap(generated_text, chunk_text)
        if cleaned_chunk and first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started_at
        generated_text += cleaned_chunk

    total_seconds = time.perf_counter() - started_at
    completion_tokens = completion_token_count_from_response(final_response, tokenizer, generated_text)
    metric = {
        "request_id": request_id,
        "round_index": round_index,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_token_seconds,
        "generation_seconds": total_seconds,
    }
    return generated_text.strip(), completion_tokens, metric


async def generate_one_with_latency(
    engine: Any,
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

            available_context_tokens = args.sglang_context_length - current_prompt_tokens
            if available_context_tokens <= 0:
                stop_reason = "context_length_exceeded"
                break

            request_max_tokens = min(remaining_tokens, available_context_tokens)
            generated_text, round_tokens, request_metric = await generate_text_with_sglang_metrics(
                engine=engine,
                tokenizer=tokenizer,
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
    Dict[str, Any],
]:
    import sglang as sgl
    from nl2sql_gspo.model_utils import load_tokenizer

    tokenizer_source = resolve_vllm_tokenizer_source(args.model_name_or_path)
    tokenizer = load_tokenizer(tokenizer_source)
    rows, filtered_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)
    if any(not should_use_agentic_tool_loop(row, args.max_tool_rounds) for row in rows):
        print("[sglang-latency] warning: some rows do not have tools; they will still use the tool-loop renderer")

    print(
        "[sglang-latency] loading SGLang engine "
        f"tp={args.sglang_tp_size} concurrency={args.sglang_concurrency} "
        f"context_length={args.sglang_context_length} "
        f"radix_cache={'off' if args.disable_radix_cache else 'on'}"
    )
    engine_kwargs = {
        "model_path": args.model_name_or_path,
        "tokenizer_path": tokenizer_source,
        "trust_remote_code": True,
        "tp_size": args.sglang_tp_size,
        "context_length": args.sglang_context_length,
        "mem_fraction_static": args.sglang_mem_fraction_static,
        "disable_radix_cache": args.disable_radix_cache,
        "log_level": "info",
    }
    if args.sglang_max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.sglang_max_running_requests
    if args.sglang_max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = args.sglang_max_total_tokens

    engine_load_started_at = time.monotonic()
    engine = sgl.Engine(**engine_kwargs)
    engine_load_seconds = time.monotonic() - engine_load_started_at
    semaphore = asyncio.Semaphore(max(1, args.sglang_concurrency))
    log_each_example = should_log_each_example(len(rows))
    completed = 0

    async def generate_row(row_index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed
        async with semaphore:
            source_idx = row.get("source_idx", row_index)
            if log_each_example:
                print(
                    f"[sglang-latency] generating sample {row_index + 1}/{len(rows)} "
                    f"idx={source_idx} db_id={row.get('db_id', '')}"
                )
            generated = await generate_one_with_latency(
                engine=engine,
                tokenizer=tokenizer,
                row=row,
                args=args,
            )
            completed += 1
            if should_log_progress_tick(completed - 1, len(rows)):
                print(f"[sglang-latency] generated {completed}/{len(rows)} examples")
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
        "spec_decode_stats": None,
    }
    return rows, official_predictions, detailed_predictions, filtered_rows, latency_rows, timing_metadata


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
    print(f"[sglang-latency] loaded rows={len(rows)} diff_rows={len(diff_rows)} output_dir={output_dir}")

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
    latency_summary["backend"] = "sglang"
    latency_summary["sglang_tp_size"] = args.sglang_tp_size
    latency_summary["sglang_concurrency"] = args.sglang_concurrency
    latency_summary["sglang_context_length"] = args.sglang_context_length
    latency_summary["sglang_mem_fraction_static"] = args.sglang_mem_fraction_static
    latency_summary["sglang_max_running_requests"] = args.sglang_max_running_requests
    latency_summary["sglang_max_total_tokens"] = args.sglang_max_total_tokens
    latency_summary["radix_cache"] = "off" if args.disable_radix_cache else "on"
    summary["latency_summary"] = latency_summary

    write_jsonl(output_dir / "eval_results.jsonl", per_example_results)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "latency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(latency_summary, handle, ensure_ascii=False, indent=2)

    report_rows = build_per_example_report_rows(detailed_predictions, per_example_results)
    write_summary_markdown(summary, output_dir / "eval_summary.md", args, len(rows))
    write_run_report_markdown(summary, report_rows, output_dir / "run_report.md", args, len(rows))
    write_summary_csv(summary["by_difficulty"], output_dir / "eval_summary_by_difficulty.csv")
    write_summary_csv(summary["by_db"], output_dir / "eval_summary_by_db.csv")

    total = summary["total"]
    print(
        "[sglang-latency] complete "
        f"accuracy={total['accuracy']:.2f}% ({total['correct']}/{total['count']}) "
        f"generation_seconds={generation_seconds:.2f} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
