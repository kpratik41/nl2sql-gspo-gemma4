import argparse
import csv
import json
import multiprocessing as mp
import os
import sqlite3
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.sql_utils import extract_sql, get_database_path


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
    if args.inference_backend == "transformers":
        print(f"[run] transformers_device_map={args.transformers_device_map}")
        print(f"[run] transformers_data_parallel_size={args.transformers_data_parallel_size}")
    if args.inference_backend == "vllm":
        print(f"[run] vllm_tensor_parallel_size={args.vllm_tensor_parallel_size}")
        print(f"[run] vllm_data_parallel_size={args.vllm_data_parallel_size}")
        print(f"[run] vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
        print(f"[run] vllm_max_model_len={args.vllm_max_model_len}")


def load_diff_rows(diff_json_path: str) -> List[Dict[str, Any]]:
    with open(diff_json_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    return loaded if isinstance(loaded, list) else [loaded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_backend", type=str, choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="outputs/dev-20251106-schema.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/dev_20251106.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_inference")
    parser.add_argument("--max_prompt_length", type=int, default=30000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--eval_timeout", type=float, default=120.0)
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--transformers_device_map", type=str, choices=["none", "auto"], default="none")
    parser.add_argument("--transformers_data_parallel_size", type=int, default=0)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    parser.add_argument("--vllm_data_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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
        prompt_text = render_prompt(tokenizer, prompt_messages)
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


def plan_transformers_worker_devices(data_parallel_size: int) -> List[str]:
    visible_devices = get_visible_devices()
    worker_count = data_parallel_size if data_parallel_size > 0 else len(visible_devices)

    if len(visible_devices) < worker_count:
        raise ValueError(
            "Not enough visible GPUs for the requested transformers multiprocessing setup: "
            f"need {worker_count} workers, but CUDA_VISIBLE_DEVICES exposes "
            f"{len(visible_devices)} ({','.join(visible_devices)})."
        )

    return visible_devices[:worker_count]


def prepare_rows_for_generation(rows: List[Dict[str, Any]], tokenizer, max_prompt_length: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows, skipped_rows = filter_rows_by_prompt_length(rows, tokenizer, max_prompt_length)
    print(f"[inference] running generation for {len(rows)} prompts")

    return rows, skipped_rows


def _transformers_generate_worker(
    queue,
    shard_id: int,
    device: str,
    rows: List[Dict[str, Any]],
    worker_config: Dict[str, Any],
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        import torch

        from nl2sql_gspo.model_utils import load_inference_model_and_tokenizer

        model, tokenizer = load_inference_model_and_tokenizer(
            worker_config["model_name_or_path"],
            device_map=None,
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            generation_device = torch.device("cuda:0")
        else:
            generation_device = torch.device("cpu")

        model.to(generation_device)
        model.eval()
        do_sample = worker_config["temperature"] > 0.0

        results = []
        for row in rows:
            prompt_text = row["prompt_text"]
            tokenized = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=False,
            )
            tokenized = {key: value.to(generation_device) for key, value in tokenized.items()}
            prompt_token_count = int(row["prompt_tokens"])

            with torch.inference_mode():
                output_ids = model.generate(
                    **tokenized,
                    max_new_tokens=worker_config["max_new_tokens"],
                    do_sample=do_sample,
                    temperature=worker_config["temperature"] if do_sample else None,
                    top_p=worker_config["top_p"] if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][prompt_token_count:]
            prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(
                {
                    "source_idx": row.get("source_idx", -1),
                    "db_id": row.get("db_id", ""),
                    "prompt_tokens": prompt_token_count,
                    "prediction_text": prediction_text,
                    "pred_sql": extract_sql(prediction_text),
                    "completion_token_count": int(generated_ids.shape[0]),
                }
            )

        queue.put({"status": "ok", "shard_id": shard_id, "results": results})
    except Exception:
        queue.put({"status": "error", "shard_id": shard_id, "error": traceback.format_exc()})


def generate_predictions_with_transformers_data_parallel(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    data_parallel_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    worker_devices = plan_transformers_worker_devices(data_parallel_size)
    row_shards = shard_rows_for_data_parallel(rows, len(worker_devices))
    active_shards = [
        (shard_id, device, shard_rows)
        for shard_id, (device, shard_rows) in enumerate(zip(worker_devices, row_shards))
        if shard_rows
    ]

    print(
        "[inference] loading transformers workers in multi-process data-parallel mode "
        f"workers={len(active_shards)} devices={worker_devices}"
    )

    worker_config = {
        "model_name_or_path": args.model_name_or_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    collect_error: Optional[BaseException] = None

    for shard_id, device, shard_rows in active_shards:
        print(
            f"[inference] starting transformers shard {shard_id + 1}/{len(active_shards)} "
            f"gpu={device} prompts={len(shard_rows)}"
        )
        process = ctx.Process(
            target=_transformers_generate_worker,
            args=(queue, shard_id, device, shard_rows, worker_config),
        )
        process.start()
        processes.append(process)

    collected_results: Dict[int, Dict[str, Any]] = {}
    try:
        for _ in processes:
            message = queue.get()
            if message.get("status") != "ok":
                collect_error = RuntimeError(
                    "Transformers data-parallel worker failed"
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
        if collect_error is None and process.exitcode in (0, None):
            continue
        if process.exitcode not in (0, None):
            raise RuntimeError(f"Transformers data-parallel worker exited with code {process.exitcode}")

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    log_each_example = should_log_each_example(len(rows))

    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        generated = collected_results.get(source_idx)
        if generated is None:
            raise RuntimeError(f"Missing transformers generation result for idx={source_idx}")

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
            tokenizer=llm_config["model_name_or_path"],
            trust_remote_code=True,
            tensor_parallel_size=llm_config["tensor_parallel_size"],
            distributed_executor_backend="mp",
            gpu_memory_utilization=llm_config["gpu_memory_utilization"],
            max_model_len=llm_config["max_model_len"],
            dtype="bfloat16",
        )
        sampling_params = SamplingParams(
            temperature=llm_config["temperature"],
            top_p=llm_config["top_p"],
            max_tokens=llm_config["max_new_tokens"],
        )
        prompt_texts = [row["prompt_text"] for row in rows]
        outputs = llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)

        results = []
        for row, request_output in zip(rows, outputs):
            first_output = request_output.outputs[0] if request_output.outputs else None
            prediction_text = (first_output.text or "").strip() if first_output else ""
            completion_token_count = len(first_output.token_ids) if first_output else 0
            results.append(
                {
                    "source_idx": row.get("source_idx", -1),
                    "db_id": row.get("db_id", ""),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prediction_text": prediction_text,
                    "pred_sql": extract_sql(prediction_text),
                    "completion_token_count": completion_token_count,
                }
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
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": vllm_max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
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


def generate_predictions_with_transformers(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import torch

    from nl2sql_gspo.model_utils import load_inference_model_and_tokenizer, load_tokenizer

    tokenizer = load_tokenizer(args.model_name_or_path)
    rows, skipped_rows = prepare_rows_for_generation(rows, tokenizer, args.max_prompt_length)

    if args.transformers_device_map == "auto":
        model, _ = load_inference_model_and_tokenizer(
            args.model_name_or_path,
            device_map="auto",
        )
        generation_device = next(model.parameters()).device
        print(f"[inference] model loaded on device={generation_device} using device_map='auto'")
    else:
        data_parallel_size = args.transformers_data_parallel_size
        if data_parallel_size != 1 and len(get_visible_devices()) > 1:
            return generate_predictions_with_transformers_data_parallel(rows, args, data_parallel_size)

        model, _ = load_inference_model_and_tokenizer(
            args.model_name_or_path,
            device_map=None,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            generation_device = torch.device("cuda:0")
        else:
            generation_device = torch.device("cpu")
        model.to(generation_device)
        print(f"[inference] model loaded on device={generation_device} without device_map")

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []

    do_sample = args.temperature > 0.0
    log_each_example = should_log_each_example(len(rows))

    for idx, row in enumerate(rows):
        prompt_text = row["prompt_text"]
        tokenized = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=False,
        )
        tokenized = {key: value.to(generation_device) for key, value in tokenized.items()}
        prompt_token_count = int(row["prompt_tokens"])
        source_idx = row.get("source_idx", idx)
        db_id = row.get("db_id", "")

        if log_each_example:
            print(
                f"[inference] generating sample {idx + 1}/{len(rows)} "
                f"idx={source_idx} db_id={db_id} prompt_tokens={prompt_token_count}"
            )

        with torch.inference_mode():
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_token_count:]
        prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        pred_sql = extract_sql(prediction_text)
        completion_token_count = int(generated_ids.shape[0])

        official_predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{db_id}"
        detailed_predictions.append(
            {
                "idx": source_idx,
                "db_id": db_id,
                "prediction_text": prediction_text,
                "pred_sql": pred_sql,
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "prompt_tokens": prompt_token_count,
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
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    prompt_texts = [row["prompt_text"] for row in rows]
    outputs = llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []
    log_each_example = should_log_each_example(len(rows))

    for idx, (row, request_output) in enumerate(zip(rows, outputs)):
        source_idx = row.get("source_idx", idx)
        db_id = row.get("db_id", "")
        prompt_token_count = int(row["prompt_tokens"])

        if log_each_example:
            print(
                f"[inference] generating sample {idx + 1}/{len(rows)} "
                f"idx={source_idx} db_id={db_id} prompt_tokens={prompt_token_count}"
            )

        first_output = request_output.outputs[0] if request_output.outputs else None
        prediction_text = (first_output.text or "").strip() if first_output else ""
        pred_sql = extract_sql(prediction_text)
        completion_token_count = len(first_output.token_ids) if first_output else 0

        official_predictions[str(source_idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{db_id}"
        detailed_predictions.append(
            {
                "idx": source_idx,
                "db_id": db_id,
                "prediction_text": prediction_text,
                "pred_sql": pred_sql,
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "prompt_tokens": prompt_token_count,
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


def generate_predictions(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --skip_generation is set")

    if args.inference_backend == "transformers":
        return generate_predictions_with_transformers(rows, args)

    if args.inference_backend == "vllm":
        return generate_predictions_with_vllm(rows, args)

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


def write_summary_markdown(summary: Dict[str, Any], markdown_path: Path) -> None:
    execution_stats = summary["execution_stats"]
    content = [
        "# BIRD Dev Execution Accuracy Summary",
        "",
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
        ordered_results = executor.map(
            lambda example: evaluate_one(
                example["predicted_sql"],
                example["gold_sql"],
                example["db_path"],
                timeout_s,
            ),
            prepared_examples,
        )

        for example, eval_result in zip(prepared_examples, ordered_results):
            idx = example["idx"]
            source_idx = example["source_idx"]
            db_id = example["db_id"]
            difficulty = example["difficulty"]

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
                "gold_executed": bool(eval_result["gold_executed"]),
                "pred_error": eval_result["pred_error"],
                "gold_error": eval_result["gold_error"],
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
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    print_run_configuration(args, output_dir)

    rows = load_rows(args.input_file, args.num_examples)
    print(f"[run] loaded {len(rows)} input rows")

    predictions_path = output_dir / "predict_dev.json"
    details_path = output_dir / "prediction_details.jsonl"
    filtered_path = output_dir / "filtered_examples.jsonl"
    per_example_eval_path = output_dir / "eval_results.jsonl"
    summary_path = output_dir / "eval_summary.json"
    summary_markdown_path = output_dir / "eval_summary.md"
    difficulty_csv_path = output_dir / "eval_summary_by_difficulty.csv"
    db_csv_path = output_dir / "eval_summary_by_db.csv"
    diff_rows = load_diff_rows(args.diff_json_path)
    print(f"[run] loaded {len(diff_rows)} diff rows")

    if args.skip_generation:
        official_predictions = load_predictions(predictions_path)
        filtered_rows: List[Dict[str, Any]] = []
    else:
        rows, official_predictions, detailed_predictions, filtered_rows = generate_predictions(rows, args)
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(official_predictions, handle, ensure_ascii=False, indent=2)

        with details_path.open("w", encoding="utf-8") as handle:
            for record in detailed_predictions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        with filtered_path.open("w", encoding="utf-8") as handle:
            for record in filtered_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    per_example_results, summary = evaluate_predictions(
        rows=rows,
        predictions=official_predictions,
        database_dir=args.database_dir,
        diff_rows=diff_rows,
        timeout_s=args.eval_timeout,
        eval_workers=args.eval_workers,
    )

    with per_example_eval_path.open("w", encoding="utf-8") as handle:
        for record in per_example_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_summary_markdown(summary, summary_markdown_path)
    write_summary_csv(summary["by_difficulty"], difficulty_csv_path)
    write_summary_csv(summary["by_db"], db_csv_path)
    print_summary_tables(summary)
    print(f"Saved official BIRD predictions to {predictions_path}")
    print(f"Saved filtered-example report to {filtered_path}")
    print(f"Saved per-example evaluation to {per_example_eval_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved markdown summary to {summary_markdown_path}")
    print(f"Saved difficulty CSV summary to {difficulty_csv_path}")
    print(f"Saved DB CSV summary to {db_csv_path}")


if __name__ == "__main__":
    main()