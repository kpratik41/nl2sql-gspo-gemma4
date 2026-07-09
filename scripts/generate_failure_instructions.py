import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.model_utils import load_tokenizer
from nl2sql_gspo.schema_utils import extract_columns_from_sql, extract_tables_from_sql
from nl2sql_gspo.sql_utils import bird_execute_sql, bird_get_gold_rows, bird_result_match, extract_sql


HARD_TIMEOUT_BUFFER_S = 15.0
QUESTION_TAG_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.IGNORECASE | re.DOTALL)
QUESTION_LINE_RE = re.compile(r"(?:^|\n)Question:\s*(.*?)(?:\n[A-Z][^\n]*:|$)", re.DOTALL)
AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)
LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)
SET_OP_RE = re.compile(r"\b(union|intersect|except)\b", re.IGNORECASE)
NULL_LOGIC_RE = re.compile(r"\bis\s+null\b|\bis\s+not\s+null\b|\bcoalesce\s*\(|\bifnull\s*\(", re.IGNORECASE)
LITERAL_RE = re.compile(r"'(?:''|[^'])*'|\b\d+(?:\.\d+)?\b")


PATTERN_LIBRARY: Dict[str, Dict[str, str]] = {
    "empty_or_unparsed_sql": {
        "title": "Always emit one complete executable SQL query",
        "rule": "Return exactly one complete read-only SQLite query in the final answer and avoid empty, partial, or prose-only answers.",
        "why": "The generated answer did not contain extractable SQL.",
    },
    "execution_failure": {
        "title": "Prefer valid SQLite syntax over risky variants",
        "rule": "Favor simple valid SQLite syntax and verify clause order, aliases, parentheses, and function names before finalizing the query.",
        "why": "The predicted SQL failed to execute at all.",
    },
    "missing_required_tables": {
        "title": "Include every required table in the join path",
        "rule": "Bring in every table needed to connect the asked entities, especially bridge tables and lookup tables required by the schema.",
        "why": "Gold referenced tables that the prediction omitted.",
    },
    "extra_unneeded_tables": {
        "title": "Do not introduce unrelated joins",
        "rule": "Avoid adding extra tables or joins that are not needed for the requested result, because they often change row counts or filter behavior.",
        "why": "The prediction introduced tables not present in the gold SQL.",
    },
    "missing_required_columns": {
        "title": "Use all columns needed by the question",
        "rule": "Make sure every key select, filter, grouping, ordering, and join column required by the question appears in the query.",
        "why": "Gold referenced columns that the prediction omitted.",
    },
    "extra_unneeded_columns": {
        "title": "Avoid unnecessary projected or filter columns",
        "rule": "Do not add extra columns to SELECT, WHERE, GROUP BY, or ORDER BY unless the question explicitly requires them.",
        "why": "The prediction used columns that were not part of the gold SQL.",
    },
    "aggregation_mismatch": {
        "title": "Match aggregation intent exactly",
        "rule": "Use the exact aggregation requested by the question: distinguish count, sum, avg, min, and max instead of approximating with a different aggregate.",
        "why": "Aggregate-function usage differed between prediction and gold SQL.",
    },
    "distinct_mismatch": {
        "title": "Use DISTINCT only when uniqueness is requested",
        "rule": "Add DISTINCT only when the question asks for unique values, and avoid DISTINCT when duplicates are meaningful.",
        "why": "DISTINCT usage differed between prediction and gold SQL.",
    },
    "group_by_mismatch": {
        "title": "Group by the correct entity keys",
        "rule": "When aggregating by entity, group on the exact key columns implied by the question and avoid grouping on extra attributes.",
        "why": "GROUP BY columns differed between prediction and gold SQL.",
    },
    "order_by_mismatch": {
        "title": "Respect ranking and sort direction",
        "rule": "Match the requested ordering exactly, including which column to sort by and whether the direction should be ascending or descending.",
        "why": "ORDER BY specification differed between prediction and gold SQL.",
    },
    "limit_mismatch": {
        "title": "Apply LIMIT only when the question asks for top-k",
        "rule": "Use LIMIT only when the question requests a bounded number of rows such as top, highest, first, or latest.",
        "why": "LIMIT usage differed between prediction and gold SQL.",
    },
    "join_path_mismatch": {
        "title": "Choose the correct join path and join count",
        "rule": "Follow the schema’s intended foreign-key path and avoid skipping bridge tables or adding redundant joins that distort the result set.",
        "why": "The number or structure of joins differed between prediction and gold SQL.",
    },
    "filter_literal_mismatch": {
        "title": "Copy filter literals and constants exactly",
        "rule": "Carry over names, years, statuses, thresholds, and other filter literals exactly from the question or hint, and bind them to the correct columns.",
        "why": "Literal values used by the filters differed between prediction and gold SQL.",
    },
    "null_handling_mismatch": {
        "title": "Handle NULLs only when the question requires it",
        "rule": "Use IS NULL, IS NOT NULL, COALESCE, or IFNULL only when null semantics are explicitly relevant to the requested answer.",
        "why": "NULL-handling logic differed between prediction and gold SQL.",
    },
    "set_operation_mismatch": {
        "title": "Use UNION or EXCEPT only for explicit set-combination tasks",
        "rule": "Reserve UNION, INTERSECT, and EXCEPT for questions that truly ask to combine, intersect, or subtract sets.",
        "why": "Set-operation usage differed between prediction and gold SQL.",
    },
    "over_restrictive_filtering": {
        "title": "Avoid over-restricting the result set",
        "rule": "Be careful not to add extra predicates that wipe out valid rows when the gold answer is non-empty.",
        "why": "The prediction executed but returned an empty set while gold returned rows.",
    },
    "under_restrictive_filtering": {
        "title": "Do not omit required constraints",
        "rule": "Keep every condition needed to narrow the result to the requested subset; missing filters often produce extra rows that look plausible but are wrong.",
        "why": "The prediction executed and returned rows when gold’s set was empty.",
    },
    "other_result_set_mismatch": {
        "title": "Match the exact result set, not just the rough intent",
        "rule": "Before finalizing, check that the query structure would produce exactly the requested row set rather than a nearby but broader or narrower answer.",
        "why": "The prediction executed but still produced a different result set without a more specific heuristic match.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="outputs/gemma4_31b_gspo_bird/checkpoint-70")
    parser.add_argument("--input_file", type=str, default="outputs/train-6601-schema-filtered.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/train_databases")
    parser.add_argument("--output_dir", type=str, default="outputs/train_failure_instructions_ckpt70")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=17)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--eval_timeout", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=32)
    parser.add_argument("--num_rules", type=int, default=12)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    parser.add_argument("--vllm_data_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--vllm_max_model_len", type=int, default=24576)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--prediction_samples_file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool, skip_generation: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if skip_generation:
            output_dir.mkdir(parents=True, exist_ok=True)
            return
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already contains files. Use --overwrite to reuse it."
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def print_configuration(args: argparse.Namespace, output_dir: Path) -> None:
    print("[run] starting train-failure instruction mining")
    print(f"[run] model_name_or_path={args.model_name_or_path}")
    print(f"[run] input_file={args.input_file}")
    print(f"[run] database_dir={args.database_dir}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] sample_size={args.sample_size}")
    print(f"[run] sample_seed={args.sample_seed}")
    print(f"[run] num_generations={args.num_generations}")
    print(f"[run] temperature={args.temperature}")
    print(f"[run] top_p={args.top_p}")
    print(f"[run] max_prompt_length={args.max_prompt_length}")
    print(f"[run] max_new_tokens={args.max_new_tokens}")
    print(f"[run] eval_timeout={args.eval_timeout}")
    print(f"[run] eval_workers={args.eval_workers}")
    print(f"[run] num_rules={args.num_rules}")
    print(f"[run] vllm_tensor_parallel_size={args.vllm_tensor_parallel_size}")
    print(f"[run] vllm_data_parallel_size={args.vllm_data_parallel_size}")
    print(f"[run] vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
    print(f"[run] vllm_max_model_len={args.vllm_max_model_len}")
    print(f"[run] skip_generation={args.skip_generation}")
    print(f"[run] prediction_samples_file={args.prediction_samples_file or '<default>'}")
    print(f"[run] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")


def load_rows(input_file: str) -> List[Dict[str, Any]]:
    input_path = Path(input_file)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            raw_rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with input_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        raw_rows = loaded if isinstance(loaded, list) else [loaded]

    rows = [normalize_record(row) for row in raw_rows]
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


def sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def preview_text(text: str, max_chars: int = 160) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    lines = []
    for message in messages:
        lines.append(f"{message.get('role', 'user').upper()}: {message.get('content', '')}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def get_generation_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt_messages = row.get("prompt") or []
    if prompt_messages:
        return prompt_messages
    return [message for message in (row.get("messages") or []) if message.get("role") != "assistant"]


def extract_question_text(row: Dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"]).strip()
    user_text = ""
    for message in get_generation_messages(row):
        if message.get("role") == "user":
            user_text = str(message.get("content", ""))
            break
    tagged = QUESTION_TAG_RE.search(user_text)
    if tagged:
        return " ".join(tagged.group(1).split())
    lined = QUESTION_LINE_RE.search(user_text)
    if lined:
        return " ".join(lined.group(1).split())
    return preview_text(user_text, max_chars=220)


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


def filter_rows_by_prompt_length(
    rows: List[Dict[str, Any]], tokenizer, max_prompt_length: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    for row in rows:
        prompt_messages = get_generation_messages(row)
        prompt_text = render_prompt(tokenizer, prompt_messages)
        prompt_token_count = len(tokenizer(prompt_text, truncation=False)["input_ids"])
        prepared_row = dict(row)
        prepared_row["question_text"] = extract_question_text(row)
        prepared_row["prompt_text"] = prompt_text
        prepared_row["prompt_tokens"] = prompt_token_count
        if prompt_token_count > max_prompt_length:
            skipped_rows.append(
                {
                    "idx": row.get("source_idx", -1),
                    "db_id": row.get("db_id", ""),
                    "question": prepared_row["question_text"],
                    "prompt_tokens": prompt_token_count,
                    "max_prompt_length": max_prompt_length,
                }
            )
            continue
        kept_rows.append(prepared_row)
    print(
        f"[filter] kept {len(kept_rows)} prompts under max_prompt_length={max_prompt_length}; "
        f"skipped {len(skipped_rows)}"
    )
    return kept_rows, skipped_rows


def get_visible_devices() -> List[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return [str(idx) for idx in range(8)]
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
        visible_devices[offset : offset + tensor_parallel_size]
        for offset in range(0, required_devices, tensor_parallel_size)
    ]


def shard_rows_for_data_parallel(rows: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for row_index, row in enumerate(rows):
        shards[row_index % num_shards].append(row)
    return shards


def _vllm_generate_worker(
    queue,
    shard_id: int,
    device_group: List[str],
    rows: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    shard_file_path: str,
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
        sampling_params = SamplingParams(
            n=llm_config["num_generations"],
            temperature=llm_config["temperature"],
            top_p=llm_config["top_p"],
            max_tokens=llm_config["max_new_tokens"],
        )
        prompt_texts = [row["prompt_text"] for row in rows]
        outputs = llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)

        results: List[Dict[str, Any]] = []
        for row, request_output in zip(rows, outputs):
            generations: List[Dict[str, Any]] = []
            for sample_idx, output in enumerate(request_output.outputs[: llm_config["num_generations"]]):
                prediction_text = (output.text or "").strip()
                generations.append(
                    {
                        "sample_idx": sample_idx,
                        "prediction_text": prediction_text,
                        "pred_sql": extract_sql(prediction_text),
                        "completion_token_count": len(output.token_ids),
                    }
                )
            results.append(
                {
                    "idx": row.get("source_idx", -1),
                    "db_id": row.get("db_id", ""),
                    "question": row.get("question_text", ""),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "gold_sql": extract_sql(row.get("gold_sql", "")),
                    "generations": generations,
                }
            )
        shard_path = Path(shard_file_path)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = shard_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(shard_path)
        queue.put(
            {
                "status": "ok",
                "shard_id": shard_id,
                "output_path": str(shard_path),
                "num_rows": len(results),
            }
        )
    except Exception:
        error_text = traceback.format_exc()
        try:
            error_path = Path(shard_file_path).with_suffix(".error.txt")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(error_text, encoding="utf-8")
        except Exception:
            pass
        queue.put(
            {
                "status": "error",
                "shard_id": shard_id,
                "error": error_text,
            }
        )


def generate_predictions_with_vllm_n(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tokenizer = load_tokenizer(args.model_name_or_path)
    rows, skipped_rows = filter_rows_by_prompt_length(rows, tokenizer, args.max_prompt_length)
    if not rows:
        return [], skipped_rows

    tensor_parallel_size = args.vllm_tensor_parallel_size
    data_parallel_size = args.vllm_data_parallel_size
    vllm_max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)
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
        "num_generations": args.num_generations,
    }

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    expected_shard_paths: Dict[int, Path] = {}
    for shard_id, device_group, shard_rows in active_shards:
        print(
            f"[inference] starting vLLM shard {shard_id + 1}/{len(active_shards)} "
            f"gpus={','.join(device_group)} prompts={len(shard_rows)}"
        )
        shard_path = output_dir / "generation_shards" / f"shard_{shard_id:02d}.jsonl"
        expected_shard_paths[shard_id] = shard_path
        process = ctx.Process(
            target=_vllm_generate_worker,
            args=(result_queue, shard_id, device_group, shard_rows, llm_config, str(shard_path)),
        )
        process.start()
        processes.append(process)

    shard_files: Dict[int, Path] = {}
    completed_shards: Set[int] = set()
    try:
        while len(completed_shards) < len(active_shards):
            try:
                message = result_queue.get(timeout=30)
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue

            shard_id = int(message.get("shard_id", -1))
            if message.get("status") != "ok":
                raise RuntimeError(
                    "vLLM failure-pattern worker failed"
                    + (f" (shard {shard_id})" if shard_id >= 0 else "")
                    + ":\n"
                    + message.get("error", "unknown error")
                )
            completed_shards.add(shard_id)
            shard_files[shard_id] = Path(message["output_path"])
            print(
                f"[inference] shard {shard_id + 1}/{len(active_shards)} finished "
                f"rows={message.get('num_rows', '?')}"
            )
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    missing_shards = []
    for shard_id, shard_path in expected_shard_paths.items():
        if shard_path.exists():
            shard_files[shard_id] = shard_path
        else:
            missing_shards.append(shard_id)
    if missing_shards:
        error_chunks = []
        for shard_id in missing_shards:
            error_path = expected_shard_paths[shard_id].with_suffix(".error.txt")
            if error_path.exists():
                error_chunks.append(f"shard {shard_id}:\n{error_path.read_text(encoding='utf-8')}")
            else:
                error_chunks.append(f"shard {shard_id}: no shard output file written")
        raise RuntimeError("Generation shards incomplete:\n" + "\n".join(error_chunks))

    collected_results: Dict[int, Dict[str, Any]] = {}
    for shard_id in sorted(shard_files):
        with shard_files[shard_id].open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = json.loads(line)
                collected_results[result["idx"]] = result

    prediction_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_idx = row.get("source_idx", idx)
        generated = collected_results.get(source_idx)
        if generated is None:
            raise RuntimeError(f"Missing vLLM generation result for idx={source_idx}")
        prediction_rows.append(generated)
        completed = idx + 1
        if completed == 1 or completed == len(rows) or completed % 50 == 0:
            print(f"[inference] generated {completed}/{len(rows)} prompts")

    return prediction_rows, skipped_rows


def _rows_to_hashable_set(rows: Optional[List[Tuple[Any, ...]]]) -> frozenset:
    if not rows:
        return frozenset()
    hashable: List[Tuple[Any, ...]] = []
    for row in rows:
        try:
            hashable.append(tuple(row))
        except Exception:
            hashable.append((repr(row),))
    try:
        return frozenset(hashable)
    except TypeError:
        return frozenset(tuple(repr(cell) for cell in row) for row in hashable)


def _run_with_hard_timeout(fn, *args, soft_timeout_s: float, **kwargs):
    hard_timeout_s = float(soft_timeout_s) + HARD_TIMEOUT_BUFFER_S
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=hard_timeout_s)
        except concurrent.futures.TimeoutError:
            return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def evaluate_generation(
    predicted_sql: str,
    gold_sql: str,
    db_id: str,
    database_dir: str,
    timeout_s: float,
) -> Dict[str, Any]:
    pred_result = _run_with_hard_timeout(
        bird_execute_sql,
        sql=predicted_sql,
        db_id=db_id,
        database_dir=database_dir,
        timeout_s=timeout_s,
        soft_timeout_s=timeout_s,
    )
    if pred_result is None:
        pred_executed, pred_rows, pred_error = (
            False,
            None,
            f"hard-timeout after {timeout_s + HARD_TIMEOUT_BUFFER_S:.1f}s",
        )
    else:
        pred_executed, pred_rows, pred_error = pred_result

    gold_result = _run_with_hard_timeout(
        bird_get_gold_rows,
        gold_sql=gold_sql,
        db_id=db_id,
        database_dir=database_dir,
        timeout_s=timeout_s,
        soft_timeout_s=timeout_s,
    )
    if gold_result is None:
        gold_executed, gold_row_set, gold_error = (
            False,
            None,
            f"hard-timeout after {timeout_s + HARD_TIMEOUT_BUFFER_S:.1f}s",
        )
    else:
        gold_executed, gold_row_set, gold_error = gold_result

    pred_row_set = _rows_to_hashable_set(pred_rows) if pred_executed else None
    matched = pred_executed and gold_executed and bird_result_match(pred_rows, gold_row_set)
    if pred_executed and gold_executed:
        status = "ok" if matched else "mismatch"
    else:
        parts = []
        if pred_error:
            parts.append(f"pred_error: {pred_error}")
        if gold_error:
            parts.append(f"gold_error: {gold_error}")
        status = "; ".join(parts) if parts else "error"

    return {
        "res": int(matched),
        "status": status,
        "pred_executed": bool(pred_executed),
        "gold_executed": bool(gold_executed),
        "pred_error": pred_error,
        "gold_error": gold_error,
        "pred_sql_extracted": bool(predicted_sql.strip()),
        "gold_sql_extracted": bool(gold_sql.strip()),
        "pred_row_count": len(pred_row_set) if pred_row_set is not None else None,
        "gold_row_count": len(gold_row_set) if gold_row_set is not None else None,
    }


def evaluate_generations(
    prediction_rows: List[Dict[str, Any]],
    database_dir: str,
    timeout_s: float,
    max_workers: int,
) -> List[Dict[str, Any]]:
    jobs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for row in prediction_rows:
        for generation in row["generations"]:
            jobs.append((row, generation))

    def _run_job(job: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        row, generation = job
        result = evaluate_generation(
            predicted_sql=generation["pred_sql"],
            gold_sql=row["gold_sql"],
            db_id=row["db_id"],
            database_dir=database_dir,
            timeout_s=timeout_s,
        )
        return {
            "idx": row["idx"],
            "db_id": row["db_id"],
            "question": row.get("question", ""),
            "prompt_tokens": row.get("prompt_tokens", 0),
            "sample_idx": generation["sample_idx"],
            "prediction_text": generation["prediction_text"],
            "pred_sql": generation["pred_sql"],
            "gold_sql": row["gold_sql"],
            **result,
        }

    results: List[Dict[str, Any]] = []
    worker_count = max(1, min(max_workers, 64))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        total_jobs = len(jobs)
        futures = [executor.submit(_run_job, job) for job in jobs]
        for job_index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if job_index == 1 or job_index == total_jobs or job_index % 200 == 0:
                print(f"[evaluation] scored {job_index}/{total_jobs} candidate generations")
    return results


def normalize_identifier_list(raw: str) -> Tuple[str, ...]:
    parts = []
    for chunk in raw.split(","):
        token = chunk.strip().lower()
        token = re.sub(r"\basc\b|\bdesc\b", "", token).strip()
        token = token.replace("`", "")
        if "." in token:
            token = token.split(".")[-1]
        if token:
            parts.append(token)
    return tuple(parts)


def extract_clause_signature(sql: str, clause_name: str) -> Tuple[str, ...]:
    sql_lower = (sql or "").lower()
    clause_map = {
        "group by": r"group\s+by\s+(.*?)(?=\border\s+by\b|\bhaving\b|\blimit\b|\bunion\b|\bexcept\b|\bintersect\b|$)",
        "order by": r"order\s+by\s+(.*?)(?=\blimit\b|\bunion\b|\bexcept\b|\bintersect\b|$)",
    }
    match = re.search(clause_map[clause_name], sql_lower, re.DOTALL)
    if not match:
        return tuple()
    return normalize_identifier_list(match.group(1))


def extract_aggregate_signature(sql: str) -> Tuple[str, ...]:
    return tuple(sorted(func.lower() for func in AGG_RE.findall(sql or "")))


def extract_literal_signature(sql: str) -> Tuple[str, ...]:
    literals = []
    for token in LITERAL_RE.findall(sql or ""):
        literals.append(token.strip().lower())
    return tuple(sorted(literals))


def extract_set_op_signature(sql: str) -> Tuple[str, ...]:
    return tuple(sorted(op.lower() for op in SET_OP_RE.findall(sql or "")))


def count_joins(sql: str) -> int:
    return len(re.findall(r"\bjoin\b", sql or "", re.IGNORECASE))


def has_distinct(sql: str) -> bool:
    return bool(re.search(r"\bselect\s+distinct\b", sql or "", re.IGNORECASE))


def extract_limit_signature(sql: str) -> Optional[str]:
    match = LIMIT_RE.search(sql or "")
    return match.group(1) if match else None


def uses_null_logic(sql: str) -> bool:
    return bool(NULL_LOGIC_RE.search(sql or ""))


def classify_error_patterns(result: Dict[str, Any]) -> List[str]:
    patterns: Set[str] = set()
    pred_sql = result.get("pred_sql", "") or ""
    gold_sql = result.get("gold_sql", "") or ""

    if not result.get("pred_sql_extracted", False):
        patterns.add("empty_or_unparsed_sql")
    if not result.get("pred_executed", False):
        patterns.add("execution_failure")

    pred_tables = extract_tables_from_sql(pred_sql)
    gold_tables = extract_tables_from_sql(gold_sql)
    if gold_tables - pred_tables:
        patterns.add("missing_required_tables")
    if pred_tables - gold_tables:
        patterns.add("extra_unneeded_tables")

    pred_columns = extract_columns_from_sql(pred_sql)
    gold_columns = extract_columns_from_sql(gold_sql)
    if gold_columns - pred_columns:
        patterns.add("missing_required_columns")
    if pred_columns - gold_columns:
        patterns.add("extra_unneeded_columns")

    if extract_aggregate_signature(pred_sql) != extract_aggregate_signature(gold_sql):
        patterns.add("aggregation_mismatch")
    if has_distinct(pred_sql) != has_distinct(gold_sql):
        patterns.add("distinct_mismatch")
    if extract_clause_signature(pred_sql, "group by") != extract_clause_signature(gold_sql, "group by"):
        patterns.add("group_by_mismatch")
    if extract_clause_signature(pred_sql, "order by") != extract_clause_signature(gold_sql, "order by"):
        patterns.add("order_by_mismatch")
    if extract_limit_signature(pred_sql) != extract_limit_signature(gold_sql):
        patterns.add("limit_mismatch")
    if count_joins(pred_sql) != count_joins(gold_sql):
        patterns.add("join_path_mismatch")
    if extract_literal_signature(pred_sql) != extract_literal_signature(gold_sql):
        patterns.add("filter_literal_mismatch")
    if uses_null_logic(pred_sql) != uses_null_logic(gold_sql):
        patterns.add("null_handling_mismatch")
    if extract_set_op_signature(pred_sql) != extract_set_op_signature(gold_sql):
        patterns.add("set_operation_mismatch")

    pred_row_count = result.get("pred_row_count")
    gold_row_count = result.get("gold_row_count")
    if result.get("pred_executed") and result.get("gold_executed") and not result.get("res"):
        if pred_row_count == 0 and (gold_row_count or 0) > 0:
            patterns.add("over_restrictive_filtering")
        if (pred_row_count or 0) > 0 and gold_row_count == 0:
            patterns.add("under_restrictive_filtering")

    if not patterns:
        patterns.add("other_result_set_mismatch")
    return sorted(patterns)


def build_heterogeneous_sets(results: List[Dict[str, Any]], num_generations: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_idx[result["idx"]].append(result)

    heterogeneous_prompts: List[Dict[str, Any]] = []
    heterogeneous_wrong_generations: List[Dict[str, Any]] = []

    for idx, prompt_results in by_idx.items():
        prompt_results.sort(key=lambda item: item["sample_idx"])
        num_correct = sum(int(item["res"]) for item in prompt_results)
        if 0 < num_correct < num_generations:
            first = prompt_results[0]
            heterogeneous_prompts.append(
                {
                    "idx": idx,
                    "db_id": first["db_id"],
                    "question": first.get("question", ""),
                    "gold_sql": first["gold_sql"],
                    "prompt_tokens": first.get("prompt_tokens", 0),
                    "num_generations": len(prompt_results),
                    "num_correct": num_correct,
                    "num_wrong": len(prompt_results) - num_correct,
                }
            )
            for item in prompt_results:
                if not item["res"]:
                    enriched = dict(item)
                    enriched["patterns"] = classify_error_patterns(item)
                    heterogeneous_wrong_generations.append(enriched)

    heterogeneous_prompts.sort(key=lambda item: (item["num_wrong"], item["idx"]), reverse=True)
    return heterogeneous_prompts, heterogeneous_wrong_generations


def summarize_patterns(
    heterogeneous_prompts: List[Dict[str, Any]],
    wrong_generations: List[Dict[str, Any]],
    num_rules: int,
) -> List[Dict[str, Any]]:
    pattern_stats: Dict[str, Dict[str, Any]] = {}
    for wrong in wrong_generations:
        for pattern in wrong["patterns"]:
            stats = pattern_stats.setdefault(
                pattern,
                {"count": 0, "prompt_ids": set(), "examples": []},
            )
            stats["count"] += 1
            stats["prompt_ids"].add(wrong["idx"])
            if len(stats["examples"]) < 3:
                stats["examples"].append(
                    {
                        "idx": wrong["idx"],
                        "db_id": wrong["db_id"],
                        "question": wrong.get("question", ""),
                        "sample_idx": wrong["sample_idx"],
                        "status": wrong["status"],
                        "pred_sql": wrong["pred_sql"],
                        "gold_sql": wrong["gold_sql"],
                    }
                )

    total_wrong = max(1, len(wrong_generations))
    total_prompts = max(1, len(heterogeneous_prompts))
    ranked_patterns: List[Dict[str, Any]] = []
    for pattern, stats in pattern_stats.items():
        meta = PATTERN_LIBRARY.get(pattern, PATTERN_LIBRARY["other_result_set_mismatch"])
        ranked_patterns.append(
            {
                "pattern": pattern,
                "title": meta["title"],
                "rule": meta["rule"],
                "why": meta["why"],
                "wrong_generation_count": stats["count"],
                "wrong_generation_pct": 100.0 * stats["count"] / total_wrong,
                "heterogeneous_prompt_count": len(stats["prompt_ids"]),
                "heterogeneous_prompt_pct": 100.0 * len(stats["prompt_ids"]) / total_prompts,
                "examples": stats["examples"],
            }
        )

    ranked_patterns.sort(
        key=lambda item: (
            item["wrong_generation_count"],
            item["heterogeneous_prompt_count"],
            item["title"],
        ),
        reverse=True,
    )
    return ranked_patterns[: max(1, num_rules)]


def build_markdown(
    args: argparse.Namespace,
    sampled_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    heterogeneous_prompts: List[Dict[str, Any]],
    wrong_generations: List[Dict[str, Any]],
    ranked_patterns: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Failure Instruction Candidates",
        "",
        "## Run Summary",
        "",
        f"- model_name_or_path: {args.model_name_or_path}",
        f"- input_file: {args.input_file}",
        f"- database_dir: {args.database_dir}",
        f"- sampled_prompts: {len(sampled_rows)}",
        f"- prompts_used_for_generation: {len(prediction_rows)}",
        f"- prompts_skipped_for_length: {len(skipped_rows)}",
        f"- num_generations_per_prompt: {args.num_generations}",
        f"- heterogeneous_prompts: {len(heterogeneous_prompts)}",
        f"- wrong_generations_from_heterogeneous_prompts: {len(wrong_generations)}",
        "",
        "## Prompt-Ready Rules",
        "",
    ]

    for idx, pattern in enumerate(ranked_patterns, start=1):
        lines.append(f"{idx}. {pattern['rule']}")

    lines.extend(["", "## Pattern Details", ""])
    for idx, pattern in enumerate(ranked_patterns, start=1):
        lines.append(f"### Rule {idx}: {pattern['title']}")
        lines.append("")
        lines.append(f"- observed_wrong_generations: {pattern['wrong_generation_count']} ({pattern['wrong_generation_pct']:.2f}%)")
        lines.append(
            f"- observed_heterogeneous_prompts: {pattern['heterogeneous_prompt_count']} ({pattern['heterogeneous_prompt_pct']:.2f}%)"
        )
        lines.append(f"- heuristic: {pattern['why']}")
        lines.append(f"- instruction: {pattern['rule']}")
        example = pattern["examples"][0] if pattern["examples"] else None
        if example is not None:
            lines.append(f"- example_db_id: {example['db_id']}")
            lines.append(f"- example_idx: {example['idx']}")
            lines.append(f"- example_question: {example['question']}")
            lines.append(f"- example_status: {example['status']}")
            lines.append("")
            lines.append("```sql")
            lines.append(f"-- gold\n{example['gold_sql']}")
            lines.append("")
            lines.append(f"-- wrong prediction\n{example['pred_sql']}")
            lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite, skip_generation=args.skip_generation)
    print_configuration(args, output_dir)

    sampled_path = output_dir / "sampled_rows.jsonl"
    filtered_path = output_dir / "filtered_examples.jsonl"
    generations_path = Path(args.prediction_samples_file) if args.prediction_samples_file else output_dir / "prediction_samples.jsonl"
    eval_path = output_dir / "generation_eval_results.jsonl"
    hetero_prompt_path = output_dir / "heterogeneous_prompts.jsonl"
    hetero_wrong_path = output_dir / "heterogeneous_wrong_generations.jsonl"
    summary_path = output_dir / "pattern_summary.json"
    markdown_path = output_dir / "failure_instruction_rules.md"

    if args.skip_generation:
        if not generations_path.exists():
            raise FileNotFoundError(
                f"Prediction samples file {generations_path} does not exist; cannot use --skip_generation."
            )
        prediction_rows = load_jsonl(generations_path)
        sampled_rows = load_jsonl(sampled_path) if sampled_path.exists() else []
        skipped_rows = load_jsonl(filtered_path) if filtered_path.exists() else []
        print(f"[run] loaded {len(prediction_rows)} saved prediction rows from {generations_path}")
    else:
        rows = load_rows(args.input_file)
        print(f"[run] loaded {len(rows)} normalized input rows")
        sampled_rows = sample_rows(rows, args.sample_size, args.sample_seed)
        print(f"[run] sampled {len(sampled_rows)} rows")
        write_jsonl(sampled_path, sampled_rows)

        prediction_rows, skipped_rows = generate_predictions_with_vllm_n(sampled_rows, args, output_dir=output_dir)
        if not prediction_rows:
            raise RuntimeError("No prompts remained after prompt-length filtering.")
        write_jsonl(filtered_path, skipped_rows)
        write_jsonl(generations_path, prediction_rows)
        print(f"[run] wrote crash-safe generation samples to {generations_path}")

    evaluation_rows = evaluate_generations(
        prediction_rows=prediction_rows,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        max_workers=args.eval_workers,
    )

    heterogeneous_prompts, wrong_generations = build_heterogeneous_sets(
        evaluation_rows,
        num_generations=args.num_generations,
    )
    ranked_patterns = summarize_patterns(
        heterogeneous_prompts=heterogeneous_prompts,
        wrong_generations=wrong_generations,
        num_rules=args.num_rules,
    )
    write_jsonl(eval_path, evaluation_rows)
    write_jsonl(hetero_prompt_path, heterogeneous_prompts)
    write_jsonl(hetero_wrong_path, wrong_generations)

    summary = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "input_file": args.input_file,
            "database_dir": args.database_dir,
            "sample_size": args.sample_size,
            "sample_seed": args.sample_seed,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_data_parallel_size": args.vllm_data_parallel_size,
        },
        "counts": {
            "sampled_prompts": len(sampled_rows),
            "prompts_used_for_generation": len(prediction_rows),
            "prompts_skipped_for_length": len(skipped_rows),
            "evaluated_generations": len(evaluation_rows),
            "heterogeneous_prompts": len(heterogeneous_prompts),
            "wrong_generations_from_heterogeneous_prompts": len(wrong_generations),
        },
        "rules": ranked_patterns,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    markdown_text = build_markdown(
        args=args,
        sampled_rows=sampled_rows,
        skipped_rows=skipped_rows,
        prediction_rows=prediction_rows,
        heterogeneous_prompts=heterogeneous_prompts,
        wrong_generations=wrong_generations,
        ranked_patterns=ranked_patterns,
    )
    markdown_path.write_text(markdown_text, encoding="utf-8")

    print(f"[summary] heterogeneous prompts: {len(heterogeneous_prompts)}")
    print(f"[summary] wrong generations from heterogeneous prompts: {len(wrong_generations)}")
    for idx, rule in enumerate(ranked_patterns, start=1):
        print(
            f"[summary] rule {idx}: {rule['title']} | "
            f"wrong_generations={rule['wrong_generation_count']} "
            f"heterogeneous_prompts={rule['heterogeneous_prompt_count']}"
        )
    print(f"Saved sampled rows to {sampled_path}")
    print(f"Saved filtered-example report to {filtered_path}")
    print(f"Saved generation samples to {generations_path}")
    print(f"Saved per-generation evaluation results to {eval_path}")
    print(f"Saved heterogeneous prompt summary to {hetero_prompt_path}")
    print(f"Saved heterogeneous wrong generations to {hetero_wrong_path}")
    print(f"Saved pattern summary to {summary_path}")
    print(f"Saved markdown instruction candidates to {markdown_path}")


if __name__ == "__main__":
    main()
