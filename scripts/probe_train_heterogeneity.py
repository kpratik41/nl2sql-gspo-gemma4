#!/usr/bin/env python
"""Probe result-reward heterogeneity on training prompts.

This script samples training rows, generates ``num_generations`` completions
per prompt, executes predicted and gold SQL with the same BIRD-style semantics
used by the training reward, and reports the distribution of result-correct
counts from 0..num_generations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from nl2sql_gspo.model_utils import load_tokenizer
from nl2sql_gspo.sql_utils import (
    bird_execute_sql,
    bird_get_gold_rows,
    bird_result_match,
    extract_sql,
    is_safe_readonly_sql,
)
from scripts.run_inference_bird import load_rows, prepare_rows_for_generation, preview_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local_vllm", "server"], default="local_vllm")
    parser.add_argument("--model_name_or_path", default="google/gemma-4-31B-it")
    parser.add_argument("--train_file", default="outputs/train-6601-schema-filtered.jsonl")
    parser.add_argument("--database_dir", default="databases")
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--num_prompts", type=int, default=96)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=17)

    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=20000)

    parser.add_argument("--vllm_server_base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=8)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_max_model_len", type=int, default=24576)

    parser.add_argument("--exec_timeout_s", type=float, default=60.0)
    parser.add_argument("--eval_workers", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true", help="Prepare prompts but skip generation/evaluation.")
    return parser.parse_args()


def select_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.sample_strategy == "random":
        rng = random.Random(args.sample_seed)
        if args.num_prompts > len(rows):
            raise ValueError(f"Requested {args.num_prompts} prompts, but only {len(rows)} rows are available")
        return rng.sample(rows, args.num_prompts)

    start = max(0, args.sample_start)
    end = start + args.num_prompts
    selected = rows[start:end]
    if len(selected) < args.num_prompts:
        raise ValueError(
            f"Requested rows [{start}:{end}], but only got {len(selected)} from {len(rows)} rows"
        )
    return selected


def generate_with_local_vllm(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[List[str]]:
    from vllm import LLM, SamplingParams

    print(
        "[gen] loading local vLLM "
        f"model={args.model_name_or_path} tp={args.vllm_tensor_parallel_size} "
        f"max_model_len={args.vllm_max_model_len} visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(
        [row["prompt_text"] for row in rows],
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    grouped: List[List[str]] = []
    for request_output in outputs:
        grouped.append([(output.text or "").strip() for output in request_output.outputs])
    return grouped


def generate_with_server(rows: List[Dict[str, Any]], tokenizer, args: argparse.Namespace) -> List[List[str]]:
    from trl.generation.vllm_client import VLLMClient

    print(f"[gen] using TRL vLLM server at {args.vllm_server_base_url}")
    client = VLLMClient(base_url=args.vllm_server_base_url, connection_timeout=600)
    response = client.generate(
        [row["prompt_text"] for row in rows],
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        logprobs=None,
    )
    completion_ids = response["completion_ids"]
    expected = len(rows) * args.num_generations
    if len(completion_ids) != expected:
        raise RuntimeError(
            f"Server returned {len(completion_ids)} completions, expected {expected} "
            f"({len(rows)} prompts * {args.num_generations})"
        )

    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    grouped = []
    for offset in range(0, len(texts), args.num_generations):
        grouped.append([text.strip() for text in texts[offset : offset + args.num_generations]])
    return grouped


def evaluate_completion(
    row: Dict[str, Any],
    generation_index: int,
    prediction_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    db_id = row.get("db_id", "")
    gold_sql = extract_sql(row.get("gold_sql", ""))
    pred_sql = extract_sql(prediction_text)

    gold_ok, gold_rows, gold_error = bird_get_gold_rows(
        gold_sql=gold_sql,
        db_id=db_id,
        database_dir=args.database_dir,
        timeout_s=args.exec_timeout_s,
    )
    pred_safe = is_safe_readonly_sql(pred_sql)
    pred_ok, pred_rows, pred_error = bird_execute_sql(
        sql=pred_sql,
        db_id=db_id,
        database_dir=args.database_dir,
        timeout_s=args.exec_timeout_s,
    )
    result_correct = bool(pred_ok and gold_ok and bird_result_match(pred_rows, gold_rows))
    return {
        "generation_index": generation_index,
        "prediction_text": prediction_text,
        "pred_sql": pred_sql,
        "pred_sql_extracted": bool(pred_sql.strip()),
        "pred_sql_safe_readonly": pred_safe,
        "pred_sql_executed": bool(pred_ok),
        "pred_error": pred_error,
        "gold_sql": gold_sql,
        "gold_sql_executed": bool(gold_ok),
        "gold_error": gold_error,
        "result_correct": result_correct,
    }


def evaluate_all(
    rows: List[Dict[str, Any]],
    grouped_predictions: List[List[str]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    futures = []
    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
        for prompt_index, (row, predictions) in enumerate(zip(rows, grouped_predictions)):
            for generation_index, prediction_text in enumerate(predictions):
                futures.append(
                    (
                        prompt_index,
                        pool.submit(evaluate_completion, row, generation_index, prediction_text, args),
                    )
                )

        per_prompt_generations: List[List[Dict[str, Any]]] = [[] for _ in rows]
        completed = 0
        for prompt_index, future in futures:
            result = future.result()
            per_prompt_generations[prompt_index].append(result)
            completed += 1
            if completed == 1 or completed == len(futures) or completed % 100 == 0:
                print(f"[eval] evaluated {completed}/{len(futures)} completions")

    per_prompt: List[Dict[str, Any]] = []
    correct_count_hist = Counter()
    exec_count_hist = Counter()

    total_pred_extracted = 0
    total_pred_safe = 0
    total_pred_executed = 0
    total_result_correct = 0
    total_gold_executed_prompts = 0

    for row, generations in zip(rows, per_prompt_generations):
        generations.sort(key=lambda item: item["generation_index"])
        correct_count = sum(1 for item in generations if item["result_correct"])
        exec_count = sum(1 for item in generations if item["pred_sql_executed"])
        gold_ok = bool(generations and generations[0]["gold_sql_executed"])

        correct_count_hist[correct_count] += 1
        exec_count_hist[exec_count] += 1
        total_pred_extracted += sum(1 for item in generations if item["pred_sql_extracted"])
        total_pred_safe += sum(1 for item in generations if item["pred_sql_safe_readonly"])
        total_pred_executed += exec_count
        total_result_correct += correct_count
        total_gold_executed_prompts += int(gold_ok)

        per_prompt.append(
            {
                "source_idx": row.get("source_idx", -1),
                "db_id": row.get("db_id", ""),
                "prompt_tokens": row.get("prompt_tokens", None),
                "gold_sql": extract_sql(row.get("gold_sql", "")),
                "gold_sql_executed": gold_ok,
                "gold_error": generations[0]["gold_error"] if generations else "missing generations",
                "correct_count": correct_count,
                "exec_count": exec_count,
                "heterogeneous_result_reward": 0 < correct_count < args.num_generations,
                "prompt_preview": preview_text(row.get("prompt_text", ""), max_chars=240),
                "generations": generations,
            }
        )

    total_completions = len(rows) * args.num_generations
    heterogeneous_prompts = sum(
        1 for item in per_prompt if item["heterogeneous_result_reward"]
    )
    summary = {
        "backend": args.backend,
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "database_dir": args.database_dir,
        "num_prompts": len(rows),
        "num_generations": args.num_generations,
        "total_completions": total_completions,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "heterogeneous_prompt_count": heterogeneous_prompts,
        "heterogeneous_prompt_rate": heterogeneous_prompts / max(1, len(rows)),
        "all_wrong_prompt_count": correct_count_hist.get(0, 0),
        "all_correct_prompt_count": correct_count_hist.get(args.num_generations, 0),
        "correct_count_histogram": {
            str(i): correct_count_hist.get(i, 0) for i in range(args.num_generations + 1)
        },
        "exec_count_histogram": {
            str(i): exec_count_hist.get(i, 0) for i in range(args.num_generations + 1)
        },
        "gold_sql_executed_prompts": total_gold_executed_prompts,
        "gold_sql_failed_prompts": len(rows) - total_gold_executed_prompts,
        "pred_sql_extracted_completions": total_pred_extracted,
        "pred_sql_safe_readonly_completions": total_pred_safe,
        "pred_sql_executed_completions": total_pred_executed,
        "result_correct_completions": total_result_correct,
        "pred_sql_extracted_rate": total_pred_extracted / max(1, total_completions),
        "pred_sql_executed_rate": total_pred_executed / max(1, total_completions),
        "result_correct_rate": total_result_correct / max(1, total_completions),
    }
    return per_prompt, summary


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/train_heterogeneity_probe_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[probe] configuration")
    for key, value in vars(args).items():
        print(f"[probe] {key}={value}")
    print(f"[probe] output_dir={output_dir}")
    print(f"[probe] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    tokenizer = load_tokenizer(args.model_name_or_path)
    all_rows = load_rows(args.train_file, num_examples=-1)
    selected_rows = select_rows(all_rows, args)
    rows, skipped_rows = prepare_rows_for_generation(
        selected_rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
    )
    if skipped_rows:
        skipped_path = output_dir / "skipped_prompts.jsonl"
        with skipped_path.open("w", encoding="utf-8") as handle:
            for row in skipped_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[probe] wrote skipped prompts to {skipped_path}")

    if args.dry_run:
        print("[probe] dry_run set; skipping generation/evaluation")
        return

    if args.backend == "local_vllm":
        grouped_predictions = generate_with_local_vllm(rows, args)
    else:
        grouped_predictions = generate_with_server(rows, tokenizer, args)

    if len(grouped_predictions) != len(rows):
        raise RuntimeError(f"Got predictions for {len(grouped_predictions)} prompts, expected {len(rows)}")
    bad_group_sizes = [len(group) for group in grouped_predictions if len(group) != args.num_generations]
    if bad_group_sizes:
        raise RuntimeError(f"Some prompts did not receive {args.num_generations} generations: {bad_group_sizes[:5]}")

    per_prompt, summary = evaluate_all(rows, grouped_predictions, args)

    per_prompt_path = output_dir / "per_prompt.jsonl"
    with per_prompt_path.open("w", encoding="utf-8") as handle:
        for item in per_prompt:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[probe] wrote {summary_path}")
    print(f"[probe] wrote {per_prompt_path}")


if __name__ == "__main__":
    main()
