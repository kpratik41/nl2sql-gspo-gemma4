import argparse
import json
import multiprocessing as mp
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nl2sql_gspo.data import normalize_record
from nl2sql_gspo.sql_utils import extract_sql, get_database_path


BIRD_SPLIT_MARKER = "\t----- bird -----\t"


def load_diff_rows(diff_json_path: str) -> List[Dict[str, Any]]:
    with open(diff_json_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    return loaded if isinstance(loaded, list) else [loaded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="outputs/dev-20251106-schema.jsonl")
    parser.add_argument("--database_dir", type=str, default="databases/dev_databases")
    parser.add_argument("--diff_json_path", type=str, default="data/bird_dev_data/raw/dev_20251106.json")
    parser.add_argument("--output_dir", type=str, default="outputs/bird_dev_inference")
    parser.add_argument("--max_prompt_length", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--eval_timeout", type=float, default=30.0)
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

    return rows


def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def generate_predictions(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --skip_generation is set")

    import torch

    from nl2sql_gspo.model_utils import load_inference_model_and_tokenizer

    model, tokenizer = load_inference_model_and_tokenizer(args.model_name_or_path)
    generation_device = next(model.parameters()).device

    official_predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, Any]] = []

    do_sample = args.temperature > 0.0

    for idx, row in enumerate(rows):
        prompt_messages = row.get("prompt") or row.get("messages") or []
        prompt_text = render_prompt(tokenizer, prompt_messages)

        tokenized = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
        )
        tokenized = {key: value.to(generation_device) for key, value in tokenized.items()}
        prompt_token_count = int(tokenized["input_ids"].shape[1])

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
        db_id = row.get("db_id", "")

        official_predictions[str(idx)] = f"{pred_sql}{BIRD_SPLIT_MARKER}{db_id}"
        detailed_predictions.append(
            {
                "idx": idx,
                "db_id": db_id,
                "prediction_text": prediction_text,
                "pred_sql": pred_sql,
                "gold_sql": row.get("gold_sql", ""),
                "prompt_tokens": prompt_token_count,
            }
        )

        if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(rows):
            print(f"[inference] generated {idx + 1}/{len(rows)} prompts")

    return official_predictions, detailed_predictions


def load_predictions(predictions_path: Path) -> Dict[str, str]:
    with predictions_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _execute_query_pair(queue, predicted_sql: str, ground_sql: str, db_path: str) -> None:
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        pred_rows = cursor.fetchall()
        cursor.execute(ground_sql)
        gold_rows = cursor.fetchall()
        conn.close()
        queue.put({"res": int(set(pred_rows) == set(gold_rows)), "status": "ok"})
    except Exception as exc:
        queue.put({"res": 0, "status": f"error: {exc}"})


def evaluate_one(predicted_sql: str, ground_sql: str, db_path: str, timeout_s: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_execute_query_pair, args=(queue, predicted_sql, ground_sql, db_path))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"res": 0, "status": "timeout"}

    if not queue.empty():
        return queue.get()

    return {"res": 0, "status": "error: no result"}


def build_group_summary(
    results: List[Dict[str, Any]],
    group_key: str,
    group_order: List[str] | None = None,
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


def write_summary_markdown(summary: Dict[str, Any], markdown_path: Path) -> None:
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
    ]

    markdown_path.write_text("\n".join(content), encoding="utf-8")


def evaluate_predictions(
    rows: List[Dict[str, Any]],
    predictions: Dict[str, str],
    database_dir: str,
    diff_rows: List[Dict[str, Any]],
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_example_results: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        packed_prediction = predictions.get(str(idx), "")
        if BIRD_SPLIT_MARKER in packed_prediction:
            predicted_sql, predicted_db_id = packed_prediction.split(BIRD_SPLIT_MARKER, 1)
        else:
            predicted_sql = packed_prediction
            predicted_db_id = row.get("db_id", "")

        db_id = predicted_db_id or row.get("db_id", "")
        difficulty = diff_rows[idx].get("difficulty", "unknown") if idx < len(diff_rows) else "unknown"
        db_path = get_database_path(db_id=db_id, database_dir=database_dir)
        eval_result = evaluate_one(predicted_sql, row.get("gold_sql", ""), db_path, timeout_s)
        per_example_results.append(
            {
                "idx": idx,
                "db_id": db_id,
                "difficulty": difficulty,
                "pred_sql": predicted_sql,
                "gold_sql": row.get("gold_sql", ""),
                "res": int(eval_result["res"]),
                "status": eval_result["status"],
            }
        )

        if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(rows):
            print(f"[evaluation] scored {idx + 1}/{len(rows)} predictions")

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

    rows = load_rows(args.input_file, args.num_examples)

    predictions_path = output_dir / "predict_dev.json"
    details_path = output_dir / "prediction_details.jsonl"
    per_example_eval_path = output_dir / "eval_results.jsonl"
    summary_path = output_dir / "eval_summary.json"
    summary_markdown_path = output_dir / "eval_summary.md"
    diff_rows = load_diff_rows(args.diff_json_path)

    if args.skip_generation:
        official_predictions = load_predictions(predictions_path)
    else:
        official_predictions, detailed_predictions = generate_predictions(rows, args)
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(official_predictions, handle, ensure_ascii=False, indent=2)

        with details_path.open("w", encoding="utf-8") as handle:
            for record in detailed_predictions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    per_example_results, summary = evaluate_predictions(
        rows=rows,
        predictions=official_predictions,
        database_dir=args.database_dir,
        diff_rows=diff_rows,
        timeout_s=args.eval_timeout,
    )

    with per_example_eval_path.open("w", encoding="utf-8") as handle:
        for record in per_example_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_summary_markdown(summary, summary_markdown_path)
    print_summary_tables(summary)
    print(f"Saved official BIRD predictions to {predictions_path}")
    print(f"Saved per-example evaluation to {per_example_eval_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved markdown summary to {summary_markdown_path}")


if __name__ == "__main__":
    main()