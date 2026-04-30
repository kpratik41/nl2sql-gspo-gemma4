import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi


DEFAULT_TRAIN_INPUT = Path("data/bird_train_data/raw/train-6601.jsonl")
DEFAULT_DEV_INPUT = Path("data/bird_dev_data/raw/dev_20251106.json")
DEFAULT_TRAIN_OUTPUT = Path("data/bird_train_data/raw/train-6601-few-shot.jsonl")
DEFAULT_DEV_OUTPUT = Path("data/bird_dev_data/raw/dev_20251106-few-shot.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    return items


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, payload: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in payload:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_train_index(train_data: list[dict]) -> tuple[BM25Okapi, dict[str, list[int]]]:
    train_texts = [f"{item.get('question', '')} {item.get('evidence', '')}" for item in train_data]
    tokenized_corpus = [tokenize(text) for text in train_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    indices_by_db: dict[str, list[int]] = {}
    for index, item in enumerate(train_data):
        db_id = item.get("db_id", "")
        if isinstance(db_id, str) and db_id:
            indices_by_db.setdefault(db_id, []).append(index)

    all_indices = list(range(len(train_data)))
    eligible_other_db_indices = {
        db_id: [index for index in all_indices if index not in same_db_indices]
        for db_id, same_db_indices in indices_by_db.items()
    }
    return bm25, eligible_other_db_indices


def format_few_shot_examples(train_data: list[dict], ranked_indices: list[int], scores) -> list[dict]:
    few_shot_examples: list[dict] = []
    for rank, index in enumerate(ranked_indices, start=1):
        train_item = train_data[index]
        few_shot_examples.append(
            {
                "rank": rank,
                "bm25_score": round(float(scores[index]), 4),
                "db_id": train_item.get("db_id", ""),
                "question": train_item.get("question", ""),
                "evidence": train_item.get("evidence", ""),
                "SQL": train_item.get("SQL", ""),
            }
        )
    return few_shot_examples


def attach_inference_few_shots(train_data: list[dict], dev_data: list[dict], bm25: BM25Okapi, top_n: int) -> list[dict]:
    output: list[dict] = []
    for dev_item in dev_data:
        query_text = f"{dev_item.get('question', '')} {dev_item.get('evidence', '')}".strip()
        scores = bm25.get_scores(tokenize(query_text))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_n]

        output.append(
            {
                "question_id": dev_item.get("question_id"),
                "db_id": dev_item.get("db_id", ""),
                "question": dev_item.get("question", ""),
                "evidence": dev_item.get("evidence", ""),
                "SQL": dev_item.get("SQL", ""),
                "difficulty": dev_item.get("difficulty", ""),
                "few_shot_examples": format_few_shot_examples(train_data, ranked_indices, scores),
            }
        )

    return output


def attach_training_few_shots(
    train_data: list[dict],
    bm25: BM25Okapi,
    eligible_other_db_indices: dict[str, list[int]],
    top_n: int,
) -> list[dict]:
    output: list[dict] = []
    for train_item in train_data:
        db_id = train_item.get("db_id", "")
        query_text = f"{train_item.get('question', '')} {train_item.get('evidence', '')}".strip()
        scores = bm25.get_scores(tokenize(query_text))

        candidate_indices = eligible_other_db_indices.get(db_id, list(range(len(train_data))))
        ranked_indices = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)[:top_n]

        output.append(
            {
                "db_id": train_item.get("db_id", ""),
                "question": train_item.get("question", ""),
                "evidence": train_item.get("evidence", ""),
                "SQL": train_item.get("SQL", ""),
                "few_shot_examples": format_few_shot_examples(train_data, ranked_indices, scores),
            }
        )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BM25-based few-shot files for dev inference and train-time "
            "self-supervision."
        )
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV_OUTPUT)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be >= 1")

    if args.train_input.suffix.lower() != ".jsonl":
        raise ValueError(f"Train file must be .jsonl, got: {args.train_input}")

    if args.dev_input.suffix.lower() != ".json":
        raise ValueError(f"Dev file must be .json, got: {args.dev_input}")

    train_data = load_jsonl(args.train_input)
    dev_data = load_json(args.dev_input)
    if not isinstance(dev_data, list):
        raise ValueError(f"Expected a JSON array in {args.dev_input}")

    bm25, eligible_other_db_indices = build_train_index(train_data)
    print(f"BM25 index built over {len(train_data)} training examples.")

    train_output = attach_training_few_shots(
        train_data=train_data,
        bm25=bm25,
        eligible_other_db_indices=eligible_other_db_indices,
        top_n=args.top_n,
    )
    write_jsonl(args.train_output, train_output)
    print(f"Saved {len(train_output)} train entries with top-{args.top_n} few-shot examples -> {args.train_output}")

    dev_output = attach_inference_few_shots(
        train_data=train_data,
        dev_data=dev_data,
        bm25=bm25,
        top_n=args.top_n,
    )
    write_json(args.dev_output, dev_output)
    print(f"Saved {len(dev_output)} dev entries with top-{args.top_n} few-shot examples -> {args.dev_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())