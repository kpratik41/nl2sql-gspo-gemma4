import argparse
import json
import re
from pathlib import Path


DEFAULT_REFERENCE_INPUT = Path("data/bird_train_data/raw/train-6601.jsonl")
DEFAULT_DEV_INPUT = Path("data/bird_dev_data/raw/bird_dev.json")
DEFAULT_DEV_OUTPUT = Path("data/bird_dev_data/raw/bird_dev-few-shot.json")


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_retrieval_index(reference_data: list[dict]):
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rank-bm25 is required for few-shot generation. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    reference_texts = [f"{item.get('question', '')} {item.get('evidence', '')}" for item in reference_data]
    tokenized_corpus = [tokenize(text) for text in reference_texts]
    return BM25Okapi(tokenized_corpus)


def format_few_shot_examples(reference_data: list[dict], ranked_indices: list[int], scores) -> list[dict]:
    few_shot_examples: list[dict] = []
    for rank, index in enumerate(ranked_indices, start=1):
        reference_item = reference_data[index]
        few_shot_examples.append(
            {
                "rank": rank,
                "bm25_score": round(float(scores[index]), 4),
                "db_id": reference_item.get("db_id", ""),
                "question": reference_item.get("question", ""),
                "evidence": reference_item.get("evidence", ""),
                "SQL": reference_item.get("SQL", ""),
            }
        )
    return few_shot_examples


def attach_inference_few_shots(reference_data: list[dict], dev_data: list[dict], bm25, top_n: int) -> list[dict]:
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
                "few_shot_examples": format_few_shot_examples(reference_data, ranked_indices, scores),
            }
        )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BM25-based few-shot files for dev inference."
    )
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=DEFAULT_REFERENCE_INPUT,
        help="JSONL examples used as the retrieval corpus for few-shot prompts.",
    )
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV_OUTPUT)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be >= 1")

    if args.reference_input.suffix.lower() != ".jsonl":
        raise ValueError(f"Reference file must be .jsonl, got: {args.reference_input}")

    if args.dev_input.suffix.lower() != ".json":
        raise ValueError(f"Dev file must be .json, got: {args.dev_input}")

    reference_data = load_jsonl(args.reference_input)
    dev_data = load_json(args.dev_input)
    if not isinstance(dev_data, list):
        raise ValueError(f"Expected a JSON array in {args.dev_input}")

    bm25 = build_retrieval_index(reference_data)
    print(f"BM25 index built over {len(reference_data)} reference examples.")

    dev_output = attach_inference_few_shots(
        reference_data=reference_data,
        dev_data=dev_data,
        bm25=bm25,
        top_n=args.top_n,
    )
    write_json(args.dev_output, dev_output)
    print(f"Saved {len(dev_output)} dev entries with top-{args.top_n} few-shot examples -> {args.dev_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
