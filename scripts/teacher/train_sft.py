#!/usr/bin/env python3
"""Stage A4 — masked multi-turn SFT on the RFT dataset.

Loss is attributed to assistant spans only. System turns, user turns and tool
responses are labelled ``-100``; see ``scripts/teacher/sft_masking.py`` for how
the spans are located.

Sequences are **never truncated**. A record longer than ``--max_seq_len`` is
dropped, as is any record with no supervised assistant tokens, and both counts
are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from teacher.sft_masking import IGNORE_INDEX, build_supervised_example  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default="google/gemma-4-31B-it")
    parser.add_argument("--train_file", default="outputs/teacher/rft/train_rft_31b.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", default="configs/ds_zero3_bf16_no_scheduler.json")
    parser.add_argument("--max_seq_len", type=int, default=20480)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--cache_file", default=None, help="Optional path to cache the tokenized dataset.")
    return parser.parse_args()


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for feature in features:
            ids = feature["input_ids"]
            lab = feature["labels"]
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def build_dataset(args: argparse.Namespace, tokenizer) -> List[Dict[str, Any]]:
    cache_path = Path(args.cache_file) if args.cache_file else None
    if cache_path and cache_path.exists():
        print(f"[sft] loading tokenized cache {cache_path}")
        with cache_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    rows = [json.loads(line) for line in open(args.train_file, encoding="utf-8") if line.strip()]
    examples: List[Dict[str, Any]] = []
    dropped_long = 0
    dropped_unsupervised = 0
    boundary_failures = 0

    for position, row in enumerate(rows):
        input_ids, labels, stats = build_supervised_example(tokenizer, row["messages"], row.get("tools"))
        boundary_failures += stats["n_boundary_failures"]
        if stats["n_tokens"] > args.max_seq_len:
            dropped_long += 1
            continue
        if stats["n_supervised_tokens"] == 0:
            dropped_unsupervised += 1
            continue
        examples.append({"input_ids": input_ids, "labels": labels})
        if (position + 1) % 500 == 0:
            print(f"[sft] tokenized {position + 1}/{len(rows)}", flush=True)

    print(
        f"[sft] dataset: {len(examples)} usable / {len(rows)} records "
        f"(dropped_over_{args.max_seq_len}={dropped_long}, dropped_unsupervised={dropped_unsupervised}, "
        f"boundary_failures={boundary_failures})"
    )
    if not examples:
        raise SystemExit("no usable training examples")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example) + "\n")
        print(f"[sft] cached tokenized dataset to {cache_path}")
    return examples


def main() -> None:
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from nl2sql_gspo.model_utils import load_tokenizer

    args = parse_args()
    tokenizer = load_tokenizer(args.model_name_or_path)
    examples = build_dataset(args, tokenizer)

    print(f"[sft] loading model {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        report_to=[r for r in args.report_to.split(",") if r and r != "none"] or "none",
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=Collator(pad_token_id=pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[sft] saved final model to {args.output_dir}")


if __name__ == "__main__":
    main()
