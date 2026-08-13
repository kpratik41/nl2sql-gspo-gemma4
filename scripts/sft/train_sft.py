#!/usr/bin/env python3
"""A4: STaR-style masked multi-turn SFT warm-start on the RFT dataset.

Each training row is ``{prompt, messages, tools, ...}`` where ``messages`` =
hint-free [system, user] + the verified agentic transcript (assistant / tool
turns). We render the whole conversation with the tokenizer chat template
(tools included) and supervise ONLY the assistant turns; system / user / tool
turns are masked (label = -100). This teaches the student the verified
reasoning -> tool-call -> final-SQL trajectory while never showing the
privileged hint (it is not in the prompt).

DeepSpeed ZeRO-3 is enabled by passing ``--deepspeed`` to ``TrainingArguments``
BEFORE the model is loaded, so ``from_pretrained`` shards weights at init
(zero.Init) instead of materializing the full 31B on every rank.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# repo imports (PYTHONPATH = _pkgs_trl14:CODE:CODE/src)
from nl2sql_gspo.model_utils import load_tokenizer


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(*a: Any) -> None:
    if is_main():
        print(*a, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Masked multi-turn SFT (Path A4)."
    )
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument(
        "--train_file",
        required=True,
        help="RFT jsonl from build_rft_from_traces.py.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--deepspeed",
        required=True,
        help="DeepSpeed ZeRO-3 config json.",
    )
    p.add_argument("--max_seq_len", type=int, default=20480)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_examples(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_seq_len: int,
) -> List[Dict[str, List[int]]]:
    """Render + mask each RFT row into {input_ids, labels}. Assistant-only loss."""

    def ct_ids(
        messages: List[Dict[str, Any]],
        tools,
        add_generation_prompt: bool,
    ) -> List[int]:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

    examples: List[Dict[str, List[int]]] = []
    n_too_long = 0
    n_no_supervision = 0

    for row in rows:
        messages = row.get("messages")
        tools = row.get("tools")

        if not messages:
            continue

        full = ct_ids(
            messages,
            tools,
            add_generation_prompt=False,
        )

        if len(full) > max_seq_len:
            n_too_long += 1
            continue

        labels = [-100] * len(full)

        for k, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            prev_gen = ct_ids(
                messages[:k],
                tools,
                add_generation_prompt=True,
            )
            through = ct_ids(
                messages[: k + 1],
                tools,
                add_generation_prompt=False,
            )

            start = _lcp(prev_gen, full)
            end = min(len(through), len(full))

            for i in range(start, end):
                labels[i] = full[i]

        if not any(label != -100 for label in labels):
            n_no_supervision += 1
            continue

        examples.append(
            {
                "input_ids": full,
                "labels": labels,
            }
        )

    log(
        f"[sft] built {len(examples)} examples "
        f"(dropped {n_too_long} > max_seq_len={max_seq_len}, "
        f"{n_no_supervision} no-supervision)"
    )

    if examples:
        sup = sum(
            sum(1 for x in ex["labels"] if x != -100)
            for ex in examples
        )
        tot = sum(
            len(ex["input_ids"])
            for ex in examples
        )
        lens = sorted(
            len(ex["input_ids"])
            for ex in examples
        )

        log(
            f"[sft] supervised tokens={sup}/{tot} "
            f"({100 * sup / max(1, tot):.1f}%) | "
            f"seq_len p50={lens[len(lens) // 2]} max={lens[-1]}"
        )

    return examples


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(
        self,
        features: List[Dict[str, List[int]]],
    ) -> Dict[str, torch.Tensor]:
        max_len = max(
            len(f["input_ids"])
            for f in features
        )

        input_ids, labels, attention = [], [], []

        for feature in features:
            ids = feature["input_ids"]
            lab = feature["labels"]
            pad = max_len - len(ids)

            input_ids.append(
                ids + [self.pad_token_id] * pad
            )
            labels.append(
                lab + [-100] * pad
            )
            attention.append(
                [1] * len(ids) + [0] * pad
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention,
                dtype=torch.long,
            ),
        }


def main() -> None:
    args = parse_args()

    # 1) TrainingArguments FIRST -> activates HfDeepSpeedConfig zero.Init for from_pretrained.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=[args.report_to]
        if args.report_to and args.report_to != "none"
        else [],
        run_name=args.run_name,
        seed=args.seed,
        ddp_timeout=36000,
        dataloader_num_workers=2,
        logging_first_step=True,
    )

    tokenizer = load_tokenizer(args.model_name_or_path)

    rows = read_jsonl(args.train_file)
    log(f"[sft] loaded {len(rows)} rft rows from {args.train_file}")

    examples = build_examples(
        rows,
        tokenizer,
        args.max_seq_len,
    )

    if not examples:
        raise RuntimeError(
            "no SFT examples after rendering/masking"
        )

    from datasets import Dataset

    train_dataset = Dataset.from_list(examples)

    # 2) load model under zero.Init (sharded); use_cache off for grad checkpointing.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PadCollator(
            pad_token_id=tokenizer.pad_token_id
        ),
        processing_class=tokenizer,
    )

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.save_model(args.output_dir)

    if is_main():
        tokenizer.save_pretrained(args.output_dir)
        print(
            f"[sft] saved final model -> {args.output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()