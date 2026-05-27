#!/usr/bin/env python3
"""Analyze where pass@k oracle success does not translate to temp-0 success."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_RUN = Path(
    "outputs/training/train-6601-schema-bare-tool/gemma-4-E4B-it/"
    "grpo_deepspeed_p15500_c8000_g16_t1p2_bs4_ga8_lr2e-6_e4b_bare_lr2e6_maskfix_20260526_044450"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sc(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    by_ckpt: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    if not path.exists():
        return by_ckpt
    for row in load_jsonl(path):
        by_ckpt[int(row["ckpt"])][int(row["idx"])] = row
    return by_ckpt


def passk_dir(passk_root: Path, ckpt: int) -> Path:
    return passk_root / f"maskfix_ckpt-{ckpt}_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k"


def summarize_checkpoint(
    ckpt: int,
    passk_root: Path,
    train_run: Path,
    sc_by_ckpt: dict[int, dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_examples = load_jsonl(passk_dir(passk_root, ckpt) / "passk_per_example.jsonl")
    temp0_rows = load_jsonl(train_run / f"checkpoint-{ckpt}" / "eval_results.jsonl")
    sc_rows = sc_by_ckpt.get(ckpt, {})
    if len(per_examples) != len(temp0_rows):
        raise ValueError(f"ckpt-{ckpt}: passk/temp0 length mismatch")

    hist = Counter(int(row["num_correct"]) for row in per_examples)
    sample_rows: list[dict[str, Any]] = []

    for passk_row, temp0_row in zip(per_examples, temp0_rows):
        idx = int(passk_row["idx"])
        correct_count = int(passk_row["num_correct"])
        temp0_correct = int(temp0_row.get("res") == 1)
        sc = sc_rows.get(idx, {})
        sc2_correct = int(sc.get("option2_correct") == 1)
        sample_rows.append(
            {
                "ckpt": ckpt,
                "idx": idx,
                "db_id": passk_row.get("db_id", ""),
                "passk_correct_count": correct_count,
                "passk_any_correct": int(correct_count > 0),
                "passk_all_correct": int(correct_count == 16),
                "temp0_correct": temp0_correct,
                "sc2_correct": sc2_correct,
                "oracle_available_but_temp0_wrong": int(correct_count > 0 and not temp0_correct),
                "rare_correct_temp0_wrong": int(0 < correct_count <= 4 and not temp0_correct),
                "temp0_wrong_sc2_correct": int((not temp0_correct) and sc2_correct),
                "temp0_correct_sc2_wrong": int(temp0_correct and not sc2_correct),
                "largest_sc_group_16": sc.get("largest_group_16"),
                "num_sc_groups_16": sc.get("num_groups_16"),
                "num_valid_sc_groups_16": sc.get("num_valid_groups_16"),
                "option2_source": sc.get("option2_source"),
            }
        )

    total = len(per_examples)
    temp0_correct = sum(row["temp0_correct"] for row in sample_rows)
    sc2_correct = sum(row["sc2_correct"] for row in sample_rows)
    summary = {
        "ckpt": ckpt,
        "examples": total,
        "temp0_correct": temp0_correct,
        "temp0_accuracy": 100.0 * temp0_correct / total,
        "pass16_any_correct": sum(row["passk_any_correct"] for row in sample_rows),
        "pass16_any_accuracy": 100.0 * sum(row["passk_any_correct"] for row in sample_rows) / total,
        "pass16_all_wrong": hist[0],
        "pass16_all_correct": hist[16],
        "oracle_available_but_temp0_wrong": sum(
            row["oracle_available_but_temp0_wrong"] for row in sample_rows
        ),
        "rare_correct_temp0_wrong": sum(row["rare_correct_temp0_wrong"] for row in sample_rows),
        "sc2_correct": sc2_correct,
        "sc2_accuracy": 100.0 * sc2_correct / total,
        "temp0_wrong_sc2_correct": sum(row["temp0_wrong_sc2_correct"] for row in sample_rows),
        "temp0_correct_sc2_wrong": sum(row["temp0_correct_sc2_wrong"] for row in sample_rows),
        "avg_correct_count": sum(int(row["num_correct"]) for row in per_examples) / total,
        "median_correct_count": sorted(hist.elements())[total // 2],
        "correct_count_hist": {str(i): hist[i] for i in range(17)},
    }
    return summary, sample_rows


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Pass@K Selection Analysis",
        "",
        "| ckpt | temp0 acc | pass@16 oracle | SC option2 | all wrong | all correct | oracle but temp0 wrong | rare correct temp0 wrong | SC rescues | SC harms | avg correct / 16 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {ckpt} | `{temp0_accuracy:.2f}%` | `{pass16_any_accuracy:.2f}%` | "
            "`{sc2_accuracy:.2f}%` | `{pass16_all_wrong}` | `{pass16_all_correct}` | "
            "`{oracle_available_but_temp0_wrong}` | `{rare_correct_temp0_wrong}` | "
            "`{temp0_wrong_sc2_correct}` | `{temp0_correct_sc2_wrong}` | "
            "`{avg_correct_count:.2f}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "`rare correct temp0 wrong` means 1-4 of the 16 sampled generations were correct, but the temperature-0 run was wrong.",
            "`SC rescues` means temperature-0 was wrong while self-consistency option 2 was correct; `SC harms` is the reverse.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", default="0,20,40,60,80,100,120")
    parser.add_argument("--passk-root", type=Path, default=Path("outputs/passk"))
    parser.add_argument("--train-run", type=Path, default=DEFAULT_TRAIN_RUN)
    parser.add_argument(
        "--self-consistency-jsonl",
        type=Path,
        default=Path("outputs/analysis/maskfix_self_consistency/self_consistency_per_example.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/maskfix_passk_selection"),
    )
    args = parser.parse_args()

    ckpts = [int(item) for item in args.ckpts.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sc_by_ckpt = load_sc(args.self_consistency_jsonl)

    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for ckpt in ckpts:
        summary, rows = summarize_checkpoint(ckpt, args.passk_root, args.train_run, sc_by_ckpt)
        summaries.append(summary)
        sample_rows.extend(rows)

    (args.output_dir / "selection_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_markdown(args.output_dir / "selection_summary.md", summaries)

    with (args.output_dir / "selection_per_example.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    hard_cases = [
        row
        for row in sample_rows
        if row["rare_correct_temp0_wrong"] or row["oracle_available_but_temp0_wrong"]
    ]
    hard_cases.sort(key=lambda row: (row["passk_correct_count"], row["ckpt"], row["idx"]))
    with (args.output_dir / "hard_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(hard_cases)

    print(f"wrote {args.output_dir / 'selection_summary.md'}")
    print(f"wrote {args.output_dir / 'selection_per_example.csv'}")
    print(f"wrote {args.output_dir / 'hard_cases.csv'}")


if __name__ == "__main__":
    main()
