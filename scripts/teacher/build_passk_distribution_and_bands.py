#!/usr/bin/env python3
"""Build pass@16 distribution artifacts and SFT target bands.

This is the small replacement for the deleted "target index" stage.  It assumes
pass@k has already been run on the hint-free / bare-tool training file and reads
the merged ``passk_per_example.jsonl`` produced by ``scripts/run_passk_bird.py``.

Outputs:
  - passk16_distribution.json / .csv / .md
  - passk16_distribution_by_db.csv
  - target_idx_all_wrong.{txt,json}
  - target_idx_mixed.{txt,json}
  - target_idx_all_correct_capped.{txt,json}
  - target_idx_selftrace.{txt,json}
  - target_idx_summary.json

The all-wrong teacher target excludes examples whose gold SQL failed execution
when a merged ``passk_candidates.jsonl`` is available, because those examples
cannot be verified against gold during teacher-trace generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 0..K pass@K correctness distribution and split train "
            "examples into all-wrong, mixed, and capped all-correct bands."
        )
    )
    parser.add_argument(
        "--passk-dir",
        type=Path,
        default=None,
        help=(
            "Merged pass@k directory, or the pass@k root containing merged/. "
            "Used to find passk_per_example.jsonl and passk_candidates.jsonl."
        ),
    )
    parser.add_argument(
        "--per-example",
        type=Path,
        default=None,
        help="Explicit path to merged passk_per_example.jsonl.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to merged passk_candidates.jsonl. If present, "
            "gold-SQL execution failures are excluded from teacher all-wrong ids."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/teacher"),
        help="Directory where distribution artifacts and target id files are written.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Expected number of sampled candidates per example.",
    )
    parser.add_argument(
        "--all-correct-cap-per-db",
        type=int,
        default=16,
        help=(
            "Maximum number of 16/K all-correct examples kept per database for "
            "the self-trace anchor. Use -1 to disable the cap."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic per-database all-correct sampling.",
    )
    parser.add_argument(
        "--keep-gold-failures",
        action="store_true",
        help=(
            "Do not drop all-wrong examples whose gold SQL failed execution. "
            "Normally leave this off for teacher-trace generation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def resolve_passk_paths(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Path]]:
    per_example = args.per_example
    candidates = args.candidates
    passk_dir = args.passk_dir

    if passk_dir is not None:
        if (passk_dir / "passk_per_example.jsonl").exists():
            merged_dir = passk_dir
        elif (passk_dir / "merged" / "passk_per_example.jsonl").exists():
            merged_dir = passk_dir / "merged"
        else:
            raise FileNotFoundError(
                f"Could not find passk_per_example.jsonl under {passk_dir}"
            )
        per_example = per_example or (merged_dir / "passk_per_example.jsonl")
        candidate_path = merged_dir / "passk_candidates.jsonl"
        if candidates is None and candidate_path.exists():
            candidates = candidate_path
    elif per_example is None:
        raise SystemExit("Provide --passk-dir or --per-example")

    assert per_example is not None
    if not per_example.exists():
        raise FileNotFoundError(per_example)
    if candidates is not None and not candidates.exists():
        raise FileNotFoundError(candidates)

    return per_example, candidates, passk_dir


def int_field(row: Dict[str, Any], key: str, *, default: Optional[int] = None) -> int:
    value = row.get(key, default)
    if value is None:
        raise ValueError(f"missing integer field {key!r} in row: {row}")
    return int(value)


def percent(numer: int, denom: int) -> float:
    return 100.0 * numer / denom if denom else 0.0


def load_gold_failure_ids(candidates_path: Optional[Path]) -> List[int]:
    if candidates_path is None:
        return []

    gold_ok_by_idx: Dict[int, bool] = {}
    for row in read_jsonl(candidates_path):
        idx = int_field(row, "idx")
        gold_ok_by_idx[idx] = gold_ok_by_idx.get(idx, False) or bool(row.get("gold_executed"))

    return sorted(idx for idx, gold_ok in gold_ok_by_idx.items() if not gold_ok)


def choose_capped_all_correct(
    all_correct_rows: Sequence[Dict[str, Any]],
    cap_per_db: int,
    seed: int,
) -> List[int]:
    by_db: Dict[str, List[int]] = defaultdict(list)
    for row in all_correct_rows:
        by_db[str(row.get("db_id", ""))].append(int_field(row, "idx"))

    rng = random.Random(seed)
    chosen: List[int] = []
    for db_id in sorted(by_db):
        ids = sorted(by_db[db_id])
        if cap_per_db >= 0 and len(ids) > cap_per_db:
            ids = sorted(rng.sample(ids, cap_per_db))
        chosen.extend(ids)
    return sorted(chosen)


def write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_ids(output_dir: Path, stem: str, ids: Sequence[int], overwrite: bool) -> None:
    ids = sorted(int(idx) for idx in ids)
    write_text(output_dir / f"{stem}.txt", "".join(f"{idx}\n" for idx in ids), overwrite)
    write_json(output_dir / f"{stem}.json", ids, overwrite)


def build_distribution_rows(hist: Counter, total: int, num_generations: int) -> List[Dict[str, Any]]:
    cumulative = 0
    rows: List[Dict[str, Any]] = []
    for num_correct in range(num_generations + 1):
        count = int(hist.get(num_correct, 0))
        cumulative += count
        rows.append(
            {
                "correct_out_of_k": num_correct,
                "examples": count,
                "share": percent(count, total),
                "cumulative": percent(cumulative, total),
            }
        )
    return rows


def build_by_db_rows(per_examples: Sequence[Dict[str, Any]], num_generations: int) -> List[Dict[str, Any]]:
    by_db: Dict[str, Counter] = defaultdict(Counter)
    for row in per_examples:
        db_id = str(row.get("db_id", ""))
        num_correct = int_field(row, "num_correct")
        if num_correct == 0:
            bucket = "all_wrong"
        elif num_correct == num_generations:
            bucket = "all_correct"
        else:
            bucket = "mixed"
        by_db[db_id]["examples"] += 1
        by_db[db_id][bucket] += 1

    rows: List[Dict[str, Any]] = []
    for db_id in sorted(by_db):
        counts = by_db[db_id]
        total = int(counts["examples"])
        rows.append(
            {
                "db_id": db_id,
                "examples": total,
                "all_wrong": int(counts["all_wrong"]),
                "mixed": int(counts["mixed"]),
                "all_correct": int(counts["all_correct"]),
                "all_wrong_share": percent(int(counts["all_wrong"]), total),
                "mixed_share": percent(int(counts["mixed"]), total),
                "all_correct_share": percent(int(counts["all_correct"]), total),
            }
        )
    return rows


def markdown_report(
    summary: Dict[str, Any],
    distribution_rows: Sequence[Dict[str, Any]],
    by_db_rows: Sequence[Dict[str, Any]],
) -> str:
    k = int(summary["num_generations"])
    lines = [
        f"# pass@{k} Training Distribution and Target Bands",
        "",
        "## Inputs",
        "",
        f"- per_example: `{summary['inputs']['per_example']}`",
        f"- candidates: `{summary['inputs'].get('candidates') or ''}`",
        f"- num_generations: `{summary['num_generations']}`",
        "",
        "## Correct-Candidate Distribution",
        "",
        f"| correct out of {k} | examples | share | cumulative |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in distribution_rows:
        lines.append(
            f"| {row['correct_out_of_k']} | `{row['examples']}` | "
            f"`{row['share']:.2f}%` | `{row['cumulative']:.2f}%` |"
        )

    buckets = summary["bands"]
    lines.extend(
        [
            "",
            "## SFT Target Bands",
            "",
            "| band | examples | output | note |",
            "| --- | ---: | --- | --- |",
            (
                f"| all-wrong teacher | `{buckets['all_wrong_teacher']['count']}` | "
                "`target_idx_all_wrong.{txt,json}` | "
                f"0/{k}, excluding gold execution failures unless requested |"
            ),
            (
                f"| mixed self-trace | `{buckets['mixed']['count']}` | "
                f"`target_idx_mixed.{{txt,json}}` | 1..{k - 1}/{k} |"
            ),
            (
                f"| all-correct capped | `{buckets['all_correct_capped']['count']}` | "
                "`target_idx_all_correct_capped.{txt,json}` | "
                f"{k}/{k}, cap per DB = `{summary['all_correct_cap_per_db']}` |"
            ),
            (
                f"| combined self-trace | `{buckets['selftrace']['count']}` | "
                "`target_idx_selftrace.{txt,json}` | mixed + capped all-correct |"
            ),
            "",
            "## DAPO Buckets Before Capping",
            "",
            "| bucket | examples | share |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in ("all_wrong_raw", "mixed", "all_correct_raw"):
        bucket = buckets[key]
        lines.append(f"| {bucket['label']} | `{bucket['count']}` | `{bucket['share']:.2f}%` |")

    lines.extend(
        [
            "",
            "## Databases With Most Mixed Examples",
            "",
            "| db_id | examples | mixed | mixed share |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(by_db_rows, key=lambda item: (-item["mixed"], item["db_id"]))[:10]:
        lines.append(
            f"| `{row['db_id']}` | `{row['examples']}` | `{row['mixed']}` | "
            f"`{row['mixed_share']:.2f}%` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    per_example_path, candidates_path, passk_dir = resolve_passk_paths(args)
    per_examples = read_jsonl(per_example_path)
    if not per_examples:
        raise ValueError(f"{per_example_path} contained no examples")

    rows_by_idx = {int_field(row, "idx"): row for row in per_examples}
    if len(rows_by_idx) != len(per_examples):
        raise ValueError("Duplicate idx values found in passk_per_example.jsonl")

    total = len(per_examples)
    hist = Counter(int_field(row, "num_correct") for row in per_examples)
    invalid = sorted(k for k in hist if k < 0 or k > args.num_generations)
    if invalid:
        raise ValueError(f"num_correct outside 0..{args.num_generations}: {invalid}")

    all_wrong_raw = sorted(int_field(row, "idx") for row in per_examples if int_field(row, "num_correct") == 0)
    mixed = sorted(
        int_field(row, "idx")
        for row in per_examples
        if 0 < int_field(row, "num_correct") < args.num_generations
    )
    all_correct_rows = [
        row for row in per_examples if int_field(row, "num_correct") == args.num_generations
    ]
    all_correct_raw = sorted(int_field(row, "idx") for row in all_correct_rows)
    all_correct_capped = choose_capped_all_correct(
        all_correct_rows,
        cap_per_db=args.all_correct_cap_per_db,
        seed=args.seed,
    )

    gold_failure_ids = load_gold_failure_ids(candidates_path)
    gold_failure_set = set(gold_failure_ids)
    if args.keep_gold_failures:
        all_wrong_teacher = all_wrong_raw
    else:
        all_wrong_teacher = [idx for idx in all_wrong_raw if idx not in gold_failure_set]

    selftrace = sorted(set(mixed) | set(all_correct_capped))
    distribution_rows = build_distribution_rows(hist, total, args.num_generations)
    by_db_rows = build_by_db_rows(per_examples, args.num_generations)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_json = {
        "inputs": {
            "passk_dir": str(passk_dir) if passk_dir else None,
            "per_example": str(per_example_path),
            "candidates": str(candidates_path) if candidates_path else None,
        },
        "num_generations": args.num_generations,
        "total_examples": total,
        "distribution": distribution_rows,
        "by_db_csv": str(output_dir / "passk16_distribution_by_db.csv"),
        "candidate_correct_total": sum(k * int(v) for k, v in hist.items()),
        "candidate_count_expected": total * args.num_generations,
        "candidate_accuracy_from_distribution": percent(
            sum(k * int(v) for k, v in hist.items()),
            total * args.num_generations,
        ),
        "pass_at_16_oracle": percent(total - int(hist.get(0, 0)), total),
        "pass_at_k_oracle": percent(total - int(hist.get(0, 0)), total),
        "all_correct_cap_per_db": args.all_correct_cap_per_db,
        "seed": args.seed,
        "gold_execution_failures": {
            "count": len(gold_failure_ids),
            "ids": gold_failure_ids,
        },
        "bands": {
            "all_wrong_raw": {
                "label": f"all-wrong (0/{args.num_generations})",
                "count": len(all_wrong_raw),
                "share": percent(len(all_wrong_raw), total),
            },
            "all_wrong_teacher": {
                "label": "all-wrong teacher targets",
                "count": len(all_wrong_teacher),
                "share": percent(len(all_wrong_teacher), total),
                "excluded_gold_failure_count": len(set(all_wrong_raw) & gold_failure_set),
            },
            "mixed": {
                "label": f"mixed (1..{args.num_generations - 1}/{args.num_generations})",
                "count": len(mixed),
                "share": percent(len(mixed), total),
            },
            "all_correct_raw": {
                "label": f"all-correct ({args.num_generations}/{args.num_generations})",
                "count": len(all_correct_raw),
                "share": percent(len(all_correct_raw), total),
            },
            "all_correct_capped": {
                "label": "all-correct capped",
                "count": len(all_correct_capped),
                "share": percent(len(all_correct_capped), total),
                "dropped_by_cap": len(all_correct_raw) - len(all_correct_capped),
            },
            "selftrace": {
                "label": "mixed + capped all-correct",
                "count": len(selftrace),
                "share": percent(len(selftrace), total),
            },
        },
        "outputs": {
            "distribution_json": str(output_dir / "passk16_distribution.json"),
            "distribution_csv": str(output_dir / "passk16_distribution.csv"),
            "distribution_md": str(output_dir / "passk16_distribution.md"),
            "distribution_by_db_csv": str(output_dir / "passk16_distribution_by_db.csv"),
            "target_idx_all_wrong": str(output_dir / "target_idx_all_wrong.txt"),
            "target_idx_all_wrong_json": str(output_dir / "target_idx_all_wrong.json"),
            "target_idx_mixed": str(output_dir / "target_idx_mixed.txt"),
            "target_idx_mixed_json": str(output_dir / "target_idx_mixed.json"),
            "target_idx_all_correct_capped": str(output_dir / "target_idx_all_correct_capped.txt"),
            "target_idx_all_correct_capped_json": str(output_dir / "target_idx_all_correct_capped.json"),
            "target_idx_selftrace": str(output_dir / "target_idx_selftrace.txt"),
            "target_idx_selftrace_json": str(output_dir / "target_idx_selftrace.json"),
            "target_idx_summary": str(output_dir / "target_idx_summary.json"),
        },
    }

    write_json(output_dir / "passk16_distribution.json", distribution_json, args.overwrite)
    write_csv(
        output_dir / "passk16_distribution.csv",
        distribution_rows,
        ["correct_out_of_k", "examples", "share", "cumulative"],
        args.overwrite,
    )
    write_csv(
        output_dir / "passk16_distribution_by_db.csv",
        by_db_rows,
        [
            "db_id",
            "examples",
            "all_wrong",
            "mixed",
            "all_correct",
            "all_wrong_share",
            "mixed_share",
            "all_correct_share",
        ],
        args.overwrite,
    )
    write_text(
        output_dir / "passk16_distribution.md",
        markdown_report(distribution_json, distribution_rows, by_db_rows),
        args.overwrite,
    )
    write_ids(output_dir, "target_idx_all_wrong", all_wrong_teacher, args.overwrite)
    write_ids(output_dir, "target_idx_mixed", mixed, args.overwrite)
    write_ids(output_dir, "target_idx_all_correct_capped", all_correct_capped, args.overwrite)
    write_ids(output_dir, "target_idx_selftrace", selftrace, args.overwrite)
    write_json(output_dir / "target_idx_summary.json", distribution_json, args.overwrite)

    print(f"[bands] read {total} examples from {per_example_path}")
    print(
        "[bands] raw buckets: "
        f"all_wrong={len(all_wrong_raw)} mixed={len(mixed)} all_correct={len(all_correct_raw)}"
    )
    print(
        "[bands] targets: "
        f"teacher_all_wrong={len(all_wrong_teacher)} "
        f"all_correct_capped={len(all_correct_capped)} "
        f"selftrace={len(selftrace)}"
    )
    if gold_failure_ids:
        print(
            f"[bands] gold execution failures={len(gold_failure_ids)} "
            f"(excluded from all-wrong teacher targets: {len(set(all_wrong_raw) & gold_failure_set)})"
        )
    print(f"[bands] wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
