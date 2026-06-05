#!/usr/bin/env python3
"""Validate pass@k shard candidate completeness before merge/SC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nl2sql_gspo.prompt_builder import read_json_or_jsonl
from nl2sql_gspo.sample_plan import expand_sample_plan


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_path(shard_dir: Path) -> Path:
    final_path = shard_dir / "passk_candidates_raw.jsonl"
    incremental_path = shard_dir / "passk_candidates_raw.incremental.jsonl"
    if final_path.exists() and final_path.stat().st_size > 0:
        return final_path
    if incremental_path.exists() and incremental_path.stat().st_size > 0:
        return incremental_path
    raise FileNotFoundError(f"No non-empty candidate file found in {shard_dir}")


def source_indices(input_file: str, limit: int) -> List[int]:
    rows = read_json_or_jsonl(input_file)
    if limit >= 0:
        rows = rows[:limit]
    return list(range(len(rows)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--sample_plan", default="")
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards != len(args.shard_dirs):
        raise ValueError(
            f"--num_shards={args.num_shards} but received {len(args.shard_dirs)} shard dirs"
        )

    sample_specs = expand_sample_plan(
        args.sample_plan,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    expected_sample_ids = {int(spec.sample_id) for spec in sample_specs}
    all_indices = source_indices(args.input_file, args.limit)

    seen_global: Set[Tuple[int, int]] = set()
    total_expected = 0
    total_seen = 0

    for shard_index, raw_shard_dir in enumerate(args.shard_dirs):
        shard_dir = Path(raw_shard_dir)
        path = candidate_path(shard_dir)
        rows = read_jsonl(path)
        seen: Set[Tuple[int, int]] = set()
        for row in rows:
            idx = int(row["idx"])
            sample_id = int(row["sample_id"])
            key = (idx, sample_id)
            if key in seen:
                raise ValueError(f"Duplicate candidate in {path}: idx={idx} sample_id={sample_id}")
            if key in seen_global:
                raise ValueError(f"Duplicate candidate across shards: idx={idx} sample_id={sample_id}")
            if idx % args.num_shards != shard_index:
                raise ValueError(
                    f"Candidate idx={idx} is in shard {shard_index}, but idx % {args.num_shards} = {idx % args.num_shards}"
                )
            if sample_id not in expected_sample_ids:
                raise ValueError(
                    f"Unexpected sample_id={sample_id} in {path}; expected {sorted(expected_sample_ids)}"
                )
            seen.add(key)
            seen_global.add(key)

        shard_indices = [idx for idx in all_indices if idx % args.num_shards == shard_index]
        expected = {
            (idx, sample_id)
            for idx in shard_indices
            for sample_id in expected_sample_ids
        }
        missing = sorted(expected - seen)[:10]
        extra = sorted(seen - expected)[:10]
        if missing or extra:
            raise ValueError(
                f"Shard {shard_index} candidate count mismatch for {path}: "
                f"seen={len(seen)} expected={len(expected)} "
                f"missing_preview={missing} extra_preview={extra}"
            )
        print(f"[validate-passk] shard {shard_index}: {len(seen)}/{len(expected)} candidates OK ({path})")
        total_seen += len(seen)
        total_expected += len(expected)

    if total_seen != total_expected:
        raise ValueError(f"Total candidate mismatch: seen={total_seen} expected={total_expected}")
    print(f"[validate-passk] all shards complete: {total_seen}/{total_expected} candidates")


if __name__ == "__main__":
    main()
