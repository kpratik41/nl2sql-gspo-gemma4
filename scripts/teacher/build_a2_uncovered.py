#!/usr/bin/env python3
"""Build the A2-uncovered target-id file for A2b.

Given the original all-wrong teacher target ids and one or more A2/A2b
``teacher_traces.jsonl`` files, this writes the ids that still do not have any
kept trace.  A2b should run on these uncovered ids instead of re-sampling
targets already recovered by greedy A2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write all-wrong target ids not recovered by teacher traces."
    )
    parser.add_argument(
        "--all-wrong-ids",
        type=Path,
        default=Path("outputs/teacher/target_idx_all_wrong.json"),
        help="Original all-wrong target ids, JSON list or newline text.",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        default=[Path("outputs/teacher/a2_greedy_tp2_shards4/merged/teacher_traces.jsonl")],
        help="One or more teacher_traces.jsonl files to treat as recovered sources.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/teacher/target_idx_all_wrong_a2_uncovered"),
        help="Output path prefix. Writes .txt, .json, and .summary.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    return parser.parse_args()


def load_ids(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return sorted(int(x) for x in json.loads(text))
    ids: List[int] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: expected integer source_idx") from exc
    return sorted(ids)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def recovered_ids(trace_paths: Sequence[Path]) -> Set[int]:
    recovered: Set[int] = set()
    for path in trace_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            if row.get("kept"):
                recovered.add(int(row["idx"]))
    return recovered


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


def main() -> None:
    args = parse_args()
    all_wrong = load_ids(args.all_wrong_ids)
    all_wrong_set = set(all_wrong)
    recovered = recovered_ids(args.traces)
    recovered_in_scope = sorted(all_wrong_set & recovered)
    uncovered = sorted(all_wrong_set - recovered)

    output_prefix = args.output_prefix
    write_text(
        output_prefix.with_suffix(".txt"),
        "".join(f"{idx}\n" for idx in uncovered),
        args.overwrite,
    )
    write_json(output_prefix.with_suffix(".json"), uncovered, args.overwrite)
    write_json(
        output_prefix.with_suffix(".summary.json"),
        {
            "all_wrong_ids": str(args.all_wrong_ids),
            "traces": [str(path) for path in args.traces],
            "n_all_wrong": len(all_wrong),
            "n_recovered_in_scope": len(recovered_in_scope),
            "n_uncovered": len(uncovered),
            "coverage_rate": len(recovered_in_scope) / max(1, len(all_wrong)),
            "output_txt": str(output_prefix.with_suffix(".txt")),
            "output_json": str(output_prefix.with_suffix(".json")),
        },
        args.overwrite,
    )

    print(
        "[a2-uncovered] "
        f"all_wrong={len(all_wrong)} recovered={len(recovered_in_scope)} "
        f"uncovered={len(uncovered)}"
    )
    print(f"[a2-uncovered] wrote {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
