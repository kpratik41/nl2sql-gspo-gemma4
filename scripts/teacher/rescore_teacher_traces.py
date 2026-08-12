#!/usr/bin/env python3
"""Re-apply the current teacher trace leak/copy policy to saved traces."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from teacher.teacher_hint import detect_text_leakage, summarize_copy_rate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score teacher_traces.jsonl with the current leakage policy."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--drop-copy-first",
        action="store_true",
        help="Mark exact first-tool-call gold copies as not kept.",
    )
    parser.add_argument(
        "--drop-near-copy",
        action="store_true",
        help="Mark near-copy traces as not kept.",
    )
    parser.add_argument("--overwrite", action="store_true")
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


def assistant_texts(row: Dict[str, Any]) -> List[str]:
    return [
        str(turn.get("content", ""))
        for turn in row.get("transcript", [])
        if turn.get("role") == "assistant"
    ]


def should_keep(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if not row.get("verified"):
        return False
    if row.get("leak_reasons"):
        return False
    flags = row.get("copy_flags") or {}
    if args.drop_copy_first and flags.get("copy_first_call"):
        return False
    if args.drop_near_copy and flags.get("near_copy"):
        return False
    return True


def summarize(rows: List[Dict[str, Any]], elapsed_s: float, args: argparse.Namespace) -> Dict[str, Any]:
    kept_by_idx: Dict[int, bool] = {}
    for row in rows:
        kept_by_idx[int(row["idx"])] = kept_by_idx.get(int(row["idx"]), False) or bool(row.get("kept"))

    kept_flags = [row.get("copy_flags") or {} for row in rows if row.get("kept")]
    leak_counter = Counter()
    for row in rows:
        for reason in row.get("leak_reasons") or []:
            leak_counter[reason] += 1

    return {
        "input": str(args.input),
        "policy": {
            "drop_copy_first": bool(args.drop_copy_first),
            "drop_near_copy": bool(args.drop_near_copy),
        },
        "n_targets": len({int(row["idx"]) for row in rows}),
        "n_samples_total": len(rows),
        "n_verified_samples": sum(1 for row in rows if row.get("verified")),
        "n_kept_samples": sum(1 for row in rows if row.get("kept")),
        "n_leaked_samples": sum(1 for row in rows if row.get("leak_reasons")),
        "targets_with_kept_trace": sum(1 for value in kept_by_idx.values() if value),
        "target_coverage_rate": sum(1 for value in kept_by_idx.values() if value) / max(1, len(kept_by_idx)),
        "copy_rate_over_kept": summarize_copy_rate(kept_flags),
        "leak_reasons": dict(leak_counter),
        "stop_reasons": dict(Counter(row.get("stop_reason", "") for row in rows)),
        "elapsed_s": round(elapsed_s, 1),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists and is non-empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = read_jsonl(args.input)
    for row in rows:
        row["leak_reasons"] = detect_text_leakage(
            assistant_texts(row),
            str(row.get("gold_sql", "")),
        )
        row["kept"] = should_keep(row, args)

    summary = summarize(rows, time.time() - t0, args)

    traces_path = args.output_dir / "teacher_traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (int(item["idx"]), int(item["sample_id"]))):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"traces -> {traces_path}")


if __name__ == "__main__":
    main()
