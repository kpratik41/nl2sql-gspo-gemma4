#!/usr/bin/env python3
"""Stage A3 — assemble hint-free student RFT records from verified traces.

Inputs are one or more ``teacher_traces.jsonl`` files produced by
``gen_teacher_traces.py`` (A2 greedy, A2b sampled, A3a self-traces). All the
rejection already happened during generation: only samples with ``kept == true``
(verified against gold **and** free of hard leaks) are considered here.

This stage does three things:

1. **Deduplicate.** Generation produces many verified traces for the same
   question — around 7 on average. Training on all of them would show easy
   questions up to 9 times and hard ones once. Keep ``--max_per_idx`` per id.
2. **Rank.** Prefer a trace that did not simply copy gold, then fewer tool
   rounds, then the earliest sample.
3. **Reformat.** Emit ``{prompt, messages, metadata}`` where ``prompt`` is the
   original hint-free system/user turns and ``messages`` is that prompt followed
   by the assistant/tool transcript.

A final leak scan runs over the assembled records as a last gate before the data
reaches training.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from teacher.teacher_hint import detect_leaks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="One or more teacher_traces.jsonl files (A2, A2b, A3a).",
    )
    parser.add_argument("--output", required=True, help="Output RFT JSONL path.")
    parser.add_argument("--summary", default=None, help="Defaults to <output dir>/rft_build_summary.json.")
    parser.add_argument("--max_per_idx", type=int, default=1)
    parser.add_argument(
        "--drop_copy_first_call",
        action="store_true",
        help="Exclude traces whose first tool call already contained gold SQL (reasoning-only dataset).",
    )
    parser.add_argument(
        "--require_tool_rounds",
        type=int,
        default=1,
        help="Drop traces with fewer than this many tool rounds. 0 disables the check.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_key(trace: Dict[str, Any]) -> tuple:
    """Lower is better: non-copy first, then fewer tool rounds, then earliest sample."""
    return (
        1 if trace.get("copy_flags") else 0,
        int(trace.get("tool_rounds", 0)),
        int(trace.get("sample_id", 0)),
    )


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else out_path.parent / "rft_build_summary.json"

    by_idx: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    per_source: Dict[str, int] = {}
    dropped_copy_first = 0
    dropped_few_rounds = 0

    for trace_path in args.traces:
        rows = load_jsonl(Path(trace_path))
        kept_here = 0
        for row in rows:
            if not row.get("kept"):
                continue
            if args.drop_copy_first_call and "copy_first_call" in (row.get("copy_flags") or []):
                dropped_copy_first += 1
                continue
            if args.require_tool_rounds and int(row.get("tool_rounds", 0)) < args.require_tool_rounds:
                dropped_few_rounds += 1
                continue
            row["_source"] = trace_path
            by_idx[int(row["idx"])].append(row)
            kept_here += 1
        per_source[trace_path] = kept_here
        print(f"[rft] {trace_path}: {kept_here} eligible kept traces")

    records: List[Dict[str, Any]] = []
    band_counts: collections.Counter = collections.Counter()
    copy_counts: collections.Counter = collections.Counter()

    for idx in sorted(by_idx):
        candidates = sorted(by_idx[idx], key=rank_key)[: max(1, args.max_per_idx)]
        for trace in candidates:
            band = "teacher" if trace.get("hint_strategy") == "full_sql" else "self"
            band_counts[band] += 1
            copy_counts["copy_free" if not trace.get("copy_flags") else "copy_flagged"] += 1
            records.append(
                {
                    "prompt": trace["prompt"],
                    "messages": list(trace["prompt"]) + list(trace["transcript"]),
                    "db_id": trace.get("db_id", ""),
                    "gold_sql": trace.get("gold_sql", ""),
                    "evidence": trace.get("evidence", ""),
                    "question": trace.get("question", ""),
                    "tools": trace.get("tools") or [],
                    "source_idx": idx,
                    "band": band,
                    "hint_strategy": trace.get("hint_strategy", ""),
                    "teacher_final_sql": trace.get("final_sql", ""),
                    "teacher_tool_rounds": int(trace.get("tool_rounds", 0)),
                    "teacher_tool_calls": int(trace.get("tool_call_count", 0)),
                    "copy_flags": trace.get("copy_flags") or [],
                    "soft_leaks": trace.get("soft_leaks") or [],
                    "sample_id": int(trace.get("sample_id", 0)),
                }
            )

    # Final gate: nothing that reaches training may carry the privileged hint or
    # a hard leak, regardless of what generation-time screening concluded.
    hint_in_prompt = 0
    hard_leak_records = 0
    for record in records:
        if any("internal_reference" in (m.get("content") or "") for m in record["messages"]):
            hint_in_prompt += 1
        assistant_texts = [
            (m.get("content") or "") + " " + (m.get("reasoning") or "")
            for m in record["messages"]
            if m.get("role") == "assistant"
        ]
        if detect_leaks(assistant_texts, record["gold_sql"])["hard"]:
            hard_leak_records += 1

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    rounds = [r["teacher_tool_rounds"] for r in records]
    summary = {
        "traces": list(args.traces),
        "eligible_kept_per_source": per_source,
        "max_per_idx": args.max_per_idx,
        "drop_copy_first_call": args.drop_copy_first_call,
        "require_tool_rounds": args.require_tool_rounds,
        "dropped_copy_first_call": dropped_copy_first,
        "dropped_below_tool_rounds": dropped_few_rounds,
        "distinct_ids": len(by_idx),
        "records": len(records),
        "band_counts": dict(band_counts),
        "copy_counts": dict(copy_counts),
        "mean_tool_rounds": round(sum(rounds) / len(rounds), 3) if rounds else 0.0,
        "final_gate": {
            "records_with_hint_in_messages": hint_in_prompt,
            "records_with_hard_leak": hard_leak_records,
        },
        "output": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if hint_in_prompt or hard_leak_records:
        raise SystemExit("FINAL GATE FAILED: privileged hint or hard leak present in assembled records")
    print(f"[rft] wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
