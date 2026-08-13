#!/usr/bin/env python3
"""A3: turn verified teacher traces into hint-free student RFT records.

Input:
  - ``teacher_traces.jsonl`` from A2 (gen_gold_conditioned_traces.py)
  - the ORIGINAL bare-tool train file (hint-free system+user prompt, gold_sql)

For every teacher trace with ``kept == True`` (verified vs gold AND no text
leakage), this builds a student SFT/RFT record whose:
  - ``prompt`` = the ORIGINAL hint-free [system, user] (exactly what the student
    sees at test time - the privileged reference is NOT here),
  - ``messages`` = prompt + the teacher's structured assistant/tool transcript.

The privileged hint therefore never appears in training data. Copy/transcription
traces are kept by default (coverage > elegance for the all-wrong band) but can
be dropped with ``--drop_copy_first_call`` to produce a reasoning-only variant.
Tool-response turns (role="tool") are meant to be MASKED during SFT loss.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_train_rows(train_file: str) -> Dict[int, Dict[str, Any]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    with open(train_file, "r", encoding="utf-8") as handle:
        for source_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_idx[int(row.get("source_idx", source_idx))] = row
    return by_idx


def hint_free_prompt(
    train_row: Dict[str, Any],
) -> Optional[List[Dict[str, str]]]:
    """The original system+user prompt (NO privileged reference)."""

    prompt = train_row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return [
            dict(m)
            for m in prompt
            if m.get("role") in ("system", "user")
        ]

    messages = train_row.get("messages")
    if isinstance(messages, list) and messages:
        return [
            dict(m)
            for m in messages
            if m.get("role") in ("system", "user")
        ]

    return None


def pick_best_trace(
    traces: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prefer a genuine-reasoning trace over a pure transcription copy, then
    prefer fewer tool rounds (cleaner), then earliest sample."""

    def key(t: Dict[str, Any]):
        copy = t.get("copy_flags", {}) or {}
        return (
            1 if copy.get("copy_first_call") else 0,  # non-copy first
            int(t.get("tool_rounds", 0)),              # fewer rounds
            int(t.get("sample_id", 0)),
        )

    return sorted(traces, key=key)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build hint-free student RFT records from teacher traces (Path A3)."
    )
    p.add_argument(
        "--teacher_traces",
        required=True,
        nargs="+",
        help="One or more teacher_traces.jsonl files (unioned by idx; best trace per idx kept).",
    )
    p.add_argument(
        "--train_file",
        required=True,
        help="Original bare-tool train jsonl (hint-free prompts).",
    )
    p.add_argument(
        "--output_file",
        required=True,
        help="Output RFT jsonl.",
    )
    p.add_argument(
        "--max_per_idx",
        type=int,
        default=1,
        help="Max kept traces per train idx.",
    )
    p.add_argument(
        "--drop_copy_first_call",
        action="store_true",
        help="Exclude traces whose first executed SQL == gold verbatim (transcription).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    traces: List[Dict[str, Any]] = []
    for trace_path in args.teacher_traces:
        traces.extend(read_jsonl(trace_path))

    train_by_idx = index_train_rows(args.train_file)

    kept = [t for t in traces if t.get("kept")]
    if args.drop_copy_first_call:
        kept = [
            t
            for t in kept
            if not (t.get("copy_flags", {}) or {}).get("copy_first_call")
        ]

    # group kept traces by idx
    by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for t in kept:
        by_idx.setdefault(int(t["idx"]), []).append(t)

    records: List[Dict[str, Any]] = []
    skipped_no_prompt = 0
    skipped_empty_transcript = 0
    copy_counter = Counter()
    db_counter = Counter()

    for idx, group in by_idx.items():
        group = sorted(
            group,
            key=lambda t: (
                1
                if (t.get("copy_flags", {}) or {}).get("copy_first_call")
                else 0,
                int(t.get("tool_rounds", 0)),
                int(t.get("sample_id", 0)),
            ),
        )
        chosen = group[: max(1, args.max_per_idx)]

        train_row = train_by_idx.get(idx)
        if train_row is None:
            skipped_no_prompt += 1
            continue

        prompt = hint_free_prompt(train_row)
        if not prompt:
            skipped_no_prompt += 1
            continue

        for t in chosen:
            transcript = t.get("transcript") or []

            # need at least one assistant turn with a final answer
            assistant_turns = [
                m
                for m in transcript
                if m.get("role") == "assistant"
            ]
            if not assistant_turns:
                skipped_empty_transcript += 1
                continue

            messages = [dict(m) for m in prompt] + [
                {
                    "role": m["role"],
                    "content": m["content"],
                }
                for m in transcript
            ]

            copy = t.get("copy_flags", {}) or {}
            copy_counter["copy_first_call"] += int(
                bool(copy.get("copy_first_call"))
            )
            copy_counter["near_copy"] += int(
                bool(copy.get("near_copy"))
            )
            copy_counter["zero_verify_copy"] += int(
                bool(copy.get("zero_verify_copy"))
            )
            db_counter[t.get("db_id", "")] += 1

            records.append(
                {
                    "db_id": train_row.get(
                        "db_id",
                        t.get("db_id", ""),
                    ),
                    "gold_sql": train_row.get(
                        "gold_sql",
                        t.get("gold_sql", ""),
                    ),
                    "evidence": train_row.get("evidence", ""),
                    "question": train_row.get(
                        "question",
                        t.get("question", ""),
                    ),
                    "tools": train_row.get("tools"),
                    "prompt": prompt,
                    "messages": messages,
                    "source_idx": idx,
                    "teacher_final_sql": t.get("final_sql", ""),
                    "teacher_tool_rounds": t.get("tool_rounds", 0),
                    "copy_flags": copy,
                }
            )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    summary = {
        "teacher_traces": args.teacher_traces,
        "train_file": args.train_file,
        "output_file": str(out_path),
        "n_kept_traces_input": len(kept),
        "n_unique_idx": len(by_idx),
        "n_rft_records": n,
        "max_per_idx": args.max_per_idx,
        "drop_copy_first_call": args.drop_copy_first_call,
        "skipped_no_prompt": skipped_no_prompt,
        "skipped_empty_transcript": skipped_empty_transcript,
        "copy_counts": dict(copy_counter),
        "copy_first_call_rate": (
            copy_counter["copy_first_call"] / n
        ) if n else 0.0,
        "distinct_db_ids": len(db_counter),
    }

    (out_path.parent / "rft_build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("rft dataset ->", out_path)


if __name__ == "__main__":
    main()