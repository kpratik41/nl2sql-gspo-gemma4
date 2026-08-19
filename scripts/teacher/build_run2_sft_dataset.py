#!/usr/bin/env python3
"""Build Run 2 SFT/RFT JSONL from pass@16 student traces and teacher traces."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nl2sql_gspo.inference_tool_executor import (  # noqa: E402
    extract_tool_calls,
    strip_decoded_thought_prefix,
    text_before_first_tool_call,
)
from teacher.teacher_hint import detect_text_leakage  # noqa: E402


def detect_leaks(
    assistant_texts,
    gold_sql: str,
    **_kwargs,
):
    """Compat shim for the current ``teacher_hint`` API.

    The old ``detect_leaks`` returned ``{"hard": [...], "soft": [...]}``; the
    current module exposes hard reasons only, as a flat list. Both call sites
    below read ``["hard"]`` for diagnostic counters, so the soft bucket is
    reported empty rather than reconstructed.
    """
    return {"hard": detect_text_leakage(assistant_texts, gold_sql), "soft": []}

TOOL_RESPONSE_RE = re.compile(
    r"<\|tool_response\>response:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{value:(?P<value>.*?)\}<tool_response\|>",
    re.DOTALL,
)
STRICT_FORBIDDEN_PHRASE_RE = re.compile(
    r"\b("
    r"internal\s+reference|internal_reference|do[_\s-]?not[_\s-]?reveal|"
    r"gold\s+(sql|query|answer|standard)|ground[\s_-]?truth|"
    r"reference\s+(sql|query|solution|answer)"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_jsonl", default="outputs/train-6601-schema-bare-tool.jsonl")
    parser.add_argument(
        "--passk_per_example",
        default="outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp2_shards4/merged/passk_per_example.jsonl",
    )
    parser.add_argument(
        "--passk_candidates",
        default="outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp2_shards4/merged/passk_candidates.jsonl",
    )
    parser.add_argument(
        "--a2_traces",
        default="outputs/teacher/a2_greedy_tp2_shards4/merged/teacher_traces.jsonl",
    )
    parser.add_argument(
        "--a2b_traces",
        default="outputs/teacher/a2b_uncovered_tp2_shards4/merged/teacher_traces.jsonl",
    )
    parser.add_argument("--output_dir", default="outputs/teacher/rft_run2")
    parser.add_argument("--all_correct_count", type=int, default=1394)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_source_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for position, row in enumerate(read_jsonl(path)):
        rows[int(row.get("source_idx", position))] = row
    return rows


def load_passk_bands(path: Path) -> Tuple[List[int], List[int], List[int]]:
    mixed: List[int] = []
    all_correct: List[int] = []
    all_wrong: List[int] = []
    for row in read_jsonl(path):
        idx = int(row["idx"])
        n_correct = int(row["num_correct"])
        if n_correct == 0:
            all_wrong.append(idx)
        elif n_correct == 16:
            all_correct.append(idx)
        else:
            mixed.append(idx)
    return mixed, all_correct, all_wrong


def parse_tool_response(match: re.Match[str]) -> Dict[str, Any]:
    name = match.group("name")
    value_text = match.group("value")
    try:
        value = json.loads(value_text)
    except json.JSONDecodeError:
        value = {"error": "parse_tool_response_failed", "raw": value_text}
    return {"name": name, "response": {"value": value}}


def strip_leading_thought_label(text: str) -> str:
    """Drop decoded Gemma thought-channel label before storing structured reasoning."""
    return strip_decoded_thought_prefix(text)


def collapse_leading_thought_labels(text: str) -> str:
    """Keep one decoded thought label in plain assistant content."""
    stripped = (text or "").strip()
    if not (stripped == "thought" or stripped.startswith("thought\n")):
        return stripped
    body = strip_leading_thought_label(stripped)
    return "thought" if not body else f"thought\n{body}"


def prediction_text_to_transcript(text: str) -> List[Dict[str, Any]]:
    transcript: List[Dict[str, Any]] = []
    cursor = 0
    for match in TOOL_RESPONSE_RE.finditer(text or ""):
        assistant_text = (text or "")[cursor : match.start()].strip()
        tool_calls = extract_tool_calls(assistant_text)
        if tool_calls:
            reasoning = strip_leading_thought_label(text_before_first_tool_call(assistant_text))
            transcript.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": reasoning,
                    "tool_calls": [
                        {
                            "id": call.get("id") or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": (call.get("function") or {}).get("name", ""),
                                "arguments": (call.get("function") or {}).get("arguments") or {},
                            },
                        }
                        for index, call in enumerate(tool_calls)
                    ],
                    "tool_responses": [parse_tool_response(match)],
                }
            )
        elif assistant_text:
            transcript.append({"role": "assistant", "content": assistant_text})
        cursor = match.end()

    tail = (text or "")[cursor:].strip()
    if tail:
        transcript.append({"role": "assistant", "content": tail})
    return transcript


def normalize_structured_transcript(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls") and isinstance(item.get("reasoning"), str):
            item["reasoning"] = strip_leading_thought_label(item["reasoning"])
        if item.get("role") == "assistant" and not item.get("tool_calls") and isinstance(item.get("content"), str):
            item["content"] = collapse_leading_thought_labels(item["content"])
        normalized.append(item)
    return normalized


def passk_rank_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    text = row.get("prediction_text") or ""
    transcript = prediction_text_to_transcript(text)
    malformed = int(not transcript or not row.get("pred_sql"))
    return (
        int(row.get("stop_reason") != "finished"),
        malformed,
        -int(row.get("tool_rounds", 0) >= 1),
        int(row.get("tool_rounds", 0)),
        int(row.get("completion_token_count", 0)),
    )


def teacher_rank_key(row: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        int(bool(row.get("copy_flags"))),
        int(row.get("tool_rounds", 0)),
        int(row.get("sample_id", 0)),
    )


def make_passk_record(
    candidate: Dict[str, Any],
    source_row: Dict[str, Any],
    band: str,
    source_path: str,
) -> Dict[str, Any]:
    transcript = prediction_text_to_transcript(candidate.get("prediction_text") or "")
    idx = int(candidate["idx"])
    return {
        "prompt": source_row["prompt"],
        "messages": list(source_row["prompt"]) + normalize_structured_transcript(transcript),
        "db_id": candidate.get("db_id", source_row.get("db_id", "")),
        "gold_sql": candidate.get("gold_sql", source_row.get("gold_sql", "")),
        "evidence": source_row.get("evidence", ""),
        "question": source_row.get("question", ""),
        "tools": source_row.get("tools") or [],
        "source_idx": idx,
        "band": band,
        "hint_strategy": "none",
        "teacher_final_sql": candidate.get("pred_sql", ""),
        "teacher_tool_rounds": int(candidate.get("tool_rounds", 0)),
        "teacher_tool_calls": int(candidate.get("tool_call_count", 0)),
        "copy_flags": [],
        "soft_leaks": [],
        "sample_id": int(candidate.get("sample_id", 0)),
        "source_trace_file": source_path,
        "source_trace_type": "passk_student",
    }


def make_teacher_record(trace: Dict[str, Any], band: str, source_path: str) -> Dict[str, Any]:
    transcript = normalize_structured_transcript(trace["transcript"])
    return {
        "prompt": trace["prompt"],
        "messages": list(trace["prompt"]) + transcript,
        "db_id": trace.get("db_id", ""),
        "gold_sql": trace.get("gold_sql", ""),
        "evidence": trace.get("evidence", ""),
        "question": trace.get("question", ""),
        "tools": trace.get("tools") or [],
        "source_idx": int(trace["idx"]),
        "band": band,
        "hint_strategy": trace.get("hint_strategy", ""),
        "teacher_final_sql": trace.get("final_sql", ""),
        "teacher_tool_rounds": int(trace.get("tool_rounds", 0)),
        "teacher_tool_calls": int(trace.get("tool_call_count", 0)),
        "copy_flags": trace.get("copy_flags") or [],
        "soft_leaks": trace.get("soft_leaks") or [],
        "sample_id": int(trace.get("sample_id", 0)),
        "source_trace_file": source_path,
        "source_trace_type": "teacher",
    }


def record_text(record: Dict[str, Any]) -> str:
    return "\n".join(
        (message.get("content") or "") + "\n" + (message.get("reasoning") or "")
        for message in record.get("messages") or []
    )


def split_strict_phrase_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for record in records:
        if STRICT_FORBIDDEN_PHRASE_RE.search(record_text(record)):
            removed.append(
                {
                    "band": record.get("band", ""),
                    "source_idx": record.get("source_idx"),
                    "source_trace_file": record.get("source_trace_file", ""),
                }
            )
        else:
            clean.append(record)
    return clean, removed


def select_passk_records(
    candidates_path: Path,
    source_rows: Dict[int, Dict[str, Any]],
    mixed_ids: List[int],
    all_correct_ids: List[int],
    all_correct_count: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    selected_all_correct_ids = sorted(rng.sample(sorted(all_correct_ids), all_correct_count))
    targets = {idx: "mixed_pass16" for idx in mixed_ids}
    targets.update({idx: "all_correct_pass16" for idx in selected_all_correct_ids})

    candidates_by_idx: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    passk_hard_phrase_candidates = 0
    for row in read_jsonl(candidates_path):
        idx = int(row["idx"])
        if idx not in targets or int(row.get("correct", 0)) != 1:
            continue
        leaks = detect_leaks([row.get("prediction_text") or ""], row.get("gold_sql", ""))
        if leaks["hard"]:
            passk_hard_phrase_candidates += 1
        candidates_by_idx[idx].append(row)

    mixed_records: List[Dict[str, Any]] = []
    all_correct_records: List[Dict[str, Any]] = []
    missing: List[int] = []
    for idx in sorted(targets):
        rows = candidates_by_idx.get(idx) or []
        if not rows:
            missing.append(idx)
            continue
        chosen = sorted(rows, key=passk_rank_key)[0]
        record = make_passk_record(chosen, source_rows[idx], targets[idx], str(candidates_path))
        if targets[idx] == "mixed_pass16":
            mixed_records.append(record)
        else:
            all_correct_records.append(record)

    summary = {
        "mixed_target_ids": len(mixed_ids),
        "all_correct_available_ids": len(all_correct_ids),
        "all_correct_selected_ids": len(selected_all_correct_ids),
        "mixed_records": len(mixed_records),
        "all_correct_records": len(all_correct_records),
        "missing_selected_ids": missing,
        "passk_hard_phrase_candidates": passk_hard_phrase_candidates,
        "selection_seed": seed,
    }
    if missing:
        raise SystemExit(f"missing selected pass@16 records for ids: {missing[:20]}")
    return mixed_records, all_correct_records, summary


def select_teacher_records(a2_path: Path, a2b_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    a2_by_idx: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in read_jsonl(a2_path):
        if row.get("kept"):
            a2_by_idx[int(row["idx"])].append(row)
    a2_records = [
        make_teacher_record(sorted(rows, key=teacher_rank_key)[0], "a2_teacher", str(a2_path))
        for _, rows in sorted(a2_by_idx.items())
    ]

    a2b_by_idx: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in read_jsonl(a2b_path):
        if row.get("kept"):
            a2b_by_idx[int(row["idx"])].append(row)
    a2b_records = [
        make_teacher_record(sorted(rows, key=teacher_rank_key)[0], "a2b_teacher", str(a2b_path))
        for _, rows in sorted(a2b_by_idx.items())
    ]

    overlap = sorted({r["source_idx"] for r in a2_records} & {r["source_idx"] for r in a2b_records})
    if overlap:
        raise SystemExit(f"A2b was expected to target uncovered ids, but overlap exists: {overlap[:20]}")

    summary = {
        "a2_records": len(a2_records),
        "a2b_records": len(a2b_records),
        "a2_a2b_overlap": len(overlap),
    }
    return a2_records, a2b_records, summary


def final_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    hint_in_messages = 0
    hard_leak_records = 0
    strict_forbidden_phrase_records = 0
    empty_messages = 0
    band_counts: collections.Counter[str] = collections.Counter()
    for record in records:
        band_counts[record["band"]] += 1
        messages = record.get("messages") or []
        if not messages:
            empty_messages += 1
        if any("internal_reference" in (message.get("content") or "") for message in messages):
            hint_in_messages += 1
        text = record_text(record)
        if STRICT_FORBIDDEN_PHRASE_RE.search(text):
            strict_forbidden_phrase_records += 1
        assistant_texts = [(message.get("content") or "") + " " + (message.get("reasoning") or "") for message in messages if message.get("role") == "assistant"]
        if record.get("hint_strategy") == "full_sql" and detect_leaks(assistant_texts, record.get("gold_sql", ""))["hard"]:
            hard_leak_records += 1
    summary = {
        "records": len(records),
        "distinct_ids": len({int(record["source_idx"]) for record in records}),
        "band_counts": dict(band_counts),
        "records_with_internal_reference": hint_in_messages,
        "records_with_hard_leak": hard_leak_records,
        "records_with_empty_messages": empty_messages,
        "strict_forbidden_phrase_records": strict_forbidden_phrase_records,
    }
    if hint_in_messages or hard_leak_records or strict_forbidden_phrase_records or empty_messages:
        raise SystemExit(f"final gate failed: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_source_rows(Path(args.source_jsonl))
    mixed_ids, all_correct_ids, all_wrong_ids = load_passk_bands(Path(args.passk_per_example))

    mixed_records, all_correct_records, passk_summary = select_passk_records(
        Path(args.passk_candidates),
        source_rows,
        mixed_ids,
        all_correct_ids,
        args.all_correct_count,
        args.seed,
    )
    a2_records, a2b_records, teacher_summary = select_teacher_records(Path(args.a2_traces), Path(args.a2b_traces))

    mixed_records, removed_mixed = split_strict_phrase_records(mixed_records)
    all_correct_records, removed_all_correct = split_strict_phrase_records(all_correct_records)
    a2_records, removed_a2 = split_strict_phrase_records(a2_records)
    a2b_records, removed_a2b = split_strict_phrase_records(a2b_records)

    removed_records = removed_mixed + removed_all_correct + removed_a2 + removed_a2b
    removed_counts = collections.Counter(record["band"] for record in removed_records)

    student_records = mixed_records + all_correct_records
    teacher_records = a2_records + a2b_records
    combined_sorted = student_records + teacher_records
    combined_shuffled = list(combined_sorted)
    random.Random(args.shuffle_seed).shuffle(combined_shuffled)

    paths = {
        "mixed": output_dir / "run2_mixed_pass16_records.jsonl",
        "all_correct": output_dir / "run2_all_correct_pass16_records.jsonl",
        "student": output_dir / "run2_student_pass16_records.jsonl",
        "teacher": output_dir / "run2_teacher_records.jsonl",
        "combined_sorted": output_dir / "train_rft_31b_run2.sorted.jsonl",
        "combined_shuffled": output_dir / "train_rft_31b_run2.shuffled.jsonl",
        "summary": output_dir / "run2_build_summary.json",
        "strict_phrase_removed": output_dir / "run2_strict_phrase_removed_records.json",
    }

    counts = {
        "mixed": write_jsonl(paths["mixed"], mixed_records),
        "all_correct": write_jsonl(paths["all_correct"], all_correct_records),
        "student": write_jsonl(paths["student"], student_records),
        "teacher": write_jsonl(paths["teacher"], teacher_records),
        "combined_sorted": write_jsonl(paths["combined_sorted"], combined_sorted),
        "combined_shuffled": write_jsonl(paths["combined_shuffled"], combined_shuffled),
    }

    gate_summary = final_gate(combined_shuffled)
    summary = {
        "inputs": {
            "source_jsonl": args.source_jsonl,
            "passk_per_example": args.passk_per_example,
            "passk_candidates": args.passk_candidates,
            "a2_traces": args.a2_traces,
            "a2b_traces": args.a2b_traces,
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "passk_bands": {
            "mixed": len(mixed_ids),
            "all_correct": len(all_correct_ids),
            "all_wrong": len(all_wrong_ids),
        },
        "counts": counts,
        "passk_selection": passk_summary,
        "teacher_selection": teacher_summary,
        "shuffle_seed": args.shuffle_seed,
        "final_gate": gate_summary,
        "strict_phrase_filter": {
            "enabled": True,
            "removed_records": len(removed_records),
            "removed_by_band": dict(removed_counts),
            "pattern": STRICT_FORBIDDEN_PHRASE_RE.pattern,
            "removed_examples_path": str(paths["strict_phrase_removed"]),
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["strict_phrase_removed"].write_text(json.dumps(removed_records, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
