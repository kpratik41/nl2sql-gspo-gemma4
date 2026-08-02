#!/usr/bin/env python3
"""Privileged-hint injection, leak detection and copy detection for Stage A2.

The teacher is shown the gold SQL as an internal reference it must not reveal.
Anything it produces is then screened before the trace is allowed into the SFT
set:

* **Leaks** are split into HARD and SOFT. Only HARD leaks reject a trace. A
  blanket "any leak drops" rule was tried first in the original pipeline and
  discarded roughly a third of otherwise-usable traces.
* **Copy flags** are recorded but never reject a trace on their own; they let
  Stage A3 prefer reasoned traces over transcribed ones.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

SQL_CODE_BLOCK_RE = re.compile(r"<sql_code>.*?</sql_code>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"call:\w+\s*\{.*?\}", re.IGNORECASE | re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<\|?tool_response\|?>.*?(?:</\|?tool_response\|?>|$)", re.IGNORECASE | re.DOTALL)
CANDIDATE_SQL_RE = re.compile(r"^\s*\w*sql\w*\s*[:=].*$", re.IGNORECASE | re.MULTILINE)
SQL_STATEMENT_RE = re.compile(r"\b(SELECT|WITH)\b.*?(?:;|$)", re.IGNORECASE | re.DOTALL)

# Explicit references to the privileged material. These are unambiguous giveaways.
HARD_LEAK_PATTERNS = [
    re.compile(r"internal_reference", re.IGNORECASE),
    re.compile(r"do_not_reveal", re.IGNORECASE),
    re.compile(r"\bgold\s+(sql|query|answer|standard)\b", re.IGNORECASE),
    re.compile(r"\bground[\s_-]?truth\b", re.IGNORECASE),
    re.compile(r"\b(the|a|this)\s+reference\s+(sql|query|solution|answer)\b", re.IGNORECASE),
    re.compile(r"\breference\s+implementation\b", re.IGNORECASE),
]

# Incidental phrasing that merely talks about expected output. Harmless: a model
# solving the task independently says these things too.
SOFT_LEAK_PATTERNS = [
    re.compile(r"\bexpected\s+(output|result|rows|answer)\b", re.IGNORECASE),
    re.compile(r"\bprovided\b", re.IGNORECASE),
    re.compile(r"\bas\s+expected\b", re.IGNORECASE),
]

PRIVILEGED_INSTRUCTION = (
    "You have been given an internal reference solution purely so you can "
    "self-check your own work. Solve the question independently and show your "
    "own reasoning. You must NEVER quote, mention, paraphrase, cite, or reveal "
    "the existence of the internal reference. Do not say words like "
    "'reference', 'gold', or 'ground truth'. Write your answer exactly as you "
    "would if no reference had been given."
)


def normalize_sql(sql: str) -> str:
    """Whitespace/case/punctuation-insensitive form for similarity comparison."""
    if not sql:
        return ""
    text = sql.strip().rstrip(";")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("`", "").replace('"', "'")
    return text.strip().lower()


def sql_similarity(left: str, right: str) -> float:
    a, b = normalize_sql(left), normalize_sql(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def strip_sql_regions(text: str) -> str:
    """Natural-language prose only.

    Everything the model writes *as SQL* is removed first: ``<sql_code>``
    blocks, ``call:tool{...}`` payloads, tool responses, ``CandidateSQL=``
    scratch-pad lines, and any bare SELECT/WITH statement. Writing SQL that
    resembles gold is the task succeeding, not leaking — it is what the
    ``near_copy`` copy flag exists to record. Only what remains is prose in
    which a mention of the reference would be a genuine giveaway.
    """
    cleaned = text or ""
    for pattern in (SQL_CODE_BLOCK_RE, TOOL_CALL_RE, TOOL_RESPONSE_RE, CANDIDATE_SQL_RE, SQL_STATEMENT_RE):
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


# Back-compat alias.
strip_sql_blocks = strip_sql_regions


def inject_privileged_hint(
    messages: List[Dict[str, str]],
    gold_sql: str,
    strategy: str = "full_sql",
) -> List[Dict[str, str]]:
    """Return a teacher-only copy of ``messages``.

    ``strategy="none"`` returns an unmodified copy, which is how the self-trace
    (A3a) runs harvest student behaviour with the identical loop.
    """
    out = [dict(message) for message in messages]
    if strategy == "none" or not gold_sql:
        return out

    if strategy != "full_sql":
        raise ValueError(f"Unknown hint strategy: {strategy}")

    reference_block = (
        "\n\n<internal_reference_do_not_reveal>\n"
        f"{gold_sql.strip()}\n"
        "</internal_reference_do_not_reveal>"
    )

    system_indexes = [i for i, m in enumerate(out) if m.get("role") == "system"]
    if system_indexes:
        i = system_indexes[0]
        out[i]["content"] = f"{PRIVILEGED_INSTRUCTION}\n\n{out[i].get('content', '')}"
    else:
        out.insert(0, {"role": "system", "content": PRIVILEGED_INSTRUCTION})

    user_indexes = [i for i, m in enumerate(out) if m.get("role") == "user"]
    if user_indexes:
        i = user_indexes[-1]
        out[i]["content"] = f"{out[i].get('content', '')}{reference_block}"
    else:
        out.append({"role": "user", "content": reference_block.strip()})

    return out


def detect_leaks(
    assistant_texts: Sequence[str],
    gold_sql: str,
    gold_tables: Optional[Sequence[str]] = None,
    gold_columns: Optional[Sequence[str]] = None,
    prose_identifier_threshold: int = 2,
    near_copy_threshold: float = 0.95,
    enable_identifier_heuristic: bool = False,
) -> Dict[str, Any]:
    """Classify leakage across every assistant turn.

    Returns ``{"hard": [...], "soft": [...]}``. A trace is rejected iff ``hard``
    is non-empty.

    ``enable_identifier_heuristic`` is **off by default and should stay off**
    for these prompts. The idea was to flag "gold-exclusive" tables/columns
    appearing in prose, but the full database schema is already in the prompt,
    so naming gold's tables is exactly what solving the question looks like. On
    a smoke test it flagged 8/8 traces, every one a false positive. Enable it
    only for prompt formats that withhold the schema.
    """
    hard: List[str] = []
    soft: List[str] = []
    joined = "\n".join(t for t in assistant_texts if t)
    prose = strip_sql_regions(joined)

    for pattern in HARD_LEAK_PATTERNS:
        if pattern.search(joined):
            hard.append(f"explicit_mention:{pattern.pattern}")

    # Gold SQL verbalized as prose rather than emitted as SQL.
    gold_norm = normalize_sql(gold_sql)
    if gold_norm:
        prose_norm = normalize_sql(prose)
        if gold_norm and gold_norm in prose_norm:
            hard.append("gold_sql_in_prose")
        else:
            for chunk in re.split(r"[.!?\n]", prose):
                if len(chunk) >= 40 and sql_similarity(chunk, gold_sql) >= near_copy_threshold:
                    hard.append("gold_sql_near_copy_in_prose")
                    break

    if enable_identifier_heuristic:
        prose_lower = prose.lower()
        exclusive_hits = [
            name
            for name in list(gold_tables or []) + list(gold_columns or [])
            if name and len(name) >= 4 and re.search(rf"\b{re.escape(name.lower())}\b", prose_lower)
        ]
        if len(exclusive_hits) >= prose_identifier_threshold:
            hard.append(f"gold_identifiers_in_prose:{','.join(sorted(set(exclusive_hits))[:6])}")

    for pattern in SOFT_LEAK_PATTERNS:
        if pattern.search(joined):
            soft.append(f"incidental:{pattern.pattern}")

    return {"hard": hard, "soft": soft}


def detect_copy(
    gold_sql: str,
    final_sql: str,
    first_tool_sql: str = "",
    tool_rounds: int = 0,
    near_copy_threshold: float = 0.95,
) -> List[str]:
    """Copy heuristics. Recorded, never rejecting on their own."""
    flags: List[str] = []
    final_sim = sql_similarity(final_sql, gold_sql)
    first_sim = sql_similarity(first_tool_sql, gold_sql) if first_tool_sql else 0.0

    if first_sim >= near_copy_threshold:
        flags.append("copy_first_call")
    if final_sim >= near_copy_threshold and tool_rounds == 0:
        flags.append("zero_verify_copy")
    if final_sim >= near_copy_threshold:
        flags.append("near_copy")
    return flags
