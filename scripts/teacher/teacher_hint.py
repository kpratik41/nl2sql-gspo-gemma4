"""Privileged-teacher hint injection + leakage/copy detection (Path A2/A3 core).

This module is pure logic (no GPU, no vLLM) so it can be unit-tested directly.

Two responsibilities:

1. ``inject_privileged_hint`` - take a standard bare-tool prompt (system+user,
   NO gold) and return a *teacher* prompt that additionally exposes the gold
   SQL as an internal reference. The teacher is instructed to solve as if it
   derived the answer itself and to NEVER verbalize the reference. Only the
   ``prompt``/``messages`` used for teacher *generation* carry the hint; the
   saved student SFT record uses the original hint-free prompt.

2. Leakage + copy detection, applied to the generated trajectory *before* it
   becomes training data:
   - ``detect_text_leakage``: the assistant text (scratchpad / tool calls /
     final answer) must never reference the hidden reference / gold / provided
     SQL.  Public prompt hints and normal verification language such as
     "expected output columns" are not treated as privileged leakage.
   - ``detect_copy``: even without verbalizing, a trajectory that just pastes
     the gold as its first tool call (or final-answers gold with zero tool
     verification) is transcription, not reasoning. These are flagged (and
     counted as ``copy_rate``) rather than silently trusted.

The dev pass@k gate remains the ultimate leakage check: if train accuracy
jumps but dev pass@1 / pass@16 does not, the traces leaked or memorized.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Privileged instruction block (v1: full gold SQL, non-verbalized)
# ---------------------------------------------------------------------------

PRIVILEGED_INSTRUCTION = """
PRIVILEGED SELF-CHECK (internal only)
A verified reference solution for THIS problem is available to you at the end of
the user message. It is provided ONLY to guide and self-check your own reasoning.
Hard requirements:
- Solve the problem as if you derived it independently: draft your own SQL,
  verify it with sqlite_query, and finalize through the normal workflow.
- NEVER mention, quote, describe, paraphrase, or allude to the reference, a
  "gold"/"provided"/"given"/"expected" SQL or answer, or the fact that any
  reference exists, anywhere in scratch_pad, tool calls, or the final answer.
- Your trajectory must read exactly like a fully independent solve.
Violating the non-disclosure rule makes the trajectory unusable.
""".strip()

_REFERENCE_BLOCK = "\n\n<internal_reference_do_not_reveal>\n{gold_sql}\n</internal_reference_do_not_reveal>"


# ---------------------------------------------------------------------------
# Hint injection
# ---------------------------------------------------------------------------

def inject_privileged_hint(
    prompt_messages: Sequence[Dict[str, Any]],
    gold_sql: str,
    *,
    strategy: str = "full_sql",
) -> List[Dict[str, Any]]:
    """Return a teacher prompt (deep-copied) with the privileged reference.

    ``strategy`` currently supports ``"full_sql"`` (v1). The privileged
    instruction is prepended to the system turn and the reference block is
    appended to the last user turn. The original ``prompt_messages`` are not
    mutated.
    """

    if strategy != "full_sql":
        raise ValueError(f"unsupported hint strategy: {strategy!r}")
    if not gold_sql or not str(gold_sql).strip():
        raise ValueError("gold_sql is required to inject a privileged hint")

    messages = copy.deepcopy(list(prompt_messages))
    if not messages:
        raise ValueError("prompt_messages is empty")

    # 1) prepend privileged instruction to the system turn (or create one).
    sys_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "system"),
        None,
    )
    if sys_idx is None:
        messages.insert(
            0,
            {"role": "system", "content": PRIVILEGED_INSTRUCTION},
        )
    else:
        base = str(messages[sys_idx].get("content", ""))
        messages[sys_idx]["content"] = f"{PRIVILEGED_INSTRUCTION}\n\n{base}"

    # 2) append the reference block to the last user turn.
    user_idx = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        None,
    )
    if user_idx is None:
        raise ValueError(
            "prompt_messages has no user turn to attach the reference to"
        )

    user_content = str(messages[user_idx].get("content", ""))
    messages[user_idx]["content"] = (
        user_content
        + _REFERENCE_BLOCK.format(gold_sql=str(gold_sql).strip())
    )
    return messages


# ---------------------------------------------------------------------------
# SQL normalization
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    """Whitespace/case/punctuation-normalized SQL for copy comparison."""

    text = (sql or "").strip().rstrip(";")
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip("`").strip()
    return text.lower()


# ---------------------------------------------------------------------------
# Text leakage detection
# ---------------------------------------------------------------------------

_LEAK_PHRASE_RE = re.compile(
    r"\b("
    r"reference(?:\s+(?:solution|sql|query|answer))?"
    r"|gold(?:en)?\s+(?:sql|query|answer|solution|standard)"
    r"|ground.?truth"
    r"|provided\s+(?:sql|query|solution|answer)"
    r"|given\s+(?:sql|query|solution|answer)"
    r"|expected\s+(?:sql|query)"
    r"|as\s+(?:shown|provided|given)\s+(?:in\s+)?(?:the\s+)?(?:reference|solution)"
    r"|internal_reference"
    r"|do_not_reveal"
    r")\b",
    re.IGNORECASE,
)


def detect_text_leakage(
    assistant_texts: Sequence[str],
    gold_sql: str,
) -> List[str]:
    """Return a list of leakage reasons; empty means clean.

    Flags explicit references to the hidden privileged block, gold SQL, or
    provided/given SQL.  It deliberately does not flag:

    * public ``<hint>`` references from the original prompt;
    * "expected output/columns" wording used by the normal verifier prompt;
    * correct SQL appearing in ``CandidateSQL=`` or ``sqlite_query`` tool calls.

    Copy/transcription concerns are handled by ``detect_copy`` instead.
    """

    reasons: List[str] = []
    for text in assistant_texts:
        text = text or ""

        phrase = _LEAK_PHRASE_RE.search(text)
        if phrase:
            reasons.append(f"phrase:{phrase.group(1).lower()}")

    # dedupe, preserve order
    seen: set = set()
    unique = [
        r for r in reasons
        if not (r in seen or seen.add(r))
    ]
    return unique


# ---------------------------------------------------------------------------
# Copy / transcription detection
# ---------------------------------------------------------------------------

def _edit_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0,1] (1.0 == identical)."""

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    la, lb = len(a), len(b)
    prev = list(range(lb + 1))

    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]

        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = cur

    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def detect_copy(
    gold_sql: str,
    first_tool_sql: Optional[str],
    final_sql: str,
    *,
    had_tool_calls: bool,
    near_copy_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Return copy/transcription flags for one verified trajectory.

    - ``copy_first_call``: the first executed SQL equals gold verbatim
      (normalized) -> pure transcription, no independent drafting.
    - ``zero_verify_copy``: no tool calls at all but the final SQL equals gold
      -> suspicious for a hard (all-wrong) item.
    - ``near_copy``: first executed SQL is >= ``near_copy_threshold`` similar to
      gold -> likely transcription with cosmetic edits.
    """

    gold_norm = normalize_sql(gold_sql)
    first_norm = normalize_sql(first_tool_sql) if first_tool_sql else ""
    final_norm = normalize_sql(final_sql)

    copy_first_call = bool(
        gold_norm
        and first_norm
        and first_norm == gold_norm
    )

    zero_verify_copy = bool(
        (not had_tool_calls)
        and gold_norm
        and final_norm == gold_norm
    )

    near = (
        _edit_ratio(first_norm, gold_norm)
        if (first_norm and gold_norm)
        else 0.0
    )
    near_copy = bool(near >= near_copy_threshold)

    return {
        "copy_first_call": copy_first_call,
        "zero_verify_copy": zero_verify_copy,
        "near_copy": near_copy,
        "first_call_sim_to_gold": round(near, 4),
    }


def summarize_copy_rate(
    flags: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate copy flags across kept trajectories into rates."""

    n = len(flags)
    if n == 0:
        return {
            "n": 0,
            "copy_first_call_rate": 0.0,
            "zero_verify_copy_rate": 0.0,
            "near_copy_rate": 0.0,
        }

    return {
        "n": n,
        "copy_first_call_rate": (
            sum(1 for f in flags if f.get("copy_first_call")) / n
        ),
        "zero_verify_copy_rate": (
            sum(1 for f in flags if f.get("zero_verify_copy")) / n
        ),
        "near_copy_rate": (
            sum(1 for f in flags if f.get("near_copy")) / n
        ),
    }
