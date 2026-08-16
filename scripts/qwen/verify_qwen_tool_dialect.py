#!/usr/bin/env python3
"""Verify the Qwen3.8 tool-call dialect end to end.

Checks, in order:
  1. Gemma parsing is unchanged (regression guard for existing runs).
  2. Qwen XML tool calls parse into the structure TRL's tool loop expects.
  3. The pre-existing Gemma-only parser finds ZERO calls in Qwen output --
     i.e. the bug this dialect layer fixes.
  4. extract_sql() refuses to scrape draft SQL out of an unfinished Qwen
     tool call (reward-hacking guard).
  5. format_reward's <sql_code> leak guard fires on Qwen tool syntax.
  6. A parsed call round-trips through the REAL Qwen3.8-27B chat template and
     re-parses identically -- this is what keeps multi-turn rollouts stable.

Run:  PYTHONPATH=src .venv/bin/python scripts/qwen/verify_qwen_tool_dialect.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nl2sql_gspo.tool_dialects import (  # noqa: E402
    GEMMA_DIALECT,
    QWEN_DIALECT,
    contains_any_tool_call_marker,
    get_dialect,
    set_dialect,
)

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.8-27B")

GEMMA_TEXT = (
    "<scratch_pad>check the literal</scratch_pad>\n"
    'call:sqlite_query{db_id:<|"|>california_schools<|"|>,'
    'sql:<|"|>SELECT COUNT(*) FROM schools WHERE County = \'Alameda\'<|"|>,max_return_rows:10}'
)

QWEN_TEXT = """<think>
I should verify the county literal before answering.
</think>

Let me check the exact stored spelling first.
<tool_call>
<function=sqlite_query>
<parameter=db_id>
california_schools
</parameter>
<parameter=sql>
SELECT COUNT(*) FROM schools WHERE County = 'Alameda'
</parameter>
<parameter=max_return_rows>
10
</parameter>
</function>
</tool_call>"""

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


# 1. Gemma regression -------------------------------------------------------
gemma_calls = GEMMA_DIALECT.parse_tool_calls(GEMMA_TEXT)
check("gemma: parses 1 call", len(gemma_calls) == 1, f"got {len(gemma_calls)}")
if gemma_calls:
    fn = gemma_calls[0]["function"]
    check("gemma: name", fn["name"] == "sqlite_query", fn["name"])
    check("gemma: db_id", fn["arguments"].get("db_id") == "california_schools",
          repr(fn["arguments"].get("db_id")))
    check("gemma: sql preserved",
          "COUNT(*)" in str(fn["arguments"].get("sql", "")),
          repr(fn["arguments"].get("sql"))[:70])
    check("gemma: int coercion", fn["arguments"].get("max_return_rows") == 10,
          repr(fn["arguments"].get("max_return_rows")))

# 2. Qwen parsing -----------------------------------------------------------
qwen_calls = QWEN_DIALECT.parse_tool_calls(QWEN_TEXT)
check("qwen: parses 1 call", len(qwen_calls) == 1, f"got {len(qwen_calls)}")
if qwen_calls:
    fn = qwen_calls[0]["function"]
    check("qwen: name", fn["name"] == "sqlite_query", fn["name"])
    check("qwen: db_id", fn["arguments"].get("db_id") == "california_schools",
          repr(fn["arguments"].get("db_id")))
    check("qwen: sql preserved",
          fn["arguments"].get("sql") == "SELECT COUNT(*) FROM schools WHERE County = 'Alameda'",
          repr(fn["arguments"].get("sql")))
    check("qwen: int coercion", fn["arguments"].get("max_return_rows") == 10,
          repr(fn["arguments"].get("max_return_rows")))
    check("qwen: call has type=function", qwen_calls[0].get("type") == "function")

# 3. The bug being fixed ----------------------------------------------------
cross = GEMMA_DIALECT.parse_tool_calls(QWEN_TEXT)
check("gemma parser finds 0 calls in Qwen output (the bug)", len(cross) == 0,
      f"got {len(cross)}")

# 4. Unfinished-rollout SQL guard ------------------------------------------
set_dialect("qwen")
from nl2sql_gspo.sql_utils import extract_sql  # noqa: E402

check("qwen: marker detected in unfinished rollout",
      contains_any_tool_call_marker(QWEN_TEXT))
check("qwen: extract_sql refuses draft SQL from tool call",
      extract_sql(QWEN_TEXT) == "", repr(extract_sql(QWEN_TEXT))[:70])

finished = (
    "<scratch_pad>done</scratch_pad>\n"
    "<relevant_tables>schools</relevant_tables>\n"
    "<relevant_columns>schools.County</relevant_columns>\n"
    "<final_answer><sql_code>SELECT COUNT(*) FROM schools</sql_code></final_answer>"
)
check("qwen: extract_sql accepts a finished answer",
      "SELECT COUNT(*)" in extract_sql(finished), repr(extract_sql(finished)))

# 5. format_reward leak guard ----------------------------------------------
check("qwen: sql_leak_markers include XML tool syntax",
      "<tool_call>" in get_dialect().sql_leak_markers
      and "<function=" in get_dialect().sql_leak_markers,
      str(get_dialect().sql_leak_markers))

# 6. Real chat-template round trip -----------------------------------------
try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    call = qwen_calls[0]
    messages = [
        {"role": "system", "content": "You are an NL2SQL assistant."},
        {"role": "user", "content": "How many schools are in Alameda county?"},
        {
            "role": "assistant",
            "content": "Let me check the exact stored spelling first.",
            "reasoning_content": "I should verify the county literal.",
            "tool_calls": [
                {"id": call["id"], "type": "function", "function": call["function"]}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "sqlite_query",
            "content": '{"columns": ["COUNT(*)"], "rows": [[91]]}',
        },
    ]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=None
    )
    check("template: emits <tool_call>", "<tool_call>" in rendered)
    check("template: emits <function=sqlite_query>", "<function=sqlite_query>" in rendered)
    check("template: emits <tool_response>", "<tool_response>" in rendered)
    check("template: opens assistant turn", rendered.rstrip().endswith("<think>"),
          repr(rendered[-40:]))

    reparsed = QWEN_DIALECT.parse_tool_calls(rendered)
    check("template: re-parses to 1 call", len(reparsed) == 1, f"got {len(reparsed)}")
    if reparsed:
        check("template: round-trip preserves name",
              reparsed[0]["function"]["name"] == call["function"]["name"])
        check("template: round-trip preserves sql",
              reparsed[0]["function"]["arguments"].get("sql")
              == call["function"]["arguments"].get("sql"),
              repr(reparsed[0]["function"]["arguments"].get("sql"))[:70])
        check("template: round-trip preserves db_id",
              reparsed[0]["function"]["arguments"].get("db_id")
              == call["function"]["arguments"].get("db_id"))
except Exception as exc:  # noqa: BLE001
    check(f"chat-template round trip ({type(exc).__name__})", False, str(exc)[:160])

# 7. Trainer bridge -------------------------------------------------------
# _attach_native_tool_calls is what actually turns generated text into the
# structured tool_calls list TRL's GRPO tool loop dispatches on. It touches no
# instance state, so it can be exercised unbound without building a trainer.
try:
    from nl2sql_gspo.dynamic_sampling_trainer import DynamicSamplingGRPOTrainer

    attach = DynamicSamplingGRPOTrainer._attach_native_tool_calls
    stats = DynamicSamplingGRPOTrainer._native_tool_parse_stats

    set_dialect("qwen")
    completions = [[{"role": "assistant", "content": QWEN_TEXT}]]
    attached = attach(None, completions)
    check("trainer: attaches Qwen tool calls", attached == 1, f"attached={attached}")
    check("trainer: tool_calls on the message",
          len(completions[0][-1].get("tool_calls") or []) == 1)
    if completions[0][-1].get("tool_calls"):
        tc = completions[0][-1]["tool_calls"][0]
        check("trainer: tool_call shape is TRL-compatible",
              tc.get("type") == "function"
              and "name" in tc.get("function", {})
              and isinstance(tc["function"].get("arguments"), dict),
              str(tc)[:90])
    qstats = stats(None, [[{"role": "assistant", "content": QWEN_TEXT}]])
    check("trainer: parse stats see the call",
          qstats["raw_seq"] == 1 and qstats["parsed_calls"] == 1, str(qstats))

    # Same text under the Gemma dialect must attach nothing -- proving the
    # dialect switch, not luck, is what makes the Qwen path work.
    set_dialect("gemma")
    gem_completions = [[{"role": "assistant", "content": QWEN_TEXT}]]
    check("trainer: gemma dialect attaches 0 on Qwen text",
          attach(None, gem_completions) == 0)

    # And the Gemma path still works, unchanged.
    gem_ok = [[{"role": "assistant", "content": GEMMA_TEXT}]]
    check("trainer: gemma dialect still attaches Gemma calls",
          attach(None, gem_ok) == 1)
    set_dialect("qwen")
except Exception as exc:  # noqa: BLE001
    check(f"trainer bridge ({type(exc).__name__})", False, str(exc)[:160])

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
