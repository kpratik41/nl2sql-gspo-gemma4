#!/usr/bin/env python3
"""Checks for the tool loop guard and the empty-result hint.

The property that matters most is isolation: a guard that leaks dedup state
between rollouts would suppress a different sample's legitimate query. Run this
before enabling NL2SQL_TOOL_LOOP_GUARD anywhere, and again before ever turning
it on for RL.

    python scripts/qwen/verify_tool_loop_guard.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from nl2sql_gspo import tool_loop_guard as g
    from nl2sql_gspo.inference_tool_executor import _augment_tool_response, _is_empty_result

    # ---- disabled by default -------------------------------------------------
    os.environ.pop(g.ENV_VAR, None)
    check("disabled when env var unset", not g.is_enabled())
    with g.rollout_scope("r0") as scope:
        check("disabled: rollout_scope yields None", scope is None)
        g.record("sqlite_query", {"sql": "SELECT 1"}, {"rows": [[1]]})
        check("disabled: repeat is not suppressed", g.check("sqlite_query", {"sql": "SELECT 1"}) is None)

    os.environ[g.ENV_VAR] = "1"
    check("enabled when env var set", g.is_enabled())

    # ---- no scope means no dedup (the RL safety property) --------------------
    g.record("sqlite_query", {"sql": "SELECT 2"}, {"rows": [[2]]})
    check(
        "enabled but unscoped: nothing recorded, nothing suppressed",
        g.check("sqlite_query", {"sql": "SELECT 2"}) is None,
    )

    # ---- basic dedup ---------------------------------------------------------
    with g.rollout_scope("r1") as scope:
        args = {"db_id": "x", "sql": "SELECT COUNT(*) FROM t WHERE a = 'z'"}
        check("first call passes through", g.check("sqlite_query", args) is None)
        g.record("sqlite_query", args, {"rows": [[0]], "columns": ["c"]})
        notice = g.check("sqlite_query", args)
        check("identical repeat is suppressed", isinstance(notice, dict) and notice.get("repeated_call") is True)
        check(
            "notice carries the previous result summary",
            notice.get("previous_result", {}).get("row_count") == 1
            and notice["previous_result"].get("first_value") == 0,
            str(notice.get("previous_result")),
        )
        check("notice tells the model what to do instead", "sqlite_peek" in notice.get("message", ""))
        check("scope counted the suppression", scope.suppressed == 1)

        reindented = {"db_id": "x", "sql": "SELECT   COUNT(*)\n  FROM t\n WHERE a = 'z'"}
        check("whitespace-only difference still counts as a repeat", g.check("sqlite_query", reindented) is not None)

        changed = {"db_id": "x", "sql": "SELECT COUNT(*) FROM t WHERE a = 'Z'"}
        check("changed literal case is a DIFFERENT query", g.check("sqlite_query", changed) is None)

        other_tool = {"db_id": "x", "table": "t", "columns": ["a"]}
        check("different tool name is not a repeat", g.check("sqlite_peek", other_tool) is None)

    # ---- isolation between scopes -------------------------------------------
    args = {"db_id": "x", "sql": "SELECT 3"}
    with g.rollout_scope("r2"):
        g.record("sqlite_query", args, {"rows": [[3]]})
    with g.rollout_scope("r3"):
        check("a new scope does not inherit the previous scope's store", g.check("sqlite_query", args) is None)

    # ---- isolation across concurrent asyncio tasks ---------------------------
    async def rollout(tag: str, results: dict) -> None:
        with g.rollout_scope(tag):
            a = {"db_id": "d", "sql": f"SELECT '{tag}'"}
            shared = {"db_id": "d", "sql": "SELECT shared"}
            g.record("sqlite_query", a, {"rows": [[tag]]})
            g.record("sqlite_query", shared, {"rows": [[1]]})
            await asyncio.sleep(0.01)  # force interleaving
            results[tag] = (
                g.check("sqlite_query", a) is not None,          # own call: suppressed
                g.check("sqlite_query", {"db_id": "d", "sql": "SELECT 'other'"}) is None,
                g.check("sqlite_query", shared) is not None,     # own copy of a shared query
            )

    async def drive() -> dict:
        out: dict = {}
        await asyncio.gather(*(rollout(f"t{i}", out) for i in range(8)))
        return out

    res = asyncio.run(drive())
    check("8 concurrent rollouts each dedup their own calls", all(r[0] for r in res.values()))
    check("concurrent rollouts do not see each other's calls", all(r[1] for r in res.values()))
    check("a query issued by every rollout is deduped per rollout", all(r[2] for r in res.values()))

    # ---- guard_tool_callable (the RL seam) ----------------------------------
    calls = {"n": 0}

    def fake_sqlite_query(db_id=None, sql=None):
        calls["n"] += 1
        return {"rows": [[calls["n"]]], "columns": ["c"]}

    wrapped = g.guard_tool_callable(fake_sqlite_query)
    with g.rollout_scope("r4"):
        wrapped(db_id="d", sql="SELECT 1")
        second = wrapped(db_id="d", sql="SELECT 1")
        check("wrapped callable suppresses the repeat", isinstance(second, dict) and second.get("repeated_call"))
        check("wrapped callable did not re-execute the tool", calls["n"] == 1, f"executions={calls['n']}")
        wrapped(db_id="d", sql="SELECT 2")
        check("wrapped callable still runs a different query", calls["n"] == 2, f"executions={calls['n']}")

    calls["n"] = 0
    wrapped(db_id="d", sql="SELECT 1")
    wrapped(db_id="d", sql="SELECT 1")
    check("wrapped callable is pass-through with no scope (RL default)", calls["n"] == 2, f"executions={calls['n']}")

    # ---- async tools must stay async (regression guard) ---------------------
    # gen_tools' tools are coroutine functions. A sync wrapper around one makes
    # iscoroutinefunction() False while still returning a coroutine, so callers
    # that dispatch on that check never await it and the coroutine's repr gets
    # fed back into the rollout as the tool result.
    acalls = {"n": 0}

    async def fake_async_query(db_id=None, sql=None):
        acalls["n"] += 1
        return {"rows": [[acalls["n"]]], "columns": ["c"]}

    awrapped = g.guard_tool_callable(fake_async_query)
    check("async tool stays a coroutine function after wrapping", asyncio.iscoroutinefunction(awrapped))

    async def drive_async():
        with g.rollout_scope("r5"):
            first = await awrapped(db_id="d", sql="SELECT 1")
            second = await awrapped(db_id="d", sql="SELECT 1")
            third = await awrapped(db_id="d", sql="SELECT 2")
        return first, second, third

    first, second, third = asyncio.run(drive_async())
    check("async wrapper returns real results, not a coroutine repr", first.get("rows") == [[1]], repr(first)[:80])
    check("async wrapper suppresses the repeat", isinstance(second, dict) and second.get("repeated_call") is True)
    check("async wrapper did not re-execute on repeat", acalls["n"] == 2, f"executions={acalls['n']}")
    check("async wrapper still runs a different query", third.get("rows") == [[2]], repr(third)[:80])

    # ---- empty-result hint ---------------------------------------------------
    check("empty row set is an empty result", _is_empty_result({"rows": []}))
    check("lone 0 aggregate is an empty result", _is_empty_result({"rows": [[0]]}))
    check("lone NULL aggregate is an empty result", _is_empty_result({"rows": [[None]]}))
    check("lone non-zero aggregate is NOT empty", not _is_empty_result({"rows": [[7]]}))
    check("multi-row result is NOT empty", not _is_empty_result({"rows": [[0], [1]]}))
    check("multi-column single row is NOT empty", not _is_empty_result({"rows": [[0, 1]]}))

    empty = _augment_tool_response("sqlite_query", {"rows": [], "columns": ["c"]})
    check("empty result gets the hint", "empty_result_hint" in empty)
    check("empty result does NOT get the column reminder", "column_coverage_reminder" not in empty)
    check("hint names both diagnostic tools", "sqlite_peek" in empty["empty_result_hint"] and "bm25_search_sqlite" in empty["empty_result_hint"])

    zero = _augment_tool_response("sqlite_query", {"rows": [[0]], "columns": ["c"]})
    check("zero count gets the hint", "empty_result_hint" in zero)

    normal = _augment_tool_response("sqlite_query", {"rows": [[7]], "columns": ["c"]})
    check("normal result keeps the column reminder", "column_coverage_reminder" in normal)
    check("normal result has no hint", "empty_result_hint" not in normal)

    err = _augment_tool_response("sqlite_query", {"error": "boom", "columns": ["c"]})
    check("error response is left alone", "empty_result_hint" not in err and "column_coverage_reminder" not in err)

    peek = _augment_tool_response("sqlite_peek", {"rows": [], "columns": ["c"]})
    check("non-sqlite_query tools are left alone", "empty_result_hint" not in peek)

    os.environ.pop(g.ENV_VAR, None)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
