"""Stop a rollout from re-issuing a tool call it has already made.

Why this exists
---------------
On BIRD dev at temperature 0, 70 of 1534 examples exhausted the 8-round tool
budget, and they scored 27.1% against 71.51% overall. The cause is not
exploration: 64 of those 70 re-issued a *byte-identical* SQL query, three of
them nine times in a row, against 2.1% of the examples that finished normally.

The mechanism is a fixed point in greedy decoding. Once the model emits a bare
tool call with no ``<scratch_pad>`` reasoning (69 of 70 capped rows do this from
round 2 onward, versus roughly half of finished rows), the context ends with
*(bare tool call, its result)*. That prefix deterministically regenerates the
same bare tool call, which returns the same result, forever. The trigger is
usually a result the model does not believe -- 0 rows, ``COUNT(*) = 0``, or
NULL -- which occurred as the first result in 41% of capped rows versus 2.2% of
finished ones.

Re-executing an identical call cannot produce new information, so this guard
short-circuits it and returns a note saying so. That gives the model something
new in context, which is the one thing the loop deprives it of.

Scope and safety
----------------
State lives on a scope object held in a :class:`contextvars.ContextVar`, never
in a module-level dict. Two consequences matter:

* Concurrency is safe by construction. ``contextvars`` are copied per asyncio
  task and propagated through ``asyncio.to_thread``, so concurrent rollouts in
  the same process each see their own store.
* **No active scope means no deduplication.** A caller that has not opened a
  scope gets ordinary pass-through behaviour rather than sharing a global
  store. This is the failure mode that matters for RL, where rollouts run in
  parallel batches: leaking dedup state across samples in a batch would
  silently corrupt training, so the design makes the unscoped case inert
  instead of wrong.

Enablement
----------
Off unless ``NL2SQL_TOOL_LOOP_GUARD`` is truthy. Eval runners opt in; the RL
trainer deliberately does not, until the eval side has shown the guard helps.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

ENV_VAR = "NL2SQL_TOOL_LOOP_GUARD"

_WHITESPACE_RE = re.compile(r"\s+")


def is_enabled() -> bool:
    """True when the guard is switched on for this process."""

    return (os.environ.get(ENV_VAR, "") or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Scope:
    """Per-rollout record of which tool calls have already been executed."""

    rollout_id: str
    seen: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    suppressed: int = 0


_active: contextvars.ContextVar[Optional[_Scope]] = contextvars.ContextVar(
    "nl2sql_tool_loop_guard_scope", default=None
)


@contextlib.contextmanager
def rollout_scope(rollout_id: str) -> Iterator[Optional[_Scope]]:
    """Open a deduplication scope for one rollout.

    Entering is cheap and always safe: when the guard is disabled this yields
    ``None`` and installs nothing, so callers can wrap unconditionally.
    """

    if not is_enabled():
        yield None
        return

    scope = _Scope(rollout_id=str(rollout_id))
    token = _active.set(scope)
    try:
        yield scope
    finally:
        # Dropping the scope drops its store. Nothing survives into the next
        # rollout, which is what keeps parallel batches independent.
        _active.reset(token)


def current_scope() -> Optional[_Scope]:
    return _active.get()


def fingerprint(name: str, arguments: Any) -> str:
    """Identity of a tool call for repeat detection.

    Whitespace inside string arguments is collapsed so that a query re-emitted
    with different indentation still counts as the same query. Nothing else is
    normalised -- case and punctuation are load-bearing in SQL literals, and
    folding them would merge genuinely different queries.
    """

    def canon(value: Any) -> Any:
        if isinstance(value, str):
            return _WHITESPACE_RE.sub(" ", value).strip()
        if isinstance(value, dict):
            return {k: canon(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [canon(v) for v in value]
        return value

    return json.dumps({"name": name, "arguments": canon(arguments)}, sort_keys=True, default=str)


def _summarize(response: Any) -> Dict[str, Any]:
    """Compact description of a previous result, for the repeat notice."""

    if not isinstance(response, dict):
        return {"result": "non-dict response"}
    if response.get("error"):
        return {"error": response.get("error"), "message": str(response.get("message", ""))[:200]}
    rows = response.get("rows")
    if isinstance(rows, list):
        summary: Dict[str, Any] = {"row_count": len(rows)}
        if rows and isinstance(rows[0], list) and len(rows[0]) == 1:
            summary["first_value"] = rows[0][0]
        return summary
    return {"result": "no rows in response"}


def repeat_notice(name: str, previous: Dict[str, Any]) -> Dict[str, Any]:
    """The response substituted for a duplicate call."""

    return {
        "repeated_call": True,
        "previous_result": previous,
        "message": (
            f"This exact {name} call was already executed earlier in this conversation and "
            "returned the result summarized above. It was not run again, because repeating an "
            "identical call cannot return anything new. Either change the call -- inspect a "
            "column's real values with sqlite_peek, or find the correct literal with "
            "bm25_search_sqlite -- or give your final answer now."
        ),
    }


def check(name: str, arguments: Any) -> Optional[Dict[str, Any]]:
    """Return a repeat notice if this call was already made, else ``None``."""

    scope = _active.get()
    if scope is None:
        return None
    previous = scope.seen.get(fingerprint(name, arguments))
    if previous is None:
        return None
    scope.suppressed += 1
    return repeat_notice(name, previous)


def record(name: str, arguments: Any, response: Any) -> None:
    """Remember a call and a summary of what it returned."""

    scope = _active.get()
    if scope is None:
        return
    scope.seen.setdefault(fingerprint(name, arguments), _summarize(response))


def guard_tool_callable(tool: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a plain tool callable so repeats short-circuit.

    This is the seam for the RL trainer, whose tools are invoked directly by
    TRL rather than through :mod:`inference_tool_executor`. It is inert until
    both the env var is set and the caller opens a :func:`rollout_scope` per
    rollout -- neither of which the trainer does today, by design.
    """

    import asyncio
    import functools

    name = getattr(tool, "__name__", "tool")

    # The async branch is not optional. gen_tools' tools are coroutine
    # functions, and a sync wrapper around one makes
    # asyncio.iscoroutinefunction() report False while still returning a
    # coroutine. Callers that dispatch on that check then never await it, and
    # the *repr* of the coroutine object gets fed back into the rollout as the
    # tool result -- '"<coroutine object sqlite_query at 0x...>"'. Rollouts keep
    # running and rewards keep being computed, so it corrupts training silently.
    if asyncio.iscoroutinefunction(tool):

        @functools.wraps(tool)
        async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
            if args:
                # Positional args would need the tool's signature to fingerprint
                # reliably. Every caller in this repo passes keywords, so rather
                # than guess, fall through unguarded.
                return await tool(*args, **kwargs)
            notice = check(name, kwargs)
            if notice is not None:
                return notice
            result = await tool(**kwargs)
            record(name, kwargs, result)
            return result

        return wrapped_async

    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            return tool(*args, **kwargs)
        notice = check(name, kwargs)
        if notice is not None:
            return notice
        result = tool(**kwargs)
        record(name, kwargs, result)
        return result

    return wrapped
