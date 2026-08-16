"""Execute native model tool calls during standalone inference.

Parsing and rendering are delegated to the active :mod:`tool_dialects` dialect
so the same executor drives Gemma-4's ``call:name{...}`` syntax and Qwen3.8's
XML ``<tool_call><function=...>`` syntax. The dialect defaults to Gemma, so
existing Gemma callers are unaffected.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from nl2sql_gspo import tool_loop_guard
from nl2sql_gspo.tool_dialects import (
    GEMMA_TOOL_CALL_RE,
    KNOWN_TOOL_ARG_KEYS,
    get_dialect,
)

# Back-compat alias: this module historically exported the Gemma regex.
TOOL_CALL_RE = GEMMA_TOOL_CALL_RE


def configure_tool_env(database_dir: str) -> str:
    """Configure database roots consumed by ``gen_tools.py``."""

    roots = []

    def add(path: str | None) -> None:
        if not path:
            return
        expanded = str(Path(path).expanduser())
        if expanded not in roots:
            roots.append(expanded)

    add(database_dir)
    add(os.path.join(database_dir, "train_databases"))
    add(os.path.join(database_dir, "dev_databases"))
    add("databases")
    add("databases/train_databases")
    add("databases/dev_databases")

    existing = os.environ.get("BIRD_DB_ROOTS")
    if existing:
        for raw in existing.split(os.pathsep):
            add(raw)

    os.environ["BIRD_DB_ROOTS"] = os.pathsep.join(roots)
    return os.environ["BIRD_DB_ROOTS"]



async def _execute_tool(name: str, arguments: Dict[str, Any], timeout_s: float) -> Any:
    import gen_tools

    tool = getattr(gen_tools, name, None)
    if tool is None:
        return {"error": "unknown_tool", "message": f"Tool '{name}' is not available."}

    try:
        return await asyncio.wait_for(tool(**arguments), timeout=float(timeout_s))
    except Exception as exc:
        return {"error": "tool_exception", "message": str(exc)}


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract native tool calls from generated text using the active dialect."""

    return get_dialect().parse_tool_calls(text)


def strip_decoded_thought_prefix(text: str) -> str:
    """Remove Gemma thought-channel labels left after special-token decoding."""
    stripped = (text or "").strip()
    while stripped == "thought" or stripped.startswith("thought\n"):
        if stripped == "thought":
            return ""
        stripped = stripped[len("thought\n") :].lstrip()
    return stripped


def text_before_first_tool_call(text: str) -> str:
    """Return model text before the first tool call, useful as reasoning."""

    dialect = get_dialect()
    match = dialect.tool_call_marker_re.search(text or "")
    if not match:
        return strip_decoded_thought_prefix(text or "")
    return strip_decoded_thought_prefix((text or "")[: match.start()])


def format_tool_response(name: str, response: Any) -> str:
    return get_dialect().render_tool_response(name, response)


def _augment_tool_response(name: str, response: Any) -> Any:
    if name != "sqlite_query" or not isinstance(response, dict) or response.get("error"):
        return response
    if "columns" not in response:
        return response

    augmented = dict(response)
    if _is_empty_result(response):
        # The column-coverage reminder is the wrong advice here: when a query
        # matches nothing, the columns are usually fine and the predicate is
        # not. Telling the model to re-check its SELECT list gives it nothing
        # new, so it re-emits the same query -- 41% of the rollouts that
        # exhausted the tool budget hit an empty or zero first result, against
        # 2.2% of those that finished normally.
        augmented["empty_result_hint"] = (
            "This query matched no data. That usually means a literal does not match the "
            "stored values, or a column's stored format differs from what the predicate "
            "assumes -- for example a status stored as '+'/'-' rather than "
            "'positive'/'negative', or a number stored as text. Do not re-run this query "
            "unchanged. Inspect the filtered column with sqlite_peek, or find the correct "
            "literal with bm25_search_sqlite, then revise the predicate."
        )
        return augmented

    augmented["column_coverage_reminder"] = (
        "Before final_answer, compare these returned columns against your "
        "ExpectedOutputColumns. If any requested attribute is missing, revise "
        "the SELECT list and call sqlite_query again."
    )
    return augmented


def _is_empty_result(response: Dict[str, Any]) -> bool:
    """True when a query returned nothing meaningful.

    Two shapes count. A plain empty row set is the obvious one. The other is a
    lone aggregate cell of 0 or NULL: ``SELECT COUNT(*) ... WHERE <bad literal>``
    returns one row containing 0, which is "no matching data" wearing a result's
    clothing, and it drove the worst observed loops.
    """

    rows = response.get("rows")
    if not isinstance(rows, list):
        return False
    if not rows:
        return True
    if len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1:
        return rows[0][0] in (0, 0.0, "0", None)
    return False


def execute_tool_calls(tool_calls: List[Dict[str, Any]], timeout_s: float = 60.0) -> List[Dict[str, Any]]:
    """Execute parsed tool calls and return responses in chat-template shape."""

    if not tool_calls:
        return []

    async def run_all() -> list[tuple[str, Any]]:
        responses = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            arguments = function.get("arguments") or {}

            # Inert unless the guard is enabled AND the caller opened a
            # rollout_scope, so nothing changes for callers that do neither.
            notice = tool_loop_guard.check(name, arguments)
            if notice is not None:
                responses.append((name, notice))
                continue

            response = await _execute_tool(name, arguments, timeout_s)
            tool_loop_guard.record(name, arguments, response)
            responses.append((name, _augment_tool_response(name, response)))
        return responses

    responses = asyncio.run(run_all())
    return [
        {
            "name": name,
            "response": {"value": response},
            "raw_response": response,
            "rendered": format_tool_response(name, response),
        }
        for name, response in responses
    ]


def extract_and_execute_tools(text: str, timeout_s: float = 60.0) -> str:
    """Append compact tool responses after any tool calls found in ``text``."""

    tool_calls = extract_tool_calls(text)
    if not tool_calls:
        return text

    responses = execute_tool_calls(tool_calls, timeout_s)
    rendered = [text.rstrip()]
    for response in responses:
        rendered.append(response["rendered"])

    return "\n".join(rendered)
