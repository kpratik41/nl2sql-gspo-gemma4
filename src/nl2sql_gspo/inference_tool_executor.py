"""Execute native Gemma tool calls during standalone inference."""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


TOOL_CALL_RE = re.compile(
    r"(?:<\|tool_call\>\s*)?call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{(?P<args>.*?)\}(?:<tool_call\|>)?",
    re.DOTALL,
)

KNOWN_TOOL_ARG_KEYS = {
    "columns",
    "column",
    "db_id",
    "limit",
    "max_return_rows",
    "query",
    "sql",
    "table",
    "top_k",
    "where",
}


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


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    if value.startswith("<|\"|>") and value.endswith("<|\"|>"):
        return value[len("<|\"|>") : -len("<|\"|>")]
    if (value[0], value[-1]) in {("\"", "\""), ("'", "'")}:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value[1:-1]
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                if value.startswith("[") and value.endswith("]"):
                    inner = value[1:-1].strip()
                    if not inner:
                        return []
                    return [_parse_scalar(item.strip()) for item in inner.split(",")]
                return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_args(args_text: str) -> Dict[str, Any]:
    """Parse Gemma's compact ``key:value`` argument format."""

    args: Dict[str, Any] = {}
    key = ""
    value = ""
    in_key = True
    quote: str | None = None
    depth = 0
    parts = []

    def is_next_arg_separator(offset: int) -> bool:
        remaining = args_text[offset + 1 :]
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", remaining)
        return bool(match and match.group(1) in KNOWN_TOOL_ARG_KEYS)

    for offset, char in enumerate(args_text):
        if quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)

        if not quote and depth == 0 and char == ":" and in_key:
            in_key = False
            continue
        if not quote and depth == 0 and char == "," and not in_key:
            if key.strip().strip('"').strip("'") == "sql" and not is_next_arg_separator(offset):
                value += char
                continue
            parts.append((key.strip(), value.strip()))
            key, value, in_key = "", "", True
            continue

        if in_key:
            key += char
        else:
            value += char

    if key.strip():
        parts.append((key.strip(), value.strip()))

    for raw_key, raw_value in parts:
        cleaned_key = raw_key.strip().strip('"').strip("'")
        if cleaned_key:
            args[cleaned_key] = _parse_scalar(raw_value)
    return args


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
    """Extract Gemma-style tool calls from generated text."""

    calls: List[Dict[str, Any]] = []
    for index, match in enumerate(TOOL_CALL_RE.finditer(text or "")):
        name = match.group("name")
        arguments = _split_args(match.group("args"))
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return calls


def strip_decoded_thought_prefix(text: str) -> str:
    """Remove Gemma thought-channel labels left after special-token decoding."""
    stripped = (text or "").strip()
    while stripped == "thought" or stripped.startswith("thought\n"):
        if stripped == "thought":
            return ""
        stripped = stripped[len("thought\n") :].lstrip()
    return stripped


def text_before_first_tool_call(text: str) -> str:
    """Return model text before the first tool call, useful as Gemma reasoning."""

    match = TOOL_CALL_RE.search(text or "")
    if not match:
        return strip_decoded_thought_prefix(text or "")
    return strip_decoded_thought_prefix((text or "")[: match.start()])


def format_tool_response(name: str, response: Any) -> str:
    response_json = json.dumps(response, ensure_ascii=False, default=str)
    return f"<|tool_response>response:{name}{{value:{response_json}}}<tool_response|>"


def _augment_tool_response(name: str, response: Any) -> Any:
    if name != "sqlite_query" or not isinstance(response, dict) or response.get("error"):
        return response
    if "columns" not in response:
        return response

    augmented = dict(response)
    augmented["column_coverage_reminder"] = (
        "Before final_answer, compare these returned columns against your "
        "ExpectedOutputColumns. If any requested attribute is missing, revise "
        "the SELECT list and call sqlite_query again."
    )
    return augmented


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
            response = await _execute_tool(name, arguments, timeout_s)
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
