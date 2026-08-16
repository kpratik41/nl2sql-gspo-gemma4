"""Model-specific tool-call surface syntax ("dialects").

The RL stack was written for Gemma-4, which emits compact native tool calls::

    call:sqlite_query{db_id:<|"|>california_schools<|"|>,sql:<|"|>SELECT ...<|"|>}

Qwen3.8-27B (Qwen3.5 architecture) ships a completely different, XML-shaped
format in its chat template::

    <tool_call>
    <function=sqlite_query>
    <parameter=db_id>
    california_schools
    </parameter>
    <parameter=sql>
    SELECT ...
    </parameter>
    </function>
    </tool_call>

Nothing in the trainer parsed the XML form, so a Qwen rollout would produce
plain text that never became a structured ``tool_calls`` list, the GRPO tool
loop would never fire, and the model would train with tools silently disabled.

This module isolates every format-dependent regex and renderer behind a
``ToolDialect`` so the trainer, reward functions and SQL extractors work for
both models. Gemma stays the default, so existing Gemma runs are unchanged.

The Qwen regexes and scalar coercion are copied from
``scripts/run_inference_bird_qwen_async.py``, which is the implementation that
produced the validated Qwen3.8-27B pass@16 eval numbers.
"""

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern


# --------------------------------------------------------------------------
# Gemma-4 native format
# --------------------------------------------------------------------------

GEMMA_TOOL_CALL_RE = re.compile(
    r"(?:<\|tool_call\>\s*)?call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{(?P<args>.*?)\}(?:<tool_call\|>)?",
    re.DOTALL,
)
GEMMA_TOOL_CALL_MARKER_RE = re.compile(
    r"(?:<\|tool_call\>\s*)?call:[A-Za-z_][A-Za-z0-9_]*\{",
    re.IGNORECASE,
)
GEMMA_RAW_CALL_NAME_RE = re.compile(r"call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{")
GEMMA_TOOL_RESPONSE_RE = re.compile(
    r"<\|tool_response\>|response:[A-Za-z_][A-Za-z0-9_]*\{", re.IGNORECASE
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


# --------------------------------------------------------------------------
# Qwen3.5 / Qwen3.8 XML format
# --------------------------------------------------------------------------

QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*"
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<body>.*?)"
    r"</function>\s*"
    r"</tool_call>",
    re.DOTALL,
)
QWEN_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<value>.*?)"
    r"\s*</parameter>",
    re.DOTALL,
)
QWEN_TOOL_CALL_MARKER_RE = re.compile(
    r"<tool_call>\s*<function=[A-Za-z_][A-Za-z0-9_]*>", re.IGNORECASE
)
QWEN_RAW_CALL_NAME_RE = re.compile(r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>")
QWEN_TOOL_RESPONSE_RE = re.compile(r"</?tool_response>", re.IGNORECASE)


def _gemma_parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[len('<|"|>') : -len('<|"|>')]
    if (value[0], value[-1]) in {('"', '"'), ("'", "'")}:
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
                    return [_gemma_parse_scalar(item.strip()) for item in inner.split(",")]
                return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _gemma_split_args(args_text: str) -> Dict[str, Any]:
    """Parse Gemma's compact ``key:value`` argument format."""

    args: Dict[str, Any] = {}
    key = ""
    value = ""
    in_key = True
    quote: Optional[str] = None
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
            args[cleaned_key] = _gemma_parse_scalar(raw_value)
    return args


def _qwen_parse_parameter_value(value: str) -> Any:
    """Coerce one ``<parameter=...>`` body to a Python scalar.

    Mirrors ``parse_parameter_value`` in the validated Qwen eval runner.
    """

    stripped = value.strip()
    if not stripped:
        return ""
    if stripped in {"true", "false"}:
        return stripped == "true"
    if stripped == "null":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _gemma_parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for index, match in enumerate(GEMMA_TOOL_CALL_RE.finditer(text or "")):
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": _gemma_split_args(match.group("args")),
                },
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return calls


def _qwen_parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for index, match in enumerate(QWEN_TOOL_CALL_RE.finditer(text or "")):
        arguments: Dict[str, Any] = {}
        for param in QWEN_PARAMETER_RE.finditer(match.group("body") or ""):
            arguments[param.group("name")] = _qwen_parse_parameter_value(param.group("value"))
        calls.append(
            {
                "id": f"call_{index}_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": arguments,
                },
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return calls


def _gemma_render_tool_response(name: str, response: Any) -> str:
    response_json = json.dumps(response, ensure_ascii=False, default=str)
    return f"<|tool_response>response:{name}{{value:{response_json}}}<tool_response|>"


def _qwen_render_tool_response(name: str, response: Any) -> str:
    """Render a tool result the way the Qwen chat template renders ``role: tool``.

    The template wraps tool messages in ``<tool_response>`` inside a user turn,
    so standalone (non-chat-template) rendering must match to avoid prompt drift
    between training rollouts and inference.
    """

    response_json = json.dumps(response, ensure_ascii=False, default=str)
    return f"<tool_response>\n{response_json}\n</tool_response>"


@dataclass(frozen=True)
class ToolDialect:
    """Everything that differs between model tool-call surface syntaxes."""

    name: str
    parse_tool_calls: Callable[[str], List[Dict[str, Any]]]
    render_tool_response: Callable[[str, Any], str]
    tool_call_re: Pattern[str]
    tool_call_marker_re: Pattern[str]
    raw_call_name_re: Pattern[str]
    tool_response_re: Pattern[str]
    # Substrings that must never appear inside <sql_code> (format-reward guard).
    sql_leak_markers: tuple = ()
    # Sampling stop strings that end the assistant turn right after a call.
    stop_sequences: tuple = ()

    def text_before_first_tool_call(self, text: str) -> str:
        match = self.tool_call_marker_re.search(text or "")
        if not match:
            return (text or "").strip()
        return (text or "")[: match.start()].strip()


GEMMA_DIALECT = ToolDialect(
    name="gemma",
    parse_tool_calls=_gemma_parse_tool_calls,
    render_tool_response=_gemma_render_tool_response,
    tool_call_re=GEMMA_TOOL_CALL_RE,
    tool_call_marker_re=GEMMA_TOOL_CALL_MARKER_RE,
    raw_call_name_re=GEMMA_RAW_CALL_NAME_RE,
    tool_response_re=GEMMA_TOOL_RESPONSE_RE,
    sql_leak_markers=("call:", "```"),
    stop_sequences=(),
)

QWEN_DIALECT = ToolDialect(
    name="qwen",
    parse_tool_calls=_qwen_parse_tool_calls,
    render_tool_response=_qwen_render_tool_response,
    tool_call_re=QWEN_TOOL_CALL_RE,
    tool_call_marker_re=QWEN_TOOL_CALL_MARKER_RE,
    raw_call_name_re=QWEN_RAW_CALL_NAME_RE,
    tool_response_re=QWEN_TOOL_RESPONSE_RE,
    sql_leak_markers=("<tool_call>", "<function=", "```"),
    stop_sequences=("</tool_call>",),
)

DIALECTS: Dict[str, ToolDialect] = {
    GEMMA_DIALECT.name: GEMMA_DIALECT,
    QWEN_DIALECT.name: QWEN_DIALECT,
}

_ENV_VAR = "NL2SQL_TOOL_DIALECT"
_DEFAULT_DIALECT = "gemma"


def detect_dialect_name(model_name_or_path: str) -> str:
    """Guess the dialect from a model path. Falls back to the Gemma default."""

    lowered = (model_name_or_path or "").lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return _DEFAULT_DIALECT


def set_dialect(name: str) -> ToolDialect:
    """Select the active dialect process-wide.

    Written to the environment so DataLoader workers and any subprocess that
    re-imports this module inherit the same choice.
    """

    key = (name or "").strip().lower()
    if key not in DIALECTS:
        raise ValueError(
            f"Unknown tool dialect {name!r}. Available: {sorted(DIALECTS)}"
        )
    os.environ[_ENV_VAR] = key
    return DIALECTS[key]


def get_dialect(name: Optional[str] = None) -> ToolDialect:
    """Return the active dialect (explicit arg > env var > Gemma default)."""

    key = (name or os.environ.get(_ENV_VAR) or _DEFAULT_DIALECT).strip().lower()
    if key not in DIALECTS:
        raise ValueError(
            f"Unknown tool dialect {key!r}. Available: {sorted(DIALECTS)}"
        )
    return DIALECTS[key]


def contains_any_tool_call_marker(text: str) -> bool:
    """True if ``text`` starts a tool call in *any* known dialect.

    Used as a safety guard rather than a parser: a rollout that opened a tool
    call but never reached a final answer must not have draft SQL scraped out
    of its scratchpad or out of the call's own ``sql`` argument. Checking every
    dialect keeps that guard intact even if the active dialect is misconfigured.
    """

    if not text:
        return False
    return any(dialect.tool_call_marker_re.search(text) for dialect in DIALECTS.values())
