"""Action space for the computer-use agent, plus tolerant parsing of model output.

The model is asked to emit a single JSON object per step. We parse tolerantly
because VLMs wrap JSON in prose, fences, or <tool_call> tags depending on mood.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class ActionError(ValueError):
    """Model emitted something we could not turn into an action."""


@dataclass
class Action:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    # Free-text the model produced alongside the action. Kept for trajectory logs.
    reasoning: str = ""

    def __str__(self) -> str:
        arg_s = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({arg_s})"


# name -> (required args, optional args). Keep this small; every action you add
# is another thing the model can get wrong.
SCHEMA: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "click_id":     (("id",), ("button",)),
    "click":        (("x", "y"), ("button",)),
    "double_click": (("x", "y"), ()),
    "type":         (("text",), ("enter",)),
    "key":          (("keys",), ()),
    "scroll":       (("dy",), ("x", "y", "dx")),
    "goto":         (("url",), ()),
    "back":         ((), ()),
    "wait":         ((), ("ms",)),
    "finish":       (("answer",), ()),
    "give_up":      (("reason",), ()),
}

TERMINAL = {"finish", "give_up"}

ACTION_DOCS = """\
click_id(id)                 - PREFERRED. Click the element with that numbered badge.
click(point, button="left")  - click at point [x, y]. Only when no badge covers the target.
double_click(point)          - double-click at point [x, y].
type(text, enter=false)      - type text into the focused element. Set enter=true to press Enter after.
key(keys)                    - press a key combo, e.g. "Enter", "Control+a", "Escape", "Tab".
scroll(dy, point=null)       - scroll vertically by dy pixels (positive = down). Optionally centred on point [x, y].
goto(url)                    - navigate directly to a URL.
back()                       - browser back.
wait(ms=1000)                - wait for the page to settle.
finish(answer)               - the task is complete; answer is your result string.
give_up(reason)              - the task cannot be completed; explain why."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_BARE_Y = re.compile(r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?=[},])')


def _candidate_blobs(text: str) -> list[str]:
    """Ordered candidate JSON strings, most-likely-correct first."""
    # Reasoning traces contain hypothetical actions the model then rejects.
    # Parsing one of those would execute a discarded plan, so drop them first.
    text = _THINK.sub("", text)
    out: list[str] = []
    out += [m.strip() for m in _TOOL_TAG.findall(text)]
    out += [m.strip() for m in _FENCE.findall(text)]
    # Last resort: balanced-brace scan, last object wins (models often restate).
    depth, start = 0, None
    found: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                found.append(text[start : i + 1])
                start = None
            depth = max(depth, 0)
    out += list(reversed(found))

    # Observed failure: {"x": 500, 490} -- a bare second number where "y" should
    # be, which is not valid JSON. Repair it rather than losing the whole step.
    repaired = [_BARE_Y.sub(r'"x": \1, "y": \2', b) for b in out]
    out += [r for r, o in zip(repaired, out) if r != o]
    return out


_POINT_KEYS = ("point", "coordinate", "coord", "position", "xy")


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Fold the several ways the model expresses a point into flat x/y.

    Qwen grounds natively as an [x, y] pair, so it emits `{"point": [x, y]}` or
    even `{"x": [x, y]}` regardless of what the prompt asks for. Meet it where it
    is rather than fighting the prior.
    """
    args = dict(args)
    for key in _POINT_KEYS:
        v = args.pop(key, None)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            args.setdefault("x", v[0])
            args.setdefault("y", v[1])
    # {"x": [132, 138]} -- both coordinates packed into x.
    x = args.get("x")
    if isinstance(x, (list, tuple)) and len(x) == 2 and args.get("y") is None:
        args["x"], args["y"] = x[0], x[1]
    # A nested [[x, y]] box/point, occasionally emitted for grounding.
    if isinstance(args.get("x"), (list, tuple)) and len(args["x"]) == 1:
        inner = args["x"][0]
        if isinstance(inner, (list, tuple)) and len(inner) == 2:
            args["x"], args["y"] = inner[0], inner[1]
    return args


def _coerce(obj: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Accept the several shapes models emit for 'call this action'."""
    # {"action": "click", "x": 1, "y": 2} or {"action": "click", "args"/"arguments": {...}}
    for key in ("action", "name", "function", "tool"):
        if key in obj and isinstance(obj[key], str):
            name = obj[key]
            for akey in ("args", "arguments", "parameters", "action_input"):
                if isinstance(obj.get(akey), dict):
                    return name, dict(obj[akey])
            args = {k: v for k, v in obj.items() if k not in
                    ("action", "name", "function", "tool", "reasoning", "thought")}
            return name, args
    # {"click": {"x": 1, "y": 2}}
    if len(obj) == 1:
        (k, v), = obj.items()
        if k in SCHEMA and isinstance(v, dict):
            return k, dict(v)
    return None


def parse_action(text: str) -> Action:
    """Extract an Action from raw model output. Raises ActionError if impossible."""
    errors: list[str] = []
    for blob in _candidate_blobs(text):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        coerced = _coerce(obj)
        if coerced is None:
            continue
        name, args = coerced
        name = name.strip()
        if name not in SCHEMA:
            errors.append(f"unknown action {name!r}")
            continue
        args = _normalize_args(args)
        required, optional = SCHEMA[name]
        missing = [r for r in required if r not in args]
        if missing:
            errors.append(f"{name} missing required arg(s): {', '.join(missing)}")
            continue
        allowed = set(required) | set(optional)
        args = {k: v for k, v in args.items() if k in allowed}
        reasoning = ""
        for rk in ("reasoning", "thought"):
            if isinstance(obj.get(rk), str):
                reasoning = obj[rk]
                break
        return Action(name=name, args=args, reasoning=reasoning)

    detail = f" ({'; '.join(errors)})" if errors else ""
    raise ActionError(f"no valid action found in model output{detail}")
