#!/usr/bin/env python3
"""Assistant-only label masking for multi-turn tool-calling SFT.

Loss must be attributed **only** to tokens the model itself would generate:

* system / user turns          -> masked
* tool responses               -> masked (environment output; supervising it
                                  teaches the model to hallucinate query results)
* assistant reasoning + tool call -> supervised
* final assistant answer       -> supervised

The transcript format puts ``tool_calls`` *and* ``tool_responses`` inside a
single assistant message, so a naive "supervise everything the assistant message
rendered" rule would train on tool output. The boundary is found by rendering
the same message twice, once with its ``tool_responses`` stripped.

Sequences are never truncated. A record that does not fit ``max_seq_len`` is
reported so the caller can drop it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

IGNORE_INDEX = -100


def _render(tokenizer, messages: List[Dict[str, Any]], tools, add_generation_prompt: bool) -> List[int]:
    out = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(out, dict):
        return list(out["input_ids"])
    if hasattr(out, "input_ids"):
        return list(out.input_ids)
    return list(out)


def _strip_tool_responses(message: Dict[str, Any]) -> Dict[str, Any]:
    trimmed = dict(message)
    if trimmed.get("tool_responses"):
        trimmed["tool_responses"] = []
    return trimmed


def _longest_common_prefix(left: List[int], right: List[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def tool_response_opener_ids(tokenizer) -> List[int]:
    """Token ids the template emits to open a tool-response block.

    The chat template appends this marker to any assistant turn carrying a tool
    call, even when the responses list is empty. It is inserted by the harness at
    inference time, not produced by the model, so it must not be supervised.
    """
    ids: List[int] = []
    for marker in ("<|tool_response>", "<|tool_response|>"):
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            continue
        if len(encoded) == 1:
            ids.append(encoded[0])
    return ids


def build_supervised_example(
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Return ``(input_ids, labels, stats)`` with only assistant spans supervised.

    Walks the conversation prefix by prefix. For each assistant message the
    supervised span runs from the end of the generation prompt to the end of the
    assistant turn *excluding* any tool response the same message carries.
    """
    input_ids = _render(tokenizer, messages, tools, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(input_ids)
    opener_ids = set(tool_response_opener_ids(tokenizer))

    supervised_tokens = 0
    supervised_spans = 0
    boundary_failures = 0
    trimmed_openers = 0

    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        # Where the model would start generating this turn. The generation
        # prompt can speculatively open and close an empty channel, so it is not
        # a strict prefix of the real render -- take the longest common prefix.
        prefix_ids = _render(tokenizer, messages[:i], tools, add_generation_prompt=True)
        start = _longest_common_prefix(prefix_ids, input_ids)

        # Where the assistant turn ends, before any tool response it carries.
        upto_ids = _render(
            tokenizer,
            list(messages[:i]) + [_strip_tool_responses(message)],
            tools,
            add_generation_prompt=False,
        )
        end = len(upto_ids)

        # Drop the trailing tool-response opener the template appends to any
        # turn with a tool call; the harness emits it, not the model.
        while end > start and input_ids[end - 1] in opener_ids:
            end -= 1
            trimmed_openers += 1

        if start >= end or end > len(input_ids) or input_ids[:end] != upto_ids[:end]:
            boundary_failures += 1
            continue

        for position in range(start, end):
            labels[position] = input_ids[position]
        supervised_tokens += end - start
        supervised_spans += 1

    stats = {
        "n_tokens": len(input_ids),
        "n_supervised_tokens": supervised_tokens,
        "n_supervised_spans": supervised_spans,
        "n_boundary_failures": boundary_failures,
        "n_trimmed_tool_response_openers": trimmed_openers,
        "supervised_fraction": round(supervised_tokens / len(input_ids), 4) if input_ids else 0.0,
    }
    return input_ids, labels, stats
