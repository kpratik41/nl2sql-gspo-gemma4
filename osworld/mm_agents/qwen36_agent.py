"""Qwen3.6-35B-A3B agent for OSWorld 2.0.

Subclasses the shipped Qwen3.5-VL agent rather than forking it, so upstream fixes
to prompting, folding, and pyautogui generation keep flowing through. Only the
two Qwen3.6-specific behaviours are overridden.

Serve the model with vLLM, then:

    from mm_agents.qwen36_agent import Qwen36Agent
    agent = Qwen36Agent(base_url="http://<gpu-host>:8000/v1", api_key="EMPTY")
"""
from __future__ import annotations

import re
from typing import List, Tuple

from mm_agents.qwen35vl_agent import Qwen35VLAgent

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

# Qwen3.6 emits reasoning inside <think>...</think>. The base agent's tool-call
# extractor scans the entire response with re.finditer, so a hypothetical call the
# model drafts *and then rejects* while reasoning would be executed as if it were
# the decision. Strip the reasoning before parsing.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Qwen36Agent(Qwen35VLAgent):
    """Qwen3.6 variant. Identical to the base agent except for reasoning handling.

    enable_thinking is a real trade-off on this benchmark, not a formality:

      True  - matches how the reported SOTA numbers were produced (Opus with
              maximum thinking) and should help with the constraint tracking and
              state recovery that OSWorld 2.0 is built to stress. Costs
              10-30x more decode tokens per step, so a 500-step episode gets
              much slower and more expensive.
      False - ~0.5s steps. Sends an empty prefilled <think></think> via the chat
              template. Expect worse long-horizon behaviour.

    Default is True because the benchmark rewards deliberation and max_tokens
    here (32768) is large enough to hold it. Measure both on a few tasks before
    committing to a full run.
    """

    def __init__(self, *args, enable_thinking: bool = True, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(*args, model=model, **kwargs)
        self.enable_thinking = enable_thinking

    def call_llm(self, payload: dict, model: str) -> str:
        # The base class forwards payload["extra_body"] to the OpenAI client but
        # never sets it, which is the hook we need: vLLM passes chat_template_kwargs
        # through to the Jinja template, where enable_thinking=false prefills an
        # empty <think></think>.
        payload = dict(payload)
        extra_body = dict(payload.get("extra_body") or {})
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        template_kwargs.setdefault("enable_thinking", self.enable_thinking)
        extra_body["chat_template_kwargs"] = template_kwargs
        payload["extra_body"] = extra_body
        return super().call_llm(payload, model)

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        return super().parse_response(
            _THINK.sub("", response or ""),
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )
