"""Thin client for an OpenAI-compatible VLM endpoint (vLLM).

Accepts a list of base URLs so you can point at N vLLM replicas and get
round-robin load spreading for free once you scale past one GPU.
"""
from __future__ import annotations

import base64
import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


@dataclass
class VLMConfig:
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    base_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8000/v1"])
    api_key: str = "EMPTY"
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 1024
    timeout_s: float = 180.0
    # Qwen3.6 reasons by default and will happily spend >10k tokens deliberating
    # before emitting an action -- and can fall into repetition loops when the
    # screen does not match the instruction. An agent loop wants a fast, terse
    # decision per step, so thinking is off unless you deliberately enable it.
    enable_thinking: bool = False


class VLMClient:
    def __init__(self, cfg: VLMConfig | None = None):
        from openai import OpenAI

        self.cfg = cfg or VLMConfig()
        self._clients = [
            OpenAI(base_url=u, api_key=self.cfg.api_key, timeout=self.cfg.timeout_s, max_retries=2)
            for u in self.cfg.base_urls
        ]
        self._rr = itertools.cycle(range(len(self._clients)))
        self._lock = threading.Lock()

    def _next_client(self):
        with self._lock:
            return self._clients[next(self._rr)]

    def chat(self, messages: Iterable[dict[str, Any]], **overrides) -> str:
        """Returns the assistant's text. Reasoning-style output is folded in."""
        client = self._next_client()
        thinking = overrides.get("enable_thinking", self.cfg.enable_thinking)
        resp = client.chat.completions.create(
            model=overrides.get("model", self.cfg.model),
            messages=list(messages),
            temperature=overrides.get("temperature", self.cfg.temperature),
            top_p=overrides.get("top_p", self.cfg.top_p),
            max_tokens=overrides.get("max_tokens", self.cfg.max_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        msg = resp.choices[0].message
        text = msg.content or ""
        # Some builds surface chain-of-thought separately; keep it for the log
        # but put the visible content last so action parsing sees it.
        thinking = getattr(msg, "reasoning_content", None)
        if thinking and not text.strip():
            text = thinking
        return text

    def health(self) -> bool:
        try:
            self._clients[0].models.list()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Message construction

def user_message(text: str, png: bytes | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if png is not None:
        content.append({"type": "image_url", "image_url": {"url": png_data_url(png)}})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def strip_images(message: dict[str, Any], placeholder: str) -> dict[str, Any]:
    """Replace image parts with a text stub -- the core context-budget lever.

    A 1280x800 screenshot costs on the order of a thousand vision tokens. Keeping
    every one of them makes step N cost O(N) images. Keep the recent few; stub
    the rest.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message
    kept = [c for c in content if c.get("type") != "image_url"]
    if len(kept) == len(content):
        return message
    return {**message, "content": [{"type": "text", "text": placeholder}] + kept}
