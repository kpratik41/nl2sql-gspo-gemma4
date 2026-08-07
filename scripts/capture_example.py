#!/usr/bin/env python3
"""Run one task while recording the exact messages sent to the LLM.

The normal trajectory log keeps model *output* but not the outgoing prompt.
This wraps VLMClient.chat to capture both sides verbatim, so a run can be
documented turn by turn.

    .venv/bin/python scripts/capture_example.py > runs/capture.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import Agent, AgentConfig, BrowserEnv, VLMClient, VLMConfig

TASK = ("Search Wikipedia for 'Hedy Lamarr' and tell me what invention she is "
        "credited as a co-inventor of.")
START_URL = "https://en.wikipedia.org"


def redact(messages):
    """Replace image payloads with a description; keep everything else verbatim."""
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            out.append({"role": m["role"], "content": c})
            continue
        parts = []
        for p in c:
            if p.get("type") == "image_url":
                b64 = p["image_url"]["url"].split(",", 1)[1]
                nbytes = len(b64) * 3 // 4
                parts.append({"type": "image", "bytes": nbytes})
            else:
                parts.append(p)
        out.append({"role": m["role"], "content": parts})
    return out


def main() -> int:
    turns = []
    client = VLMClient(VLMConfig())
    original = client.chat

    def recording_chat(messages, **kw):
        msgs = list(messages)
        t0 = time.time()
        resp = original(msgs, **kw)
        turns.append({
            "turn": len(turns) + 1,
            "latency_s": round(time.time() - t0, 3),
            "request": redact(msgs),
            "response": resp,
        })
        return resp

    client.chat = recording_chat

    env = BrowserEnv(start_url=START_URL)
    with env:
        result = Agent(client, env, AgentConfig(max_steps=12, verbose=False)).run(TASK)

    json.dump({
        "task": TASK,
        "start_url": START_URL,
        "model": client.cfg.model,
        "sampling": {
            "temperature": client.cfg.temperature,
            "top_p": client.cfg.top_p,
            "max_tokens": client.cfg.max_tokens,
            "enable_thinking": client.cfg.enable_thinking,
        },
        "result": {"status": result.status, "answer": result.answer,
                   "steps": result.steps, "trajectory": result.trajectory_dir},
        "turns": turns,
    }, sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
