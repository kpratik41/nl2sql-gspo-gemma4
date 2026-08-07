#!/usr/bin/env python3
"""Run one computer-use task.

    .venv/bin/python run_task.py "Find the current top story on news.ycombinator.com"
    .venv/bin/python run_task.py --start-url https://example.com "Summarise this page" --steps 10
"""
from __future__ import annotations

import argparse
import sys

from harness import Agent, AgentConfig, BrowserEnv, CoordSpace, VLMClient, VLMConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task", help="natural-language task for the agent")
    p.add_argument("--start-url", default="https://www.google.com")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--keep-images", type=int, default=3,
                   help="screenshots kept at full res in context (cost dial)")
    p.add_argument("--coord-space", default="norm_1000",
                   choices=[c.value for c in CoordSpace],
                   help="run scripts/calibrate.py to determine this")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--base-url", action="append", default=None,
                   help="vLLM endpoint; repeat for multiple replicas")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--headed", action="store_true", help="show the browser (needs a display)")
    args = p.parse_args()

    client = VLMClient(VLMConfig(
        model=args.model,
        base_urls=args.base_url or ["http://127.0.0.1:8000/v1"],
    ))
    if not client.health():
        print(f"No vLLM server reachable at {client.cfg.base_urls}.\n"
              f"Start one with:  ./scripts/serve.sh", file=sys.stderr)
        return 1

    env = BrowserEnv(width=args.width, height=args.height,
                     headless=not args.headed, start_url=args.start_url)
    cfg = AgentConfig(max_steps=args.steps, keep_images=args.keep_images,
                      coord_space=CoordSpace(args.coord_space))

    print(f"task: {args.task}\nstart: {args.start_url}\n")
    with env:
        result = Agent(client, env, cfg).run(args.task)
    return 0 if result.status == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
