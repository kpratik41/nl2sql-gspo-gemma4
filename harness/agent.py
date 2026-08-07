"""The agent loop: observe -> think -> act -> repeat."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import ACTION_DOCS, TERMINAL, Action, ActionError, parse_action
from .coords import CoordSpace
from .env import Env, Observation
from .model import VLMClient, strip_images, user_message
from .trajectory import Step, Trajectory

SYSTEM_PROMPT = """\
You are a computer-use agent operating a web browser. Each turn you receive a \
screenshot of the current {width}x{height} browser viewport, and you reply with \
exactly one action.

Available actions:
{actions}

Rules:
- Reply with ONE JSON object and nothing else, in this shape:
  {{"reasoning": "<one short sentence on what you see and why this action>", "action": "<name>", "args": {{...}}}}
- Interactive elements are outlined with a small coloured number badge. To click
  one, use its number: {{"reasoning": "...", "action": "click_id", "args": {{"id": 7}}}}
  This is far more reliable than aiming at a pixel -- always prefer it.
- Only if no badge covers your target, give a location as "point": [x, y], e.g.
  {{"reasoning": "empty canvas area", "action": "click", "args": {{"point": [512, 331]}}}}
- Look at the screenshot before every action. Do not assume the page changed the \
way you expected -- verify, then act.
- Prefer clicking visible UI over guessing URLs, but goto() is fine for a known site.
- If a click did nothing, do not repeat it identically. Scroll, wait, or try a \
different target.
- When the task is done, call finish(answer). If it is genuinely impossible, call \
give_up(reason). Do not loop forever.\
"""


@dataclass
class AgentConfig:
    max_steps: int = 25
    # Screenshots kept at full resolution. Older ones become text stubs. This is
    # the main context-cost dial: cost per step is O(keep_images), not O(steps).
    keep_images: int = 3
    # Verified by scripts/calibrate.py against Qwen3.6-35B-A3B-FP8: the model
    # grounds in 0..1000 normalised space, NOT screenshot pixels. Re-run
    # calibration if you change model or preprocessing.
    coord_space: CoordSpace = CoordSpace.NORM_1000
    runs_dir: str = "runs"
    verbose: bool = True


@dataclass
class Result:
    status: str          # "finished" | "gave_up" | "max_steps" | "error"
    answer: str
    steps: int
    trajectory_dir: str


class Agent:
    def __init__(self, client: VLMClient, env: Env, cfg: AgentConfig | None = None):
        self.client = client
        self.env = env
        self.cfg = cfg or AgentConfig()

    # -- coordinate handling ----------------------------------------------
    def _to_pixels(self, action: Action, obs: Observation) -> Action:
        a = dict(action.args)
        if "x" in a and "y" in a and a["x"] is not None and a["y"] is not None:
            try:
                x, y = self.cfg.coord_space.to_pixels(
                    float(a["x"]), float(a["y"]), obs.width, obs.height
                )
            except (TypeError, ValueError):
                return action
            # Clamp: an off-screen click is a silent no-op in Playwright, which
            # looks like "the model did nothing" and wastes debugging time.
            a["x"] = max(0.0, min(x, obs.width - 1))
            a["y"] = max(0.0, min(y, obs.height - 1))
        return Action(name=action.name, args=a, reasoning=action.reasoning)

    # -- prompt ------------------------------------------------------------
    def _observation_text(self, obs: Observation, step: int, task: str) -> str:
        parts = [f"Step {step + 1}/{self.cfg.max_steps}."]
        if obs.info.get("url"):
            parts.append(f"Current URL: {obs.info['url']}")
        if obs.info.get("title"):
            parts.append(f"Page title: {obs.info['title']}")
        if obs.info.get("tabs", 1) > 1:
            parts.append(f"({obs.info['tabs']} tabs open; you are on the newest.)")
        marks = obs.info.get("marks") or []
        if marks:
            # The badge numbers are legible in the image, but the labels often are
            # not at this resolution. Listing them as text costs a few hundred
            # tokens and removes most of the OCR guesswork.
            listing = "  ".join(f"[{m['id']}] {m['label']}" for m in marks[:120] if m["label"])
            parts.append(f"Numbered elements on screen:\n{listing}")
        if obs.error:
            parts.append(f"The previous action FAILED with: {obs.error}")
        parts.append(f"Task: {task}")
        parts.append("Reply with one JSON action.")
        return "\n".join(parts)

    def _prune(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the most recent `keep_images` screenshots at full res."""
        image_idxs = [
            i for i, m in enumerate(messages)
            if isinstance(m.get("content"), list)
            and any(c.get("type") == "image_url" for c in m["content"])
        ]
        stale = set(image_idxs[: max(0, len(image_idxs) - self.cfg.keep_images)])
        return [
            strip_images(m, "[earlier screenshot omitted to save context]") if i in stale else m
            for i, m in enumerate(messages)
        ]

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)

    # -- main loop ---------------------------------------------------------
    def run(self, task: str) -> Result:
        traj = Trajectory(
            Path(self.cfg.runs_dir),
            task,
            config={
                "model": self.client.cfg.model,
                "max_steps": self.cfg.max_steps,
                "keep_images": self.cfg.keep_images,
                "coord_space": self.cfg.coord_space.value,
            },
        )
        obs = self.env.reset(task)
        messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                width=obs.width, height=obs.height, actions=ACTION_DOCS
            ),
        }]
        status, answer = "max_steps", ""

        for i in range(self.cfg.max_steps):
            shot = traj.save_screenshot(i, obs.png)
            messages.append(user_message(self._observation_text(obs, i, task), obs.png))
            messages = self._prune(messages)
            n_images = sum(
                1 for m in messages if isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"])
            )

            t0 = time.time()
            try:
                raw = self.client.chat(messages)
            except Exception as e:
                traj.add(Step(index=i, screenshot=shot, url=obs.info.get("url", ""),
                              error=f"model call failed: {e}", latency_s=time.time() - t0))
                traj.finish("error", str(e))
                return Result("error", str(e), i, str(traj))
            latency = time.time() - t0

            step = Step(index=i, screenshot=shot, url=obs.info.get("url", ""),
                        raw_output=raw, latency_s=round(latency, 2), prompt_images=n_images)

            try:
                action = parse_action(raw)
            except ActionError as e:
                step.parse_error = str(e)
                traj.add(step)
                self._log(f"  [{i+1}] unparseable output: {e}")
                messages.append({"role": "assistant", "content": raw})
                messages.append(user_message(
                    f"That was not a valid action ({e}). Reply with exactly one JSON "
                    f'object: {{"reasoning": "...", "action": "...", "args": {{...}}}}'
                ))
                obs = self.env.observe() if hasattr(self.env, "observe") else obs
                continue

            step.action = str(action)
            step.action_json = {"name": action.name, "args": action.args,
                                "reasoning": action.reasoning}
            self._log(f"  [{i+1}] {action}  ({latency:.1f}s)"
                      + (f" - {action.reasoning}" if action.reasoning else ""))

            if action.name in TERMINAL:
                answer = str(action.args.get("answer") or action.args.get("reason") or "")
                status = "finished" if action.name == "finish" else "gave_up"
                traj.add(step)
                break

            executed = self._to_pixels(action, obs)
            obs = self.env.step(executed)
            step.error = obs.error
            if obs.error:
                self._log(f"       ! {obs.error}")
            traj.add(step)

            messages.append({"role": "assistant", "content": raw})

        traj.finish(status, answer)
        self._log(f"\n{status}: {answer or '(no answer)'}\ntrajectory: {traj}")
        return Result(status, answer, len(traj.steps), str(traj))
