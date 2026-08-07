"""Trajectory logging.

Write this from day one. Debugging a computer-use agent without replay is
guesswork, and if you later want SFT/RL data, this format *is* the dataset --
retrofitting it after the fact is painful.

Layout:
    runs/<ts>_<slug>/
        meta.json        task, config, final status
        steps.jsonl      one record per step
        step_000.png     the screenshot the model saw at that step
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return (s[:n] or "task").rstrip("-")


@dataclass
class Step:
    index: int
    screenshot: str            # filename relative to the run dir
    url: str = ""
    raw_output: str = ""       # exactly what the model emitted
    action: str = ""           # parsed, stringified
    action_json: dict[str, Any] = field(default_factory=dict)
    error: str | None = None   # env error from executing this action
    parse_error: str | None = None
    latency_s: float = 0.0
    prompt_images: int = 0     # how many images were in context this step


class Trajectory:
    def __init__(self, root: Path | str, task: str, config: dict[str, Any] | None = None):
        self.task = task
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{ts}_{_slug(task)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[Step] = []
        self._t0 = time.time()
        self._meta: dict[str, Any] = {
            "task": task,
            "started": ts,
            "config": config or {},
            "status": "running",
        }
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self._meta, indent=2))

    def save_screenshot(self, index: int, png: bytes) -> str:
        name = f"step_{index:03d}.png"
        (self.dir / name).write_bytes(png)
        return name

    def add(self, step: Step) -> None:
        self.steps.append(step)
        with (self.dir / "steps.jsonl").open("a") as f:
            f.write(json.dumps(asdict(step)) + "\n")

    def finish(self, status: str, result: str = "") -> None:
        self._meta.update(
            status=status,
            result=result,
            steps=len(self.steps),
            wall_s=round(time.time() - self._t0, 2),
        )
        self._write_meta()

    def __str__(self) -> str:
        return str(self.dir)
