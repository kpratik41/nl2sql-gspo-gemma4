"""Shared plumbing for the experiment scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
DATA = ROOT / "data"

for _d in (RESULTS, FIGURES, DATA):
    _d.mkdir(parents=True, exist_ok=True)

# A fixed non-secret demo master secret keeps the whole study reproducible.
# A deployment reads this from a secrets manager instead -- see synthmark.keys.
DEMO_MASTER_SECRET = os.environ.get(
    "SYNTHMARK_MASTER_SECRET", "synthmark-demo-master-secret-not-for-production-use"
)

PRIMARY_KEY_ID = "eval/primary/v1"
OTHER_KEY_ID = "eval/other-business-unit/v1"

MODEL_ID = os.environ.get("SYNTHMARK_MODEL", "google/gemma-4-E4B-it")


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
