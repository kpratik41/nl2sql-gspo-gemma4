#!/usr/bin/env python3
"""Determine which coordinate convention the model actually outputs.

Renders a page of labelled targets at known pixel positions, asks the model to
click each one, then scores every candidate coordinate space by mean error.

Run this before trusting any click. Thirty seconds here saves a day of "the
agent clicks slightly up and to the left of everything".

    .venv/bin/python scripts/calibrate.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.actions import ActionError, parse_action
from harness.coords import CoordSpace
from harness.env import BrowserEnv
from harness.model import VLMClient, VLMConfig, user_message

W, H = 1280, 800

# (label, centre_x, centre_y) -- spread across the viewport, avoiding edges so a
# coordinate-space mismatch shows up as a large, unambiguous error.
#
# Labels are SINGLE digits on purpose. Two-character labels ("A1", "J9") get read
# as just their first glyph, and asking the model to click a label it cannot find
# derails it into a repetition loop -- which measures prompt design, not
# coordinates.
TARGETS = [
    ("1", 160, 120), ("2", 640, 120), ("3", 1120, 120),
    ("4", 160, 400), ("5", 640, 400), ("6", 1120, 400),
    ("7", 160, 680), ("8", 640, 680), ("9", 1120, 680),
]
BTN_W, BTN_H = 120, 64


def build_page() -> str:
    boxes = "".join(
        f'<div class="t" style="left:{x - BTN_W // 2}px;top:{y - BTN_H // 2}px">{label}</div>'
        for label, x, y in TARGETS
    )
    return (
        "data:text/html;charset=utf-8,"
        + (
            "<html><head><style>"
            "body{margin:0;background:#f2f2f4;font-family:system-ui,sans-serif;"
            f"width:{W}px;height:{H}px;overflow:hidden}}"
            ".t{position:absolute;width:%dpx;height:%dpx;background:#2563eb;color:#fff;"
            "display:flex;align-items:center;justify-content:center;font-size:28px;"
            "font-weight:700;border-radius:10px}" % (BTN_W, BTN_H)
            + "</style></head><body>" + boxes + "</body></html>"
        ).replace("#", "%23").replace("\n", "")
    )


PROMPT = """\
This is a {w}x{h} screenshot containing labelled blue buttons.

Reply with ONE JSON object and nothing else:
{{"action": "click", "args": {{"point": [<x>, <y>]}}}}

Click the exact centre of the button labelled "{label}"."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    args = ap.parse_args()

    client = VLMClient(VLMConfig(model=args.model, base_urls=[args.base_url],
                                 temperature=0.0, enable_thinking=False))
    if not client.health():
        print(f"No vLLM server at {args.base_url}. Start it with ./scripts/serve.sh", file=sys.stderr)
        return 1

    env = BrowserEnv(width=W, height=H, start_url=build_page())
    with env:
        obs = env.reset("")
        Path("runs").mkdir(exist_ok=True)
        Path("runs/calibration.png").write_bytes(obs.png)
        print(f"Rendered {W}x{H} calibration page -> runs/calibration.png\n")

        preds: list[tuple[str, float, float, int, int]] = []
        for label, tx, ty in TARGETS:
            raw = client.chat([user_message(PROMPT.format(w=W, h=H, label=label), obs.png)])
            try:
                a = parse_action(raw)
                px, py = float(a.args["x"]), float(a.args["y"])
            except (ActionError, KeyError, TypeError, ValueError) as e:
                print(f"  {label}: unparseable ({e}) :: {raw[:80]!r}")
                continue
            preds.append((label, px, py, tx, ty))
            print(f"  {label}: model said ({px:>7.1f}, {py:>7.1f})   true ({tx}, {ty})")

    if not preds:
        print("\nNo usable predictions -- check the server and the raw output above.", file=sys.stderr)
        return 1

    # Median, not mean: a single misidentified target (model clicked the wrong
    # button) is a grounding miss, not evidence about the coordinate space, and
    # one such outlier swamps a mean over nine points.
    tol = max(BTN_W, BTN_H) / 2
    print(f"\n{'space':<12} {'median err':>12} {'mean err':>10} {'on-target':>11}")
    print("-" * 47)
    scores: list[tuple[float, CoordSpace, list[float]]] = []
    for space in CoordSpace:
        errs = []
        for _, px, py, tx, ty in preds:
            cx, cy = space.to_pixels(px, py, W, H)
            errs.append(((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5)
        med = statistics.median(errs)
        hits = sum(1 for e in errs if e <= tol)
        scores.append((med, space, errs))
        print(f"{space.value:<12} {med:>12.1f} {statistics.mean(errs):>10.1f} "
              f"{hits:>7}/{len(errs)}")

    scores.sort(key=lambda s: s[0])
    med, best, errs = scores[0]
    hits = sum(1 for e in errs if e <= tol)
    print(f"\nBest fit: CoordSpace.{best.name}  (median error {med:.1f}px, "
          f"{hits}/{len(errs)} on target)")

    misses = [(lbl, e) for (lbl, *_), e in zip(preds, errs) if e > tol]
    if misses:
        print("\nOff-target predictions (likely misidentified targets, not a "
              "coordinate-space problem):")
        for lbl, e in misses:
            print(f"  target {lbl}: {e:.0f}px off")

    if med > tol:
        print("\nWARNING: the median miss exceeds half a button, so this is a systematic\n"
              "         problem -- grounding or preprocessing, not just the coord space.\n"
              "         Inspect runs/calibration.png and the raw predictions above.")
    else:
        print(f"\nUse:  --coord-space {best.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
