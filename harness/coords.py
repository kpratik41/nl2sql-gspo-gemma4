"""Model-output coordinate space -> screen pixels.

This is the single most common source of silent breakage in a computer-use
harness: the model points at the right thing, but in a coordinate convention
you guessed wrong about, so every click lands slightly (or wildly) off.

Do not guess. Run scripts/calibrate.py and set COORD_SPACE from the result.
"""
from __future__ import annotations

from enum import Enum


class CoordSpace(str, Enum):
    PIXEL = "pixel"          # coords are pixels in the image the model was shown
    NORM_1000 = "norm_1000"  # coords are 0..999 relative
    NORM_1 = "norm_1"        # coords are 0.0..1.0 relative

    def to_pixels(self, x: float, y: float, img_w: int, img_h: int) -> tuple[float, float]:
        if self is CoordSpace.PIXEL:
            return float(x), float(y)
        if self is CoordSpace.NORM_1000:
            return float(x) / 1000.0 * img_w, float(y) / 1000.0 * img_h
        return float(x) * img_w, float(y) * img_h


def rescale(x: float, y: float, from_w: int, from_h: int, to_w: int, to_h: int) -> tuple[float, float]:
    """Map a point between two image sizes (used when we downscale before sending)."""
    if from_w == to_w and from_h == to_h:
        return x, y
    return x * (to_w / from_w), y * (to_h / from_h)


def guess_space(samples: list[tuple[float, float]], img_w: int, img_h: int) -> CoordSpace:
    """Heuristic fallback when you have raw model coords but no calibration run.

    Only a hint -- ambiguous whenever the viewport is near 1000px wide, which is
    exactly the common case. Prefer calibrate.py.
    """
    if not samples:
        return CoordSpace.PIXEL
    xs = [abs(x) for x, _ in samples]
    ys = [abs(y) for _, y in samples]
    if max(xs + ys) <= 1.5:
        return CoordSpace.NORM_1
    # If any coord exceeds the image bounds, it cannot be pixel space.
    if max(xs) > img_w * 1.02 or max(ys) > img_h * 1.02:
        return CoordSpace.NORM_1000
    return CoordSpace.PIXEL
