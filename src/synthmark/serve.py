"""A detection service, so that teams without ML infrastructure can check text.

Deployment notes
----------------
This service holds secret key material in memory.  Whoever can call it can
learn whether a given text carries the watermark; whoever can read its
configuration can *forge* the watermark.  Accordingly:

* Terminate TLS in front of it and require authenticated callers.
* Do not log request bodies.  Text submitted for checking is, by construction,
  text somebody is suspicious about; it may be sensitive.
* The response deliberately reports a probability-flavoured verdict and the
  number of tokens scored, never a bare boolean.  A detector answers "is this
  consistent with our watermark", not "who wrote this".

Run with::

    uvicorn synthmark.serve:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .detect import Calibration, Detector
from .keys import WatermarkKey, derive_key, load_master_secret

# --------------------------------------------------------------------- schemas


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Candidate text to check.")
    key_id: str | None = Field(
        None, description="Which watermark key to test against. Defaults to the service default."
    )
    method: Literal["mean", "weighted_mean"] = "mean"
    target_fpr: float = Field(
        0.01, gt=0, lt=1, description="False-positive rate the decision threshold is set to."
    )


class DetectResponse(BaseModel):
    verdict: Literal["watermark_detected", "no_watermark_detected", "text_too_short"]
    score: float
    num_tokens_scored: int
    p_value: float | None
    empirical_p_value: float | None
    threshold: float | None
    target_fpr: float
    key_id: str
    key_fingerprint: str
    interpretation: str


MIN_TOKENS = 40
"""Below this, the detector has too little signal for any verdict to be meaningful.

Refusing to answer is better than returning a confident-looking number on
50 characters of text, which is where false accusations come from.
"""


def _interpret(score: float, n: int, detected: bool, target_fpr: float) -> str:
    if not detected:
        return (
            f"No watermark detected at a {target_fpr:.1%} false-positive threshold over "
            f"{n} scored tokens. This is not evidence that the text is human-written: "
            "short, heavily edited, paraphrased, or low-entropy text (facts, code, "
            "structured output) carries little or no detectable watermark."
        )
    return (
        f"Statistically consistent with text generated using this watermark key, over "
        f"{n} scored tokens, at a {target_fpr:.1%} false-positive threshold. This "
        "indicates the model was likely involved in producing this text; it does not "
        "establish authorship, and cannot distinguish text the model wrote from text "
        "the model edited."
    )


# ------------------------------------------------------------------ app factory


def build_app(
    keys: dict[str, WatermarkKey],
    tokenizer,
    *,
    default_key_id: str,
    calibrations: dict[str, Calibration] | None = None,
    device: str = "cpu",
):
    """Build a FastAPI app serving one or more watermark keys.

    Multiple keys let a single service cover several business units without any
    of them being able to detect (or forge) another's watermark -- the keys are
    independent, so a text marked by one is null-distributed under another.
    """
    from fastapi import FastAPI, HTTPException

    detectors = {kid: Detector(k, tokenizer, device=device) for kid, k in keys.items()}
    calibrations = calibrations or {}

    app = FastAPI(
        title="synthmark detection service",
        description="Checks whether text carries a SynthID-Text watermark from a known key.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "keys": [
                {"key_id": kid, "fingerprint": k.fingerprint, "depth": k.depth}
                for kid, k in keys.items()
            ],
            "default_key_id": default_key_id,
            "calibrated": sorted(calibrations),
        }

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest) -> DetectResponse:
        key_id = req.key_id or default_key_id
        detector = detectors.get(key_id)
        if detector is None:
            raise HTTPException(404, f"unknown key_id {key_id!r}")

        result = detector.detect(
            req.text,
            method=req.method,
            calibration=calibrations.get(key_id),
            target_fpr=req.target_fpr,
        )

        if result.num_tokens_scored < MIN_TOKENS:
            return DetectResponse(
                verdict="text_too_short",
                score=result.score,
                num_tokens_scored=result.num_tokens_scored,
                p_value=result.p_value,
                empirical_p_value=result.empirical_p_value,
                threshold=result.threshold,
                target_fpr=req.target_fpr,
                key_id=key_id,
                key_fingerprint=result.key_fingerprint or "",
                interpretation=(
                    f"Only {result.num_tokens_scored} tokens could be scored; at least "
                    f"{MIN_TOKENS} are needed for a meaningful verdict. Submit more text."
                ),
            )

        # Fall back to the analytic p-value when no calibration is loaded.
        detected = result.is_watermarked
        if detected is None:
            detected = (result.p_value is not None) and (result.p_value < req.target_fpr)

        return DetectResponse(
            verdict="watermark_detected" if detected else "no_watermark_detected",
            score=result.score,
            num_tokens_scored=result.num_tokens_scored,
            p_value=result.p_value,
            empirical_p_value=result.empirical_p_value,
            threshold=result.threshold,
            target_fpr=req.target_fpr,
            key_id=key_id,
            key_fingerprint=result.key_fingerprint or "",
            interpretation=_interpret(
                result.score, result.num_tokens_scored, detected, req.target_fpr
            ),
        )

    return app


def _default_app():
    """Module-level app for ``uvicorn synthmark.serve:app``.

    Configured entirely through environment variables so no secret ever needs to
    live in the source tree:

    ``SYNTHMARK_MASTER_SECRET``  master secret (required)
    ``SYNTHMARK_KEY_IDS``        comma-separated key labels to serve
    ``SYNTHMARK_MODEL``          tokenizer to load
    ``SYNTHMARK_CALIBRATION``    optional path to a calibration JSON
    """
    from transformers import AutoTokenizer

    master = load_master_secret()
    key_ids = [k.strip() for k in os.environ.get("SYNTHMARK_KEY_IDS", "default/v1").split(",") if k.strip()]
    model_id = os.environ.get("SYNTHMARK_MODEL", "google/gemma-4-E4B-it")

    keys = {kid: derive_key(master, kid) for kid in key_ids}
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    calibrations = {}
    cal_path = os.environ.get("SYNTHMARK_CALIBRATION")
    if cal_path:
        cal = Calibration.load(cal_path)
        calibrations = {kid: cal for kid in key_ids}

    return build_app(keys, tokenizer, default_key_id=key_ids[0], calibrations=calibrations)


class _LazyApp:
    """Defer building the app (and loading the tokenizer) until first request."""

    def __init__(self):
        self._app = None

    def __call__(self, scope, receive, send):
        if self._app is None:
            self._app = _default_app()
        return self._app(scope, receive, send)


app = _LazyApp()
