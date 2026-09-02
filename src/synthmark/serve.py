"""The central detection service.

This is the only component that should ever hold watermark keys in a multi-team
deployment.  The reason is not tidiness, it is that **a watermark key is
symmetric**: whoever can use it to detect can also use it to forge.  Handing a
key to a downstream team so they can "just check locally" also hands them the
ability to manufacture text that your own detector will confirm as
platform-generated.  So the key stays here, behind an API, and consumers get
verdicts rather than key material.

Serving several models
----------------------
Each served model has its own key (see :mod:`synthmark.registry`), which is what
makes attribution possible: a text marked by the Nemotron key is
null-distributed under the Gemma key, so a hit identifies *which* model wrote
it, not merely that one of ours did.  Two endpoints follow from that:

``POST /detect``
    "Is this text marked by *this* key?"  Use it when you already know which
    model is in question -- the common case, since the caller usually has the
    request log.

``POST /attribute``
    "Is this text marked by *any* of our keys, and if so which?"  Use it for
    incident response, when the origin is exactly what you are trying to
    establish.  Testing many keys at once inflates the false-positive rate, so
    this endpoint applies a Bonferroni correction; see :func:`_attribute`.

Deployment notes
----------------
* Terminate TLS in front of it and require authenticated callers.
* Do not log request bodies.  Text submitted for checking is, by construction,
  text somebody is suspicious about; it may be sensitive.
* The response reports a probability-flavoured verdict and the number of tokens
  scored, never a bare boolean.  A detector answers "is this consistent with
  our watermark", not "who wrote this".
* It needs no GPU and no model weights -- only tokenizers -- so it can run on
  ordinary compute, far away from the serving fleet.

Run with::

    export SYNTHMARK_MASTER_SECRET="$(vault read -field=value secret/synthmark/master)"
    uvicorn synthmark.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .detect import Calibration, Detector
from .keys import WatermarkKey, load_master_secret
from .registry import KeyEntry, KeyRegistry

MIN_TOKENS = 40
"""Below this, the detector has too little signal for any verdict to be meaningful.

Refusing to answer is better than returning a confident-looking number on
50 characters of text, which is where false accusations come from.
"""


# --------------------------------------------------------------------- schemas


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Candidate text to check.")
    key_id: str | None = Field(
        None, description="Which watermark key to test against. Defaults to the service default."
    )
    model_id: str | None = Field(
        None,
        description="Alternative to key_id: name the model and the service picks its active key.",
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
    model_id: str
    interpretation: str


class AttributeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Candidate text of unknown origin.")
    method: Literal["mean", "weighted_mean"] = "mean"
    target_fpr: float = Field(
        0.01,
        gt=0,
        lt=1,
        description=(
            "False-positive rate for the scan as a whole, not per key. The per-key "
            "threshold is this divided by the number of keys tested."
        ),
    )
    include_retired: bool = Field(
        True,
        description=(
            "Test retired keys too. Leave this on: text generated under a retired key "
            "still exists, and excluding it is how an old incident becomes unattributable."
        ),
    )


class KeyScore(BaseModel):
    key_id: str
    model_id: str
    key_fingerprint: str
    status: str
    score: float
    num_tokens_scored: int
    p_value: float | None
    empirical_p_value: float | None
    threshold: float | None
    matched: bool


class AttributeResponse(BaseModel):
    verdict: Literal["attributed", "no_match", "text_too_short", "ambiguous"]
    best: KeyScore | None
    candidates: list[KeyScore]
    keys_tested: int
    target_fpr: float
    per_key_alpha: float
    interpretation: str


# ------------------------------------------------------------- served keys


@dataclass
class ServedKey:
    """One key the service can test against, with everything needed to do so."""

    entry: KeyEntry
    key: WatermarkKey
    detector: Detector
    calibration: Calibration | None = None

    @property
    def key_id(self) -> str:
        return self.entry.key_id


def build_served(
    registry: KeyRegistry,
    master_secret: bytes | str,
    *,
    tokenizer_loader: Callable[[str], object] | None = None,
    calibrations: dict[str, Calibration] | None = None,
    device: str = "cpu",
    verify_fingerprints: bool = True,
) -> dict[str, ServedKey]:
    """Turn a registry plus a master secret into ready-to-use detectors.

    Tokenizers are loaded once per distinct tokenizer id and shared, so serving
    ten keys across three model families costs three tokenizer loads, not ten.

    Args:
        tokenizer_loader: Injected for testing; defaults to
            ``transformers.AutoTokenizer.from_pretrained``.
        verify_fingerprints: Check each derived key against the fingerprint
            recorded in the registry.  Leave this on -- see
            :meth:`synthmark.registry.KeyEntry.resolve`.
    """
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = AutoTokenizer.from_pretrained

    calibrations = calibrations or {}
    tokenizers = {tid: tokenizer_loader(tid) for tid in registry.tokenizers()}

    served: dict[str, ServedKey] = {}
    for entry in registry:
        key = entry.resolve(master_secret, verify=verify_fingerprints)
        served[entry.key_id] = ServedKey(
            entry=entry,
            key=key,
            detector=Detector(key, tokenizers[entry.tokenizer], device=device),
            calibration=calibrations.get(entry.key_id),
        )
    return served


# ---------------------------------------------------------------- interpreting


def _interpret(n: int, detected: bool, target_fpr: float) -> str:
    if not detected:
        return (
            f"No watermark detected at a {target_fpr:.2%} false-positive threshold over "
            f"{n} scored tokens. This is not evidence that the text is human-written: "
            "short, heavily edited, paraphrased, or low-entropy text (facts, code, "
            "structured output) carries little or no detectable watermark."
        )
    return (
        f"Statistically consistent with text generated using this watermark key, over "
        f"{n} scored tokens, at a {target_fpr:.2%} false-positive threshold. This "
        "indicates the model was likely involved in producing this text; it does not "
        "establish authorship, and cannot distinguish text the model wrote from text "
        "the model edited."
    )


def _score_one(sk: ServedKey, text: str, method: str, alpha: float) -> tuple[KeyScore, object]:
    """Score ``text`` under one key and package it, without deciding a verdict."""
    result = sk.detector.detect(
        text, method=method, calibration=sk.calibration, target_fpr=alpha
    )
    detected = result.is_watermarked
    if detected is None:
        # No empirical calibration for this key: fall back to the analytic null.
        detected = (result.p_value is not None) and (result.p_value < alpha)
    return (
        KeyScore(
            key_id=sk.entry.key_id,
            model_id=sk.entry.model_id,
            key_fingerprint=sk.key.fingerprint,
            status=sk.entry.status,
            score=result.score,
            num_tokens_scored=result.num_tokens_scored,
            p_value=result.p_value,
            empirical_p_value=result.empirical_p_value,
            threshold=result.threshold,
            matched=bool(detected) and result.num_tokens_scored >= MIN_TOKENS,
        ),
        result,
    )


# ------------------------------------------------------------------ app factory


def build_app(
    served: dict[str, ServedKey],
    *,
    default_key_id: str | None = None,
    title: str = "synthmark detection service",
):
    """Build a FastAPI app over a set of served keys.

    Independent keys are what make this safe to run across business units and
    model families at once: a text marked under one key is null-distributed
    under every other, so no consumer learns anything about traffic that is not
    theirs, and a hit names the model.
    """
    from fastapi import FastAPI, HTTPException

    if not served:
        raise ValueError("at least one key must be served")
    if default_key_id is None:
        default_key_id = next(iter(served))
    if default_key_id not in served:
        raise ValueError(f"default_key_id {default_key_id!r} is not among the served keys")

    by_model = {sk.entry.model_id: kid for kid, sk in served.items() if sk.entry.is_active}

    app = FastAPI(
        title=title,
        description="Checks whether text carries a SynthID-Text watermark from a known key.",
        version="0.2.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "keys": [
                {
                    "key_id": sk.entry.key_id,
                    "fingerprint": sk.key.fingerprint,
                    "model_id": sk.entry.model_id,
                    "status": sk.entry.status,
                    "depth": sk.key.depth,
                    "calibrated": sk.calibration is not None,
                }
                for sk in served.values()
            ],
            "default_key_id": default_key_id,
        }

    def _route(key_id: str | None, model_id: str | None) -> ServedKey:
        if key_id and model_id:
            raise HTTPException(400, "supply key_id or model_id, not both")
        if model_id:
            resolved = by_model.get(model_id)
            if resolved is None:
                raise HTTPException(404, f"no active key for model {model_id!r}")
            return served[resolved]
        sk = served.get(key_id or default_key_id)
        if sk is None:
            raise HTTPException(404, f"unknown key_id {key_id!r}")
        return sk

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest) -> DetectResponse:
        sk = _route(req.key_id, req.model_id)
        ks, _ = _score_one(sk, req.text, req.method, req.target_fpr)

        if ks.num_tokens_scored < MIN_TOKENS:
            verdict, interpretation = "text_too_short", (
                f"Only {ks.num_tokens_scored} tokens could be scored; at least "
                f"{MIN_TOKENS} are needed for a meaningful verdict. Submit more text."
            )
        else:
            verdict = "watermark_detected" if ks.matched else "no_watermark_detected"
            interpretation = _interpret(ks.num_tokens_scored, ks.matched, req.target_fpr)

        return DetectResponse(
            verdict=verdict,
            score=ks.score,
            num_tokens_scored=ks.num_tokens_scored,
            p_value=ks.p_value,
            empirical_p_value=ks.empirical_p_value,
            threshold=ks.threshold,
            target_fpr=req.target_fpr,
            key_id=ks.key_id,
            key_fingerprint=ks.key_fingerprint,
            model_id=ks.model_id,
            interpretation=interpretation,
        )

    @app.post("/attribute", response_model=AttributeResponse)
    def attribute(req: AttributeRequest) -> AttributeResponse:
        return _attribute(served, req)

    return app


def _attribute(served: dict[str, ServedKey], req: AttributeRequest) -> AttributeResponse:
    """Scan every key and report which, if any, the text is marked by.

    The correction matters more than it looks.  Each key is an independent test,
    so scanning ``N`` keys at a per-key false-positive rate of ``alpha`` gives a
    scan-level rate of ``1 - (1 - alpha)^N``: at the usual 1% and 40 keys, a
    third of unwatermarked texts would be attributed to *something*.  Testing
    each key at ``target_fpr / N`` instead holds the scan-level rate at
    ``target_fpr``, which is the number that belongs in a report.

    This is also the reason to keep the registry small.  Every additional key
    axis -- per desk, per environment, per deployment -- makes each individual
    test stricter and therefore every genuine watermark harder to find.
    """
    candidates_pool = [
        sk for sk in served.values() if req.include_retired or sk.entry.is_active
    ]
    n = len(candidates_pool)
    if n == 0:
        return AttributeResponse(
            verdict="no_match",
            best=None,
            candidates=[],
            keys_tested=0,
            target_fpr=req.target_fpr,
            per_key_alpha=req.target_fpr,
            interpretation="No keys were eligible for testing.",
        )

    per_key_alpha = req.target_fpr / n
    scored = [_score_one(sk, req.text, req.method, per_key_alpha)[0] for sk in candidates_pool]
    scored.sort(key=lambda ks: ks.score, reverse=True)

    max_tokens = max(ks.num_tokens_scored for ks in scored)
    if max_tokens < MIN_TOKENS:
        return AttributeResponse(
            verdict="text_too_short",
            best=None,
            candidates=scored,
            keys_tested=n,
            target_fpr=req.target_fpr,
            per_key_alpha=per_key_alpha,
            interpretation=(
                f"Only {max_tokens} tokens could be scored under the best-matching tokenizer; "
                f"at least {MIN_TOKENS} are needed. Submit more text."
            ),
        )

    matches = [ks for ks in scored if ks.matched]
    if not matches:
        return AttributeResponse(
            verdict="no_match",
            best=None,
            candidates=scored,
            keys_tested=n,
            target_fpr=req.target_fpr,
            per_key_alpha=per_key_alpha,
            interpretation=(
                f"No watermark from any of {n} known keys, at a {req.target_fpr:.2%} "
                f"false-positive rate across the whole scan ({per_key_alpha:.3%} per key). "
                "This is not evidence of human authorship: paraphrased, heavily edited, "
                "short, or low-entropy text carries little detectable watermark, and a "
                "model served without watermarking leaves none at all."
            ),
        )

    best = matches[0]
    if len(matches) > 1:
        return AttributeResponse(
            verdict="ambiguous",
            best=best,
            candidates=scored,
            keys_tested=n,
            target_fpr=req.target_fpr,
            per_key_alpha=per_key_alpha,
            interpretation=(
                f"{len(matches)} keys matched: {', '.join(m.key_id for m in matches)}. "
                "Independent keys should not both fire on one text, so treat this as a "
                "configuration fault rather than a finding -- most likely two entries "
                "derived from the same label, or a text assembled from several sources. "
                "Investigate before relying on it."
            ),
        )

    return AttributeResponse(
        verdict="attributed",
        best=best,
        candidates=scored,
        keys_tested=n,
        target_fpr=req.target_fpr,
        per_key_alpha=per_key_alpha,
        interpretation=(
            f"Consistent with generation by {best.model_id} under key {best.key_id}, over "
            f"{best.num_tokens_scored} scored tokens, at a {req.target_fpr:.2%} "
            f"false-positive rate across all {n} keys tested. This indicates the model was "
            "likely involved in producing the text; it does not establish authorship, and "
            "cannot distinguish text the model wrote from text the model edited."
        ),
    )


# --------------------------------------------------------------- env-driven app


def _default_app():
    """Module-level app for ``uvicorn synthmark.serve:app``.

    Configured entirely through the environment so no secret ever lives in the
    source tree or an image layer:

    ``SYNTHMARK_MASTER_SECRET``   master secret (required)
    ``SYNTHMARK_KEY_REGISTRY``    path to the key registry JSON (required)
    ``SYNTHMARK_CALIBRATION_DIR`` optional dir of ``<key_id with / as _>.json``
    ``SYNTHMARK_DEVICE``          defaults to cpu, which is all detection needs
    """
    master = load_master_secret()
    registry_path = os.environ.get("SYNTHMARK_KEY_REGISTRY")
    if not registry_path:
        raise RuntimeError(
            "SYNTHMARK_KEY_REGISTRY is not set. Point it at the key registry JSON "
            "describing which key marks which model."
        )
    registry = KeyRegistry.load(registry_path)

    calibrations: dict[str, Calibration] = {}
    cal_dir = os.environ.get("SYNTHMARK_CALIBRATION_DIR")
    if cal_dir:
        for entry in registry:
            path = Path(cal_dir) / f"{entry.key_id.replace('/', '_')}.json"
            if path.exists():
                calibrations[entry.key_id] = Calibration.load(path)

    served = build_served(
        registry,
        master,
        calibrations=calibrations,
        device=os.environ.get("SYNTHMARK_DEVICE", "cpu"),
    )
    return build_app(served)


class _LazyApp:
    """Defer building the app (and loading tokenizers) until first request."""

    def __init__(self):
        self._app = None

    def __call__(self, scope, receive, send):
        if self._app is None:
            self._app = _default_app()
        return self._app(scope, receive, send)


app = _LazyApp()
