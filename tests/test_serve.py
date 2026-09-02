"""The detection service must be safe to expose to non-specialists.

The behaviours tested here are the ones that stop a detection API from being
misused: refusing a verdict on text that is too short, isolating keys from each
other, attributing a text to the right model out of several, correcting for the
multiple tests an attribution scan performs, and never returning a bare boolean
without the context needed to read it.
"""

import numpy as np
import pytest
import torch

from synthmark.config import build_processor
from synthmark_detect import Detector
from synthmark.registry import RETIRED, KeyEntry, KeyRegistry
from synthmark_detect.serve import MIN_TOKENS, build_app, build_served

MASTER = "master-secret-with-enough-entropy"
GEMMA = "google/gemma-4-E4B-it"
NEMOTRON = "nvidia/Nemotron-Super-49B"
VOCAB = 2_000


class StubTokenizer:
    """Maps whitespace-separated integers to token ids, so tests control tokens exactly."""

    eos_token_id = 999_999
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(t) for t in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


REGISTRY = KeyRegistry(entries=(
    KeyEntry("gemma-4-e4b/v1", GEMMA, depth=8),
    KeyEntry("nemotron-super-49b/v1", NEMOTRON, depth=8),
    KeyEntry("gemma-4-e4b/v0", GEMMA, depth=8, status=RETIRED),
))


@pytest.fixture(scope="module")
def served():
    return build_served(REGISTRY, MASTER, tokenizer_loader=lambda _: StubTokenizer())


@pytest.fixture(scope="module")
def client(served):
    from fastapi.testclient import TestClient

    return TestClient(build_app(served, default_key_id="gemma-4-e4b/v1"))


def plain_text(n=300, seed=0):
    """Unwatermarked token soup."""
    rng = np.random.default_rng(seed)
    return " ".join(str(int(x)) for x in rng.integers(1, VOCAB, size=n))


def watermarked_text(key, n=260, seed=0):
    """Text genuinely marked by ``key``, sampled from random logits.

    Standing in for a model this way keeps the test in milliseconds while
    exercising the real tournament sampling, which is what the detector reads.
    """
    proc = build_processor(key, "cpu")
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (1, key.ngram_len), generator=gen)
    for _ in range(n):
        scores = torch.randn(1, VOCAB, generator=gen)
        probs = torch.softmax(proc(ids, scores), dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=gen)], dim=1)
    return " ".join(str(int(t)) for t in ids[0])


# ------------------------------------------------------------------- /health


def test_health_lists_keys_without_leaking_them(client):
    body = client.get("/health").json()
    assert {k["key_id"] for k in body["keys"]} == set(REGISTRY._by_id)
    for k in body["keys"]:
        assert set(k) == {"key_id", "fingerprint", "model_id", "status", "depth", "calibrated"}


# ------------------------------------------------------------------- /detect


def test_short_text_gets_no_verdict(client):
    body = client.post("/detect", json={"text": "1 2 3 4 5 6 7 8 9 10"}).json()
    assert body["verdict"] == "text_too_short"
    assert str(MIN_TOKENS) in body["interpretation"]


def test_unwatermarked_text_is_not_flagged(client):
    body = client.post("/detect", json={"text": plain_text()}).json()
    assert body["verdict"] == "no_watermark_detected"
    assert body["num_tokens_scored"] > MIN_TOKENS
    # The negative must be caveated, not read as "a human wrote this".
    assert "not evidence" in body["interpretation"]


def test_watermarked_text_is_detected(client, served):
    text = watermarked_text(served["gemma-4-e4b/v1"].key)
    body = client.post("/detect", json={"text": text, "key_id": "gemma-4-e4b/v1"}).json()
    assert body["verdict"] == "watermark_detected"
    assert body["p_value"] < 1e-6


def test_watermark_is_invisible_to_another_models_key(client, served):
    """Gemma's watermark must be null under Nemotron's key, or attribution is meaningless."""
    text = watermarked_text(served["gemma-4-e4b/v1"].key)
    body = client.post("/detect", json={"text": text, "key_id": "nemotron-super-49b/v1"}).json()
    assert body["verdict"] == "no_watermark_detected"


def test_routing_by_model_id_picks_the_active_key(client, served):
    text = watermarked_text(served["gemma-4-e4b/v1"].key)
    body = client.post("/detect", json={"text": text, "model_id": GEMMA}).json()
    assert body["key_id"] == "gemma-4-e4b/v1"  # the active one, not the retired v0
    assert body["verdict"] == "watermark_detected"


def test_unknown_key_and_model_are_rejected(client):
    assert client.post("/detect", json={"text": plain_text(), "key_id": "nope/v1"}).status_code == 404
    assert client.post("/detect", json={"text": plain_text(), "model_id": "nope"}).status_code == 404


def test_key_id_and_model_id_together_are_rejected(client):
    r = client.post("/detect", json={"text": plain_text(), "key_id": "gemma-4-e4b/v1",
                                     "model_id": GEMMA})
    assert r.status_code == 400


def test_response_always_carries_context(client):
    body = client.post("/detect", json={"text": plain_text(seed=4)}).json()
    for field in ("score", "num_tokens_scored", "p_value", "target_fpr", "model_id",
                  "key_fingerprint", "interpretation"):
        assert body[field] is not None


def test_empty_text_rejected_by_schema(client):
    assert client.post("/detect", json={"text": ""}).status_code == 422


# ---------------------------------------------------------------- /attribute


def test_attribution_names_the_generating_model(client, served):
    text = watermarked_text(served["nemotron-super-49b/v1"].key, seed=7)
    body = client.post("/attribute", json={"text": text}).json()
    assert body["verdict"] == "attributed"
    assert body["best"]["key_id"] == "nemotron-super-49b/v1"
    assert body["best"]["model_id"] == NEMOTRON
    assert body["keys_tested"] == len(REGISTRY)


def test_attribution_reports_every_key_it_tested(client, served):
    """Ranked candidates are what makes a verdict auditable rather than oracular."""
    text = watermarked_text(served["gemma-4-e4b/v1"].key, seed=8)
    body = client.post("/attribute", json={"text": text}).json()
    assert [c["key_id"] for c in body["candidates"]][0] == "gemma-4-e4b/v1"
    assert len(body["candidates"]) == len(REGISTRY)
    assert sum(c["matched"] for c in body["candidates"]) == 1


def test_attribution_corrects_for_multiple_keys(client):
    """Scanning N keys at alpha each gives a scan-level FPR of 1-(1-alpha)^N, not alpha."""
    body = client.post("/attribute", json={"text": plain_text(), "target_fpr": 0.01}).json()
    assert body["per_key_alpha"] == pytest.approx(0.01 / len(REGISTRY))


def test_unwatermarked_text_is_attributed_to_nothing(client):
    body = client.post("/attribute", json={"text": plain_text(seed=11)}).json()
    assert body["verdict"] == "no_match"
    assert body["best"] is None
    assert "not evidence of human authorship" in body["interpretation"]


def test_retired_keys_are_scanned_by_default(client, served):
    """An incident under a retired key must stay attributable."""
    text = watermarked_text(served["gemma-4-e4b/v0"].key, seed=12)
    body = client.post("/attribute", json={"text": text}).json()
    assert body["verdict"] == "attributed"
    assert body["best"]["key_id"] == "gemma-4-e4b/v0"
    assert body["best"]["status"] == RETIRED

    narrowed = client.post("/attribute", json={"text": text, "include_retired": False}).json()
    assert narrowed["verdict"] == "no_match"
    assert narrowed["keys_tested"] == 2


def test_attribution_refuses_short_text(client):
    body = client.post("/attribute", json={"text": "1 2 3 4 5"}).json()
    assert body["verdict"] == "text_too_short"
    assert body["best"] is None
