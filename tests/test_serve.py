"""The detection service must be safe to expose to non-specialists.

The behaviours tested here are the ones that stop a detection API from being
misused: refusing a verdict on text that is too short, isolating keys from each
other, and never returning a bare boolean without the context needed to read it.
"""

import pytest

from synthmark.detect import Detector
from synthmark.keys import derive_key
from synthmark.serve import MIN_TOKENS, build_app

MASTER = "master-secret-with-enough-entropy"


class StubTokenizer:
    eos_token_id = 999_999
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(t) for t in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    keys = {
        "desk-a/v1": derive_key(MASTER, "desk-a/v1", depth=8),
        "desk-b/v1": derive_key(MASTER, "desk-b/v1", depth=8),
    }
    return TestClient(build_app(keys, StubTokenizer(), default_key_id="desk-a/v1"))


def long_text(n=300, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return " ".join(str(int(x)) for x in rng.integers(1, 50_000, size=n))


def test_health_lists_keys_without_leaking_them(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert {k["key_id"] for k in body["keys"]} == {"desk-a/v1", "desk-b/v1"}
    for k in body["keys"]:
        assert set(k) == {"key_id", "fingerprint", "depth"}  # no key material


def test_short_text_gets_no_verdict(client):
    r = client.post("/detect", json={"text": "1 2 3 4 5 6 7 8 9 10"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "text_too_short"
    assert str(MIN_TOKENS) in body["interpretation"]


def test_unwatermarked_text_is_not_flagged(client):
    body = client.post("/detect", json={"text": long_text()}).json()
    assert body["verdict"] == "no_watermark_detected"
    assert body["num_tokens_scored"] > MIN_TOKENS
    # The negative must be caveated, not read as "a human wrote this".
    assert "not evidence" in body["interpretation"]


def test_unknown_key_is_rejected(client):
    r = client.post("/detect", json={"text": long_text(), "key_id": "desk-z/v1"})
    assert r.status_code == 404


def test_keys_are_routed_independently(client):
    text = long_text(seed=3)
    a = client.post("/detect", json={"text": text, "key_id": "desk-a/v1"}).json()
    b = client.post("/detect", json={"text": text, "key_id": "desk-b/v1"}).json()
    assert a["score"] != b["score"]
    assert a["key_fingerprint"] != b["key_fingerprint"]


def test_response_always_carries_context(client):
    body = client.post("/detect", json={"text": long_text(seed=4)}).json()
    for field in ("score", "num_tokens_scored", "p_value", "target_fpr", "interpretation"):
        assert body[field] is not None


def test_empty_text_rejected_by_schema(client):
    assert client.post("/detect", json={"text": ""}).status_code == 422
