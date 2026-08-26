"""Key management: determinism, independence, and not leaking secrets."""

import json

import pytest

from synthmark.keys import WatermarkKey, derive_key, generate_key


def test_derivation_is_deterministic():
    a = derive_key("master-secret-with-enough-entropy", "team/v1")
    b = derive_key("master-secret-with-enough-entropy", "team/v1")
    assert a.keys == b.keys
    assert a.sampling_table_seed == b.sampling_table_seed
    assert a.fingerprint == b.fingerprint


def test_different_labels_give_independent_keys():
    a = derive_key("master-secret-with-enough-entropy", "team/v1")
    b = derive_key("master-secret-with-enough-entropy", "team/v2")
    c = derive_key("a-completely-different-master-key", "team/v1")
    assert a.keys != b.keys
    assert a.keys != c.keys
    # No shared elements is overwhelmingly likely for 30 independent uint32 draws.
    assert len(set(a.keys) & set(b.keys)) == 0


def test_master_secret_must_have_entropy():
    with pytest.raises(ValueError):
        derive_key("short", "team/v1")


def test_fingerprint_does_not_reveal_key():
    key = derive_key("master-secret-with-enough-entropy", "team/v1")
    fp = key.fingerprint
    assert len(fp) == 16
    for k in key.keys:
        assert str(k) not in fp


def test_public_summary_omits_secrets():
    key = derive_key("master-secret-with-enough-entropy", "team/v1")
    pub = key.public_summary()
    assert "keys" not in pub
    assert "sampling_table_seed" not in pub
    assert pub["fingerprint"] == key.fingerprint
    assert pub["depth"] == key.depth


def test_roundtrip_and_permissions(tmp_path):
    key = generate_key("team/v1", depth=8)
    path = key.save(tmp_path / "k.json")
    assert oct(path.stat().st_mode)[-3:] == "600"
    assert WatermarkKey.load(path).keys == key.keys
    with pytest.raises(FileExistsError):
        key.save(path)
    key.save(path, overwrite=True)


def test_validation_rejects_bad_parameters():
    with pytest.raises(ValueError):
        WatermarkKey(key_id="x", keys=())
    with pytest.raises(ValueError):
        WatermarkKey(key_id="", keys=(1, 2))
    with pytest.raises(ValueError):
        WatermarkKey(key_id="x", keys=(1, 2), ngram_len=1)
    with pytest.raises(ValueError):
        WatermarkKey(key_id="x", keys=(1, 2), sampling_table_size=1000)  # not a power of two
    with pytest.raises(ValueError):
        WatermarkKey(key_id="x", keys=(-1,))
