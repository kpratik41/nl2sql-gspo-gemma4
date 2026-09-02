"""The key registry is the thing that stops a misconfiguration from being silent.

Every failure mode of a watermark deployment -- wrong master secret, drifted
structural parameters, two models sharing a key, a retired key dropped from the
detector -- produces the same symptom: a perfectly normal-looking null score.
These tests check that each of them raises instead.
"""

import json

import pytest

from synthmark.registry import ACTIVE, RETIRED, KeyEntry, KeyRegistry, RegistryError

MASTER = "master-secret-with-enough-entropy"
OTHER_MASTER = "a-completely-different-master-secret"


def registry(**overrides) -> KeyRegistry:
    entries = overrides.pop(
        "entries",
        (
            KeyEntry("gemma-4-e4b/v1", "google/gemma-4-E4B-it", depth=8),
            KeyEntry("nemotron-super-49b/v1", "nvidia/Nemotron-Super-49B", depth=8),
        ),
    )
    return KeyRegistry(entries=entries, **overrides)


# ------------------------------------------------------------------ invariants


def test_two_models_get_independent_keys():
    """The whole point of per-model keys: a hit must name the model."""
    keys = registry().resolve_all(MASTER)
    gemma, nemotron = keys["gemma-4-e4b/v1"], keys["nemotron-super-49b/v1"]
    assert gemma.keys != nemotron.keys
    assert gemma.sampling_table_seed != nemotron.sampling_table_seed
    assert gemma.fingerprint != nemotron.fingerprint


def test_duplicate_key_id_is_rejected():
    with pytest.raises(RegistryError, match="duplicate key_id"):
        registry(entries=(
            KeyEntry("dup/v1", "model-a"),
            KeyEntry("dup/v1", "model-b"),
        ))


def test_two_active_keys_for_one_model_is_rejected():
    """Generation must be unambiguous: exactly one active key per model."""
    with pytest.raises(RegistryError, match="two active keys"):
        registry(entries=(
            KeyEntry("m/v1", "model-a"),
            KeyEntry("m/v2", "model-a"),
        ))


def test_rotation_retires_the_old_key_rather_than_replacing_it():
    reg = registry(entries=(
        KeyEntry("m/v1", "model-a", status=RETIRED),
        KeyEntry("m/v2", "model-a", status=ACTIVE),
    ))
    assert reg.for_model("model-a").key_id == "m/v2"
    # The retired key stays resolvable: text generated under it still exists.
    assert set(reg.resolve_all(MASTER)) == {"m/v1", "m/v2"}
    assert set(reg.resolve_all(MASTER, active_only=True)) == {"m/v2"}


def test_unregistered_model_raises_rather_than_defaulting():
    """Silently falling back to some other key would mark text undetectably."""
    with pytest.raises(KeyError, match="no active watermark key"):
        registry().for_model("meta-llama/Llama-3.3-70B")


def test_retired_key_is_not_offered_for_generation():
    reg = registry(entries=(KeyEntry("m/v1", "model-a", status=RETIRED),))
    with pytest.raises(KeyError):
        reg.for_model("model-a")


def test_invalid_status_is_rejected():
    with pytest.raises(RegistryError, match="status must be"):
        KeyEntry("m/v1", "model-a", status="draft")


# ---------------------------------------------------------------- fingerprints


def test_stamped_fingerprints_verify_against_their_master():
    stamped = registry().stamp_fingerprints(MASTER)
    assert all(e.fingerprint for e in stamped)
    stamped.resolve_all(MASTER)  # must not raise


def test_wrong_master_secret_fails_loudly():
    """The failure this prevents is a detector that silently detects nothing."""
    stamped = registry().stamp_fingerprints(MASTER)
    with pytest.raises(RegistryError, match="does not match the registered"):
        stamped.resolve_all(OTHER_MASTER)


def test_edited_structural_parameter_fails_loudly():
    """depth and ngram_len are part of the key; a drifted value detects nothing."""
    stamped = registry().stamp_fingerprints(MASTER)
    tampered = KeyRegistry(entries=(
        KeyEntry(**{**stamped.entries[0].to_dict(), "depth": 16}),
    ))
    with pytest.raises(RegistryError, match="does not match the registered"):
        tampered.resolve_all(MASTER)


# -------------------------------------------------------------------- storage


def test_round_trips_through_json_without_secrets(tmp_path):
    stamped = registry().stamp_fingerprints(MASTER)
    path = stamped.save(tmp_path / "registry.json")

    raw = json.loads(path.read_text())
    secret_material = {str(k) for key in stamped.resolve_all(MASTER).values() for k in key.keys}
    assert secret_material.isdisjoint(json.dumps(raw).split('"'))

    assert KeyRegistry.load(path).to_dict() == stamped.to_dict()


def test_tokenizers_are_deduplicated():
    """Ten keys over three model families should cost three tokenizer loads."""
    reg = registry(entries=(
        KeyEntry("a/v1", "google/gemma-4-E4B-it"),
        KeyEntry("a/v0", "google/gemma-4-E4B-it", status=RETIRED),
        KeyEntry("b/v1", "nvidia/Nemotron-Super-49B"),
    ))
    assert reg.tokenizers() == ("google/gemma-4-E4B-it", "nvidia/Nemotron-Super-49B")


def test_tokenizer_id_overrides_model_id():
    entry = KeyEntry("a/v1", "org/weights-only", tokenizer_id="org/tokenizer")
    assert entry.tokenizer == "org/tokenizer"
    assert KeyEntry("b/v1", "org/model").tokenizer == "org/model"


def test_example_registry_in_repo_is_valid():
    reg = KeyRegistry.load("configs/key_registry.example.json")
    assert len(reg) >= 2
    # Distinct models, so a detection hit is attributable.
    assert len({e.model_id for e in reg.active()}) == len(reg.active())
