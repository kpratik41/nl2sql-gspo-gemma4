"""Watermark key material: generation, derivation, fingerprinting, and storage.

A SynthID-Text watermark is fully determined by a small bundle of parameters.
Two of them are *secret* -- ``keys`` and ``sampling_table_seed`` -- and the rest
are *public* structural parameters that must nevertheless match exactly between
generation and detection.

Security model
--------------
Anyone holding the secret material can both *detect* the watermark and *forge*
it (i.e. generate text that a detector will flag).  The key is therefore a
signing-grade secret: treat it like an HMAC key, not like a config value.

* Store key files with mode 0600, or better, in a secrets manager / HSM and
  materialise them only in process memory.
* Prefer :func:`derive_key` over :func:`generate_key`: it turns a single master
  secret plus a non-secret label into an unlimited family of independent keys,
  so only one secret ever has to be escrowed, rotated, or backed up.
* Distribute the *detector* to consumers as a service, not as key material.
  Handing out the key to let a downstream team run detection also hands them
  the ability to fabricate watermarked text.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Keys are consumed as int64 values inside a linear-congruential hash.  We keep
# them inside the unsigned 32-bit range: large enough that guessing a single
# key is hopeless, small enough to be safely JSON round-tripped and to match the
# reference DeepMind implementation.
_KEY_MAX = 2**32
_SEED_MAX = 2**31

# Defaults follow the reference configuration from the SynthID-Text paper
# (Dathathri et al., Nature 2024) and google-deepmind/synthid-text.
DEFAULT_NGRAM_LEN = 5
DEFAULT_DEPTH = 30
DEFAULT_CONTEXT_HISTORY_SIZE = 1024
DEFAULT_SAMPLING_TABLE_SIZE = 2**16


@dataclass(frozen=True)
class WatermarkKey:
    """A complete, self-describing SynthID-Text watermark key.

    Attributes:
        key_id: Non-secret human-readable label, e.g. ``"markets-research-2026q3"``.
            Used for key rotation and for routing a detection request to the
            right key in a multi-tenant deployment.
        keys: The secret per-depth watermarking keys.  ``len(keys)`` is the
            watermarking depth: more depth means more independent Bernoulli
            samples per token and therefore a stronger detectable signal.
        ngram_len: Size of the sliding token window used to seed the g-function.
            The window is the ``ngram_len - 1`` preceding tokens plus the
            candidate token.
        context_history_size: How many recently seen contexts are remembered so
            that repeated contexts can be skipped (repeats would otherwise bias
            the watermark and degrade text quality).
        sampling_table_seed: Secret seed for the pre-computed g-value table.
        sampling_table_size: Size of that table.
        created_at: UTC ISO-8601 creation timestamp, for rotation audits.
        notes: Free-form provenance, e.g. which model family it is bound to.
    """

    key_id: str
    keys: tuple[int, ...]
    ngram_len: int = DEFAULT_NGRAM_LEN
    context_history_size: int = DEFAULT_CONTEXT_HISTORY_SIZE
    sampling_table_seed: int = 0
    sampling_table_size: int = DEFAULT_SAMPLING_TABLE_SIZE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.key_id:
            raise ValueError("key_id must be a non-empty string")
        if not self.keys:
            raise ValueError("keys must contain at least one element (depth >= 1)")
        if any(not isinstance(k, int) or not 0 <= k < _KEY_MAX for k in self.keys):
            raise ValueError(f"every key must be an int in [0, {_KEY_MAX})")
        if self.ngram_len < 2:
            raise ValueError("ngram_len must be >= 2 (one context token + the candidate)")
        if self.sampling_table_size > 2**24:
            raise ValueError("sampling_table_size must be <= 2**24")
        if self.sampling_table_size & (self.sampling_table_size - 1):
            raise ValueError("sampling_table_size should be a power of two")
        object.__setattr__(self, "keys", tuple(int(k) for k in self.keys))

    @property
    def depth(self) -> int:
        """Watermarking depth (number of independent g-functions per token)."""
        return len(self.keys)

    @property
    def fingerprint(self) -> str:
        """Short non-secret identifier of the *secret* material.

        Lets logs, result files, and audit trails record *which* key produced a
        result without recording the key itself.  It is a SHA-256 digest, so it
        is not invertible.
        """
        payload = json.dumps(
            {
                "keys": list(self.keys),
                "ngram_len": self.ngram_len,
                "sampling_table_seed": self.sampling_table_seed,
                "sampling_table_size": self.sampling_table_size,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    # ---------------------------------------------------------------- storage

    def to_dict(self, *, include_secret: bool = True) -> dict:
        data = asdict(self)
        data["keys"] = list(self.keys)
        data["depth"] = self.depth
        data["fingerprint"] = self.fingerprint
        if not include_secret:
            data.pop("keys")
            data.pop("sampling_table_seed")
        return data

    def public_summary(self) -> dict:
        """Everything safe to put in a report or a log line."""
        return self.to_dict(include_secret=False)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the key to disk as JSON with restrictive permissions."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create with 0600 from the start so the secret is never briefly world-readable.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WatermarkKey":
        with open(path) as fh:
            data = json.load(fh)
        data.pop("depth", None)
        data.pop("fingerprint", None)
        data["keys"] = tuple(data["keys"])
        return cls(**data)


def generate_key(
    key_id: str,
    *,
    depth: int = DEFAULT_DEPTH,
    ngram_len: int = DEFAULT_NGRAM_LEN,
    context_history_size: int = DEFAULT_CONTEXT_HISTORY_SIZE,
    sampling_table_size: int = DEFAULT_SAMPLING_TABLE_SIZE,
    notes: str = "",
) -> WatermarkKey:
    """Draw a fresh key from the OS cryptographic RNG.

    Use this when you are happy to escrow the resulting file.  If you would
    rather escrow a single master secret, use :func:`derive_key`.
    """
    return WatermarkKey(
        key_id=key_id,
        keys=tuple(secrets.randbelow(_KEY_MAX) for _ in range(depth)),
        ngram_len=ngram_len,
        context_history_size=context_history_size,
        sampling_table_seed=secrets.randbelow(_SEED_MAX),
        sampling_table_size=sampling_table_size,
        notes=notes,
    )


def derive_key(
    master_secret: bytes | str,
    key_id: str,
    *,
    depth: int = DEFAULT_DEPTH,
    ngram_len: int = DEFAULT_NGRAM_LEN,
    context_history_size: int = DEFAULT_CONTEXT_HISTORY_SIZE,
    sampling_table_size: int = DEFAULT_SAMPLING_TABLE_SIZE,
    notes: str = "",
) -> WatermarkKey:
    """Deterministically derive a watermark key from a master secret.

    Uses HKDF-SHA256 (RFC 5869) with ``key_id`` as the ``info`` parameter, so
    distinct labels yield cryptographically independent keys and knowledge of
    one derived key reveals nothing about the others or about the master.

    This is the recommended path for a production deployment:

    * one secret to escrow, rotate, and audit;
    * per-business-unit or per-tenant keys created on demand with no new secret
      material (``derive_key(master, "gcib-research")``);
    * rotation by versioning the label (``"gcib-research/v2"``);
    * any host holding the master can reconstruct any key, so key files never
      have to be copied between machines.

    Args:
        master_secret: The root secret.  A ``str`` is UTF-8 encoded; prefer
            raw bytes from a secrets manager.  Should be >= 32 bytes of entropy.
        key_id: Non-secret label; becomes both the derivation ``info`` and the
            key's identifier.
    """
    if isinstance(master_secret, str):
        master_secret = master_secret.encode()
    if len(master_secret) < 16:
        raise ValueError("master_secret should be at least 16 bytes of entropy")

    # HKDF-extract with a fixed, non-secret salt that domain-separates this
    # library from any other use of the same master secret.
    prk = hmac.new(b"synthmark/hkdf/v1", master_secret, hashlib.sha256).digest()

    # HKDF-expand to (4 * depth) bytes for the keys plus 4 for the table seed.
    need = 4 * depth + 4
    info = key_id.encode()
    okm, block, counter = b"", b"", 1
    while len(okm) < need:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    okm = okm[:need]

    keys = tuple(int.from_bytes(okm[i * 4 : i * 4 + 4], "big") for i in range(depth))
    seed = int.from_bytes(okm[4 * depth : 4 * depth + 4], "big") % _SEED_MAX

    return WatermarkKey(
        key_id=key_id,
        keys=keys,
        ngram_len=ngram_len,
        context_history_size=context_history_size,
        sampling_table_seed=seed,
        sampling_table_size=sampling_table_size,
        notes=notes,
    )


def load_master_secret(env_var: str = "SYNTHMARK_MASTER_SECRET") -> bytes:
    """Read a master secret from the environment, with a clear failure message."""
    raw = os.environ.get(env_var)
    if not raw:
        raise RuntimeError(
            f"{env_var} is not set. Export a high-entropy secret, e.g.\n"
            f'  export {env_var}="$(openssl rand -hex 32)"\n'
            "In production this should come from your secrets manager, not a shell profile."
        )
    return raw.encode()
