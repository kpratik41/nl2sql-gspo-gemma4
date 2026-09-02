"""The non-secret manifest that says which key belongs to which model.

A watermark key is meaningless on its own.  To use one you also need to know
*which model* it marks, *which tokenizer* segments the text, and *whether it is
still in use for generation*.  None of that is secret, and all of it has to be
identical on the generation side and the detection side or the watermark simply
does not resolve -- silently, with a perfectly normal-looking null score.

So it lives here, in a file you can check into git, review in a pull request,
and hand to an auditor:

* the **secret** is one master secret in a secrets manager;
* the **structure** is this registry;
* every key is ``derive_key(master, entry.key_id)``, reproducible on any host
  that can read the master secret, so key files never have to be copied around.

Why one key per model
---------------------
Two models served under the same key are indistinguishable: a detection hit
tells you "something of ours wrote this" but not what.  Distinct keys make
attribution possible, because a text marked by one key is null-distributed
under every other key.  Serving Nemotron and Gemma therefore means two entries
here, two labels, two derived keys -- and no extra secret to escrow.

This module is deliberately dependency-free (standard library only), so key
governance can be reviewed, tested, and automated without a GPU, a model, or
even PyTorch installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .keys import (
    DEFAULT_CONTEXT_HISTORY_SIZE,
    DEFAULT_DEPTH,
    DEFAULT_NGRAM_LEN,
    DEFAULT_SAMPLING_TABLE_SIZE,
    WatermarkKey,
    derive_key,
)

ACTIVE = "active"
"""Still used to watermark new generations."""

RETIRED = "retired"
"""No longer used for generation, but still required for detection.

Text generated under a retired key exists forever, so retiring a key means
"stop marking with it", never "stop looking for it".  Entries are therefore
only ever added to this registry, not removed.
"""


class RegistryError(ValueError):
    """Raised when a registry is internally inconsistent or does not match the master secret."""


@dataclass(frozen=True)
class KeyEntry:
    """One (key, model) binding.

    Attributes:
        key_id: The HKDF label, and the key's name.  Convention:
            ``{model-slug}/{weights-version}/v{key-epoch}``, e.g.
            ``"nvidia-nemotron-super-49b/2504/v1"``.  Rotation increments the
            key epoch and adds a new entry; it never edits an existing one.
        model_id: The model this key marks, as served.
        tokenizer_id: Tokenizer used for detection.  Defaults to ``model_id``.
            Only set it separately when the served weights and the tokenizer
            come from different repositories.
        status: :data:`ACTIVE` or :data:`RETIRED`.
        fingerprint: The expected key fingerprint.  Not secret -- it is a
            SHA-256 prefix -- but recording it turns a silent misconfiguration
            into a loud one: if the master secret or any structural parameter
            drifts, :meth:`resolve` raises instead of quietly deriving a
            different key that will detect nothing.
        ngram_len, depth, context_history_size, sampling_table_size:
            Structural parameters.  They are part of the key: a mismatch
            between generation and detection produces no signal at all.
        valid_from, valid_to: ISO-8601 dates, for the audit trail.
        notes: Free-form, e.g. the change ticket that introduced the key.
    """

    key_id: str
    model_id: str
    tokenizer_id: str | None = None
    status: str = ACTIVE
    fingerprint: str | None = None
    ngram_len: int = DEFAULT_NGRAM_LEN
    depth: int = DEFAULT_DEPTH
    context_history_size: int = DEFAULT_CONTEXT_HISTORY_SIZE
    sampling_table_size: int = DEFAULT_SAMPLING_TABLE_SIZE
    valid_from: str = ""
    valid_to: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.key_id:
            raise RegistryError("key_id must be a non-empty string")
        if not self.model_id:
            raise RegistryError(f"{self.key_id}: model_id must be a non-empty string")
        if self.status not in (ACTIVE, RETIRED):
            raise RegistryError(
                f"{self.key_id}: status must be {ACTIVE!r} or {RETIRED!r}, got {self.status!r}"
            )

    @property
    def tokenizer(self) -> str:
        """The tokenizer to detect with -- ``tokenizer_id`` if set, else ``model_id``."""
        return self.tokenizer_id or self.model_id

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE

    def resolve(self, master_secret: bytes | str, *, verify: bool = True) -> WatermarkKey:
        """Derive this entry's key from the master secret.

        Args:
            verify: If a ``fingerprint`` is recorded, check it and raise on a
                mismatch.  Leave this on.  A mismatch means the derived key is
                not the key that generated the corpus, and every downstream
                symptom of that -- undetectable text, an unattributable
                incident -- looks exactly like "there was no watermark".
        """
        key = derive_key(
            master_secret,
            self.key_id,
            depth=self.depth,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            sampling_table_size=self.sampling_table_size,
            notes=self.notes,
        )
        if verify and self.fingerprint and key.fingerprint != self.fingerprint:
            raise RegistryError(
                f"{self.key_id}: derived fingerprint {key.fingerprint} does not match the "
                f"registered {self.fingerprint}. Either the master secret is not the one this "
                "key was minted from, or a structural parameter in the registry was edited. "
                "Do not proceed: the derived key detects nothing and marks nothing detectable."
            )
        return key

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass(frozen=True)
class KeyRegistry:
    """An ordered collection of :class:`KeyEntry`, loaded from JSON.

    Invariants enforced on construction:

    * ``key_id`` is unique -- two keys with the same label are the same key.
    * At most one *active* entry per ``model_id`` -- generation must be
      unambiguous.  Rotation retires the old entry in the same commit that
      adds the new one.
    """

    entries: tuple[KeyEntry, ...] = ()
    notes: str = ""
    _by_id: dict[str, KeyEntry] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_id: dict[str, KeyEntry] = {}
        active_models: dict[str, str] = {}
        for e in self.entries:
            if e.key_id in by_id:
                raise RegistryError(f"duplicate key_id {e.key_id!r}")
            by_id[e.key_id] = e
            if e.is_active:
                if e.model_id in active_models:
                    raise RegistryError(
                        f"{e.model_id!r} has two active keys ({active_models[e.model_id]!r} and "
                        f"{e.key_id!r}). Generation would be ambiguous; retire one."
                    )
                active_models[e.model_id] = e.key_id
        object.__setattr__(self, "_by_id", by_id)

    # ------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, key_id: str) -> KeyEntry:
        try:
            return self._by_id[key_id]
        except KeyError:
            raise KeyError(f"unknown key_id {key_id!r}") from None

    def get(self, key_id: str) -> KeyEntry | None:
        return self._by_id.get(key_id)

    def active(self) -> tuple[KeyEntry, ...]:
        return tuple(e for e in self.entries if e.is_active)

    def tokenizers(self) -> tuple[str, ...]:
        """Distinct tokenizers the registry needs, so callers can load each once."""
        seen: dict[str, None] = {}
        for e in self.entries:
            seen.setdefault(e.tokenizer, None)
        return tuple(seen)

    def for_model(self, model_id: str) -> KeyEntry:
        """The active key a given model should generate with.

        This is the lookup the *serving* side makes: given the model it is about
        to run, which key does it mark with?
        """
        matches = [e for e in self.entries if e.model_id == model_id and e.is_active]
        if not matches:
            raise KeyError(
                f"no active watermark key registered for model {model_id!r}. "
                "Add an entry before serving it, or serving will emit unmarked text."
            )
        return matches[0]  # uniqueness enforced in __post_init__

    # -------------------------------------------------------------- resolving

    def resolve_all(
        self, master_secret: bytes | str, *, verify: bool = True, active_only: bool = False
    ) -> dict[str, WatermarkKey]:
        """Derive every registered key.  Detection needs retired keys too."""
        entries = self.active() if active_only else self.entries
        return {e.key_id: e.resolve(master_secret, verify=verify) for e in entries}

    def stamp_fingerprints(self, master_secret: bytes | str) -> "KeyRegistry":
        """Return a copy with every ``fingerprint`` filled in from the master secret.

        Run this once when adding an entry, then commit the result.  From then
        on the recorded fingerprints are what :meth:`resolve_all` checks against.
        """
        stamped = tuple(
            replace(e, fingerprint=e.resolve(master_secret, verify=False).fingerprint)
            for e in self.entries
        )
        return KeyRegistry(entries=stamped, notes=self.notes)

    # ---------------------------------------------------------------- storage

    def to_dict(self) -> dict:
        out: dict = {"version": 1, "keys": [e.to_dict() for e in self.entries]}
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "KeyRegistry":
        if "keys" not in data:
            raise RegistryError("registry JSON must have a top-level 'keys' list")
        return cls(
            entries=tuple(KeyEntry(**e) for e in data["keys"]),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "KeyRegistry":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> Path:
        """Write the registry as pretty JSON.  Contains no secrets by construction."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
        return path
