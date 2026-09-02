"""synthmark -- the shared contract: watermark keys, the key registry, and the g-function.

This is the package that both ends of a deployment depend on, and the reason it
is separate from the rest.  A watermark only resolves if generation and
detection agree *exactly* on the key, on ``ngram_len``, on ``depth``, and on the
g-function.  Keeping that agreement in one distribution means it is pinned by
version rather than by documentation, and a disagreement is a dependency
resolution failure rather than a silently undetectable corpus.

    pip install synthmark            # this package: keys, registry, g-function
    pip install synthmark-detect     # scoring text and the detection service
    pip install synthmark-eval       # HF generation, benchmarks, metrics

Import cost
-----------
:mod:`synthmark.keys` and :mod:`synthmark.registry` are pure standard library,
and only they are imported eagerly; :mod:`synthmark.config` is resolved on first
attribute access (PEP 562).  So a key-rotation job or a CI check on the key
registry runs with no PyTorch installed at all::

    import synthmark                                     # no torch
    reg = synthmark.KeyRegistry.load("key_registry.json")
"""

from .keys import WatermarkKey, derive_key, generate_key, load_master_secret
from .registry import KeyEntry, KeyRegistry, RegistryError

__version__ = "0.3.0"

# Attribute -> submodule it lives in.  Explicit, so a typo raises AttributeError
# here rather than a confusing ImportError later.
_LAZY = {
    "to_hf_config": "config",
    "build_processor": "config",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache, so the indirection costs one lookup
    return value


def __dir__():
    return sorted(__all__)


__all__ = [
    "WatermarkKey",
    "derive_key",
    "generate_key",
    "load_master_secret",
    "KeyEntry",
    "KeyRegistry",
    "RegistryError",
    "to_hf_config",
    "build_processor",
    "__version__",
]
