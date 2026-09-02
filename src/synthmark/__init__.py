"""synthmark -- SynthID-Text watermarking, detection and evaluation for open-weight LLMs.

Quick start::

    from synthmark import WatermarkedLM, Detector, derive_key

    key = derive_key(master_secret, "my-team/v1")
    lm = WatermarkedLM("google/gemma-4-E4B-it")

    out = lm.generate(lm.chat_prompts(["Write about the sea."]), key=key)
    result = Detector(key, lm.tokenizer).detect(out.texts[0])
    print(result.score, result.p_value)

Import cost
-----------
Only :mod:`synthmark.keys` and :mod:`synthmark.registry` are imported eagerly;
both are pure standard library.  Everything else is resolved on first attribute
access (PEP 562), so a key-management job, a CI check on the key registry, or a
non-Python caller's tooling never pays for PyTorch, and a detection deployment
never pays for the evaluation stack.
"""

from .keys import WatermarkKey, derive_key, generate_key, load_master_secret
from .registry import KeyEntry, KeyRegistry, RegistryError

__version__ = "0.2.0"

# Attribute -> submodule it lives in.  Kept explicit so a typo is an
# AttributeError at import time rather than a confusing ImportError later.
_LAZY = {
    "to_hf_config": "config",
    "build_processor": "config",
    "WatermarkedLM": "generate",
    "GenerationOutput": "generate",
    "DEFAULT_MODEL": "generate",
    "Detector": "detect",
    "DetectionResult": "detect",
    "Calibration": "detect",
    "evaluate_detection": "metrics",
    "DetectionMetrics": "metrics",
    "paired_bootstrap_diff": "metrics",
    "two_proportion_diff_ci": "metrics",
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
    "WatermarkedLM",
    "GenerationOutput",
    "DEFAULT_MODEL",
    "Detector",
    "DetectionResult",
    "Calibration",
    "evaluate_detection",
    "DetectionMetrics",
    "paired_bootstrap_diff",
    "two_proportion_diff_ci",
    "__version__",
]
