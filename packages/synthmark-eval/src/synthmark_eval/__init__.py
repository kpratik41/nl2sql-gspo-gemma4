"""synthmark-eval -- HuggingFace generation, attacks, metrics, and the learned detector.

Everything here is for *measuring* a watermark, not for serving one.  It is a
separate distribution so that neither the serving image nor the detection
service has to install it: scipy and scikit-learn alone are ~158 MB, and a
security review of a wheel entering a GPU serving image will ask about every
module in it.

:class:`~synthmark_eval.generate.WatermarkedLM` lives here rather than in the
core package for the same reason.  It wraps HuggingFace ``generate()``, which is
the research path; a vLLM deployment uses the logits-processor plugin instead
and never imports it.

For convenience this package re-exports the pieces the experiment scripts use,
so a benchmark needs one import line rather than three::

    from synthmark_eval import Detector, WatermarkedLM, derive_key, evaluate_detection
"""

from synthmark import (
    KeyEntry,
    KeyRegistry,
    WatermarkKey,
    build_processor,
    derive_key,
    generate_key,
    load_master_secret,
    to_hf_config,
)
from synthmark_detect import Calibration, DetectionResult, Detector

from .generate import DEFAULT_MODEL, GenerationOutput, WatermarkedLM
from .metrics import (
    DetectionMetrics,
    evaluate_detection,
    paired_bootstrap_diff,
    two_proportion_diff_ci,
)

__version__ = "0.3.0"

__all__ = [
    # re-exported from synthmark
    "WatermarkKey",
    "derive_key",
    "generate_key",
    "load_master_secret",
    "KeyEntry",
    "KeyRegistry",
    "to_hf_config",
    "build_processor",
    # re-exported from synthmark-detect
    "Detector",
    "DetectionResult",
    "Calibration",
    # this package
    "WatermarkedLM",
    "GenerationOutput",
    "DEFAULT_MODEL",
    "evaluate_detection",
    "DetectionMetrics",
    "paired_bootstrap_diff",
    "two_proportion_diff_ci",
    "__version__",
]
