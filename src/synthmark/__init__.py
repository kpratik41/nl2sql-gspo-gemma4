"""synthmark -- SynthID-Text watermarking, detection and evaluation for open-weight LLMs.

Quick start::

    from synthmark import WatermarkedLM, Detector, derive_key

    key = derive_key(master_secret, "my-team/v1")
    lm = WatermarkedLM("google/gemma-4-E4B-it")

    out = lm.generate(lm.chat_prompts(["Write about the sea."]), key=key)
    result = Detector(key, lm.tokenizer).detect(out.texts[0])
    print(result.score, result.p_value)
"""

from .keys import WatermarkKey, derive_key, generate_key, load_master_secret
from .config import to_hf_config, build_processor
from .generate import WatermarkedLM, GenerationOutput, DEFAULT_MODEL
from .detect import Detector, DetectionResult, Calibration
from .metrics import evaluate_detection, DetectionMetrics, paired_bootstrap_diff, two_proportion_diff_ci

__version__ = "0.1.0"

__all__ = [
    "WatermarkKey",
    "derive_key",
    "generate_key",
    "load_master_secret",
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
