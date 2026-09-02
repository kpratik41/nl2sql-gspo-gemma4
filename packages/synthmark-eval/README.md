# synthmark-eval

HuggingFace watermarked generation, attacks, metrics, and the learned Bayesian
detector — for *measuring* a watermark, not serving one.

Separate from `synthmark` and `synthmark-detect` so that neither the serving
image nor the detection service installs it: scipy and scikit-learn alone are
~158 MB.

```python
from synthmark_eval import Detector, WatermarkedLM, derive_key, evaluate_detection
```

`WatermarkedLM` wraps HuggingFace `generate()`, which is the research path. A
vLLM deployment uses the logits-processor plugin instead and never imports this.

Full documentation: <https://github.com/kpratik41/nl2sql-gspo-gemma4/tree/watermarking>
