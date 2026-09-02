"""synthmark-detect -- scoring text for a watermark, and the detection service.

Separate from :mod:`synthmark` because it has the opposite deployment shape.
Detection reads token ids and hashes them: no forward pass, no model weights, no
GPU, ~1,250 documents/s per key on ordinary CPU.  It belongs on its own
hardware, far from the serving fleet -- and it is the *only* component besides
the serving process that should ever hold a key, because a watermark key is
symmetric: whoever can detect with it can also forge with it.

    from synthmark_detect import Detector
    from synthmark_detect.serve import build_app, build_served

:mod:`synthmark_detect.serve` needs FastAPI, which is not a dependency of this
package by default::

    pip install "synthmark-detect[serve]"
"""

from .detect import Calibration, DetectionResult, Detector

__version__ = "0.3.0"

__all__ = ["Detector", "DetectionResult", "Calibration", "__version__"]
