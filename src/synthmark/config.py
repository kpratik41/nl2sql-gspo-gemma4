"""Bridge between :class:`synthmark.keys.WatermarkKey` and the HF SynthID API.

Device portability
------------------
The upstream implementation builds its g-value sampling table with a
device-local RNG::

    generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
    self.sampling_table = torch.randint(0, 2, (size,), generator=generator, device=device)

``torch.randint`` draws *different values* from CUDA and CPU generators given the
same seed.  The consequence is that the same key yields a **different watermark**
depending on which device the processor happened to be built on: text generated
on a GPU is invisible to a detector running on CPU, and the failure is silent --
the detector simply reports "no watermark" with a perfectly ordinary-looking
null score.

That would make a CPU detection service impossible, and would break detection
across a fleet with mixed hardware.  Since the sampling table is the *only*
source of randomness in the algorithm -- everything else is a deterministic
linear-congruential hash over int64 -- the fix is to always draw the table on
CPU and move it to the target device.  The classes below do that, and are what
:mod:`synthmark` uses for both generation and detection, so the two can never
disagree.

Watermarks produced with the upstream (device-dependent) code path are still
detectable by building the detector on the same device type that generated
them; see :func:`build_processor` with ``portable=False``.
"""

from __future__ import annotations

import torch
from transformers import SynthIDTextWatermarkingConfig
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

from .keys import WatermarkKey


class PortableSynthIDLogitsProcessor(SynthIDTextWatermarkLogitsProcessor):
    """SynthID logits processor whose watermark does not depend on the device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = kwargs.get("sampling_table_seed")
        size = kwargs.get("sampling_table_size")
        if seed is None or size is None:  # positional construction
            raise TypeError(
                "PortableSynthIDLogitsProcessor must be constructed with keyword arguments"
            )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        table = torch.randint(low=0, high=2, size=(size,), generator=generator, device="cpu")
        self.sampling_table = table.to(self.device)


class PortableSynthIDWatermarkingConfig(SynthIDTextWatermarkingConfig):
    """``watermarking_config`` that builds a device-independent processor."""

    def construct_processor(self, vocab_size: int, device) -> SynthIDTextWatermarkLogitsProcessor:
        return PortableSynthIDLogitsProcessor(
            ngram_len=self.ngram_len,
            keys=self.keys,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
            device=device,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
            debug_mode=self.debug_mode,
        )


def to_hf_config(key: WatermarkKey, *, portable: bool = True) -> SynthIDTextWatermarkingConfig:
    """Build the ``watermarking_config`` to hand to ``model.generate``.

    Args:
        portable: Use the device-independent sampling table (recommended, and the
            default).  Set to ``False`` only to reproduce the exact behaviour of
            upstream Transformers.
    """
    cls = PortableSynthIDWatermarkingConfig if portable else SynthIDTextWatermarkingConfig
    return cls(
        ngram_len=key.ngram_len,
        keys=list(key.keys),
        context_history_size=key.context_history_size,
        sampling_table_seed=key.sampling_table_seed,
        sampling_table_size=key.sampling_table_size,
    )


def build_processor(
    key: WatermarkKey, device: str | torch.device = "cpu", *, portable: bool = True
) -> SynthIDTextWatermarkLogitsProcessor:
    """Build the logits processor directly.

    Detection needs the same g-function that generation used, which this exposes
    via ``compute_g_values`` and the masking helpers.  The processor carries no
    per-request state, so one instance serves many detection calls.

    Args:
        portable: See :func:`to_hf_config`.  Must match whatever produced the
            text: a portable detector cannot read a non-portable watermark.
    """
    cls = PortableSynthIDLogitsProcessor if portable else SynthIDTextWatermarkLogitsProcessor
    return cls(
        ngram_len=key.ngram_len,
        keys=list(key.keys),
        sampling_table_size=key.sampling_table_size,
        sampling_table_seed=key.sampling_table_seed,
        context_history_size=key.context_history_size,
        device=torch.device(device),
    )
