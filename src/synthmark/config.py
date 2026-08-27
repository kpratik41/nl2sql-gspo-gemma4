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


class CandidateOnlySynthIDLogitsProcessor(PortableSynthIDLogitsProcessor):
    """SynthID processor that hashes only the tokens that can actually be sampled.

    Why this exists
    ---------------
    The watermark processor runs *after* top-k / top-p filtering, so by the time
    it sees the scores nearly the whole vocabulary is already ``-inf`` and
    carries zero probability.  Upstream nevertheless evaluates the g-function
    over ``arange(vocab_size)``: at each decoding step it materialises a
    ``(batch, vocab_size, depth)`` int64 tensor -- 2 GB at batch 32 with Gemma's
    262k vocabulary and depth 30 -- and makes several passes over it.  Upstream's
    own comments still read ``[batch_size, top_k, depth]``, inherited from the
    DeepMind reference, which passes only the surviving candidates.

    With ``top_k=64`` that is roughly 4,000x more hashing than the result
    requires, and the waste scales with batch size, which is why batched
    throughput collapses while single-request latency looks fine.

    Why restricting to candidates is exact, not an approximation
    -----------------------------------------------------------
    Writing ``C`` for the set of tokens with finite score:

    * ``softmax`` over the full vocabulary equals ``softmax`` over ``C`` alone --
      the ``-inf`` entries contribute nothing to numerator or denominator.
    * The tournament's ``g_mass = sum(g * probs)`` is unchanged, because tokens
      outside ``C`` have probability zero.
    * ``probs * (1 + g - g_mass)`` maps zero to zero, so tokens outside ``C``
      stay at zero and end up at the same floor value upstream assigns them.

    The returned scores are therefore mathematically identical.  They are not
    *bitwise* identical in float32: upstream sums ``g * probs`` over the whole
    vocabulary while this sums over the candidates, and floating-point addition
    is not associative.

    The right place to measure that gap is probability space, not log space.
    Measured against the reference at Gemma's vocabulary with depth 30, the
    largest probability difference for any token is ~1e-7 and the total-variation
    distance between the two sampling distributions is ~1e-7, with the most
    likely token always unchanged.  Individual *log*-probabilities can differ by
    more -- around 0.3 in the worst case -- but only for tokens the tournament
    has already driven to a probability of ~1e-19, where ``log`` is ill
    conditioned and the absolute probability gap is ~1e-19.  Nothing downstream
    can observe that.  ``tests/`` asserts the probability-space bounds.

    The g-values themselves are untouched, so watermark strength is unaffected.

    When there is nothing to skip -- unfiltered sampling, where every token has a
    finite score -- this falls back to the parent implementation rather than
    paying for a full-vocabulary ``topk``.
    """

    #: Above this fraction of the vocabulary surviving the filter, the reduction
    #: cannot pay for its own gather/scatter, so the parent path is used.
    fallback_fraction: float = 0.25

    def _update_scores_subset(
        self, sub_scores: torch.Tensor, g_values: torch.Tensor
    ) -> torch.Tensor:
        """Tournament update over a candidate subset.

        Mirrors ``update_scores`` exactly; only the vocabulary axis is shorter.
        The loop over depth is inherently sequential -- each round reweights the
        distribution the previous round produced -- so it is kept as a loop.
        """
        _, _, depth = g_values.shape
        probs = torch.softmax(sub_scores, dim=1)
        for i in range(depth):
            g_at_depth = g_values[:, :, i]
            g_mass = (g_at_depth * probs).sum(dim=1, keepdim=True)
            probs = probs * (1 + g_at_depth - g_mass)
        log_probs = torch.log(probs)
        return torch.where(
            torch.isfinite(log_probs), log_probs, torch.finfo(log_probs.dtype).min
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # State bookkeeping below mirrors the upstream __call__ exactly; only the
        # index set handed to the g-function differs.
        self._check_input_ids_shape(input_ids)
        batch_size, vocab_size = scores.shape

        if self.debug_mode:
            scores = torch.ones_like(scores)

        # Decide the path BEFORE touching state. The parent does its own context
        # and num_calls bookkeeping, so advancing state here and then delegating
        # would advance it twice and desynchronise the context window -- which
        # changes the g-values and silently corrupts the watermark.
        n_candidates = int(torch.isfinite(scores).sum(dim=1).max().item())
        if n_candidates >= int(vocab_size * self.fallback_fraction):
            return super().__call__(input_ids, scores)

        if self.state is None:
            self._init_state(batch_size)
        else:
            self.state.context = torch.concat(
                (self.state.context, input_ids[:, -1:]), dim=1
            )[:, 1:]

        self.state.num_calls += 1
        if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
            return scores

        # topk returns the largest scores, and -inf sorts last, so this is
        # guaranteed to cover every finite entry in every row.
        cand_idx = scores.topk(n_candidates, dim=1).indices
        sub_scores = scores.gather(1, cand_idx)

        ngram_keys, hash_with_context = self._compute_keys(self.state.context, cand_idx)
        g_values = self.sample_g_values(ngram_keys)
        sub_updated = self._update_scores_subset(sub_scores, g_values)

        updated_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        updated_scores.scatter_(1, cand_idx, sub_updated)

        hash_with_context = hash_with_context[:, None]
        is_repeated_context = (self.state.context_history == hash_with_context).any(
            dim=1, keepdim=True
        )
        self.state.context_history = torch.concat(
            (hash_with_context, self.state.context_history), dim=1
        )[:, :-1]

        return torch.where(is_repeated_context, input=scores, other=updated_scores)


class PortableSynthIDWatermarkingConfig(SynthIDTextWatermarkingConfig):
    """``watermarking_config`` that builds a device-independent processor.

    ``fast`` selects :class:`CandidateOnlySynthIDLogitsProcessor`, which produces
    identical scores but skips hashing tokens that top-k/top-p already ruled out.
    """

    fast: bool = True

    def construct_processor(self, vocab_size: int, device) -> SynthIDTextWatermarkLogitsProcessor:
        cls = CandidateOnlySynthIDLogitsProcessor if self.fast else PortableSynthIDLogitsProcessor
        return cls(
            ngram_len=self.ngram_len,
            keys=self.keys,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
            device=device,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
            debug_mode=self.debug_mode,
        )


def to_hf_config(
    key: WatermarkKey, *, portable: bool = True, fast: bool = True
) -> SynthIDTextWatermarkingConfig:
    """Build the ``watermarking_config`` to hand to ``model.generate``.

    Args:
        portable: Use the device-independent sampling table (recommended, and the
            default).  Set to ``False`` only to reproduce the exact behaviour of
            upstream Transformers.
        fast: Skip hashing tokens that top-k/top-p already eliminated.  Produces
            identical scores; set to ``False`` to use the full-vocabulary path.
    """
    cls = PortableSynthIDWatermarkingConfig if portable else SynthIDTextWatermarkingConfig
    cfg = cls(
        ngram_len=key.ngram_len,
        keys=list(key.keys),
        context_history_size=key.context_history_size,
        sampling_table_seed=key.sampling_table_seed,
        sampling_table_size=key.sampling_table_size,
    )
    if portable:
        cfg.fast = fast
    return cfg


def build_processor(
    key: WatermarkKey,
    device: str | torch.device = "cpu",
    *,
    portable: bool = True,
    fast: bool = False,
) -> SynthIDTextWatermarkLogitsProcessor:
    """Build the logits processor directly.

    Detection needs the same g-function that generation used, which this exposes
    via ``compute_g_values`` and the masking helpers.  The processor carries no
    per-request state, so one instance serves many detection calls.

    Args:
        portable: See :func:`to_hf_config`.  Must match whatever produced the
            text: a portable detector cannot read a non-portable watermark.
        fast: Use the candidate-only processor.  Irrelevant for detection, which
            calls ``compute_g_values`` directly rather than ``__call__``, so this
            defaults to ``False`` here.
    """
    if portable:
        cls = CandidateOnlySynthIDLogitsProcessor if fast else PortableSynthIDLogitsProcessor
    else:
        cls = SynthIDTextWatermarkLogitsProcessor
    return cls(
        ngram_len=key.ngram_len,
        keys=list(key.keys),
        sampling_table_size=key.sampling_table_size,
        sampling_table_seed=key.sampling_table_seed,
        context_history_size=key.context_history_size,
        device=torch.device(device),
    )
