"""Detection of SynthID-Text watermarks in arbitrary text.

Detection replays the watermark's g-function over a candidate token sequence
using the same secret key that generation used, and asks whether the observed
g-values look biased towards 1.  Under the null hypothesis (text that was not
watermarked with this key) every g-value is an unbiased coin flip, so the mean
sits at 0.5; watermarked text pushes it above 0.5.

Three scoring functions are provided, in increasing order of power and of setup
cost:

``mean``
    The mean of every valid g-value.  No training, no calibration, closed-form
    null distribution.  This is the honest baseline.
``weighted_mean``
    A per-depth weighted mean.  Cheap, and lets you downweight depths that carry
    less signal for a given model.
``bayesian``
    A small learned model (see :mod:`synthmark.bayesian`) that scores the whole
    g-value matrix jointly.  Strongest, especially on short text, but must be
    trained per model + key.

Every score is turned into a decision through a *calibrated threshold* rather
than a hand-picked one -- see :class:`Calibration`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .config import build_processor
from .keys import WatermarkKey


@dataclass
class DetectionResult:
    """Outcome of scoring one piece of text."""

    score: float
    """Detector statistic. For mean-based scores this is a g-value mean in [0, 1]."""

    num_tokens_scored: int
    """How many token positions contributed. Detection power grows with this."""

    z_score: float | None = None
    """Standard deviations above the null mean, under the analytic null."""

    p_value: float | None = None
    """One-sided p-value under the analytic null: P(score >= observed | not watermarked)."""

    empirical_p_value: float | None = None
    """p-value read off a calibration set of genuinely non-watermarked text."""

    is_watermarked: bool | None = None
    """Decision at the calibrated threshold, if a calibration was supplied."""

    threshold: float | None = None
    key_id: str | None = None
    key_fingerprint: str | None = None
    method: str = "mean"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Calibration:
    """Empirical null distribution of a detector statistic.

    Analytic p-values assume the g-values are independent fair coin flips.  That
    is very nearly true, but "very nearly" is not something to stake a
    false-accusation rate on.  A calibration set of real non-watermarked text --
    human writing, and output from the same model with the watermark switched
    off -- gives thresholds you can actually defend.

    Because detection power depends strongly on length, thresholds are stored
    per length bucket.
    """

    method: str
    buckets: list[int]
    """Lower edges of the token-count buckets, ascending."""

    scores: dict[int, list[float]] = field(default_factory=dict)
    """Bucket lower edge -> observed null scores."""

    key_fingerprint: str | None = None

    def _bucket_for(self, n_tokens: int) -> int:
        chosen = self.buckets[0]
        for edge in self.buckets:
            if n_tokens >= edge:
                chosen = edge
            else:
                break
        return chosen

    def threshold(self, n_tokens: int, target_fpr: float = 0.01) -> float:
        """Smallest score whose empirical false-positive rate is <= ``target_fpr``."""
        null = self.scores.get(self._bucket_for(n_tokens), [])
        if not null:
            return float("inf")
        return float(np.quantile(np.asarray(null), 1.0 - target_fpr))

    def p_value(self, score: float, n_tokens: int) -> float:
        """Fraction of null scores at or above ``score`` (add-one smoothed)."""
        null = np.asarray(self.scores.get(self._bucket_for(n_tokens), []))
        if null.size == 0:
            return float("nan")
        return float((np.sum(null >= score) + 1) / (null.size + 1))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "method": self.method,
                    "buckets": self.buckets,
                    "key_fingerprint": self.key_fingerprint,
                    "scores": {str(k): v for k, v in self.scores.items()},
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(
            method=data["method"],
            buckets=data["buckets"],
            scores={int(k): v for k, v in data["scores"].items()},
            key_fingerprint=data.get("key_fingerprint"),
        )


class Detector:
    """Scores text for the presence of a specific watermark key.

    Args:
        key: The watermark key to test for.  Detection with the wrong key
            returns a null-distributed score -- this is the property that makes
            per-tenant keys meaningful.
        tokenizer: The tokenizer of the model that produced the text.  The
            watermark lives in *token* choices, so detection requires the same
            tokenizer; a different tokenizer segments the text differently and
            destroys the signal.
        device: Where to run the g-function.  ``"cpu"`` is fine and keeps the
            detector deployable without a GPU, which matters if you want a
            detection service that non-ML teams can call.
    """

    def __init__(
        self,
        key: WatermarkKey,
        tokenizer,
        device: str | torch.device = "cpu",
        *,
        depth_weights: Sequence[float] | None = None,
    ):
        self.key = key
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.processor = build_processor(key, self.device)
        self.depth = key.depth
        self.ngram_len = key.ngram_len
        if depth_weights is None:
            # Linearly decaying weights: the reference implementation's default.
            # Whether this beats a flat mean is an empirical question for a given
            # model -- experiments/03_detectability.py measures it.
            depth_weights = np.linspace(1.0, 0.0, self.depth, endpoint=False)
        w = np.asarray(depth_weights, dtype=np.float64)
        if w.shape != (self.depth,):
            raise ValueError(f"depth_weights must have length {self.depth}")
        self.depth_weights = w / w.sum()

    # ------------------------------------------------------------ g-values

    def _tokenize(self, texts: Sequence[str], prefixes: Sequence[str] | None = None):
        """Right-pad a batch of texts and return ids plus a real-token mask.

        Args:
            texts: The candidate texts.
            prefixes: Optional per-text prefixes (e.g. the prompt that preceded
                the completion).  The watermark at position *i* depends on the
                ``ngram_len - 1`` tokens before it, so the first few completion
                tokens were watermarked using context that ran back into the
                prompt.  A real detector never has that context; supplying it is
                only useful for research comparisons.  Prefix tokens are scored
                out of the final mask either way.
        """
        ids_list, skip_list = [], []
        for i, text in enumerate(texts):
            prefix_ids = []
            if prefixes is not None and prefixes[i]:
                prefix_ids = self.tokenizer(prefixes[i], add_special_tokens=False)["input_ids"]
                # Only the trailing context window can matter.
                prefix_ids = prefix_ids[-(self.ngram_len - 1) :]
            body_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids_list.append(prefix_ids + body_ids)
            skip_list.append(len(prefix_ids))

        max_len = max(len(x) for x in ids_list)
        pad_id = self.tokenizer.pad_token_id or 0
        ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        real = torch.zeros((len(ids_list), max_len), dtype=torch.bool)
        for i, seq in enumerate(ids_list):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            real[i, len(seq) :] = False
            real[i, : len(seq)] = True
        return ids.to(self.device), real.to(self.device), skip_list

    @torch.no_grad()
    def g_values(
        self, texts: Sequence[str], prefixes: Sequence[str] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute g-values and their validity mask.

        Returns:
            ``(g, mask)`` where ``g`` has shape ``(batch, n_positions, depth)``
            and ``mask`` has shape ``(batch, n_positions)``.  A position is valid
            when it is a real (non-pad) token, its context n-gram has not been
            seen before in this text, and it does not sit at or past an EOS.
        """
        if isinstance(texts, str):
            texts = [texts]
        ids, real, skips = self._tokenize(texts, prefixes)
        n_pos = ids.shape[1] - (self.ngram_len - 1)
        if n_pos <= 0:
            empty_g = torch.zeros((len(texts), 0, self.depth), device=self.device)
            return empty_g, torch.zeros((len(texts), 0), dtype=torch.bool, device=self.device)

        g = self.processor.compute_g_values(input_ids=ids)
        mask = self.processor.compute_context_repetition_mask(input_ids=ids).bool()

        # Drop positions whose n-gram window overlaps padding: position j covers
        # source tokens [j, j + ngram_len - 1].
        real_ngram = real.unfold(dimension=1, size=self.ngram_len, step=1).all(dim=-1)
        mask &= real_ngram

        # Drop positions whose window reaches back into a supplied prefix.
        for i, skip in enumerate(skips):
            if skip:
                mask[i, :skip] = False

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos_mask = self.processor.compute_eos_token_mask(
                input_ids=ids, eos_token_id=eos_id
            )[:, self.ngram_len - 1 :].bool()
            mask &= eos_mask

        return g.float(), mask

    # -------------------------------------------------------------- scoring

    def _mean_scores(self, g: torch.Tensor, mask: torch.Tensor, weighted: bool) -> tuple[np.ndarray, np.ndarray]:
        m = mask.unsqueeze(-1).float()                      # (B, T, 1)
        n_tok = mask.sum(dim=1).cpu().numpy().astype(int)   # (B,)
        denom = m.sum(dim=1).clamp(min=1.0)                 # (B, 1)
        per_depth = (g * m).sum(dim=1) / denom              # (B, depth)
        if weighted:
            w = torch.tensor(self.depth_weights, dtype=per_depth.dtype, device=per_depth.device)
            scores = (per_depth * w).sum(dim=1)
        else:
            scores = per_depth.mean(dim=1)
        scores = scores.cpu().numpy()
        scores[n_tok == 0] = float("nan")
        return scores, n_tok

    def score(
        self,
        texts: Sequence[str] | str,
        *,
        method: str = "mean",
        prefixes: Sequence[str] | None = None,
        bayesian_model=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, num_tokens_scored)`` for a batch of texts."""
        if isinstance(texts, str):
            texts = [texts]
        g, mask = self.g_values(texts, prefixes)
        if method == "mean":
            return self._mean_scores(g, mask, weighted=False)
        if method == "weighted_mean":
            return self._mean_scores(g, mask, weighted=True)
        if method == "bayesian":
            if bayesian_model is None:
                raise ValueError("method='bayesian' requires a trained bayesian_model")
            n_tok = mask.sum(dim=1).cpu().numpy().astype(int)
            with torch.no_grad():
                dev = next(bayesian_model.parameters()).device
                out = bayesian_model(g.to(dev), mask.to(dev).float())
            posterior = out[0] if isinstance(out, tuple) else out.posterior_probabilities
            scores = posterior.detach().float().cpu().numpy()
            scores[n_tok == 0] = float("nan")
            return scores, n_tok
        raise ValueError(f"unknown method {method!r}")

    # ------------------------------------------------- single-text convenience

    def detect(
        self,
        text: str,
        *,
        method: str = "mean",
        calibration: Calibration | None = None,
        target_fpr: float = 0.01,
        prefix: str | None = None,
        bayesian_model=None,
    ) -> DetectionResult:
        """Score one text and turn it into a decision.

        Args:
            target_fpr: The false-positive rate the threshold is set to.  This is
                the number to argue about with legal and compliance: it is the
                rate at which text that was *not* watermarked gets flagged.
        """
        scores, n_tok = self.score(
            [text], method=method, prefixes=[prefix] if prefix else None, bayesian_model=bayesian_model
        )
        score, n = float(scores[0]), int(n_tok[0])

        result = DetectionResult(
            score=score,
            num_tokens_scored=n,
            method=method,
            key_id=self.key.key_id,
            key_fingerprint=self.key.fingerprint,
        )

        if method in ("mean", "weighted_mean") and n > 0:
            # Under H0 every g-value is an independent fair coin flip.  A flat
            # mean over n * depth of them has variance 0.25 / (n * depth); the
            # weighted mean has variance 0.25 * sum(w^2) / n.
            if method == "mean":
                var = 0.25 / (n * self.depth)
            else:
                var = 0.25 * float(np.sum(self.depth_weights**2)) / n
            z = (score - 0.5) / math.sqrt(var)
            result.z_score = z
            result.p_value = 0.5 * math.erfc(z / math.sqrt(2))

        if calibration is not None and n > 0:
            result.threshold = calibration.threshold(n, target_fpr)
            result.empirical_p_value = calibration.p_value(score, n)
            result.is_watermarked = bool(score >= result.threshold)

        return result

    # ---------------------------------------------------------- calibration

    def calibrate(
        self,
        negative_texts: Iterable[str],
        *,
        method: str = "mean",
        buckets: Sequence[int] = (0, 50, 100, 200, 400, 800),
        batch_size: int = 16,
        bayesian_model=None,
    ) -> Calibration:
        """Build an empirical null distribution from non-watermarked text.

        Feed this human-written text *and* unwatermarked output from the same
        model.  Both matter: the first is what a false accusation would land on,
        the second checks that the detector is keyed to the watermark rather
        than to the model's style.
        """
        texts = list(negative_texts)
        cal = Calibration(
            method=method, buckets=list(buckets), key_fingerprint=self.key.fingerprint
        )
        cal.scores = {b: [] for b in cal.buckets}
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            scores, n_tok = self.score(chunk, method=method, bayesian_model=bayesian_model)
            for s, n in zip(scores, n_tok):
                if n > 0 and not math.isnan(s):
                    cal.scores[cal._bucket_for(int(n))].append(float(s))
        return cal
