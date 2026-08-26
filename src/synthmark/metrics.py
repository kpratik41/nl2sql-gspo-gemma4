"""Detection-quality metrics, reported the way a risk reviewer will want them."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class DetectionMetrics:
    """Separation between watermarked and non-watermarked score distributions.

    ``auc`` is the headline number, but on its own it is the wrong thing to
    govern a deployment with: it averages over every operating point, including
    ones with absurd false-positive rates.  The ``tpr_at_fpr_*`` fields are the
    decision-relevant ones -- "if we accept flagging 1 in 100 innocent documents,
    what fraction of watermarked ones do we catch?"
    """

    n_positive: int
    n_negative: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    tpr_at_fpr_10pct: float
    tpr_at_fpr_1pct: float
    tpr_at_fpr_0p1pct: float
    threshold_at_fpr_1pct: float
    mean_positive: float
    mean_negative: float
    median_tokens_scored: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """True-positive rate at the strictest threshold whose FPR is <= target.

    Returns ``(tpr, threshold)``.  If no threshold achieves the target FPR
    (which happens when the negative set is smaller than ``1 / target_fpr``),
    returns the most conservative point available -- so the number is a lower
    bound on achievable TPR, never an optimistic one.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = fpr <= target_fpr
    if not ok.any():
        return 0.0, float("inf")
    idx = int(np.max(np.flatnonzero(ok)))
    return float(tpr[idx]), float(thr[idx])


def bootstrap_auc_ci(
    y_true: np.ndarray, scores: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUC, resampling each class separately.

    Stratified resampling keeps the class balance fixed, which is what we want:
    the uncertainty of interest is about the score distributions, not about how
    many of each we happened to generate.
    """
    rng = np.random.default_rng(seed)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    aucs = np.empty(n_boot)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    for i in range(n_boot):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        aucs[i] = roc_auc_score(labels, np.concatenate([p, n]))
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def evaluate_detection(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    tokens_scored: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> DetectionMetrics:
    """Full metric bundle for one detector configuration."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])

    auc = float(roc_auc_score(y, s))
    lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
    t10, _ = tpr_at_fpr(y, s, 0.10)
    t1, thr1 = tpr_at_fpr(y, s, 0.01)
    t01, _ = tpr_at_fpr(y, s, 0.001)

    return DetectionMetrics(
        n_positive=len(pos),
        n_negative=len(neg),
        auc=auc,
        auc_ci_low=lo,
        auc_ci_high=hi,
        tpr_at_fpr_10pct=t10,
        tpr_at_fpr_1pct=t1,
        tpr_at_fpr_0p1pct=t01,
        threshold_at_fpr_1pct=thr1,
        mean_positive=float(pos.mean()) if len(pos) else float("nan"),
        mean_negative=float(neg.mean()) if len(neg) else float("nan"),
        median_tokens_scored=float(np.median(tokens_scored)) if tokens_scored is not None else None,
    )


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """CI for the mean difference ``a - b`` over paired observations.

    This is the workhorse for the quality story: the claim "watermarking does
    not degrade quality" is only meaningful with an interval attached.  A CI
    that contains zero is not proof of no effect -- it bounds how large an
    effect the data can still hide, which is the honest framing.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap_diff requires equal-length paired arrays")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "n_pairs": int(len(d)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "contains_zero": bool(
            np.quantile(means, alpha / 2) <= 0 <= np.quantile(means, 1 - alpha / 2)
        ),
    }


def two_proportion_diff_ci(
    k_a: int, n_a: int, k_b: int, n_b: int, *, alpha: float = 0.05
) -> dict:
    """Newcombe interval for the difference of two independent proportions.

    Used for accuracy deltas on multiple-choice benchmarks, where the outcome is
    a count of correct answers rather than a continuous score.
    """
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)

    def wilson(k: int, n: int) -> tuple[float, float]:
        if n == 0:
            return 0.0, 1.0
        p = k / n
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
        return c - h, c + h

    l1, u1 = wilson(k_a, n_a)
    l2, u2 = wilson(k_b, n_b)
    p1, p2 = k_a / max(n_a, 1), k_b / max(n_b, 1)
    return {
        "acc_a": p1,
        "acc_b": p2,
        "diff": p1 - p2,
        "ci_low": (p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        "ci_high": (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
        "n_a": n_a,
        "n_b": n_b,
    }
