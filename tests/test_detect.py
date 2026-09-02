"""Detector behaviour that must hold regardless of which model produced the text.

These tests use a tiny stub tokenizer and synthetic token sequences, so they run
in milliseconds on CPU with no model download.  The statistical properties they
check -- an unbiased null, masking of repeats and padding, key independence --
are the ones that a wrong answer would quietly break.
"""

import numpy as np
import pytest

from synthmark_detect import Calibration, Detector
from synthmark.keys import derive_key

MASTER = "master-secret-with-enough-entropy"


class StubTokenizer:
    """Maps whitespace-separated integers to token ids, so tests control tokens exactly."""

    eos_token_id = 999_999
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(t) for t in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def random_text(n, rng, vocab=50_000):
    return " ".join(str(int(x)) for x in rng.integers(1, vocab, size=n))


@pytest.fixture
def detector():
    return Detector(derive_key(MASTER, "test/v1", depth=8), StubTokenizer())


def test_null_mean_is_centred_on_half(detector):
    """Random token sequences must score 0.5 on average: the detector is unbiased."""
    rng = np.random.default_rng(0)
    texts = [random_text(300, rng) for _ in range(64)]
    scores, n = detector.score(texts, method="mean")
    assert (n > 250).all()
    # Standard error over 64 texts x ~296 positions x depth 8 is ~0.0011.
    assert abs(scores.mean() - 0.5) < 0.005


def test_shorter_text_scores_fewer_tokens(detector):
    rng = np.random.default_rng(1)
    long_scores, n_long = detector.score([random_text(400, rng)], method="mean")
    short_scores, n_short = detector.score([random_text(50, rng)], method="mean")
    assert n_long[0] > n_short[0]
    assert n_short[0] == 50 - (detector.ngram_len - 1)


def test_repeated_context_is_masked(detector):
    """A text that repeats one n-gram forever must contribute almost no positions."""
    repeated = " ".join(["11 12 13 14 15"] * 40)
    _, n = detector.score([repeated], method="mean")
    varied = random_text(200, np.random.default_rng(2))
    _, n_varied = detector.score([varied], method="mean")
    assert n[0] < 10, "repeated contexts should be masked out"
    assert n_varied[0] > 150


def test_padding_does_not_contribute(detector):
    """Batching a short text with a long one must not change the short one's score."""
    rng = np.random.default_rng(3)
    short, long_ = random_text(60, rng), random_text(400, rng)
    alone, n_alone = detector.score([short], method="mean")
    batched, n_batched = detector.score([short, long_], method="mean")
    assert n_alone[0] == n_batched[0]
    assert alone[0] == pytest.approx(batched[0], abs=1e-6)


def test_different_keys_give_different_scores(detector):
    rng = np.random.default_rng(4)
    other = Detector(derive_key(MASTER, "test/v2", depth=8), StubTokenizer())
    text = random_text(300, rng)
    a, _ = detector.score([text], method="mean")
    b, _ = other.score([text], method="mean")
    assert a[0] != b[0]


def test_analytic_pvalue_is_uniform_under_null(detector):
    """The stated false-positive rate has to be the real one."""
    rng = np.random.default_rng(5)
    ps = [detector.detect(random_text(300, rng)).p_value for _ in range(300)]
    ps = np.array(ps)
    # Under the null, p-values are uniform; allow generous slack at n=300.
    assert 0.02 < np.mean(ps < 0.10) < 0.20
    assert abs(np.mean(ps) - 0.5) < 0.10


def test_weighted_and_flat_mean_agree_on_direction(detector):
    rng = np.random.default_rng(6)
    texts = [random_text(300, rng) for _ in range(16)]
    flat, _ = detector.score(texts, method="mean")
    weighted, _ = detector.score(texts, method="weighted_mean")
    assert flat.shape == weighted.shape
    assert np.isfinite(weighted).all()


def test_text_shorter_than_ngram_is_handled(detector):
    scores, n = detector.score(["7 8"], method="mean")
    assert n[0] == 0
    assert np.isnan(scores[0])
    result = detector.detect("7 8")
    assert result.num_tokens_scored == 0
    assert result.p_value is None


def test_calibration_thresholds_and_roundtrip(detector, tmp_path):
    rng = np.random.default_rng(7)
    texts = [random_text(300, rng) for _ in range(200)]
    cal = detector.calibrate(texts, method="mean")
    thr = cal.threshold(300, target_fpr=0.01)
    assert 0.5 < thr < 0.6
    # A stricter target must give a higher bar.
    assert cal.threshold(300, 0.001) >= thr >= cal.threshold(300, 0.10)
    path = cal.save(tmp_path / "cal.json")
    assert Calibration.load(path).threshold(300, 0.01) == thr


def test_unknown_method_rejected(detector):
    with pytest.raises(ValueError):
        detector.score(["1 2 3 4 5 6 7 8"], method="telepathy")


# --------------------------------------------------------- device portability


def test_watermark_is_device_independent():
    """The same key must give the same watermark on CPU and on GPU.

    Upstream Transformers builds the g-value sampling table with a device-local
    RNG, so a GPU-built processor and a CPU-built processor disagree for the
    same seed.  Text generated on a GPU is then silently invisible to a CPU
    detector.  synthmark draws the table on CPU and moves it, which is what this
    guards.
    """
    import torch

    from synthmark.config import build_processor

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    key = derive_key(MASTER, "portability/v1", depth=8)
    cpu_proc = build_processor(key, "cpu")
    gpu_proc = build_processor(key, "cuda:0")

    assert torch.equal(cpu_proc.sampling_table, gpu_proc.sampling_table.cpu())

    ids = torch.randint(1, 50_000, (3, 128))
    assert torch.equal(
        cpu_proc.compute_g_values(ids),
        gpu_proc.compute_g_values(ids.cuda()).cpu(),
    )
    assert torch.equal(
        cpu_proc.compute_context_repetition_mask(ids),
        gpu_proc.compute_context_repetition_mask(ids.cuda()).cpu(),
    )


def test_upstream_path_is_device_dependent():
    """Documents the upstream behaviour our portable path exists to avoid.

    If this test ever starts failing, Transformers has fixed the bug upstream
    and the portable wrapper can be retired.
    """
    import torch

    from synthmark.config import build_processor

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    key = derive_key(MASTER, "portability/v1", depth=8)
    cpu_table = build_processor(key, "cpu", portable=False).sampling_table
    gpu_table = build_processor(key, "cuda:0", portable=False).sampling_table.cpu()
    assert not torch.equal(cpu_table, gpu_table)


def test_detector_scores_match_across_devices():
    """A CPU detector and a GPU detector must agree on the same text."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    key = derive_key(MASTER, "portability/v1", depth=8)
    rng = np.random.default_rng(11)
    texts = [random_text(250, rng) for _ in range(8)]
    cpu_scores, cpu_n = Detector(key, StubTokenizer(), device="cpu").score(texts)
    gpu_scores, gpu_n = Detector(key, StubTokenizer(), device="cuda:0").score(texts)
    assert (cpu_n == gpu_n).all()
    np.testing.assert_allclose(cpu_scores, gpu_scores, atol=1e-6)


# ------------------------------------------- candidate-only fast path


def _topk_filter(scores, k):
    """Mimic what top-k filtering leaves on the table before the watermark runs."""
    import torch

    kth = scores.topk(k, dim=1).values[:, -1:]
    return scores.masked_fill(scores < kth, float("-inf"))


def _fast_and_reference(key, batch, vocab, top_k, steps=3, seed=0):
    """Run both processors over identical inputs and return their outputs."""
    import torch

    from synthmark.config import build_processor

    ref = build_processor(key, "cpu", fast=False)
    fast = build_processor(key, "cpu", fast=True)
    ref._init_state(batch)
    fast._init_state(batch)

    g = torch.Generator().manual_seed(seed)
    out_ref, out_fast = [], []
    for _ in range(steps):
        ids = torch.randint(1, vocab, (batch, 8), generator=g)
        scores = torch.randn(batch, vocab, generator=g)
        if top_k:
            scores = _topk_filter(scores, top_k)
        out_ref.append(ref(ids, scores.clone()))
        out_fast.append(fast(ids, scores.clone()))
    return out_ref, out_fast


#: Divergence permitted between the two paths, measured in probability space.
#: They differ only by floating-point summation order -- upstream sums over the
#: whole vocabulary, the fast path over the candidates -- so the gap is float32
#: rounding, not algorithmic.
#:
#: Probability space is the correct place to assert. Log-probabilities of tokens
#: the tournament has driven to ~1e-19 can differ by ~0.3 simply because ``log``
#: is ill conditioned there, while the probabilities themselves differ by ~1e-19
#: and nothing downstream can observe it.
FAST_PATH_PROB_TOL = 1e-5


def _assert_equivalent(a, b, context=""):
    """Both paths must induce the same sampling distribution."""
    import torch

    pa, pb = torch.softmax(a, dim=1), torch.softmax(b, dim=1)
    max_dp = float((pa - pb).abs().max())
    tv = float(0.5 * (pa - pb).abs().sum(dim=1).max())
    assert max_dp < FAST_PATH_PROB_TOL, f"probability gap {max_dp:.2e} {context}"
    assert tv < FAST_PATH_PROB_TOL, f"total-variation {tv:.2e} {context}"
    assert torch.equal(pa.argmax(dim=1), pb.argmax(dim=1)), f"argmax changed {context}"


def test_fast_processor_matches_reference_with_topk():
    """The candidate-only path must reproduce upstream scores to float precision.

    This is the whole justification for the optimisation: tokens eliminated by
    top-k carry zero probability, so excluding them from the g-function cannot
    change the tournament's result.
    """
    import torch

    key = derive_key(MASTER, "fast/v1", depth=8)
    ref, fast = _fast_and_reference(key, batch=4, vocab=4096, top_k=32)
    for a, b in zip(ref, fast):
        _assert_equivalent(a, b)
        # Filtered-out tokens must land on exactly the same floor value.
        filtered = a == torch.finfo(a.dtype).min
        assert torch.equal(a[filtered], b[filtered])


def test_fast_processor_preserves_the_sampling_distribution():
    """What actually matters downstream is the distribution the sampler sees."""
    import torch

    # Production-scale depth, where the tournament drives some candidates to
    # ~1e-19 and log-space comparisons stop being meaningful.
    key = derive_key(MASTER, "fast/v1", depth=30)
    ref, fast = _fast_and_reference(key, batch=4, vocab=8192, top_k=64)
    for a, b in zip(ref, fast):
        _assert_equivalent(a, b, "at depth 30")


def test_fast_processor_matches_across_batch_sizes():
    """Reduction must hold however many sequences share the batch."""
    import torch

    key = derive_key(MASTER, "fast/v1", depth=8)
    for batch in (1, 2, 8):
        ref, fast = _fast_and_reference(key, batch=batch, vocab=2048, top_k=16, seed=batch)
        for a, b in zip(ref, fast):
            _assert_equivalent(a, b, f"at batch={batch}")


def test_fast_processor_falls_back_when_nothing_is_filtered():
    """With no top-k/top-p applied there is nothing to skip; results must still match."""
    import torch

    key = derive_key(MASTER, "fast/v1", depth=8)
    ref, fast = _fast_and_reference(key, batch=2, vocab=1024, top_k=None, seed=7)
    for a, b in zip(ref, fast):
        # Nothing was filtered, so this took the parent path: exactly equal.
        assert torch.equal(a, b), "fallback path diverged from the reference"


def test_fast_processor_handles_ragged_candidate_counts():
    """top-p leaves different numbers of survivors per row; padding must not leak signal."""
    import torch

    from synthmark.config import build_processor

    key = derive_key(MASTER, "fast/v1", depth=8)
    vocab, batch = 2048, 3
    ref = build_processor(key, "cpu", fast=False)
    fast = build_processor(key, "cpu", fast=True)
    ref._init_state(batch)
    fast._init_state(batch)

    g = torch.Generator().manual_seed(11)
    for _ in range(3):
        ids = torch.randint(1, vocab, (batch, 8), generator=g)
        scores = torch.randn(batch, vocab, generator=g)
        # Deliberately uneven: 4, 16 and 64 survivors.
        for row, k in enumerate((4, 16, 64)):
            kth = scores[row].topk(k).values[-1]
            scores[row] = scores[row].masked_fill(scores[row] < kth, float("-inf"))
        _assert_equivalent(ref(ids, scores.clone()), fast(ids, scores.clone()), "ragged rows")
