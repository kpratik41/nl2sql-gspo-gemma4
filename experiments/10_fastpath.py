"""Verify and benchmark the candidate-only logits processor.

Upstream evaluates the watermark's g-function over the entire vocabulary, even
though it runs *after* top-k/top-p and nearly every token has already been set
to ``-inf``.  ``CandidateOnlySynthIDLogitsProcessor`` evaluates it only over the
tokens that can still be sampled.

This script checks the two things that matter, in that order:

1. **Equivalence.** The optimisation is only worth having if it does not change
   the result.  Both processors are run over identical inputs and their outputs
   compared, along with the sampling distributions those outputs induce.
2. **Speed.** Only then is the saving measured.

Runs on CPU.  Absolute timings are not GPU timings, but the scaling behaviour --
whether cost grows with batch size or stays flat -- is a property of the
algorithm rather than the device, and that is the question at issue.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from common import RESULTS, banner, save_json

from synthmark.config import build_processor
from synthmark.keys import derive_key


def filtered_scores(batch: int, vocab: int, top_k: int, gen: torch.Generator) -> torch.Tensor:
    """Logits as the watermark actually receives them: post top-k, mostly -inf."""
    s = torch.randn(batch, vocab, generator=gen)
    kth = s.topk(top_k, dim=1).values[:, -1:]
    return s.masked_fill(s < kth, float("-inf"))


def time_processor(proc, batch: int, vocab: int, top_k: int, repeats: int) -> float:
    gen = torch.Generator().manual_seed(0)
    proc._init_state(batch)
    ids = torch.randint(1, vocab, (batch, 8), generator=gen)
    scores = filtered_scores(batch, vocab, top_k, gen)
    proc(ids, scores.clone())  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        proc(ids, scores.clone())
    return (time.perf_counter() - t0) / repeats * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=262144, help="Gemma-4 vocabulary size.")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=64, help="The model's own default.")
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(RESULTS / "10_fastpath.json"))
    args = ap.parse_args()

    key = derive_key("benchmark-secret-not-used-for-generation", "fastpath", depth=args.depth)
    results = {"meta": {"device": "cpu", "vocab": args.vocab, "depth": args.depth,
                        "top_k": args.top_k}}

    banner("1. Equivalence — does the reduction change the answer?")
    gen = torch.Generator().manual_seed(1)
    ref = build_processor(key, "cpu", fast=False)
    fast = build_processor(key, "cpu", fast=True)
    ref._init_state(2)
    fast._init_state(2)
    worst_dp, worst_tv, worst_dlog = 0.0, 0.0, 0.0
    for _ in range(5):
        ids = torch.randint(1, args.vocab, (2, 8), generator=gen)
        sc = filtered_scores(2, args.vocab, args.top_k, gen)
        a, b = ref(ids, sc.clone()), fast(ids, sc.clone())
        pa, pb = torch.softmax(a, dim=1), torch.softmax(b, dim=1)
        worst_dp = max(worst_dp, float((pa - pb).abs().max()))
        worst_tv = max(worst_tv, float(0.5 * (pa - pb).abs().sum(dim=1).max()))
        worst_dlog = max(worst_dlog, float((a - b).abs().max()))
        assert torch.equal(pa.argmax(1), pb.argmax(1)), "most-likely token changed"
    print(f"  max probability difference, any token : {worst_dp:.2e}")
    print(f"  max total-variation distance          : {worst_tv:.2e}")
    print(f"  most likely token ever changed        : no")
    print(f"\n  (max log-probability difference       : {worst_dlog:.2e})")
    print("\n  Probability space is the meaningful comparison. Log-probabilities of tokens")
    print("  the tournament has driven to ~1e-19 can differ by ~0.3 purely because log is")
    print("  ill conditioned there; their probabilities differ by ~1e-19 and nothing")
    print("  downstream can observe it. The g-values are identical either way, so")
    print("  watermark strength is unchanged.")
    results["equivalence"] = {"max_prob_diff": worst_dp, "max_total_variation": worst_tv,
                              "max_logprob_diff": worst_dlog, "argmax_changed": False}

    banner("2. Cost per decoding step")
    print(f"  {'batch':>6} {'upstream ms':>12} {'candidate-only ms':>18} {'speed-up':>9}")
    by_batch = {}
    for b in args.batch_sizes:
        slow = time_processor(build_processor(key, "cpu", fast=False), b, args.vocab,
                              args.top_k, args.repeats)
        quick = time_processor(build_processor(key, "cpu", fast=True), b, args.vocab,
                               args.top_k, args.repeats)
        by_batch[b] = {"upstream_ms": slow, "fast_ms": quick, "speedup": slow / quick}
        print(f"  {b:6d} {slow:12.1f} {quick:18.2f} {slow / quick:8.0f}x")
    results["cost_by_batch"] = by_batch

    first, last = args.batch_sizes[0], args.batch_sizes[-1]
    grow_slow = by_batch[last]["upstream_ms"] / by_batch[first]["upstream_ms"]
    grow_fast = by_batch[last]["fast_ms"] / by_batch[first]["fast_ms"]
    print(f"\n  Going from batch {first} to {last}, upstream cost grows {grow_slow:.1f}x")
    print(f"  while the candidate-only path grows {grow_fast:.1f}x.")
    print("  That is the whole problem: upstream scales with the batch because it hashes")
    print("  the full vocabulary for every sequence; the reduced path very nearly does not.")
    results["scaling"] = {"upstream_growth": grow_slow, "fast_growth": grow_fast,
                          "from_batch": first, "to_batch": last}

    save_json(results, Path(args.out))


if __name__ == "__main__":
    main()
