"""Attribute the watermark's inference cost: model forward vs. logits processor.

The end-to-end benchmark (05_overhead.py) shows watermarked generation losing a
large fraction of *batched throughput* while barely affecting *single-request
latency*. Those are different quantities and they have different causes, so this
study separates them.

The SynthID logits processor is timed in isolation on a synthetic logits tensor,
with no model involved. The resulting per-step cost can then be compared against
the per-step cost measured end-to-end. If they match, the watermark's entire
overhead is the processor, and its scaling behaviour tells us whether that
overhead is inherent to the method or an artifact of this implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from common import RESULTS, banner, save_json

from synthmark.config import build_processor
from synthmark.keys import derive_key


def time_processor(batch: int, depth: int, vocab: int, device: str, steps: int = 40) -> float:
    """Milliseconds of processor work per decoding step, model excluded."""
    key = derive_key("benchmark-secret-not-used-for-generation", "micro", depth=depth)
    proc = build_processor(key, device)
    proc._init_state(batch)
    ids = torch.randint(1, vocab, (batch, 64), device=device)
    scores = torch.randn(batch, vocab, device=device)
    for _ in range(5):
        proc(ids, scores.clone())
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        proc(ids, scores.clone())
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / steps * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vocab", type=int, default=262144, help="Gemma-4 vocabulary size.")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 4, 16, 32])
    ap.add_argument("--depths", nargs="*", type=int, default=[1, 5, 10, 30])
    ap.add_argument("--compare", default=str(RESULTS / "05_overhead.json"),
                    help="End-to-end results to attribute against.")
    ap.add_argument("--out", default=str(RESULTS / "08_processor_cost.json"))
    args = ap.parse_args()

    banner("Where the watermark's inference cost actually goes")
    results = {"meta": {"vocab": args.vocab, "depth": args.depth, "device": args.device}}

    print("Processor cost per decoding step, model excluded:\n")
    print(f"{'batch':>6} {'ms/step':>9} {'ms per sequence':>17}")
    by_batch = {}
    for b in args.batch_sizes:
        ms = time_processor(b, args.depth, args.vocab, args.device)
        by_batch[b] = {"ms_per_step": ms, "ms_per_sequence": ms / b}
        print(f"{b:6d} {ms:9.2f} {ms / b:17.2f}")
    results["processor_by_batch"] = by_batch
    print("\nCost per sequence is flat, so the processor does not amortise across a batch.")

    print("\nProcessor cost by depth (batch 16):\n")
    print(f"{'depth':>6} {'ms/step':>9}")
    by_depth = {}
    for d in args.depths:
        ms = time_processor(16, d, args.vocab, args.device)
        by_depth[d] = ms
        print(f"{d:6d} {ms:9.2f}")
    results["processor_by_depth_batch16"] = by_depth

    compare = Path(args.compare)
    if compare.exists():
        print("\nAttribution against end-to-end measurement:\n")
        d = json.loads(compare.read_text())
        print(f"{'batch':>6} {'plain ms':>10} {'wm ms':>9} {'observed Δ':>11} "
              f"{'processor':>10} {'explained':>10}")
        attribution = {}
        for bs, v in d.get("throughput_by_batch_size", {}).items():
            b = int(bs)
            plain_ms = 1000.0 / (v["plain_tokens_per_s"] / b)
            wm_ms = 1000.0 / (v["watermarked_tokens_per_s"] / b)
            delta = wm_ms - plain_ms
            proc = by_batch.get(b, {}).get("ms_per_step") or time_processor(
                b, args.depth, args.vocab, args.device)
            attribution[b] = {"plain_ms_per_step": plain_ms, "watermarked_ms_per_step": wm_ms,
                              "observed_delta_ms": delta, "processor_ms": proc,
                              "fraction_explained": proc / delta if delta else float("nan")}
            print(f"{b:6d} {plain_ms:10.2f} {wm_ms:9.2f} {delta:11.2f} {proc:10.2f} "
                  f"{proc / delta if delta else float('nan'):9.1%}")
        results["attribution"] = attribution

        # What the overhead would be if the processor amortised like the forward pass.
        print("\nHypothetical: if the processor batched as well as the model forward does")
        print("(i.e. cost stayed near its batch-1 value), the overhead would be:\n")
        base = by_batch.get(1, {}).get("ms_per_step", float("nan"))
        print(f"{'batch':>6} {'actual':>9} {'if batched':>12}")
        hypo = {}
        for bs, v in d.get("throughput_by_batch_size", {}).items():
            b = int(bs)
            plain_ms = 1000.0 / (v["plain_tokens_per_s"] / b)
            actual = v["relative_overhead"]
            ideal = base / (plain_ms + base)
            hypo[b] = {"actual": actual, "if_amortised": ideal}
            print(f"{b:6d} {actual:9.1%} {ideal:12.1%}")
        results["hypothetical_amortised"] = hypo

    save_json(results, Path(args.out))


if __name__ == "__main__":
    main()
