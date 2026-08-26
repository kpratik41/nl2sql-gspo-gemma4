"""What does watermarking actually cost at inference time?

The public claim is "negligible speed impact, no extra tokens, no added cost".
The no-extra-tokens part is true by construction.  The speed part deserves a
measurement rather than a citation, because the cost is not model-sized: the
watermark's work is proportional to ``vocab_size x depth`` per decoding step and
is *independent of model size*.  On a large model it disappears into the noise;
on a small model with a large vocabulary it does not.

Three things are measured:

* generation throughput with and without the watermark, across batch sizes;
* the effect of watermarking **depth**, which is the tunable knob trading
  detection power against decoding cost;
* detection throughput on CPU, which decides whether a detection service needs
  a GPU at all.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from pathlib import Path

from common import DEMO_MASTER_SECRET, MODEL_ID, PRIMARY_KEY_ID, RESULTS, banner, save_json

from synthmark import Detector, WatermarkedLM, derive_key
from synthmark.data import OPEN_ENDED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 4, 16, 32])
    ap.add_argument("--depths", nargs="*", type=int, default=[1, 5, 10, 30])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Defaults to results/05_overhead.json; pass a\n                          distinct path when benchmarking a second model so the first\n                          model's numbers are not overwritten.")
    args = ap.parse_args()

    banner(f"Inference overhead | {args.model}")
    lm = WatermarkedLM(args.model, device_map=args.device)
    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)
    results: dict = {"meta": {"model": args.model, "max_new_tokens": args.max_new_tokens}}

    def throughput(prompts, k, batch_size) -> float:
        # One untimed warm-up pass, then take the best of `repeats` runs so that
        # a stray scheduling hiccup does not become the headline number.
        lm.generate(prompts[:batch_size], key=k, max_new_tokens=16, batch_size=batch_size, seed=0)
        best = 0.0
        for _ in range(args.repeats):
            out = lm.generate(
                prompts[:batch_size], key=k, max_new_tokens=args.max_new_tokens,
                batch_size=batch_size, seed=0,
            )
            best = max(best, out.tokens_per_second)
        return best

    # ---------------------------------------------------- 1. watermark on/off
    banner("1. Generation throughput, watermark on vs off")
    prompts = lm.chat_prompts((OPEN_ENDED * 4)[:32])
    print(f"{'batch':>6s} {'plain tok/s':>12s} {'wm tok/s':>10s} {'overhead':>9s}")
    tp = {}
    for bs in args.batch_sizes:
        plain = throughput(prompts, None, bs)
        wm = throughput(prompts, key, bs)
        overhead = (plain - wm) / plain if plain else float("nan")
        tp[bs] = {"plain_tokens_per_s": plain, "watermarked_tokens_per_s": wm, "relative_overhead": overhead}
        print(f"{bs:6d} {plain:12.1f} {wm:10.1f} {overhead:8.1%}")
    results["throughput_by_batch_size"] = tp

    # ------------------------------------------------------- 2. depth vs cost
    banner("2. Cost of watermarking depth (batch size 16)")
    print(f"{'depth':>6s} {'wm tok/s':>10s} {'overhead':>9s}")
    base = tp.get(16, {}).get("plain_tokens_per_s") or throughput(prompts, None, 16)
    depth_results = {}
    for depth in args.depths:
        k = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID, depth=depth)
        t = throughput(prompts, k, 16)
        depth_results[depth] = {"tokens_per_s": t, "relative_overhead": (base - t) / base}
        print(f"{depth:6d} {t:10.1f} {(base - t) / base:8.1%}")
    results["depth_cost"] = {"baseline_tokens_per_s": base, "by_depth": depth_results}
    print("\nDepth is the detection-power / decoding-cost knob: more depth means more")
    print("independent g-values per token, and more work per decoding step.")

    # ------------------------------------------------- 3. detection throughput
    banner("3. Detection throughput")
    sample = lm.generate(prompts[:16], key=key, max_new_tokens=256, batch_size=16, seed=0)
    texts = sample.texts * 4  # 64 texts
    det_results = {}
    for device in ("cpu", args.device):
        if device.startswith("cuda") and not torch.cuda.is_available():
            continue
        det = Detector(key, lm.tokenizer, device=device)
        det.score(texts[:4])  # warm up
        t0 = time.perf_counter()
        _, n = det.score(texts, method="mean")
        dt = time.perf_counter() - t0
        det_results[device] = {
            "texts_per_s": len(texts) / dt,
            "median_tokens_scored": float(np.median(n)),
            "seconds_for_64_texts": dt,
        }
        print(f"{device:8s} {len(texts) / dt:8.1f} texts/s  "
              f"({dt * 1000 / len(texts):.1f} ms per {np.median(n):.0f}-token text)")
    results["detection_throughput"] = det_results
    print("\nDetection is a hash plus a mean: it needs the tokenizer, not the model,")
    print("and runs fast enough on CPU that a detection service needs no GPU.")

    save_json(results, Path(args.out) if args.out else RESULTS / "05_overhead.json")


if __name__ == "__main__":
    main()
