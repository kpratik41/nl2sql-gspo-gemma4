"""Generate the paired watermarked / unwatermarked corpus every other study uses.

Design choices that make the downstream comparisons valid:

* **Paired by construction.** Each prompt is generated under both conditions
  with the same seed, so quality comparisons are paired and the confidence
  intervals are much tighter than an unpaired design would give.
* **Split by prompt suite.** Watermark strength is governed by how much freedom
  the model has in choosing tokens, so results are never aggregated across
  suites of different entropy.
* **Long enough to truncate.** Everything is generated at a generous token
  budget; the length/power curve is then produced by truncating this one corpus
  rather than by regenerating at each length.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import DATA, DEMO_MASTER_SECRET, MODEL_ID, PRIMARY_KEY_ID, banner, save_json
from synthmark_eval import WatermarkedLM, derive_key
from synthmark.data import SUITES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--samples-per-prompt", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--suites", nargs="*", default=list(SUITES))
    ap.add_argument("--out", default=str(DATA / "corpus.json"))
    args = ap.parse_args()

    banner(f"Corpus generation | {args.model}")
    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)
    print(f"watermark key: {key.key_id}  depth={key.depth}  ngram_len={key.ngram_len}  fp={key.fingerprint}")

    lm = WatermarkedLM(args.model, device_map="cuda:0")
    print(f"loaded {args.model}")

    records = []
    timing = {}
    t_start = time.time()

    for suite in args.suites:
        prompts_text = SUITES[suite]
        # Repeat each prompt `samples_per_prompt` times; the seed varies by
        # replicate so the samples are independent draws, and the *same* seeds
        # are reused for the unwatermarked arm to keep the pairing.
        rendered = lm.chat_prompts(prompts_text)

        for condition, k in (("watermarked", key), ("unwatermarked", None)):
            for rep in range(args.samples_per_prompt):
                seed = 10_000 + rep
                out = lm.generate(
                    rendered,
                    key=k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    batch_size=args.batch_size,
                    seed=seed,
                )
                for i, (text, ids, n) in enumerate(
                    zip(out.texts, out.token_ids, out.num_new_tokens)
                ):
                    records.append(
                        {
                            "suite": suite,
                            "condition": condition,
                            "prompt_index": i,
                            "replicate": rep,
                            "seed": seed,
                            "prompt": prompts_text[i],
                            "text": text,
                            "token_ids": ids,
                            "num_new_tokens": n,
                        }
                    )
                key_ = f"{suite}/{condition}"
                timing.setdefault(key_, {"tokens": 0, "seconds": 0.0})
                timing[key_]["tokens"] += sum(out.num_new_tokens)
                timing[key_]["seconds"] += out.wall_time_s
            done = sum(1 for r in records if r["suite"] == suite and r["condition"] == condition)
            print(f"  {suite:11s} {condition:14s} {done:4d} texts")

    meta = {
        "model": args.model,
        "key": key.public_summary(),
        "samples_per_prompt": args.samples_per_prompt,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "throughput_tokens_per_s": {
            k: v["tokens"] / v["seconds"] for k, v in timing.items() if v["seconds"] > 0
        },
        "total_texts": len(records),
        "total_wall_time_s": time.time() - t_start,
    }
    save_json({"meta": meta, "records": records}, Path(args.out))
    print(f"\n{len(records)} texts in {meta['total_wall_time_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
