"""Worked end-to-end example reproduced in report.md.

Generates on GPU, then detects on **CPU** -- which only works because synthmark
makes the watermark device-independent.
"""

import json
import sys

from synthmark import Detector, WatermarkedLM, derive_key

MASTER = "report-demo-secret-0123456789abcdef"
PROMPT = "Write three paragraphs explaining to a client why bond prices fall when interest rates rise."

key = derive_key(MASTER, "markets-research/v1")
other = derive_key(MASTER, "other-desk/v1")

lm = WatermarkedLM("google/gemma-4-E4B-it", device_map="cuda:0")
rendered = lm.chat_prompts([PROMPT])
marked = lm.generate(rendered, key=key, max_new_tokens=320, seed=11).texts[0]
plain = lm.generate(rendered, key=None, max_new_tokens=320, seed=11).texts[0]

print("=== WATERMARKED OUTPUT ===")
print(marked)
print("\n=== UNWATERMARKED OUTPUT (same prompt, same seed) ===")
print(plain)

print("\n=== DETECTION (detector runs on CPU; text was generated on GPU) ===")
rows = []
for label, k, text in (
    ("watermarked, correct key", key, marked),
    ("watermarked, wrong key", other, marked),
    ("unwatermarked, correct key", key, plain),
):
    r = Detector(k, lm.tokenizer, device="cpu").detect(text)
    rows.append({"case": label, **r.to_dict()})
    print(f"{label:30s} score={r.score:.4f}  tokens={r.num_tokens_scored:4d}  "
          f"z={r.z_score:+6.2f}  p={r.p_value:.2e}")

json.dump({"prompt": PROMPT, "watermarked": marked, "unwatermarked": plain, "detection": rows},
          open("results/07_demo.json", "w"), indent=2, default=str)
print("\nwrote results/07_demo.json")
