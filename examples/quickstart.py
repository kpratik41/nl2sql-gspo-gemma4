"""End-to-end example: mint a key, generate, detect, and check key isolation.

Run:  python examples/quickstart.py
"""

from synthmark import Detector, WatermarkedLM, derive_key

MASTER = "demo-master-secret-replace-me-with-a-real-one"

# 1. Two independent keys, derived from one escrowed secret.
key_a = derive_key(MASTER, "desk-a/v1")
key_b = derive_key(MASTER, "desk-b/v1")
print(f"desk-a fingerprint {key_a.fingerprint}   desk-b fingerprint {key_b.fingerprint}")

# 2. Generate watermarked and unwatermarked text from the same prompt and seed.
lm = WatermarkedLM("google/gemma-4-E4B-it")
prompt = lm.chat_prompts(["Write three paragraphs about the history of lighthouses."])

marked = lm.generate(prompt, key=key_a, max_new_tokens=300, seed=0).texts[0]
plain = lm.generate(prompt, key=None, max_new_tokens=300, seed=0).texts[0]

print("\n--- watermarked ---\n", marked[:400])

# 3. Detect. The same text is scored under the right key, the wrong key, and
#    the unwatermarked baseline is scored under the right key.
det_a = Detector(key_a, lm.tokenizer)
det_b = Detector(key_b, lm.tokenizer)

for label, detector, text in (
    ("watermarked, correct key  ", det_a, marked),
    ("watermarked, WRONG key    ", det_b, marked),
    ("unwatermarked, correct key", det_a, plain),
):
    r = detector.detect(text)
    print(f"{label}  score={r.score:.4f}  tokens={r.num_tokens_scored:4d}  p={r.p_value:.2e}")

print(
    "\nOnly the first line should show a score meaningfully above 0.5.\n"
    "The wrong key sees nothing -- that is what makes per-desk keys meaningful."
)
