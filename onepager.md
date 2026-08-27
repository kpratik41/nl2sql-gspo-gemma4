# Text Watermarking for LLM Output — Management Summary

Pratik Kakkar · Chandra Dhir · Anup Shirgaonkar

*Evaluation of SynthID-Text watermarking and a reusable internal toolkit. Method is
model-agnostic; tested on Gemma-4 (4B and 31B). Full detail: [report.md](report.md).*

---

## Why now

EU AI Act Article 50 requires generative AI output to be **"marked in a machine-readable
format and detectable as artificially generated."** Article 50(4) applied from **2 August
2026**; the 50(2) marking duty has a grace period to **2 December 2026** for systems already
in market. The Act applies **extraterritorially** to any provider whose output reaches EU
users, with penalties up to **€15m or 3% of worldwide turnover**.

## Bottom line

**The technology works, costs nothing in output quality, and is materially weaker than its
reputation suggests.** It is a provenance control answering *"did our systems produce this?"*
— not an anti-abuse control, and it should not be presented as one.

**Two approaches exist, and the trade-off does not disappear — it moves.** The *green-list*
approach nudges the model toward marked words: easier to detect, but it degrades output
quality. *SynthID*, which we used, never overrides the model's own judgement, so quality is
untouched — paying for that with a weaker, more fragile mark. The choice is which cost you
accept; for client-facing text, degrading quality is not one we can take.

| | Verdict | Evidence |
|---|---|---|
| Detects our own output | ✅ | 100% detection on ~400-word prose, at a 1% false-alarm rate |
| No quality or accuracy cost | ✅ | No measurable difference on either model; every interval spans zero |
| Wrongly flags human writing | ✅ Low | 1.3% against a 1% target |
| Adds user-visible latency | ✅ No | +1.7 ms per token (under 5% of a decode step) |
| Detection needs a GPU | ✅ No | 1,256 documents/second on ordinary CPU |
| Survives paraphrasing | ❌ | One rewrite pass removes it entirely: detection 100% → **0%** |
| Works on code / structured output | ❌ | Near-useless; inherent to the method, not fixable |

Serving throughput needs an optimised implementation before production use. The cause is
understood and the fix is a contained engineering task; detail in the full report.

## The finding that changes how this should be planned

**Detection gets weaker as our models get better.** On the 31B model, detection falls from
100% to **51%** at the same document length. Larger models are more confident, leaving less
randomness for the watermark to use. Both figures are for **400-token documents**, which is
all we tested; detection rises with length and the 31B was still improving sharply at that
point (44% → 64% → 80% across 200/300/400 tokens on creative prose), so longer documents
detect better.

**The consequence: detection thresholds do not transfer between models, and will silently
weaken at every model upgrade.** Re-calibration belongs on the model-promotion checklist.

## What it can and cannot be used for

| Can | Cannot |
|---|---|
| Regulatory marking under Article 50(2) | Prove **who** wrote something — it identifies a key, not a person |
| "Did our own systems generate this?" on unmodified text | Catch anyone who paraphrases to hide it |
| Keep model output out of training corpora | Judge short text, code, or structured output |
| Internal audit — was this memo or filing machine-drafted? | Prove text *is* human-written; a negative proves nothing |
| Separate desks or units via independent keys | Voice, image or video — **this study covers text only** |

Article 50(2) requires marking be effective *"as far as this is technically feasible"*. The
statute anticipates limits, so documenting them is part of a compliance argument rather than
a weakness in one.

## Recommended next steps

1. **Engage model serving** on a vLLM implementation. No vendor ships this, but the extension points exist; it is a contained engineering task, not research.
2. **Adopt key governance now** — one escrowed master secret, per-desk derived keys, detection offered as a service. Whoever holds a key can both detect *and forge* the watermark.
3. **Never deploy detection without a positive control.** Every failure mode here looks identical — a normal "no watermark detected". A known-watermarked test document, checked continuously, makes silent failure loud.

*Deliverables: `synthmark` package (generation, detection, CPU detection service, CLI), full
evaluation across two models, 27 tests. Branch `watermarking`.*
