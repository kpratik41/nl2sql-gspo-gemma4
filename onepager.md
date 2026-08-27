# Text Watermarking for LLM Output — Management Summary

Pratik Kakkar · Chandra Dhir · Anup Shirgaonkar

*Evaluation of SynthID-Text watermarking and a reusable internal toolkit. Method is
model-agnostic; tested on Gemma-4 (4B and 31B). Full detail: [report.md](report.md).*

---

## Why now

The EU AI Act Article 50 requires generative AI output to be **"marked in a machine-readable
format and detectable as artificially generated."** Article 50(4) has applied since
**2 August 2026**; the marking duty under 50(2) carries a grace period to **2 December 2026**
for systems already in market. The Act applies **extraterritorially** to any provider whose
output reaches EU users, with penalties up to **€15m or 3% of worldwide turnover**.

## Bottom line

**The technology works, costs nothing in quality, and is materially weaker than its
reputation suggests.** It is a provenance control — it answers *"did our systems produce
this?"* It is not an anti-abuse control and should not be presented as one.

| | Verdict | Evidence |
|---|---|---|
| Detects our own output | ✅ | 100% detection on ~390 words of prose, at a 1% false-alarm rate |
| No quality or accuracy cost | ✅ | No measurable difference on either model; every interval spans zero |
| Wrongly flags human writing | ✅ Low | 1.3% against a 1% target |
| Adds user-visible latency | ✅ No | +1.7 ms per token (1.7–4.2%) |
| Serving throughput cost | ⚠️ | 37–57% under batching today — but a fixable engineering issue, not inherent (see below) |
| Survives paraphrasing | ❌ | One rewrite pass removes it entirely: detection 100% → **0%** |
| Works on code / structured output | ❌ | Near-useless (AUC 0.63–0.77); inherent, not fixable |
| Detection needs a GPU | ✅ No | 1,256 documents/second on ordinary CPU |

## Three findings that change how this should be planned

**1. Detection gets weaker as our models get better.** On the 31B model, detection falls from
100% to **51%** at the same document length. Larger models are more confident, leaving less
randomness for the watermark to use. **Detection thresholds do not transfer between models
and will silently weaken at every model upgrade.** Re-calibration must be part of the
model-promotion checklist.

**2. The serving cost is an implementation defect, not a property of the method.** We
attributed 100% of the throughput loss to an unoptimised loop in the open-source library. A
properly batched implementation would cost about **4% instead of 57%**. Do not size capacity
off today's number, and do not reject watermarking on it.

**3. We found and fixed a silent-failure bug in the upstream library.** The same key produced
different watermarks on CPU versus GPU, meaning GPU-generated text was undetectable by a CPU
detector — returning a normal-looking "not watermarked" result rather than an error. Any
other team adopting this library is exposed to it.

## What it cannot do — for the compliance narrative

Not attribution — it identifies a **key**, never a person. Not proof — a negative result is
not evidence of human authorship. Not robust to a motivated actor. Not applicable to voice,
image or video: **this study covers text only.**

Article 50(2) requires marking be effective *"as far as this is technically feasible"*. The
statute anticipates limits, so documenting them is part of a compliance argument rather than
a weakness in one.

## Recommended next steps

1. **Confirm the 2 December 2026 exposure** with Legal — systems in scope, and whether Article 50(4)'s human-editorial-review exemption covers some workflows more cheaply than a technical mark.
2. **Engage model serving** on a vLLM implementation. No vendor ships this, but the extension points exist; it is a contained engineering task, not research.
3. **Adopt key governance now** — one escrowed master secret, per-desk derived keys, detection offered as a service. Whoever holds a key can both detect *and forge* the watermark.
4. **Never deploy detection without a positive control.** Every failure mode here looks identical — a normal "no watermark detected". A known-watermarked test document, checked continuously, makes silent failure loud.

*Deliverables: `synthmark` package (generation, detection, CPU detection service, CLI), full
evaluation across two models, 27 tests. Branch `watermarking`.*
