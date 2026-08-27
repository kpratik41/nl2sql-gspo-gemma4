# Watermarking Gemma-4 with SynthID-Text

**An evaluation and a reusable toolkit**

Models `google/gemma-4-E4B-it` and `google/gemma-4-31B-it` · fluency judged by
`Qwen/Qwen3.8-27B` · 2 × NVIDIA RTX PRO 6000 Blackwell (97 GB) · SynthID-Text via
Hugging Face Transformers 5.12.1 · package `synthmark` v0.1.0

---

## 1. What was done

Two deliverables.

**A package (`synthmark`).** A thin, reusable layer over the SynthID-Text implementation
that ships in Hugging Face Transformers, adding the parts a real deployment needs and the
reference code omits: key derivation and storage, calibrated decision thresholds, a
CPU-only detection service, a CLI, and an evaluation harness. It is model-agnostic; Gemma-4
is the model it was validated on.

**An evaluation.** A measurement of what the watermark actually delivers on this model —
how detectable it is, how much text it needs, whether it degrades output, what destroys it,
and what it costs — with the limitations stated rather than glossed.

One substantive bug was found and fixed along the way; see §4.

---

## 2. How the watermark works

At each decoding step the model produces a distribution over next tokens. Where several
tokens are near-equally good — after *"the weather was cold and…"*, both *"grey"* and
*"overcast"* work — an ordinary sampler picks between them with a random number. SynthID
replaces that random number with a pseudorandom function ("g-function") seeded by a secret
key and a sliding window of the preceding tokens, then runs a small tournament among the
candidates that biases the outcome towards tokens with high g-values.

Nothing is inserted into the text: no hidden characters, no markup, no extra tokens. The
signal is *which* of the equally-good tokens was chosen. Detection replays the same key
over the candidate text, computes the g-values that generation would have used, and asks
whether they are biased upward. Under the null hypothesis — text not watermarked with this
key — each g-value is an unbiased coin flip and the mean sits at 0.5.

### What that looks like on real text

Here is a passage the model actually generated, token by token. Each token gets 30
independent coin flips — one per key in the bundle — seeded by the secret key and the four
tokens immediately before it. The same passage is also scored under a *different* key.

| # | Token | Heads (of 30) | g | Running mean g | g, wrong key | Running mean, wrong key |
|---|---|---|---|---|---|---|
| 1 | ` relentless` | 17/30 | 0.567 | **0.567** | 0.567 | 0.567 |
| 2 | ` sculptor` | 13/30 | 0.433 | **0.500** | 0.633 | 0.600 |
| 3 | ` against` | 19/30 | 0.633 | **0.544** | 0.700 | 0.633 |
| 4 | ` the` | 16/30 | 0.533 | **0.542** | 0.567 | 0.617 |
| 5 | ` granite` | 15/30 | 0.500 | **0.533** | 0.467 | 0.587 |
| 6 | ` tower` | 13/30 | 0.433 | **0.517** | 0.400 | 0.556 |
| 7 | ` of` | 14/30 | 0.467 | **0.510** | 0.500 | 0.548 |
| 8 | ` the` | 11/30 | 0.367 | **0.492** | 0.567 | 0.550 |
| 9 | ` Black` | 16/30 | 0.533 | **0.496** | 0.500 | 0.544 |
| 10 | `water` | 17/30 | 0.567 | **0.503** | 0.467 | 0.537 |
| 11 | ` Point` | 14/30 | 0.467 | **0.500** | 0.600 | 0.542 |
| 12 | ` lighthouse` | 15/30 | 0.500 | **0.500** | 0.500 | 0.539 |
| 13 | `.` | 18/30 | 0.600 | **0.508** | 0.467 | 0.533 |
| 14 | ` Silas` | 19/30 | 0.633 | **0.517** | 0.600 | 0.538 |

Over the full 396-token passage: correct key **0.5372**, a different key 0.5096, null expectation 0.5000 (z = +8.11, p = 2.5e-16).

This is the whole idea in one table. **No individual token is a tell.** Token 8 comes up
0.367 under the correct key, well *below* chance; token 3 comes up 0.633 under the wrong
one. Reading any single row, you could not say which key produced the text — which is
precisely why a reader sees nothing unusual and why the watermark costs no quality.

The signal lives only in the average. Watch the two running means: they are
indistinguishable for the first dozen tokens, and only over hundreds of tokens does the
correct key's average settle above 0.5 while the wrong key's stays at chance. That single
observation explains most of §4 — why detection needs length, why one key cannot read
another's traffic, and why an edit that changes tokens costs signal.

Two properties follow directly from the mechanism, and they bound everything in §4:

- **It can only use randomness that already exists.** Where the model is nearly certain of
  the next token — arithmetic, code, fixed schemas, terse factual answers — there is no
  choice to steer and little or no watermark is embedded.
- **It cannot make the output worse.** It never introduces a token the model would not
  otherwise have sampled. Quality is preserved *at the expense of* watermark strength, not
  the other way around.

### Which watermark this is, and why the distinction matters

Transformers ships **two unrelated watermarking schemes**, and they behave very differently.
Everything in this report concerns the second one.

| | Green-list (Kirchenbauer et al.) | SynthID-Text (used here) |
|---|---|---|
| Logits processor | `WatermarkLogitsProcessor` | `SynthIDTextWatermarkLogitsProcessor` |
| Config | `WatermarkingConfig` | `SynthIDTextWatermarkingConfig` |
| Detector | `WatermarkDetector` (z-score on green-token count) | `SynthIDTextWatermarkDetector`, `BayesianDetectorModel` |
| Mechanism | Splits the vocabulary into "green" and "red" each step, adds a constant `bias` to green logits | Tournament among candidates, reweighting by g-value |
| Distortionary? | **Yes** | **No, by design** |
| Source | [arXiv 2306.04634](https://huggingface.co/papers/2306.04634) | [Nature 634 (2024)](https://www.nature.com/articles/s41586-024-08025-4) |

The green-list scheme partitions the vocabulary into green and red tokens at each step and
adds a constant `bias` (default 2.0, over a green fraction of 0.25) to the green logits. That
**does** push the model towards tokens it would not otherwise have picked: it is
distortionary, and it trades quality for detectability through that bias knob. Upstream says
so itself — the parameter's own documentation reads *"Consider lowering the `bias` if the
text generation quality degrades."*

SynthID's tournament is built the other way round. It reweights *among* the candidates the
model already found plausible while preserving total probability mass, so a token the model
gave near-zero probability stays near zero. That is precisely why §4.5 finds no quality cost,
and why there is no quality/detectability dial to tune.

**The consequence for reading this report: the "no quality degradation" result is specific to
SynthID and would not transfer to the green-list implementation.** Anyone benchmarking the
other scheme should expect a real quality/detectability trade-off, and should not cite these
numbers in support of it.

A naming trap worth flagging: the *unprefixed* classes — `WatermarkingConfig`,
`WatermarkDetector` — are the green-list scheme. `WatermarkDetector` will accept
SynthID-generated text without error and report nothing, which looks identical to genuinely
unwatermarked text. `synthmark` imports only the `SynthID*` classes and never reaches the
green-list path.

---

## 3. Experimental setup

**Corpus.** 24 prompts × 4 independent samples across six prompt suites, generated twice —
once watermarked, once not — with the *same prompts and the same seeds* under both
conditions, so every comparison is paired. 400 new tokens per generation, sampled at
temperature 1.0, top-k 64, top-p 0.95 (the model's own defaults). 1,088 texts per model.

**Two models.** Everything quality- and cost-related is measured on both `gemma-4-E4B-it`
(~4B effective) and `gemma-4-31B-it`. They share a tokenizer and a 262,144-token vocabulary,
so comparing them isolates the effect of model size. The detection, robustness and detector
studies are reported on E4B; detection strength is a property of output entropy rather than
of parameter count, and the E4B corpus already saturates at AUC 1.000.

**Fluency judge.** `Qwen/Qwen3.8-27B` — an unrelated architecture with its own tokenizer and
training data. A model cannot score its own output for this purpose (see §4.5), and a judge
from the same family would share the generator's idiosyncrasies.

The suites are split by **entropy**, because entropy is the variable that governs watermark
strength. Reporting a single aggregate number over a mixed prompt set would hide the effect
that matters most in practice.

| Suite | Content | Expected freedom in token choice |
|---|---|---|
| `creative` | Stories, descriptive writing | Highest |
| `open_ended` | Explanatory prose | High |
| `financial` | Client notes, product explanations | High |
| `factual` | Short answers pinned by facts | Low |
| `structured` | JSON with a fixed schema | Very low |
| `code` | Python, SQL, bash | Lowest |

**Key.** Watermarking depth 30, n-gram length 5 — the reference configuration from the
SynthID-Text paper. Keys are derived by HKDF-SHA256 from a single master secret, so a label
like `markets-research/v1` deterministically yields an independent key.

**Negative controls.** Three distinct kinds, because they answer different questions:

1. *Unwatermarked output from the same model* — does the detector key on the watermark, or
   merely on the model's style?
2. *Human-written text* (600 WikiText-103 passages, detokenised) — how often would we
   wrongly flag a person's writing?
3. *The same watermarked text scored with a different key* — can one desk's detector read
   another desk's traffic?

---

## 4. Results

### 4.0 Summary of claims

| Claim | Verdict | Evidence |
|---|---|---|
| The watermark is detectable with our key | ✅ | AUC 1.000, 100% detection at a 1% false-positive rate, on ~390-token free-form text |
| It is invisible to a different key | ✅ | Wrong-key AUC 0.45–0.60 (chance); mean score 0.503 either way |
| It does not degrade quality | ✅ | On **both models**: fluency, diversity and GSM8K deltas all statistically indistinguishable from zero, and opposite-signed across the two. Specific to SynthID — see §2, this does *not* transfer to the green-list scheme |
| False positives on human writing are controlled | ✅ | 1.3% observed at a nominal 1% threshold, 0% at 0.1% |
| It works on short text | ❌ | 59% detection at 100 tokens, 18% at 25 |
| It works on code and structured output | ❌ | AUC 0.77 on code, 0.55 on JSON — despite ample length |
| It survives paraphrasing | ❌ | AUC collapses 1.000 → 0.607; detection rate 0% |
| **Latency** is essentially unaffected | ✅ | +1.7 ms per token; 1.7% (31B) to 4.2% (E4B) of a decode step |
| **Batched throughput** is unaffected | ⚠️ | 39–57% loss at batch 16–32 — but this is an artifact of the reference logits processor, not of the method (§4.8) |
| Detection needs no GPU | ✅ | 1,083 texts/s on CPU (0.9 ms/text) — **but only after the fix in §4.9**; the upstream code makes CPU detection of GPU-generated text impossible |

The three ❌ rows are **inherent to the method**: they follow from the fact that a watermark
can only ride on randomness the model already had. No implementation will fix them, and they
define what the control can and cannot be used for.

The ⚠️ row is the opposite — a **fixable engineering problem** in the reference logits
processor, quantified in §4.8. It should not be read as a cost of watermarking.

### 4.1 Detection at full length

| Suite | Median tokens scored | AUC | 95% CI | TPR @1% FPR | Mean score (WM) | Mean score (plain) |
|---|---|---|---|---|---|---|
| creative | 394 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5446 | 0.4999 |
| open_ended | 390 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5256 | 0.5001 |
| financial | 380 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5229 | 0.4988 |
| factual | 6 | 0.5772 | [0.483, 0.677] | 0.028 | 0.5249 | 0.5146 |
| structured | 35 | 0.5472 | [0.441, 0.645] | 0.000 | 0.5020 | 0.4992 |
| code | 352 | 0.7692 | [0.696, 0.834] | 0.010 | 0.5047 | 0.4981 |
| **pooled (high-entropy)** | 390 | 1.0000 | [1.000, 1.000] | 1.000 | 0.5310 | 0.4996 |

On free-form prose the watermark is not merely detectable, it is **unmistakable**: every
watermarked text scores above every unwatermarked one, at ~390 tokens.

The `code` row is the important one. It has **352 scored tokens — as many as `creative` —
yet reaches only AUC 0.77**. This isolates the mechanism cleanly: the weakness of
low-entropy text is not that it is short, it is that there is no choice to encode a signal
in. Where the next token is determined, no watermark can exist. `structured` and `factual`
are weak for both reasons at once: low entropy *and* short outputs (35 and 6 scored tokens).

### 4.2 How much text is needed

| Target tokens | Median scored | AUC | TPR @1% FPR |
|---|---|---|---|
| 25 | 21 | 0.7309 | 0.181 |
| 50 | 46 | 0.8453 | 0.326 |
| 100 | 96 | 0.9582 | 0.594 |
| 200 | 195 | 0.9961 | 0.847 |
| 300 | 293 | 0.9998 | 0.993 |
| 400 | 388 | 1.0000 | 1.000 |

*(averaged over the creative, open_ended and financial suites)*

This is the table that should govern a deployment. At a 1% false-positive rate the detector
finds fewer than 6 in 10 watermarked documents at 100 tokens, and fewer than 2 in 10 at 25.

**Recommended policy: issue no verdict below ~100 scored tokens, and treat 200+ as the range
where a negative result carries real weight.** The service enforces a hard floor at 40 tokens
and returns `text_too_short` rather than a number.

The weighted-mean detector is consistently stronger than the flat mean at exactly the lengths
where it matters — at 25 tokens on `open_ended`, AUC 0.710 vs 0.627 — and identical once text
is long enough for both to saturate.

### 4.3 Key isolation

| Suite | AUC (wrong key) | Mean score (WM text) | Mean score (plain) |
|---|---|---|---|
| creative | 0.5969 | 0.5043 | 0.5027 |
| open_ended | 0.4549 | 0.5036 | 0.5040 |
| financial | 0.5190 | 0.5041 | 0.5038 |
| factual | 0.5155 | 0.5098 | 0.5101 |
| structured | 0.5260 | 0.5056 | 0.5060 |
| code | 0.4903 | 0.5031 | 0.5032 |

A detector holding the wrong key sees nothing: AUC scatters around 0.5, and watermarked and
unwatermarked text score identically (0.503 vs 0.503). Different desks or business units can
hold different keys, and a detection result is scoped strictly to the key that produced it.

### 4.4 False positives on human writing

300 WikiText-103 passages, median 132 scored tokens, mean score 0.4985 (null expectation 0.5).

| Target FPR | Observed (empirical threshold) | Observed (analytic p-value) |
|---|---|---|
| 10.0% | 0.1067 | 0.0833 |
| 1.0% | 0.0333 | 0.0133 |
| 0.1% | 0.0300 | 0.0000 |

Human prose scores 0.4985 on average against a null expectation of 0.5000 — the detector is
unbiased on writing it has never seen.

**An unexpected result: the analytic threshold beats the empirical one.** Calibrating on 300
human texts gives 3.3% false positives at a nominal 1% target, while the closed-form p-value
gives 1.3%. The reason is sample size, not principle: 300 texts split across length buckets
leaves ~150 per bucket, and a 1-in-1000 quantile cannot be estimated from 150 samples.
Empirical calibration is the right approach *given enough data* — on the order of 10⁴ texts
per length bucket — and actively harmful below that.

The analytic null was checked directly against 516 unwatermarked model outputs and is
well-calibrated and slightly conservative:

n = 516

| Nominal p threshold | Observed rate |
|---|---|
| 0.10 | 0.0814 |
| 0.01 | 0.0019 |
| 0.001 | 0.0000 |

**Token round-trip:** 66.8% of texts re-tokenise exactly; 99.5% of token positions agree.

**Use the analytic p-value unless you can calibrate at scale.**

### 4.5 Quality and accuracy

Fluency, scored by **Qwen3.8-27B** — a model from an unrelated family, with its own
architecture, tokenizer and training data, so it shares none of Gemma's idiosyncrasies:

| Model | Suite | n pairs | PPL watermarked | PPL plain | Difference [95% CI] | Significant? |
|---|---|---|---|---|---|---|
| Gemma-4-E4B | creative | 96 | 7.261 | 7.389 | -0.128 [-0.394, +0.134] | no |
| Gemma-4-E4B | open_ended | 96 | 3.424 | 3.369 | +0.055 [-0.015, +0.129] | no |
| Gemma-4-E4B | financial | 96 | 3.302 | 3.303 | -0.001 [-0.102, +0.099] | no |
| Gemma-4-31B | creative | 96 | 5.678 | 5.611 | +0.066 [-0.114, +0.249] | no |
| Gemma-4-31B | open_ended | 96 | 2.808 | 2.844 | -0.036 [-0.089, +0.018] | no |
| Gemma-4-31B | financial | 96 | 2.825 | 2.764 | +0.061 [-0.026, +0.154] | no |

Diversity and degeneracy, paired by prompt:

| Model | Suite | n pairs | distinct-2 WM | distinct-2 plain | Difference [95% CI] |
|---|---|---|---|---|---|
| Gemma-4-E4B | creative | 96 | 0.9628 | 0.9640 | -0.0012 [-0.0046, +0.0022] |
| Gemma-4-E4B | open_ended | 96 | 0.9596 | 0.9602 | -0.0006 [-0.0041, +0.0028] |
| Gemma-4-E4B | financial | 96 | 0.9350 | 0.9408 | -0.0057 [-0.0119, +0.0003] |
| Gemma-4-E4B | factual | 76 | 0.9774 | 0.9801 | -0.0027 [-0.0075, +0.0030] |
| Gemma-4-E4B | structured | 64 | 0.9621 | 0.9622 | -0.0000 [-0.0040, +0.0045] |
| Gemma-4-E4B | code | 96 | 0.9005 | 0.9007 | -0.0002 [-0.0090, +0.0084] |
| Gemma-4-31B | creative | 96 | 0.9539 | 0.9533 | +0.0006 [-0.0024, +0.0036] |
| Gemma-4-31B | open_ended | 96 | 0.9513 | 0.9528 | -0.0015 [-0.0046, +0.0016] |
| Gemma-4-31B | financial | 96 | 0.9437 | 0.9437 | -0.0000 [-0.0049, +0.0046] |
| Gemma-4-31B | factual | 85 | 1.0000 | 0.9989 | +0.0011 [+0.0000, +0.0029] |
| Gemma-4-31B | structured | 64 | 0.9666 | 0.9672 | -0.0006 [-0.0022, +0.0006] |
| Gemma-4-31B | code | 96 | 0.9249 | 0.9272 | -0.0023 [-0.0083, +0.0032] |

Task accuracy on sampled chain-of-thought:

| Model | Watermarked | Unwatermarked | Difference [95% CI] | Flips WM-only / plain-only |
|---|---|---|---|---|
| Gemma-4-E4B | 156/250 (0.624) | 161/250 (0.644) | **-0.0200** [-0.1037, +0.0641] | 15 / 20 |
| Gemma-4-31B | 233/250 (0.932) | 230/250 (0.920) | **+0.0120** [-0.0351, +0.0595] | 8 / 5 |

No measure shows a degradation, on either model. Every confidence interval contains zero.

The GSM8K result is the most informative, because the two models disagree in *direction*:
the watermarked arm is 2.0 pp **worse** on E4B and 1.2 pp **better** on the 31B. A real
systematic cost would push the same way on both. Two opposite-signed, non-significant
deltas, with near-symmetric item-level flips (15/20 and 8/5), are what sampling noise at
temperature 1.0 looks like.

The 31B interval is also tighter (±4.7 pp vs ±8.4 pp) despite the same 250 problems, because
its accuracy sits near ceiling (93% vs 62%) where binomial variance is smaller.

Two honest caveats about strength of evidence:

- The GSM8K intervals are **wide** (±8 pp on E4B, ±4.7 pp on the 31B). They rule out a large
  accuracy loss, not a small one. The fluency and diversity measures are far tighter and
  carry most of the weight.
- Perplexity is a proxy for fluency, not a measure of usefulness. It would not detect a
  failure mode that leaves text fluent but less helpful, and no automatic metric would.
  Human side-by-side rating remains the stronger design.

**A methodological note.** Multiple-choice benchmarks scored by comparing option
log-probabilities — MMLU, HellaSwag, ARC — are **unaffected by watermarking by
construction**, because no sampling occurs and the logits processor is never invoked.
Reporting "no MMLU delta" would be measuring nothing at all. Only sampled, free-form
generation can be affected, which is why GSM8K with sampled chain-of-thought was used.

### 4.6 Robustness

| Attack | Level | Median tokens | AUC | TPR @1% FPR | Mean score (WM) |
|---|---|---|---|---|---|
| none | - | 392 | 1.0000 | 1.000 | 0.5351 |
| lowercase | - | 389 | 0.9941 | 0.891 | 0.5233 |
| strip_formatting | - | 370 | 0.9999 | 0.995 | 0.5307 |
| truncate | 0.75 | 285 | 0.9995 | 0.969 | 0.5316 |
| truncate | 0.5 | 187 | 0.9932 | 0.896 | 0.5298 |
| truncate | 0.25 | 90 | 0.9433 | 0.589 | 0.5270 |
| truncate | 0.125 | 42 | 0.8625 | 0.276 | 0.5248 |
| delete_words | 0.05 | 364 | 0.9974 | 0.979 | 0.5272 |
| delete_words | 0.1 | 345 | 0.9910 | 0.911 | 0.5228 |
| delete_words | 0.2 | 307 | 0.9710 | 0.682 | 0.5153 |
| delete_words | 0.4 | 230 | 0.7575 | 0.036 | 0.5065 |
| swap_words | 0.05 | 384 | 0.9985 | 0.974 | 0.5247 |
| swap_words | 0.1 | 384 | 0.9921 | 0.859 | 0.5184 |
| swap_words | 0.2 | 384 | 0.9054 | 0.271 | 0.5099 |
| swap_words | 0.4 | 385 | 0.7196 | 0.021 | 0.5024 |
| substitute_words | 0.05 | 381 | 0.9989 | 0.953 | 0.5263 |
| substitute_words | 0.1 | 378 | 0.9929 | 0.911 | 0.5212 |
| substitute_words | 0.2 | 371 | 0.9585 | 0.703 | 0.5130 |
| substitute_words | 0.4 | 360 | 0.7491 | 0.188 | 0.5036 |
| mix_with_human | 0.75 | 378 | 0.9983 | 0.969 | 0.5239 |
| mix_with_human | 0.5 | 324 | 0.9783 | 0.688 | 0.5174 |
| mix_with_human | 0.25 | 226 | 0.8715 | 0.328 | 0.5108 |
| mix_with_human | 0.1 | 170 | 0.7034 | 0.042 | 0.5049 |
| paraphrase | 1 | 394 | 0.6074 | 0.000 | 0.5010 |
| paraphrase | 2 | 391 | 0.6451 | 0.016 | 0.5012 |
| round_trip_translate | fr | 288 | 0.9788 | 0.547 | 0.5163 |

*(unattacked baseline AUC = 1.0000)*

Every attack was applied to **both** arms and the AUC recomputed on the attacked pair, so
these numbers isolate damage to the watermark from changes to the score distribution.

Reading the table:

- **Free edits are free.** Lowercasing and stripping markdown cost almost nothing (AUC 0.994
  and 0.9999), even though they re-tokenise much of the text.
- **Casual editing survives.** Deleting, swapping or substituting up to 10% of words leaves
  AUC above 0.99. It degrades at 20% and collapses at 40% — but by then the text is visibly
  damaged, so a real editor would not go there.
- **Dilution behaves sensibly.** A document that is 25% model-written scores AUC 0.87 and is
  flagged 33% of the time. A weak signal on a mostly-human document is the *correct* answer,
  not a failure.
- **Paraphrase defeats it completely.** One pass through another model takes AUC from 1.000
  to 0.607 and detection at a 1% false-positive rate from 100% to **0%**. A second pass adds
  nothing because the first already reached the floor.
- **But round-trip translation does not.** EN→FR→EN retains AUC 0.979 and 55% detection.

That last contrast is the most useful finding here. "Passed through another model" is not
what destroys the watermark — *deliberate rewording* is. Back-translation reconstructs
similar phrasing and therefore similar token sequences; a paraphrase prompt explicitly asks
for different ones. The practical implication: the watermark survives ordinary round-tripping
through tools, and fails against anyone actively trying to remove it.

### 4.7 The learned detector

| Tokens | AUC mean | AUC weighted | AUC Bayesian | TPR@1% mean | TPR@1% Bayesian |
|---|---|---|---|---|---|
| 25 | 0.6798 | 0.7067 | 0.6833 | 0.250 | 0.133 |
| 50 | 0.8020 | 0.8563 | 0.8182 | 0.292 | 0.300 |
| 100 | 0.9454 | 0.9656 | 0.9475 | 0.525 | 0.558 |
| 200 | 0.9943 | 0.9992 | 0.9942 | 0.883 | 0.833 |
| 400 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 1.000 |

The Bayesian detector from the paper was trained on the existing corpus and evaluated on
**disjoint prompts**. It gives no meaningful improvement over the flat mean, and the simple
weighted mean beats both at every length below saturation.

This is a negative result and it is worth stating: on this model, **the extra machinery is
not worth it**. The per-depth weighted mean is training-free, has a closed-form null, and is
the better detector. A larger training corpus might change this, but the burden of proof sits
with the more complex method.

### 4.8 Inference cost: latency vs. throughput

These are two different quantities with two different answers, and conflating them is easy.

**Latency — what a single user experiences — is essentially unaffected.** At batch 1 the
watermark adds about 1.7 ms per token: 4.2% of a decode step on E4B, 1.7% on the 31B. Nobody
notices this.

**Batched throughput is a different story**, and the raw numbers look alarming:

| Batch size | Gemma-4-E4B plain | Gemma-4-E4B watermarked | Gemma-4-E4B overhead | Gemma-4-31B plain | Gemma-4-31B watermarked | Gemma-4-31B overhead |
|---|---|---|---|---|---|---|
| 1 | 26.9 | 25.8 | **4.2%** | 16.0 | 15.7 | **1.7%** |
| 4 | 99.9 | 87.2 | **12.7%** | 61.2 | 56.5 | **7.7%** |
| 16 | 398.0 | 241.4 | **39.4%** | 210.0 | 156.3 | **25.6%** |
| 32 | 770.4 | 329.0 | **57.3%** | 341.5 | 213.7 | **37.4%** |

Overhead climbs steeply with batch size on both models, and is consistently ~35% lower on
the 31B than on E4B. That much is a real measurement. But it does *not* mean the method is
expensive, and the next table shows why.

#### Where the cost actually goes

Timing the SynthID logits processor in isolation, with no model involved, accounts for
essentially all of it:

| Batch | Model forward (ms) | Watermarked step (ms) | Observed Δ | Processor alone | Explained |
|---|---|---|---|---|---|
| 1 | 37.18 | 38.82 | 1.63 | 1.76 | **108%** |
| 4 | 40.05 | 45.86 | 5.81 | 5.43 | **94%** |
| 16 | 40.20 | 66.29 | 26.09 | 26.14 | **100%** |
| 32 | 41.54 | 97.26 | 55.72 | 55.88 | **100%** |

The overhead is **100% the logits processor**. And its scaling is the whole story:

| Batch | Processor ms/step | Processor ms **per sequence** |
|---|---|---|
| 1 | 1.76 | 1.76 |
| 4 | 5.43 | 1.36 |
| 16 | 26.14 | 1.63 |
| 32 | 55.88 | 1.75 |

**The model forward is nearly flat across batch sizes (37 → 42 ms) because GPUs batch
efficiently. The watermark processor is perfectly linear — a fixed ~1.7 ms per sequence that
never amortises.** That is not a property of SynthID. It is a property of this
implementation: `update_scores` runs a Python `for i in range(depth)` loop, plus `vmap`
calls, serialised per sequence over a 262k-wide tensor.

The depth sweep confirms the loop is the cost:

| Depth | Processor ms/step |
|---|---|
| 1 | 0.54 |
| 5 | 3.11 |
| 10 | 7.18 |
| 30 | 26.16 |

Roughly 0.87 ms per depth level at batch 16 — linear in the number of Python iterations.

#### What this implies

If the processor amortised across a batch the way the model forward does, the overhead would
be flat at about 4% everywhere instead of climbing to 57%:

| Batch | Measured overhead | If it batched like the model |
|---|---|---|
| 1 | 4.2% | **4.5%** |
| 4 | 12.7% | **4.2%** |
| 16 | 39.4% | **4.2%** |
| 32 | 57.3% | **4.1%** |

So the honest conclusion is:

- **Latency is genuinely minimal**, and the widely-repeated claim to that effect is correct.
- **The batched-throughput cost measured here is real but avoidable.** It is the price of an
  unoptimised reference implementation, not of watermarking. A fused kernel that computes all
  depths in one pass, or a serving stack (vLLM, TGI) with an optimised processor, should
  recover most of the ~4%-vs-57% gap.
- **Until you have that, depth is the lever.** Depth 10 costs 15.2% instead of 39.4% on E4B
  (8.6% vs 25.6% on the 31B), and detection already saturates at AUC 1.000 by 200 tokens with
  depth 30. Depth is part of the key, so it must be fixed before keys are issued.

**Before deploying, benchmark your own serving stack.** These numbers characterise
HuggingFace `generate()`; they are not a property of SynthID, and they should not be used to
argue either for or against watermarking on a different stack.

| Depth | Gemma-4-E4B tok/s | Gemma-4-E4B overhead | Gemma-4-31B tok/s | Gemma-4-31B overhead |
|---|---|---|---|---|
| 1 | 392.7 | **1.3%** | 208.8 | **0.6%** |
| 5 | 369.1 | **7.3%** | 201.7 | **4.0%** |
| 10 | 337.7 | **15.2%** | 192.0 | **8.6%** |
| 30 | 241.1 | **39.4%** | 156.3 | **25.6%** |

| Model | Device | Texts/s | ms per text |
|---|---|---|---|
| Gemma-4-E4B | cpu | 1082.6 | 0.9 |
| Gemma-4-E4B | cuda:0 | 1241.9 | 0.8 |
| Gemma-4-31B | cpu | 1255.8 | 0.8 |
| Gemma-4-31B | cuda:0 | 1257.2 | 0.8 |

Detection is a hash and a mean. It needs the tokenizer, not the model, and CPU is within 13%
of GPU on E4B and indistinguishable on the 31B — so the detection service needs no
accelerator at all.

### 4.9 A bug worth knowing about

Upstream Transformers builds the watermark's g-value sampling table with a **device-local
RNG**:

```python
generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
self.sampling_table = torch.randint(0, 2, (size,), generator=generator, device=device)
```

`torch.randint` draws different values from CUDA and CPU generators given the same seed. The
consequence is that **the same key produces a different watermark depending on which device
the processor was built on.** Text generated on a GPU is invisible to a detector running on
CPU — and the failure is silent: the detector returns an ordinary-looking null score, not an
error.

This was found the first time generation was run on GPU and detection on CPU: the
watermarked text scored 0.4974 (p = 0.69), indistinguishable from unwatermarked.

Since the sampling table is the *only* source of randomness in the algorithm — the rest is a
deterministic linear-congruential hash over int64 — the fix is to draw the table on CPU and
move it to the target device. `synthmark` does this for both generation and detection, so
they cannot disagree. Two regression tests cover it: one asserting g-values, masks and scores
match across devices, and one documenting the upstream behaviour so we learn when it is fixed
upstream.

Anyone deploying SynthID via Transformers should check this. It would not show up in a
single-machine test, and it silently breaks CPU detection services and mixed-hardware fleets.

### 4.10 Worked example

Generated on GPU with key `markets-research/v1`, detected on **CPU** — which only works
because of the fix above.

> When interest rates in the broader economy rise, bond prices typically move in the opposite
> direction, which can seem counterintuitive at first. The core reason lies in the
> relationship between the fixed coupon payments a bond provides and the new, higher
> prevailing interest rates. […]

| Case | Score | Tokens | z | p |
|---|---|---|---|---|
| watermarked, correct key | 0.5218 | 310 | +4.21 | 1.3 × 10⁻⁵ |
| watermarked, **wrong** key | 0.4906 | 310 | −1.80 | 0.96 |
| unwatermarked, correct key | 0.5031 | 316 | +0.60 | 0.28 |

The watermarked and unwatermarked outputs read identically well; the difference lives
entirely in the statistics of which near-equivalent words were chosen.

### 4.11 Token round-trip

The watermark lives in *token* choices, but a detector is handed *text* and must re-tokenise
it. Of 400 watermarked texts, only **66.8% re-tokenise to exactly the original ids** — but
**99.5% of individual token positions agree**. The disagreements are isolated boundary
effects, and the cost to detection is negligible. Worth knowing, not worth engineering
around.

---

## 5. The package

`synthmark` is the reusable half of this work. It wraps the Transformers SynthID
implementation and adds what a deployment needs.

| Module | What it provides |
|---|---|
| `keys.py` | HKDF-SHA256 key derivation from one master secret, fingerprints, 0600 storage |
| `config.py` | Bridge to the HF API, plus the device-portability fix |
| `generate.py` | Watermarked / unwatermarked generation, perplexity scoring |
| `detect.py` | g-values, masking, three scoring methods, empirical calibration |
| `bayesian.py` | Training and use of the learned detector |
| `attacks.py` | Truncation, editing, paraphrase, translation, dilution |
| `metrics.py` | AUC, TPR@FPR, bootstrap and Newcombe confidence intervals |
| `serve.py` | FastAPI detection service, multi-key |
| `cli.py` | `keygen` · `generate` · `detect` · `calibrate` · `serve` |

Turning the watermark on is a one-argument change at the call site:

```python
out = lm.generate(prompts, key=key)      # watermarked
out = lm.generate(prompts, key=None)     # ordinary HF path
```

Three design decisions worth calling out, because they are the ones that keep the
system honest rather than merely working:

**Greedy decoding is refused, not silently unmarked.** With no sampling there is no
randomness to encode a signal in. Passing `do_sample=False` together with a key raises
rather than returning confidently unmarked text.

**The service refuses a verdict below ~40 scored tokens.** Returning a
confident-looking number on two sentences is how false accusations happen.

**Every negative result is caveated in the response body.** "No watermark detected" is
not "a human wrote this", and the API says so in the payload rather than in
documentation someone will not read.

### Key management

Keys are derived, not distributed:

```python
key = derive_key(master_secret, "markets-research/v1")
```

One escrowed master secret yields unlimited independent keys. Rotation is a new label.
Any authorised host can reconstruct any key, so key material never travels.

The critical point for a security review: **the key is signing-grade material**. Whoever
holds it can detect the watermark *and forge it*. The obvious way to let a downstream team
check text — hand them the key — also hands them the ability to fabricate text your own
detector will confirm as yours. Run detection as a service instead. See
[`docs/key-management.md`](docs/key-management.md).

---

## 6. Limitations

Stated plainly, because a detector that is oversold is worse than no detector.

- **It is not proof of authorship.** It answers "is this text statistically consistent
  with this key", nothing more. It cannot distinguish text the model wrote from text the
  model edited or a human lightly revised.
- **A negative is not evidence of human authorship.** Short, edited, paraphrased, or
  low-entropy text all produce negatives, as does any text from a different model.
- **Paraphrase defeats it.** Anyone who wants to remove the watermark can, cheaply.
  It is a provenance signal for ordinary use, not an anti-abuse control against a
  motivated adversary.
- **It requires the matching tokenizer.** Detection is defined over token sequences.
- **It is tied to the sampling configuration.** Greedy or very low-temperature decoding
  embeds little or nothing.
- **Findings here are for one model.** Detection strength depends on the model's output
  entropy; a different model needs its own measurement.

---

## 7. Relevance to the EU AI Act Code of Practice

The Code of Practice commitment is to *mark* AI-generated content in a machine-readable
way. What this evaluation supports, and what it does not:

**Supported.** Content generated through a watermarking pipeline is marked, the marking is
machine-detectable by the key holder, the mark carries no user-identifying information
(the key is per-deployment, not per-user or per-session), and the marking does not degrade
output quality.

**Not supported by watermarking alone.** Detection of edited or paraphrased content;
attribution of authorship; any claim about content the pipeline did not generate. For
files and images the appropriate mechanism is C2PA content credentials, which is a
different control and complementary to this one.

The operationally important number for governance is not AUC but the **false-positive rate
at the deployed threshold**, measured on human writing, together with the **minimum text
length** below which no verdict is issued. Both are in §4.2 and §4.4.

---

## 8. Reproducing

```bash
pip install -e '.[serve,dev]'
pytest                                   # unit tests, no model download

cd experiments
./run_all.sh                             # full evaluation
```

Individual studies:

```bash
python 01_generate_corpus.py --samples-per-prompt 4
python 02_detectability.py
python 03_quality.py --gsm8k-n 250
python 04_robustness.py
python 05_overhead.py
python 06_bayesian.py
```

Raw results are written to `results/*.json`; the logs alongside them contain the formatted
tables reproduced in this report.
