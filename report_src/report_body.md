---

## 4. Results

### 4.0 Summary of claims

| Claim | Verdict | Evidence |
|---|---|---|
| The watermark is detectable with our key | ✅ | AUC 1.000, 100% detection at a 1% false-positive rate, on ~390-token free-form text |
| It is invisible to a different key | ✅ | Wrong-key AUC 0.45–0.60 (chance); mean score 0.503 either way |
| It does not degrade quality | ✅ | Fluency, diversity and GSM8K accuracy differences all statistically indistinguishable from zero |
| False positives on human writing are controlled | ✅ | 1.3% observed at a nominal 1% threshold, 0% at 0.1% |
| It works on short text | ❌ | 59% detection at 100 tokens, 18% at 25 |
| It works on code and structured output | ❌ | AUC 0.77 on code, 0.55 on JSON — despite ample length |
| It survives paraphrasing | ❌ | AUC collapses 1.000 → 0.607; detection rate 0% |
| It is free at inference time | ❌ | 39% throughput cost at batch 16, 57% at batch 32 |
| Detection needs no GPU | ✅ | 1,083 texts/s on CPU (0.9 ms/text) |

The three ❌ rows are inherent to the method, not defects in this implementation. They
define what the control can and cannot be used for.

### 4.1 Detection at full length

{{T1}}

On free-form prose the watermark is not merely detectable, it is **unmistakable**: every
watermarked text scores above every unwatermarked one, at ~390 tokens.

The `code` row is the important one. It has **352 scored tokens — as many as `creative` —
yet reaches only AUC 0.77**. This isolates the mechanism cleanly: the weakness of
low-entropy text is not that it is short, it is that there is no choice to encode a signal
in. Where the next token is determined, no watermark can exist. `structured` and `factual`
are weak for both reasons at once: low entropy *and* short outputs (35 and 6 scored tokens).

### 4.2 How much text is needed

{{T2}}

This is the table that should govern a deployment. At a 1% false-positive rate the detector
finds fewer than 6 in 10 watermarked documents at 100 tokens, and fewer than 2 in 10 at 25.

**Recommended policy: issue no verdict below ~100 scored tokens, and treat 200+ as the range
where a negative result carries real weight.** The service enforces a hard floor at 40 tokens
and returns `text_too_short` rather than a number.

The weighted-mean detector is consistently stronger than the flat mean at exactly the lengths
where it matters — at 25 tokens on `open_ended`, AUC 0.710 vs 0.627 — and identical once text
is long enough for both to saturate.

### 4.3 Key isolation

{{T3}}

A detector holding the wrong key sees nothing: AUC scatters around 0.5, and watermarked and
unwatermarked text score identically (0.503 vs 0.503). Different desks or business units can
hold different keys, and a detection result is scoped strictly to the key that produced it.

### 4.4 False positives on human writing

{{T4}}

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

{{T5}}

**Use the analytic p-value unless you can calibrate at scale.**

### 4.5 Quality and accuracy

Fluency, scored by an independent model:

{{T7}}

Diversity and degeneracy, paired by prompt:

{{T6}}

Task accuracy on sampled chain-of-thought:

{{T8}}

No measure shows a degradation. Every confidence interval contains zero, and the GSM8K
item-level flips are near-symmetric (15 vs 20), which is what temperature-1.0 sampling noise
produces on its own.

Two honest caveats about strength of evidence:

- The GSM8K interval is **wide** (±8 pp). It rules out a large accuracy loss, not a small
  one. The fluency and diversity measures are far tighter — the `open_ended` perplexity
  interval is ±0.24 on a base of 5.5, roughly ±4% — and carry most of the weight.
- The judge is a smaller sibling of the model under test. It has different weights and a
  different argmax path, so it does not inherit the self-scoring bias, but an unrelated
  architecture would be a stronger control.

**A methodological note.** Multiple-choice benchmarks scored by comparing option
log-probabilities — MMLU, HellaSwag, ARC — are **unaffected by watermarking by
construction**, because no sampling occurs and the logits processor is never invoked.
Reporting "no MMLU delta" would be measuring nothing at all. Only sampled, free-form
generation can be affected, which is why GSM8K with sampled chain-of-thought was used.

### 4.6 Robustness

{{T9}}

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

{{T13}}

The Bayesian detector from the paper was trained on the existing corpus and evaluated on
**disjoint prompts**. It gives no meaningful improvement over the flat mean, and the simple
weighted mean beats both at every length below saturation.

This is a negative result and it is worth stating: on this model, **the extra machinery is
not worth it**. The per-depth weighted mean is training-free, has a closed-form null, and is
the better detector. A larger training corpus might change this, but the burden of proof sits
with the more complex method.

### 4.8 Inference cost

{{T10}}

**This is the finding that most contradicts the received wisdom.** "Negligible speed impact"
does not hold here: at production batch sizes the watermark costs 39–57% of throughput.

The reason is structural rather than a defect. The watermark's work per decoding step is
proportional to `vocab_size × depth` and is **independent of model size**. Gemma-4-E4B has a
262,144-token vocabulary and depth 30, so it performs 30 passes over a 262k-wide tensor per
token. On a frontier-scale model that same absolute cost disappears into a much larger
forward pass — which is why the claim is true for large models and does not transfer to small
ones. The cost also grows with batch size, because the model forward parallelises across the
batch far better than the sequential per-depth tournament does.

Depth is the lever, and it is close to linear:

{{T11}}

Given that detection already saturates at AUC 1.000 by 200–400 tokens with depth 30, **depth
10 (15% overhead) or even 5 (7%) is likely the better operating point** for long-form output,
trading headroom you are not using for throughput you are. Depth should be chosen from the
length distribution of the text you actually need to detect. Note that depth is part of the
key: changing it changes the watermark, so it must be fixed before keys are issued.

{{T12}}

Detection is a hash and a mean. It needs the tokenizer, not the model, and CPU is within 13%
of GPU — so the detection service needs no accelerator at all.

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
