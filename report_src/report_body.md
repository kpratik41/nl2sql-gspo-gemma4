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

Fluency, scored by **Qwen3.8-27B** — a model from an unrelated family, with its own
architecture, tokenizer and training data, so it shares none of Gemma's idiosyncrasies:

{{T7}}

Diversity and degeneracy, paired by prompt:

{{T6}}

Task accuracy on sampled chain-of-thought:

{{T8}}

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

### 4.8 Inference cost: latency vs. throughput

These are two different quantities with two different answers, and conflating them is easy.

**Latency — what a single user experiences — is essentially unaffected.** At batch 1 the
watermark adds about 1.7 ms per token: 4.2% of a decode step on E4B, 1.7% on the 31B. Nobody
notices this.

**Batched throughput is a different story**, and the raw numbers look alarming:

{{T10}}

Overhead climbs steeply with batch size on both models, and is consistently ~35% lower on
the 31B than on E4B. That much is a real measurement. But it does *not* mean the method is
expensive, and the next table shows why.

#### Where the cost actually goes

Timing the SynthID logits processor in isolation, with no model involved, accounts for
essentially all of it:

{{T15}}

The overhead is **100% the logits processor**. And its scaling is the whole story:

{{T14}}

**The model forward is nearly flat across batch sizes (37 → 42 ms) because GPUs batch
efficiently. The watermark processor is perfectly linear — a fixed ~1.7 ms per sequence that
never amortises.** That is not a property of SynthID. It is a property of this
implementation: `update_scores` runs a Python `for i in range(depth)` loop, plus `vmap`
calls, serialised per sequence over a 262k-wide tensor.

The depth sweep confirms the loop is the cost:

{{T17}}

Roughly 0.87 ms per depth level at batch 16 — linear in the number of Python iterations.

#### What this implies

If the processor amortised across a batch the way the model forward does, the overhead would
be flat at about 4% everywhere instead of climbing to 57%:

{{T16}}

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

{{T11}}

{{T12}}

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
