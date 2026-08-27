---

## 5. Results

### 5.0 Summary of claims

| Claim | Verdict | Evidence |
|---|---|---|
| The watermark is detectable with our key | ⚠️ | **Model-dependent, and length-dependent.** At ~390 scored tokens: E4B 100% detection at 1% FPR; 31B only **51%**. Both measured on 400-token generations, where the 31B curve is still rising steeply — longer documents detect better (§5.1a) |
| It is invisible to a different key | ✅ | Wrong-key AUC 0.45–0.60 (chance); mean score 0.503 either way |
| It does not degrade quality | ✅ | On **both models**: fluency, diversity and GSM8K deltas all statistically indistinguishable from zero, and opposite-signed across the two. Specific to SynthID — see §3, this does *not* transfer to the green-list scheme |
| False positives on human writing are controlled | ✅ | 1.3% observed at a nominal 1% threshold, 0% at 0.1% |
| It works on short text | ❌ | 59% detection at 100 tokens, 18% at 25 |
| It works on code and structured output | ❌ | AUC 0.77 on code, 0.55 on JSON — despite ample length |
| It survives paraphrasing | ❌ | AUC collapses 1.000 → 0.607; detection rate 0% |
| **Latency** is essentially unaffected | ✅ | +1.7 ms per token; 1.7% (31B) to 4.2% (E4B) of a decode step |
| **Batched throughput** is unaffected | ✅ Fixed | The reference implementation loses 39–57% at batch 16–32. Cause identified and fixed in this package — 18–92× faster on CPU; GPU re-measurement pending (§5.7) |
| Detection needs no GPU | ✅ | 1,083 texts/s on CPU (0.9 ms/text) — **but only after the fix in §5.9**; the upstream code makes CPU detection of GPU-generated text impossible |

The three ❌ rows are **inherent to the method**: they follow from the fact that a watermark
can only ride on randomness the model already had. No implementation will fix them, and they
define what the control can and cannot be used for.

The ⚠️ row is the opposite — a **fixable engineering problem** in the reference logits
processor, quantified in §5.8. It should not be read as a cost of watermarking.

### 5.1 Detection at full length

{{T1}}

On free-form prose the watermark is not merely detectable, it is **unmistakable**: every
watermarked text scores above every unwatermarked one, at ~390 tokens.

The `code` row is the important one. It has **352 scored tokens — as many as `creative` —
yet reaches only AUC 0.77**. This isolates the mechanism cleanly: the weakness of
low-entropy text is not that it is short, it is that there is no choice to encode a signal
in. Where the next token is determined, no watermark can exist. `structured` and `factual`
are weak for both reasons at once: low entropy *and* short outputs (35 and 6 scored tokens).

### 5.1a Detection is weaker on the larger model

The obvious assumption is that a watermark behaves the same way on any model from the same
family. It does not:

{{T19}}

At the same ~390 tokens, the 31B is caught **51%** of the time at a 1% false-positive rate
against E4B's **100%**. The per-token signal is roughly halved — a mean g of 0.5110 against
0.5310, over a null of 0.5000.

**Read 51% as a value at one length, not a ceiling.** Every figure here comes from 400-token
generations (~390 scored tokens), and at that point the 31B curve has not flattened: on the
creative suite, detection runs 44% → 64% → **80%** across 200, 300 and 400 tokens. Longer
documents would do better, and we did not test beyond 400 tokens, so where the 31B eventually
plateaus is unmeasured. The comparison with E4B is like-for-like — same lengths, same
prompts, same key — but neither model's asymptote is established here.

The cause follows from §3. The watermark can only ride on randomness the sampler was already
going to spend. A larger, better-trained model is *more confident*: its next-token
distributions are more peaked, fewer candidates are near-tied, and there is less residual
entropy for the tournament to steer. Higher capability means lower output entropy means a
weaker watermark. Note this is not a length effect — both corpora are ~390 scored tokens.

The length curve shifts accordingly, and the gap is widest exactly where operational
thresholds get set:

{{T20}}

**The uncomfortable implication: watermark strength degrades as models improve.** Detection
thresholds calibrated on one model do not transfer to a stronger one, and a policy tuned
today will silently weaken when the serving model is upgraded. Any deployment should
re-measure detection power per model, and treat the minimum-length rule as model-specific
rather than a property of the scheme.

What does *not* change with model size: key isolation still sits at chance (wrong-key AUC
0.37–0.51), and false positives on human text are identical, because both depend on the key
and the human corpus rather than on the generator.

### 5.2 How much text is needed

{{T2}}

This is the table that should govern a deployment. At a 1% false-positive rate the detector
finds fewer than 6 in 10 watermarked documents at 100 tokens, and fewer than 2 in 10 at 25.

**Recommended policy, for E4B: issue no verdict below ~100 scored tokens, and treat 200+ as
the range where a negative result carries real weight.** The service enforces a hard floor at
40 tokens and returns `text_too_short` rather than a number.

**These thresholds are model-specific and must be re-derived per model.** On the 31B the same
100-token point yields only 22% detection rather than 95% (§5.1a); no length in the range
tested reaches E4B's performance.

The weighted-mean detector is consistently stronger than the flat mean at exactly the lengths
where it matters — at 25 tokens on `open_ended`, AUC 0.710 vs 0.627 — and identical once text
is long enough for both to saturate.

### 5.3 Key isolation

{{T3}}

A detector holding the wrong key sees nothing: AUC scatters around 0.5, and watermarked and
unwatermarked text score identically (0.503 vs 0.503). Different desks or business units can
hold different keys, and a detection result is scoped strictly to the key that produced it.

### 5.4 False positives on human writing

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

### 5.5 Quality and accuracy

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

### 5.6 Robustness

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

### 5.7 Inference cost, and a fix

Latency and batched throughput are different quantities with different answers, and
conflating them is easy.

**Latency — what a single user experiences — is essentially unaffected.** At batch 1 the
watermark adds about 1.7 ms per token: 4.2% of a decode step on E4B, 1.7% on the 31B. Nobody
notices this.

**Batched throughput, using the reference implementation, is a different story:**

{{T10}}

Read the plain and watermarked columns as speeds — higher is faster. At batch 32 on E4B,
770 tok/s becomes 329 tok/s, so the same GPU produces well under half as much text. Overhead
climbs steeply with batch on both models and is consistently ~35% lower on the 31B.

**This is a defect in the reference implementation, not a property of the method.** The rest
of this section establishes that, and fixes it.

#### Where the cost goes

Timing the logits processor in isolation, with no model involved, accounts for essentially
all of the gap:

{{T15}}

So the overhead is **100% the logits processor**. Profiling inside it puts roughly 60% in
computing g-values and 40% in the tournament loop — and both scale linearly with batch:

{{T14}}

The reason is visible in the source. The processor evaluates the g-function over
`arange(vocab_size)`, so at every decoding step it materialises a
`(batch, vocab_size, depth)` int64 tensor — **2 GB at batch 32** with a 262k vocabulary and
depth 30 — and makes several passes over it. Upstream's own comments still read
`[batch_size, top_k, depth]`, inherited from the DeepMind reference, which passes only the
surviving candidates.

That work is genuinely proportional to `batch × vocab × depth`: about 250 million hash
computations per generated token at batch 32. **It was never going to amortise across a
batch, because it is real work per (sequence, token, depth).** Organising it better cannot
help. Not doing it can.

#### The fix

The watermark processor runs **after** top-k/top-p, so when it sees the scores roughly
262,080 of the 262,144 entries are already `-inf` with zero probability. Restricting the
g-function to the tokens that can still be sampled cuts the work by three to four orders of
magnitude — 61 thousand hashes instead of 250 million at batch 32.

This is exact rather than approximate. Softmax over the full vocabulary equals softmax over
the finite entries; the tournament's `g_mass` is unchanged because excluded tokens carry zero
probability; and the update maps zero to zero. Measured against the reference:

{{T21}}

{{T22}}

Upstream's cost grows 8.4× going from batch 1 to 8; the fixed path grows 1.7×. That flatness
is the point.

`synthmark` uses this path by default for generation
(`CandidateOnlySynthIDLogitsProcessor`), falling back to the full-vocabulary path when little
was filtered, so unfiltered sampling still behaves correctly. Detection is untouched — it
calls the g-function directly and never went through this code.

**Status and honest caveat.** Equivalence and the speed-up above are measured on CPU; the
39–57% figures in the first table are GPU measurements of the *unfixed* path. The GPU numbers
for the fixed path have not been re-measured, so no claim is made about them here. The
scaling behaviour — whether cost grows with batch — is a property of the algorithm rather
than the device, and it is what the fix changes.

#### Depth remains a lever

Independently of the above, watermarking depth trades detection power against cost:

{{T11}}

{{T17}}

Detection already saturates by 200–400 tokens at depth 30, so depth 10 is likely the better
operating point for long-form output. Depth is part of the key, so it must be fixed before
keys are issued.

**Benchmark your own serving stack before deploying.** These numbers characterise HuggingFace
`generate()`; a different stack will differ.

#### Detection costs almost nothing

{{T12}}

Detection is a hash and a mean. It needs the tokenizer, not the model, and CPU is within 13%
of GPU on E4B and indistinguishable on the 31B — so the detection service needs no
accelerator at all.

### 5.8 A bug worth knowing about

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

### 5.9 Worked example

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

### 5.10 Token round-trip

The watermark lives in *token* choices, but a detector is handed *text* and must re-tokenise
it. Of 400 watermarked texts, only **66.8% re-tokenise to exactly the original ids** — but
**99.5% of individual token positions agree**. The disagreements are isolated boundary
effects, and the cost to detection is negligible. Worth knowing, not worth engineering
around.
