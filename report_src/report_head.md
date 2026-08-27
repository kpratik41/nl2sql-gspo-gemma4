# Watermarking LLM Text Output with SynthID-Text

**An evaluation and a reusable toolkit. The method is model-agnostic; it was tested here on
Gemma-4.**

Pratik Kakkar · Chandra Dhir · Anup Shirgaonkar

Models `google/gemma-4-E4B-it` and `google/gemma-4-31B-it` · fluency judged by
`Qwen/Qwen3.8-27B` · 2 × NVIDIA RTX PRO 6000 Blackwell (97 GB) · SynthID-Text via
Hugging Face Transformers 5.12.1 · package `synthmark` v0.1.0

---

## 1. Why this exists

### The legal driver

The EU AI Act (Regulation 2024/1689) Article 50(2) requires providers of generative AI to
ensure outputs are **"marked in a machine-readable format and detectable as artificially
generated or manipulated."** Article 50(4) separately requires *deployers* to disclose when
AI-generated text is published to inform the public on matters of public interest — with an
exemption where the text underwent human editorial review and a person holds editorial
responsibility.

Three details make this operationally urgent rather than theoretical:

- **The obligation is already live.** Article 50(4) applied from 2 August 2026. The
  machine-readable marking duty under 50(2) carries an AI Omnibus grace period to
  **2 December 2026** for systems already on the market before August.
- **It reaches outside the EU.** The Act applies extraterritorially to providers whose
  outputs reach EU users, regardless of where the firm is headquartered.
- **Penalties are turnover-scaled** — up to €15 million or 3% of worldwide annual turnover.

The Code of Practice on Transparency of AI-Generated Content is a Commission-confirmed route
to demonstrating compliance. Note that Article 50(2) requires marking to be effective *"as far
as this is technically feasible"* — the statute anticipates limits, so measuring them honestly
is part of a compliance argument rather than a weakness in one.

### What watermarking actually helps with

The wider harm picture is real. The FTC's Consumer Sentinel Network logged over 330,000
fraud reports involving AI-generated content or AI-assisted social engineering in 2025, and
the economics have collapsed: attacks that once needed a team and tens of thousands of
dollars now cost tens of dollars and under an hour. The Arup case — deepfaked
video-conference participants authorising $25 million in transfers — is the canonical
example of what synthetic media enables against a large organisation.

**Scope: this report studies text watermarking only.** Nothing here was tested on, or
applies to, audio, image or video output — those need different controls entirely (C2PA
content credentials for files, audio watermarking for speech). Most of the incidents above in
fact involve voice and video, which this work does not address.

Within text, the control is also narrower than it first appears: §6 shows a single paraphrase
pass removes the mark. It is not an anti-fraud control and will not stop a motivated
attacker.

What it does deliver:

| Genuinely helps | Does not help |
|---|---|
| Regulatory provenance — machine-readable marking under Art. 50(2) | Determined adversaries — one paraphrase pass defeats it (§6) |
| Distinguishing model output from human writing in unmodified reuse | Attribution — it identifies a *key*, never a person |
| Keeping model output out of training corpora (model-collapse hygiene) | Short text, code, structured output (§5.1) |
| Internal audit: was this filing, memo or client note machine-drafted? | Proving a human *didn't* write something — a negative is not evidence |
| Per-desk keys so one unit's detector cannot read another's traffic (§5.3) | Content the pipeline did not generate |

The realistic framing is **provenance for honest use, not enforcement against dishonest
use.** It answers "did our own systems produce this?" reliably, and "did someone
adversarially pass off AI text as human?" not at all.

§8 maps these findings to the compliance obligations in detail.

---

## 2. What was done

Two deliverables.

**A package (`synthmark`).** A thin, reusable layer over the SynthID-Text implementation
that ships in Hugging Face Transformers, adding the parts a real deployment needs and the
reference code omits: key derivation and storage, calibrated decision thresholds, a
CPU-only detection service, a CLI, and an evaluation harness. It is model-agnostic; Gemma-4
is the model it was validated on.

**An evaluation.** A measurement of what the watermark actually delivers on this model —
how detectable it is, how much text it needs, whether it degrades output, what destroys it,
and what it costs — with the limitations stated rather than glossed.

One substantive bug was found and fixed along the way; see §5.

---

## 3. How the watermark works

### Two schemes ship with Transformers. This is the one we evaluated, and why

Transformers ships **two unrelated watermarking schemes**, and they behave very differently.
Everything in this report concerns the second approach.

| | Green-list (Kirchenbauer et al.) | SynthID-Text (used here) |
|---|---|---|
| Logits processor | `WatermarkLogitsProcessor` | `SynthIDTextWatermarkLogitsProcessor` |
| Config | `WatermarkingConfig` | `SynthIDTextWatermarkingConfig` |
| Detector | `WatermarkDetector` (z-score on green-token count) | `SynthIDTextWatermarkDetector`, `BayesianDetectorModel` |
| Mechanism | Splits the vocabulary into "green" and "red" each step, adds a constant `bias` to green logits | Tournament among candidates, reweighting by g-value |
| Distortionary? | **Yes** | **No, by design** |
| Source | [arXiv 2306.04634](https://huggingface.co/papers/2306.04634) | [Nature 634 (2024)](https://www.nature.com/articles/s41586-024-08025-4) |

The green-list scheme partitions the vocabulary into green and red tokens at each step and
adds a constant `bias` (default 2.0, over a green fraction of 0.25) to the green logits. **That
does push the model towards tokens it would not otherwise have picked: it is distortionary,
and it trades quality for detectability through that bias knob.** Upstream says
so itself — the parameter's own documentation reads *"Consider lowering the `bias` if the
text generation quality degrades."*

SynthID's tournament is built the other way round. **It reweights *among* the candidates the
model already found plausible while preserving total probability mass, so a token the model
gave near-zero probability stays near zero.** That is precisely why §5.5 finds no quality cost,
and why there is no quality/detectability dial to tune.

**Why we chose SynthID.** The requirement driving this work (§1) is **machine-readable
provenance that does not degrade a production service**. Only a non-distortionary scheme can
satisfy both halves of that: the green-list approach makes quality the currency it pays for
detectability with, so any deployment of it must argue about where to set `bias`, and that
argument has no good answer for a bank shipping client-facing text. SynthID removes the
question. It is also the scheme behind the EU Code of Practice signatories' deployments and
the one with a published, peer-reviewed evaluation at scale.

**Scope of the headline result: "no quality degradation" is specific to SynthID and does not
transfer to the green-list approach.** Anyone benchmarking the
other scheme should expect a real quality/detectability trade-off, and should not cite these
numbers in support of it.

A naming trap worth flagging: the *unprefixed* classes — `WatermarkingConfig`,
`WatermarkDetector` — are the green-list scheme. `WatermarkDetector` will accept
SynthID-generated text without error and report nothing, which looks identical to genuinely
unwatermarked text. `synthmark` imports only the `SynthID*` classes and never reaches the
green-list path.

---

At each decoding step the model produces a distribution over next tokens. Where several
tokens are near-equally good — after *"the weather was cold and…"*, both *"grey"* and
*"overcast"* work — an ordinary sampler picks between them with a random number. **SynthID
replaces that random number with a pseudorandom function** ("g-function") seeded by a secret
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

{{T18}}

This is the whole idea in one table. **No individual token is a tell.** Token 8 comes up
0.367 under the correct key, well *below* chance; token 3 comes up 0.633 under the wrong
one. Reading any single row, you could not say which key produced the text — which is
precisely why a reader sees nothing unusual and why the watermark costs no quality.

**The signal lives only in the average.** Watch the two running means: they are
indistinguishable for the first dozen tokens, and only over hundreds of tokens does the
correct key's average settle above 0.5 while the wrong key's stays at chance. That single
observation explains most of §5 — why detection needs length, why one key cannot read
another's traffic, and why an edit that changes tokens costs signal.

Two properties follow directly from the mechanism, and they bound everything in §5:

- **It can only use randomness that already exists.** Where the model is nearly certain of
  the next token — arithmetic, code, fixed schemas, terse factual answers — there is no
  choice to steer and little or no watermark is embedded.
- **It cannot make the output worse.** It never introduces a token the model would not
  otherwise have sampled. Quality is preserved *at the expense of* watermark strength, not
  the other way around.

## 4. Experimental setup

**Corpus.** 24 prompts × 4 independent samples across six prompt suites, generated twice —
once watermarked, once not — with the *same prompts and the same seeds* under both
conditions, so every comparison is paired. 400 new tokens per generation, sampled at
temperature 1.0, top-k 64, top-p 0.95 (the model's own defaults). 1,088 texts per model.

**Entropy, and why it governs everything.** At each decoding step the model produces a
probability distribution over all 262,144 tokens. *Entropy* measures how spread out that
distribution is:

- **High entropy** — after *"the weather was cold and…"*, perhaps twenty tokens are all
  plausible (`grey`, `overcast`, `damp`, `windy`). The sampler faces a genuine choice and
  must resolve it with a random draw.
- **Low entropy** — after *"2 + 2 ="*, essentially one token carries all the mass. There is
  no choice to make.

SynthID works by replacing the sampler's random draw with a key-derived one, so it can only
encode signal where a draw actually happens. Where the model is certain, the watermark has
nothing to steer and embeds nothing — however long the text runs. This single fact predicts
most of §5: which prompt suites are detectable, why length helps only up to a point, and why
a stronger model is harder to watermark.

**Reading AUC.** Detection results are reported as AUC and as TPR at a fixed false-positive
rate. AUC is the probability that a randomly chosen watermarked text scores higher than a
randomly chosen unwatermarked one: 0.5 is chance, 1.0 means the two score distributions do
not overlap at all.

AUC alone is the wrong number to govern a deployment with, because it is **threshold-free** —
it averages over every operating point, including ones with a false-positive rate nobody
would accept. The `code` suite makes the gap concrete: AUC 0.77 sounds workable, but at a
threshold that wrongly flags only 1 document in 100, it catches **1%** of watermarked code.
TPR at a fixed FPR answers the question that actually arises — *at a false-accusation rate we
can defend, what fraction do we catch?* — and it is the number quoted whenever a decision
depends on it.

**Two models.** Everything is measured on both `gemma-4-E4B-it` (~4B effective) and
`gemma-4-31B-it`. They share a tokenizer and a 262,144-token vocabulary, so comparing them
isolates the effect of model size. Robustness and the learned-detector comparison are
reported on E4B only; those measure how a *given* watermark signal degrades, and the E4B
corpus provides the strongest starting signal to degrade from.

**Fluency judge.** `Qwen/Qwen3.8-27B` — an unrelated architecture with its own tokenizer and
training data. A model cannot score its own output for this purpose (see §5.5), and a judge
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
