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

Two properties follow directly from the mechanism, and they bound everything in §4:

- **It can only use randomness that already exists.** Where the model is nearly certain of
  the next token — arithmetic, code, fixed schemas, terse factual answers — there is no
  choice to steer and little or no watermark is embedded.
- **It cannot make the output worse.** It never introduces a token the model would not
  otherwise have sampled. Quality is preserved *at the expense of* watermark strength, not
  the other way around.

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
