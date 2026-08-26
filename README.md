# synthmark

SynthID-Text watermarking, detection, and evaluation for open-weight LLMs.

`synthmark` wraps the [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4)
implementation that ships in Hugging Face Transformers (`SynthIDTextWatermarkingConfig`)
with the parts a production deployment actually needs and the reference code leaves out:
key management, calibrated decision thresholds, a detection service, and an evaluation
harness that measures what the watermark does and does not deliver.

Validated on `google/gemma-4-E4B-it`; the code is model-agnostic and works with any
autoregressive HF model that exposes logits.

## What the watermark is

At each decoding step the model has a distribution over next tokens. Where several tokens
are near-equally good, an ordinary sampler picks between them with a random number.
SynthID replaces that randomness with a pseudorandom function seeded by a secret key and a
sliding window of the preceding tokens. Nothing is inserted into the text — no hidden
characters, no markup. The signal lives in *which* of the equally-good tokens got chosen,
and it is recoverable only by someone who can replay the same key.

Two consequences follow directly, and they set the boundaries of everything else:

- The watermark can only use randomness that already existed. Where the model is nearly
  certain of the next token — facts, arithmetic, code, fixed schemas — there is nothing to
  steer, and little or no watermark is embedded.
- Because it never introduces a token the model would not have sampled, it cannot push
  output towards worse text. Quality is preserved *at the cost of* watermark strength,
  not the other way round.

## Install

```bash
pip install -e .            # core
pip install -e '.[serve]'   # + detection service
pip install -e '.[dev]'     # + tests
```

## Use

### Mint a key

The key is signing-grade secret material: anyone holding it can both detect **and forge**
the watermark. Prefer deriving keys from one escrowed master secret over storing many key
files.

```bash
export SYNTHMARK_MASTER_SECRET="$(openssl rand -hex 32)"
synthmark keygen markets-research/v1 --from-master --out keys/markets.json
```

```python
from synthmark import derive_key

key = derive_key(master_secret, "markets-research/v1")   # deterministic, reproducible
key.public_summary()   # safe to log: no key material, just a fingerprint
```

Different labels give cryptographically independent keys (HKDF-SHA256), so separate
business units can hold separate keys, and rotation is just a new label (`.../v2`).

### Generate

```python
from synthmark import WatermarkedLM, derive_key

lm  = WatermarkedLM("google/gemma-4-E4B-it")
key = derive_key(master_secret, "markets-research/v1")

out = lm.generate(lm.chat_prompts(["Explain duration risk to a new analyst."]), key=key)
print(out.texts[0])
```

Turning the watermark on is one argument. Passing `key=None` gives the ordinary,
unwatermarked HF path, so A/B comparisons are genuinely apples-to-apples.

Greedy decoding is refused with the watermark on, rather than silently producing unmarked
text — with no sampling there is no randomness to encode a signal in.

### Detect

```python
from synthmark import Detector

det = Detector(key, lm.tokenizer)          # CPU is fine; no model weights needed
r = det.detect(text, calibration=cal, target_fpr=0.01)

r.score               # mean g-value; 0.5 under the null
r.p_value             # analytic one-sided p-value
r.is_watermarked      # decision at a calibrated 1% false-positive threshold
r.num_tokens_scored   # detection power scales with this
```

```bash
synthmark detect --key-id markets-research/v1 --file suspect.txt
```

Detection needs the **tokenizer**, not the model. It is a hash and a mean.

> **Device portability.** Upstream Transformers builds the watermark's g-value sampling
> table with a device-local RNG, so the *same key* produces a *different watermark* on CPU
> and on GPU. Text generated on a GPU is silently invisible to a CPU detector — the
> detector returns an ordinary-looking null score rather than an error. `synthmark` draws
> the table on CPU and moves it to the target device, so generation and detection agree on
> any hardware. This is what makes a CPU-only detection service possible. See
> `synthmark.config` and `tests/test_detect.py`.

### Calibrate

Analytic p-values assume every g-value is an independent fair coin flip. That is nearly
true, and "nearly" is not something to stake a false-accusation rate on. Calibrate against
real non-watermarked text to get thresholds you can defend:

```bash
synthmark calibrate --key-id markets-research/v1 \
  --texts data/human_texts.json --out calibration.json
```

### Serve

```bash
synthmark serve --key-id desk-a/v1 desk-b/v1 --calibration calibration.json
```

One service can hold several independent keys. The response reports a score, the number of
tokens scored, and a calibrated verdict — never a bare boolean — and refuses to answer at
all below ~40 scored tokens, which is where false accusations come from.

## Layout

```
src/synthmark/
  keys.py       key generation, HKDF derivation, fingerprints, 0600 storage
  config.py     bridge to the HF SynthID API
  generate.py   watermarked / unwatermarked generation, perplexity
  detect.py     g-values, masking, scoring, empirical calibration
  bayesian.py   the learned detector from the paper
  attacks.py    edits, paraphrase, translation, dilution
  metrics.py    AUC, TPR@FPR, bootstrap and Newcombe intervals
  serve.py      FastAPI detection service
  cli.py        synthmark keygen | generate | detect | calibrate | serve
experiments/    the evaluation described in report.md
tests/          unit tests (no model download required)
```

## Reproducing the evaluation

```bash
cd experiments
python 01_generate_corpus.py --samples-per-prompt 4   # paired corpus (~30 min on 1 GPU)
python 02_detectability.py                            # separation, length, key isolation, FPR
python 03_quality.py                                  # diversity, perplexity, GSM8K accuracy
python 04_robustness.py                               # edits, paraphrase, dilution
python 05_overhead.py                                 # throughput cost
```

Findings are written to `results/` and summarised in [report.md](report.md).

## Limits

Stated up front, because a detector that is oversold is worse than none:

- It answers "is this text statistically consistent with our watermark key", not "who
  wrote this". It cannot distinguish text the model wrote from text the model edited.
- It needs length. Short text carries too little signal for a meaningful verdict.
- Low-entropy output — code, JSON, arithmetic, terse factual answers — is weakly marked or
  not marked at all. This is inherent, not a tuning problem.
- Paraphrasing through another model removes it.
- A negative result is not evidence that text is human-written.

## References

- Dathathri et al., *Scalable watermarking for identifying large language model outputs*,
  Nature 634 (2024).
- [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text)
- Hugging Face Transformers `SynthIDTextWatermarkingConfig` / `SynthIDTextWatermarkLogitsProcessor`
