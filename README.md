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

Three distributions. Install only the one whose job you are doing:

```bash
pip install synthmark                     # keys, registry, g-function  (vLLM serving image)
pip install "synthmark-detect[serve]"     # + scoring and the detection service
pip install synthmark-eval                # + benchmarks; pulls the other two
```

For development in this repo, all three editable:

```bash
pip install -e packages/synthmark -e packages/synthmark-detect -e packages/synthmark-eval
pytest -q                                 # 52 passed, 3 skipped
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
from synthmark_eval import WatermarkedLM, derive_key

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
from synthmark_detect import Detector

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

> **Batched throughput.** The reference processor evaluates the watermark's g-function over
> the entire vocabulary, even though it runs after top-k/top-p and almost every token is
> already `-inf`. That is a `(batch, vocab, depth)` tensor per decoding step — 2 GB at batch
> 32 on a 262k vocabulary — and it costs 39–57% of serving throughput. `synthmark` restricts
> the g-function to tokens that can still be sampled, which is exact (they carry zero
> probability) and 18–92× faster on CPU. See `CandidateOnlySynthIDLogitsProcessor`.

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
synthmark registry --registry configs/key_registry.json --stamp   # once, after editing
synthmark serve --registry configs/key_registry.json --calibration-dir calibrations/
```

One service holds every key in the registry, each bound to the model it marks. `POST
/detect` answers "is this marked by this key"; `POST /attribute` scans all keys and names
which model produced the text, correcting the threshold for the number of keys tested.
Responses report a score, the number of tokens scored, and a calibrated verdict — never a
bare boolean — and refuse to answer at all below ~40 scored tokens, which is where false
accusations come from.

Detection needs tokenizers but no weights and no GPU. See [docs/integration.md](docs/integration.md)
for deploying it alongside a vLLM serving platform.

## Layout

Three distributions, one repository, one commit. They share a version number and
are released together, because `keys.py` is a compatibility surface: if
`derive_key` changes, every previously watermarked document becomes undetectable.

```
packages/synthmark/            pip install synthmark
  keys.py       key generation, HKDF derivation, fingerprints, 0600 storage
  registry.py   which key marks which model; rotation and fingerprint checks
  config.py     the g-function and logits processor (bridge to the HF SynthID API)
  cli.py        synthmark keygen | generate | detect | calibrate | registry | serve

packages/synthmark-detect/     pip install synthmark-detect[serve]
  detect.py     g-values, masking, scoring, empirical calibration
  serve.py      FastAPI detection service: /detect and /attribute

packages/synthmark-eval/       pip install synthmark-eval
  generate.py   watermarked / unwatermarked generation, perplexity (HF generate())
  bayesian.py   the learned detector from the paper
  attacks.py    edits, paraphrase, translation, dilution
  metrics.py    AUC, TPR@FPR, bootstrap and Newcombe intervals

configs/        key_registry.example.json — non-secret, one key per model
deploy/         Dockerfile for the detection service
docs/           key management and serving-platform integration
experiments/    the evaluation described in watermarking_report.md
tests/          unit tests (no model download required)
```

`synthmark` is what a vLLM serving image installs: it contains the g-function and
the keys, and no detection code or HTTP service. `synthmark-detect` is what the
detection service installs. `synthmark-eval` re-exports both for the benchmarks,
so an experiment still needs one import line.

For development, install all three editable:

```bash
pip install -e packages/synthmark -e packages/synthmark-detect -e packages/synthmark-eval
```

## What is built, and what is not

Being explicit about this, because "the watermarking package" sounds like it covers
serving, and today it does not.

| | State |
|---|---|
| Key derivation, registry, rotation | **Built**, tested |
| Detection: scoring, calibration, `/detect`, `/attribute` | **Built**, tested |
| Watermarked generation via HuggingFace `generate()` | **Built**, tested, evaluated |
| Watermarked generation **under vLLM** | **Not built** — see below |

### The vLLM processor is not written yet

The *logits processor* is the code that applies the watermark: at each decoding step it
takes the model's scores over the vocabulary and steers which token is chosen.

There is exactly one implementation of it, [`synthmark/config.py`](packages/synthmark/src/synthmark/config.py),
and it is written against **HuggingFace's** interface — HF calls it as
`__call__(input_ids, scores)`.

vLLM wants the same arithmetic through a different shape. Its V1 interface is a class
with `apply(logits)`, `update_state(batch_update)`, `is_argmax_invariant()` and
`validate_params()`. **Porting the processor** means writing a second class that performs
the identical watermark computation against those four methods. Nobody has written that
file — not this repo, not upstream, not anyone. `synthmark-vllm` will be a fourth package
in `packages/` when it exists; it is deliberately absent rather than present and empty,
because an empty wheel on an internal index installs cleanly, does nothing, and burns a
version number.

Three things the port has to get right, all of which fail *silently* — no crash, no
error, just text that will not detect:

1. **Per-sequence state under continuous batching.** SynthID needs the last
   `ngram_len - 1` tokens of each sequence. vLLM adds, removes and *moves* sequences
   between row indices every step; `update_state(batch_update)` reports exactly those
   events, and the watermark state must be reindexed in lockstep or the watermark is
   computed against another request's context.
2. **Prompt tokens.** The first few generated tokens are watermarked using context that
   runs back into the prompt. vLLM V1 supplies `prompt_token_ids` on add, so this works —
   the older limitation ([vLLM #2142](https://github.com/vllm-project/vllm/issues/2142))
   is closed.
3. **`is_argmax_invariant()` must return `False`.** SynthID changes which token wins;
   claiming otherwise lets vLLM skip the processor entirely under greedy sampling.

The acceptance gate is a round trip, not a unit test: generate through vLLM, detect with
`synthmark-detect`, assert `p < 0.01` on every document, and assert no detection with the
watermark off. Run it per served model. Every failure mode above produces the same
plausible score near 0.5, so nothing cheaper catches them.

## Publishing

```bash
for p in synthmark synthmark-detect synthmark-eval; do
    python -m build --wheel --outdir dist packages/$p
done
twine upload -r internal dist/*
```

All three build as **`py3-none-any`**: any Python 3, no C-extension ABI, any OS and
architecture. One wheel each, everywhere — no manylinux matrix, no per-arch builds.

The platform-specific dependency is `torch`, which is a *dependency* rather than package
content, so which build a consumer resolves is decided by **their index URL**, not by
anything here. That is why [deploy/Dockerfile.detect](deploy/Dockerfile.detect) pins
`https://download.pytorch.org/whl/cpu` explicitly: without it, a detection container
silently pulls ~2.5 GB of CUDA runtime it can never use. Worth confirming what your
internal index resolves `torch` to before the first install.

Two things to settle before the first upload, because they are painful to change once
consumers have pinned:

- **`keys.py` is a compatibility surface, not just code.** If `derive_key` changes, every
  previously watermarked document becomes undetectable. Changes to it are major-version
  changes; the HKDF salt (`b"synthmark/hkdf/v1"`) is versioned for exactly that reason.
  All three distributions share one version and are released together.
- **`tests/test_keys.py` and `tests/test_registry.py` are the spec.** They are what a
  second implementation — a Java service, a different serving stack — validates against.

## Moving this repo to another workspace

Move the git history, not a file copy: a copy loses the commit history and the rename
tracking that keeps `git log --follow` working through the package split.

```bash
git clone <repo> && git checkout watermarking
pip install -e packages/synthmark -e packages/synthmark-detect -e packages/synthmark-eval
pytest -q                                             # 52 passed, 3 skipped
for p in synthmark synthmark-detect synthmark-eval; do
    python -m build --wheel --outdir dist packages/$p
done
```

Build the wheels *there*, not here — a build only fails where it fails: a different Python
version, no route to public PyPI, a differently configured index. Those three commands are
the smoke test for the move.

**What will not travel: `data/` (~12 MB).** The generated corpora are gitignored because
they are large and, in principle, regenerable — but regenerating them needs GPUs and the
Gemma weights. Copy `data/` across separately if you need the raw text. The *results* are
committed (`results/`, 376 KB), so every number the reports cite survives the move and the
reports rebuild without it; only re-deriving new statistics from raw text needs the corpora.

## Reproducing the evaluation

```bash
cd experiments
python 01_generate_corpus.py --samples-per-prompt 4   # paired corpus (~30 min on 1 GPU)
python 02_detectability.py                            # separation, length, key isolation, FPR
python 03_quality.py                                  # diversity, perplexity, GSM8K accuracy
python 04_robustness.py                               # edits, paraphrase, dilution
python 05_overhead.py                                 # throughput cost
```

Findings are written to `results/` and summarised in [watermarking_report.md](watermarking_report.md), with a one-page
brief in [watermarking_onepager.md](watermarking_onepager.md).

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
