# Integrating synthmark into a shared serving platform

Written for a platform team that serves several open-weight models to many internal
consumers, mostly through vLLM in containers.

The package splits into two halves with **opposite deployment shapes**, and almost every
decision below follows from that:

| | Generation side | Detection side |
|---|---|---|
| Modules | `config.py`, `generate.py` | `detect.py`, `serve.py`, `registry.py`, `keys.py` |
| Runs | inside the serving process, on the GPU | anywhere, on CPU |
| Needs | model weights | tokenizers only |
| Throughput | per token, in the sampling hot path | ~1,250 documents/s per key |
| Holds the key | yes | yes |
| Who operates it | the serving team | one central owner |

---

## 1. The library: what goes on internal PyPI

### What it is for

`synthmark` is the **shared definition of the algorithm** — how a key is derived, how the
g-function is computed, how a score becomes a verdict. It is not itself a service and not
itself a deployment. Its job is that every component in the estate agrees on those
details, because a watermark only resolves if generation and detection agree *exactly*:
same key, same `ngram_len`, same `depth`, same tokenizer. Disagree on any of them and
detection returns a normal-looking null score rather than an error.

Publishing it internally is what makes that agreement enforceable by version pin instead
of by documentation.

### Three distributions, one repository

The repository is a monorepo: one git repo, three independently publishable
packages, all built from the same commit.

```
packages/synthmark/          keys.py registry.py config.py cli.py   → the shared contract
packages/synthmark-detect/   detect.py serve.py                     → the detection service
packages/synthmark-eval/     generate.py metrics.py attacks.py ...  → benchmarks only
tests/  experiments/  scripts/  results/    → ship nowhere, stay in git as the record
```

Why three rather than one package with extras: extras control *dependencies*, not
*contents*. A wheel with an `eval` extra still contains `serve.py`, and "why does the
inference image contain an HTTP service" is a review question with no good answer.
Splitting the distributions makes the answer structural — the serving image installs
`synthmark`, whose wheel contains five files and no detector:

```
$ unzip -l synthmark-0.3.0-py3-none-any.whl | grep '\.py$'
    synthmark/__init__.py  synthmark/cli.py  synthmark/config.py
    synthmark/keys.py      synthmark/registry.py
```

Why three rather than three *branches* or three *repos*: `keys.py` must never diverge.
If `derive_key` differs by one detail between the generation build and the detection
build, every document generated becomes permanently undetectable, and nothing errors —
you get a plausible score near 0.5. In one repo that is a diff a reviewer sees and a
test round-trips. Across branches it is invisible. The three share one version number
and are released together for the same reason.

### Who installs what

```bash
pip install synthmark                     # vLLM serving image: g-function + keys
pip install "synthmark-detect[serve]"     # detection service
pip install synthmark-eval                # benchmarks; pulls the other two
```

`synthmark.keys` and `synthmark.registry` are **pure standard library**, and
`__init__.py` imports only those eagerly (`config` resolves on first attribute access).
So a key-rotation job or a CI check on the key registry runs with no PyTorch installed
at all:

```python
import synthmark            # no torch imported
reg = synthmark.KeyRegistry.load("configs/key_registry.json")
```

### Build and publish

```bash
for p in synthmark synthmark-detect synthmark-eval; do
    python -m build --wheel --outdir dist packages/$p
done
twine upload -r internal dist/*
```

All three build as `py3-none-any` — pure Python, no compiled extensions — so one wheel
each works on every OS, architecture and Python >= 3.10. The platform-specific part is
`torch`, which is a *dependency*, not part of these wheels; which torch build a consumer
resolves is controlled by their index URL, which is why `deploy/Dockerfile.detect` pins
the CPU index explicitly.

Three things to fix before the first upload, because they are painful to change after
consumers have pinned:

- **Version deliberately.** The wire format that matters is the *key derivation*. If
  `derive_key` ever changes, every previously generated document becomes undetectable.
  Treat `synthmark/keys.py` as a compatibility surface: changes to it are major-version
  changes, and the HKDF salt (`b"synthmark/hkdf/v1"`) is versioned for exactly that reason.
- **Ship the conformance tests.** `tests/test_keys.py` and `tests/test_registry.py` are
  what a second implementation (a Java service, a different serving stack) validates
  against. They are the spec.
- **Pin `transformers` with a floor, not a ceiling.** SynthID landed in 4.46; the
  device-portability fix in `config.py` overrides upstream behaviour, so upgrades need a
  round-trip test rather than a version pin to hold the line.

---

## 2. The vLLM plugin: how it works and why it is the right shape

### The mechanism

vLLM's sampler applies a chain of *logits processors* to the `(batch, vocab)` logits
tensor before sampling. In vLLM V1 a processor is a class with four methods:

```python
class SynthIDProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory): ...
    def apply(self, logits): ...              # the watermark goes here
    def update_state(self, batch_update): ... # continuous batching bookkeeping
    def is_argmax_invariant(self): return False
```

vLLM discovers it through a Python **entry point** — the same mechanism as a pytest
plugin or a console script. The package declares it:

```toml
[project.entry-points."vllm.logits_processors"]
synthid = "synthmark.vllm:SynthIDProcessor"
```

and vLLM loads it by name at engine start:

```bash
vllm serve nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
    --logits-processors synthmark.vllm:SynthIDProcessor
```

Per-request control rides on `SamplingParams.extra_args`, so a caller opts in or out
without a separate endpoint:

```python
SamplingParams(extra_args={"synthid_key_id": "nvidia-nemotron-super-49b/v1"})
```

### Why a plugin rather than a fork or a patch

The alternatives are worse in ways that compound:

- **Forking vLLM** means owning a merge every release, forever. vLLM moves weekly.
- **Patching the image** (`sed`, a `.patch` file, a vendored sampler) means the patch
  breaks silently the first time upstream refactors the sampler — and the failure mode is
  unmarked output, not a crash.
- **Monkey-patching at runtime**, which is what some existing watermark integrations do
  for Gumbel-family schemes, means the behaviour depends on import order.

A plugin is the only option where **the serving image is stock vLLM plus one wheel.** The
team upgrades vLLM on vLLM's schedule and the watermark on yours, and the interface
between them is a published, tested contract rather than a diff. Concretely, the
Dockerfile gains one line:

```dockerfile
FROM vllm/vllm-openai:v0.24.0
RUN pip install --index-url https://artifactory.example.internal/api/pypi/pypi/simple \
        "synthmark-vllm==0.3.0"
```

### What the port actually involves

`generate.py` is built on HuggingFace `generate()` and does **not** transfer. What
transfers is the g-function and the candidate-only optimisation in `config.py`, rehomed
onto the four methods above. Three things need care:

1. **Per-sequence state under continuous batching.** SynthID needs the last
   `ngram_len - 1` tokens for each sequence. Under continuous batching, sequences are
   added, removed and *moved* between row indices every step. `update_state(batch_update)`
   delivers exactly those events — added requests arrive as
   `(index, SamplingParams, prompt_token_ids, output_token_ids)` — and the watermark state
   must be reindexed in lockstep. Get this wrong and the watermark is computed against
   another request's context: no crash, no error, just text that will not detect.
2. **Prompt tokens.** The first few generated tokens are watermarked using context that
   runs back into the prompt. The old limitation where processors saw only output tokens
   ([vLLM #2142](https://github.com/vllm-project/vllm/issues/2142)) is closed in V1 —
   `prompt_token_ids` is supplied on add and processors hold live references to both lists.
3. **`is_argmax_invariant()` must return `False`.** SynthID changes which token wins.
   Claiming otherwise lets vLLM skip the processor under greedy sampling — again silently.

### The acceptance gate

Every failure mode above produces the same symptom: a plausible score near 0.5. So the
port is not done when it runs, it is done when this passes in CI:

```
generate N documents through vLLM with the watermark on
  -> detect with synthmark using the same key
  -> assert p < 0.01 on every one
generate N documents with the watermark off
  -> assert no detection
```

Run it against every served model, not once.

---

## 3. The detection service

### Why it is central, and why consumers must not run it themselves

A watermark key is **symmetric**. Whoever can use it to detect can also use it to forge:
they can generate text that your own detector will confirm as platform-generated. A key
handed to a downstream team so they can "just check locally" is a key that can
manufacture evidence.

So: one service, operated by one owner, holding the keys. Consumers get verdicts.

It is also the easy half. Detection reads token ids and hashes them — no forward pass, no
weights, no GPU, ~1,250 documents/s per key on ordinary CPU. It can run far away from the
serving fleet, on hardware nobody is competing for.

### One key per model

This is what the question "which model wrote this?" requires. Text marked by the Nemotron
key is null-distributed under the Gemma key, so a hit **names the model** rather than
merely saying "something of ours". Two models under one key are indistinguishable
forever.

Distinct keys cost nothing extra to hold: every key is
`derive_key(master_secret, key_id)`, so onboarding a model adds a registry entry, not a
secret.

`configs/key_registry.example.json` — non-secret, commit it, review it in a PR:

```json
{
  "keys": [
    {"key_id": "google-gemma-4-e4b-it/v1",     "model_id": "google/gemma-4-E4B-it",                  "status": "active"},
    {"key_id": "nvidia-nemotron-super-49b/v1", "model_id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1", "status": "active"},
    {"key_id": "google-gemma-4-e4b-it/v0",     "model_id": "google/gemma-4-E4B-it",                  "status": "retired"}
  ]
}
```

The registry enforces two invariants that a naming convention alone would not:

- **At most one active key per model.** Generation must be unambiguous. Rotation retires
  the old entry in the same commit that adds the new one.
- **`key_id` is unique.** Two entries with the same label are the same key.

And it records a **fingerprint** per entry — a SHA-256 prefix, not secret. Stamp it once:

```bash
synthmark registry --registry configs/key_registry.json --stamp
```

From then on, startup verifies that the key derived from the master secret is the key the
registry expects. A wrong secret, a rotated secret, an edited `depth` — all become a loud
startup failure instead of a service that detects nothing and says so in the language of
"no watermark found".

**Retiring is not deleting.** Text generated under a retired key still exists, so retired
keys stay in the registry and stay loaded for detection. Entries are only ever added.

### The two endpoints

`POST /detect` — *"is this marked by this key?"* The common case, when the caller already
knows which model is in question. Route by `key_id`, or by `model_id` and let the service
pick that model's active key.

`POST /attribute` — *"is this ours, and if so whose?"* For incident response, when origin
is the question. It scans every key, including retired ones, and returns a ranked list.

### The correction that makes attribution honest

Scanning `N` keys means running `N` independent tests. At the usual 1% false-positive rate
per key, the rate for the *scan* is `1 - 0.99^N` — with 40 keys, **a third of
unwatermarked texts would be attributed to something.** Scan a large enough registry and
something always lights up.

`/attribute` therefore tests each key at `target_fpr / N` (Bonferroni) and reports
`per_key_alpha` alongside the verdict, so the number in a report is the scan-level rate.

This is also the strongest argument for keeping the registry small. Every extra key axis —
per desk, per environment, per deployment — makes each individual test stricter and so
makes every genuine watermark harder to find. Tenant identity belongs in the request log,
which you already have. **Key identity should carry only what detection cannot recover
any other way: the model.**

A second matched key is reported as `ambiguous`, not as a finding. Independent keys should
never both fire; if they do, it is a configuration fault.

### Running it

```bash
export SYNTHMARK_MASTER_SECRET="$(vault read -field=value secret/synthmark/master)"
synthmark serve --registry configs/key_registry.json --calibration-dir calibrations/
```

or as a container — see `deploy/Dockerfile.detect`, which is CPU-only, pre-fetches every
tokenizer the registry names so the running container needs no network, runs as a
non-root user, and takes the master secret from the environment rather than a layer.

### Calibration is per key, not per service

Analytic thresholds assume the g-values are independent fair coin flips. That is very
nearly true, and "very nearly" is not what you want to stake a false-accusation rate on.
Empirical thresholds come from `Detector.calibrate()` over non-watermarked text and are
stored per key, because **detection strength varies by model**: we measured AUC 1.000 on
Gemma-4-E4B against 0.946 on Gemma-4-31B, since larger models are more confident, lower
entropy, and leave a weaker signal. Re-calibrate at every model onboarding. Reusing one
model's thresholds for another is how a threshold quietly stops meaning what it says.

---

## Order of work

1. **Publish `synthmark` internally.** Useful on day one for keys and detection,
   independent of the vLLM work.
2. **Put the master secret in the secrets manager and commit the key registry.** Migrating
   key custody after teams have integrated is far worse than setting it up now.
3. **Stand up the detection service.** No GPU, no port, and it gives you the oracle the
   vLLM work will be validated against.
4. **Port the processor to vLLM**, gated on the round-trip test above.

## Security constraints that do not bend

- The key can **forge** as well as detect. It never leaves the detection service and the
  serving process.
- One master secret, escrowed in a secrets manager or HSM. Never in a repo, an image
  layer, or a shell profile. Everything else is derived.
- Log `fingerprint`, never key material. `WatermarkKey.public_summary()` is the safe form.
- **The detection service must not log request bodies.** Text submitted for checking is,
  by definition, text somebody is suspicious about.
- A saved Bayesian detector directory embeds the key. Treat the directory as key material.
