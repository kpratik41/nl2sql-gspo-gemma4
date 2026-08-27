---

## 6. The package

`synthmark` is the reusable half of this work. It wraps the Transformers SynthID
implementation and adds what a deployment needs.

| Module | What it provides |
|---|---|
| `keys.py` | HKDF-SHA256 key derivation from one master secret, fingerprints, 0600 storage |
| `config.py` | Bridge to the HF API, plus the device-portability fix |
| `generate.py` | Watermarked / unwatermarked generation, perplexity scoring |
| `detect.py` | g-values, masking, three scoring methods, empirical calibration |
| `bayesian.py` | Training and use of the learned detector (optional; not used by default) |
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

## 7. Limitations

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

## 8. Mapping the findings to the obligations

§1 states the legal requirement. This section maps what was measured onto it, obligation by
obligation, so a compliance reader can see which claims the evidence supports.

| Art. 50(2) requirement | Status | Evidence |
|---|---|---|
| Output "marked in a machine-readable format" | **Met** | The mark is embedded at generation; no side-channel or metadata needed |
| "Detectable as artificially generated" | **Met, with conditions** | AUC 1.000 / 100% detection on E4B long-form prose; **0.946 / 51% on the 31B** (§5.1a) |
| "Effective… as far as technically feasible" | **Bounded, and quantified** | Limits measured and stated: length (§5.2), entropy (§5.1), paraphrase (§5.6) |
| "Robust and reliable" | **Partially** | Survives editing, formatting, translation; **not** paraphrase (§5.6) |
| Marking carries no personal data | **Met by design** | The key is per-deployment, not per-user or per-session (§5.3) |
| Marking does not degrade the service | **Met** | No quality cost on either model (§5.5); latency +1.7 ms/token (§5.8) |

**The three numbers a governance owner should actually sign off on** — not AUC, which
averages over operating points nobody would deploy:

1. **False-positive rate at the deployed threshold**, measured on human writing — 1.3% at a
   nominal 1% target (§5.4). This is the rate at which a person's own writing gets flagged.
2. **Minimum text length below which no verdict is issued** — the service refuses under 40
   scored tokens; §5.2 argues for ~100 as a policy floor on E4B.
3. **Per-model detection power**, because it does not transfer between models (§5.1a) and
   will weaken silently when the serving model is upgraded.

**Explicitly outside what this control can support:** detection of paraphrased content,
attribution to a person, and any claim about content the pipeline did not generate. For
images and files the appropriate mechanism is C2PA content credentials — a different,
complementary control.

A closing caution for anyone drafting the compliance narrative. Article 50(4)'s exemption
for human-edited text with a named editorial owner is a *process* control, and for many
internal workflows it is both cheaper and more defensible than a technical mark. Watermarking
is the right answer for machine-readable provenance at scale; it is not automatically the
right answer to every transparency obligation.

---

## 9. Reproducing

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
