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
