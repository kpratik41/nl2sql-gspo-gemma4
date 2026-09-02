# synthmark

Watermark keys, the key registry, and the SynthID-Text g-function: the contract
that both ends of a deployment share.

A watermark only resolves if generation and detection agree *exactly* on the
key, on `ngram_len`, on `depth`, and on the g-function. This package exists so
that agreement is pinned by version rather than by documentation.

```python
import synthmark                                    # no torch imported
reg = synthmark.KeyRegistry.load("key_registry.json")
key = reg.for_model("google/gemma-4-E4B-it").resolve(master_secret)
```

`synthmark.keys` and `synthmark.registry` are pure standard library. Only
`synthmark.config` needs PyTorch, and it is imported on first use, so a
key-rotation job or a CI check on the registry runs without it.

Companion distributions: `synthmark-detect` (scoring and the detection service)
and `synthmark-eval` (HuggingFace generation, attacks, metrics).

Full documentation: <https://github.com/kpratik41/nl2sql-gspo-gemma4/tree/watermarking>
