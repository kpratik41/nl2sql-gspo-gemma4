# Gemma decoder-layer replay results

Date: 2026-10-22 (America/New_York).

## Method

The same checkpoint and environment as GEMMA4_MODEL_PARITY.md were used. One BF16 eager full-model pass per length captured identical decoder inputs for layers 0, 5, 6, and 59. Tests cover 128, 1024, and 1025 tokens from the versioned SQL fixture. Each layer was copied and upcast to FP32, including normalization, Q/K/V projections, positional-embedding inputs, attention, output projection, residual paths, and MLP. This uses the exact BF16 checkpoint weight values in FP32 arithmetic; it does not recover higher-precision pretrained weights. TF32 matmul was disabled through highest FP32 matmul precision.

FP32 eager supplies the reference, controlled by FP32 SDPA with its math backend forced. BF16 candidates receive the exact same saved hidden states. The three measured stages are attention output before output projection, projected attention output, and full decoder output. Flex uses explicit causal/window BlockMasks and 32×32 tiles. FFPA is a benchmark-only causal global-layer adapter with scale 1.0; this is not AutoModel dispatch or CP validation.

## Aggregate numerical results

Maximum relative L2 over tested layers and lengths:

| Backend | Cases | Attention | Projected attention | Decoder |
|---|---:|---:|---:|---:|
| fp32_sdpa_math | 12 | 0.0000000 | 0.0000000 | 0.0000000 |
| eager | 12 | 0.0072245 | 0.0060670 | 0.0057603 |
| sdpa | 12 | 0.0060717 | 0.0054887 | 0.0056107 |
| flex_attention | 12 | 0.0060709 | 0.0054888 | 0.0055340 |
| replay_ffpa | 6 | 0.0060714 | 0.0054886 | 0.0055959 |

## Semantic controls

- Causality: 42 tests replaced the second half of the layer hidden states with perturbed values and compared the first-half decoder outputs. Maximum absolute change: 0.00000000.
- Sliding-window isolation: 6 tests at length 1025 perturbed token zero, outside the last query’s 1024-token window. Maximum last-query attention-output change: 0.00000000.
- FP32 eager versus forced FP32 SDPA math: identical outputs at all three measured stages across all 12 cases.
- These controls cover individual unpacked, unpadded layers at the tested lengths. They do not cover padded batches, cached decoding, packed documents, arbitrary contexts, gradients, or CP.

## Interpretation

The BF16 optimized paths are comparable to, and slightly closer than BF16 eager to, this FP32 layer reference in the worst relative-error summaries. The tested masks obey causality and window isolation. No local masking failure was detected. This supports investigating accumulated numerical drift as a cause of full-model differences; it does not prove that drift is the sole cause or establish acceptable RL probability errors. Max absolute and RMS errors, layer identities, input hashes, and individual invariant results are preserved in the JSON.

The earlier whole-model screening gates remain unresolved. A roughly 0.5% layer-relative error is not itself a training acceptance threshold, and the previous large target-logprob errors must not be dismissed as harmless.

## Next controlled experiment

Replay the whole model with BF16 weights/linear operations but FP32 attention score, softmax, and value accumulation, casting attention output back to BF16. Compare this with ordinary BF16 eager and optimized backends on independent natural-text/SQL examples. Include an option to keep Q/K normalization and RoPE arithmetic in FP32, changing one precision boundary at a time. This can test whether reducing local attention-rounding errors reduces the observed downstream logprob drift without requiring a full FP32 31B model. Keep same-weight teacher forcing, per-layer tracing, and per-token errors. Padding/document-boundary invariants should follow before integration.

## Reproduction

```bash
.venv-bench/bin/python benchmarks/layer_replay.py
.venv-bench/bin/python benchmarks/summarize_replay.py
```

Raw data: [benchmark_results/layer_replay/results.json](benchmark_results/layer_replay/results.json). Execution log: [benchmark_results/layer_replay.log](benchmark_results/layer_replay.log). Input fixture and preceding investigation: [GEMMA4_MODEL_PARITY.md](GEMMA4_MODEL_PARITY.md).
