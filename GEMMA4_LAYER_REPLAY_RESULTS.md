# Gemma decoder-layer replay results

## Main objective and why we are doing this

Enable **CP=2 context parallelism** and **FFPA, or another validated memory-efficient attention backend**, for **`google/gemma-4-31B-it`**, with **verl** as the target framework for eventual **full-parameter NL2SQL RL**. CP=2 distributes one long sequence across two GPU ranks to reduce per-rank sequence/activation pressure. Efficient attention avoids materializing quadratic attention-score tensors; FFPA is a candidate for faster eligible global-attention operations. We want measured capacity and speed improvements while preserving model semantics, token probabilities, and gradients. Global-attention compute/latency is not expected to become linear merely because working memory is approximately linear.

We currently have one GPU, so we validate local behavior first and design the distributed recipe for a future two-node allocation with an explicit GPUs-per-node setting. CP=2 counts GPU ranks, not nodes. Standalone kernels or successful forward execution do not establish distributed training correctness. Existing TRL/ZeRO-3 code on `consensus` is prior task work; NL2SQL integration comes after the systems checks.

**Incremental validation sequence:**

1. Establish a reproducible environment and model snapshot; benchmark single-GPU attention and full-model forward latency/memory across context lengths, with and without FFPA. Measure mask construction and cached generation separately.
2. Resolve numerical discrepancies with identical inputs/weights, teacher-forced token probabilities, FP32 references, fixed-input layer replay, and causal/window/padding/document-boundary checks. This is why we are investigating precision and masks before scaling further.
3. Validate real CP=2 forward correctness, per-rank memory, communication, and speed on separate GPUs; compare efficient-attention backends under the same allocation and effective batch.
4. Validate attention backward, model gradients, activation checkpointing, optimizer steps/state memory, and save/resume; increase context incrementally toward the provisional 40,960-total-token target.
5. Integrate the validated AutoModel/Gemma path into verl, including CP token mapping, loss normalization, actor/reference probabilities, rollout weight synchronization, and one complete resumable RL update.
6. Add NL2SQL data, executable rewards, and held-out database evaluation; measure complete RL iteration performance before choosing a production recipe.

**Success means correct long-context training with CP=2 and an efficient attention path. FFPA is optional:** retain a correct FlexAttention or other efficient baseline if FFPA is unsupported or offers an unfavorable speed/memory tradeoff. Record performance observations separately from correctness acceptance. Detailed roadmap: [GEMMA4_LONG_CONTEXT_RL.md](GEMMA4_LONG_CONTEXT_RL.md).

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
