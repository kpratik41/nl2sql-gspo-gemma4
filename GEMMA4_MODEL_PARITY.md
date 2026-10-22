# Gemma full-model numerical parity investigation

Date: 2026-10-22 (America/New_York).

## Test design

One locally cached 31B checkpoint is loaded once in BF16. The backend is switched in place; weights and token IDs are shared. Inference uses eval mode, no cache, explicit token hashes, and teacher forcing rather than sampled generation. The versioned SQL schema/question/answer fixture is repeated to reach 128, 1,023, 1,024, 1,025, and 2,048 tokens. This is a controlled synthetic fixture, not a held-out NL2SQL accuracy benchmark; no external dataset is needed for initial implementation parity.

For every length, BF16 eager supplies a candidate reference, then eager repeats as a determinism control. Compare all next-token target log probabilities, mean negative log likelihood (NLL), sampled vocabulary-distribution KL, top-1 agreement, and sampled hidden states from all 60 decoder layers. Same-QKV checks independently calculate FP32 causal/windowed attention at sampled queries in layers 0, 5, 6, and 59 using the actual model scaling and Q/K/V tensors.

## Investigation gates

Predeclared screening thresholds: absolute mean NLL delta ≤0.01 nats/token, maximum target log-probability delta ≤0.1 nat, and maximum sampled KL ≤0.001 nat. These are conservative investigation gates, not universal BF16 accuracy guarantees. Failure triggers reference/precision isolation; it does not establish which backend is incorrect. Do not relax thresholds after viewing results merely to claim success.

## Results

| Tokens | Backend | NLL delta | Max target logprob error | Max sampled KL | Last-layer relative L2 | Gate |
|---:|---|---:|---:|---:|---:|---|
| 128 | eager | 0.000000 | 0.0000 | 0.000000 | 0.000000 | pass |
| 128 | sdpa | -1.039333 | 21.7451 | 3.400868 | 0.114198 | investigate |
| 128 | flex_attention | -1.029820 | 22.9276 | 3.242771 | 0.113287 | investigate |
| 1023 | eager | 0.000000 | 0.0000 | 0.000000 | 0.000000 | pass |
| 1023 | sdpa | 0.002501 | 3.5696 | 0.216495 | 0.085866 | investigate |
| 1023 | flex_attention | 0.005524 | 2.9590 | 0.284445 | 0.053035 | investigate |
| 1024 | eager | 0.000000 | 0.0000 | 0.000000 | 0.000000 | pass |
| 1024 | sdpa | -0.000786 | 2.1279 | 0.009977 | 0.063782 | investigate |
| 1024 | flex_attention | 0.004876 | 3.1957 | 0.009578 | 0.045495 | investigate |
| 1025 | eager | 0.000000 | 0.0000 | 0.000000 | 0.000000 | pass |
| 1025 | sdpa | -0.151641 | 18.5811 | 0.863344 | 0.213921 | investigate |
| 1025 | flex_attention | -0.150965 | 18.6972 | 0.987440 | 0.227668 | investigate |
| 2048 | eager | 0.000000 | 0.0000 | 0.000000 | 0.000000 | pass |
| 2048 | sdpa | -0.010893 | 7.9666 | 0.141546 | 0.041450 | investigate |
| 2048 | flex_attention | -0.049910 | 11.6294 | 0.526606 | 0.066543 | investigate |

## Independent same-QKV attention reference

| Backend | Maximum sampled absolute error | Maximum sampled relative L2 |
|---|---:|---:|
| eager | 0.228596 | 0.005336 |
| sdpa | 0.059719 | 0.001219 |
| flex_attention | 0.060965 | 0.001295 |

Actual model attention scale in these checks: [1.0]. The earlier random-QKV benchmark used head_dim**-0.5, so it is a separate kernel exercise and cannot substitute for this test. FP32 calculations here use BF16 input Q/K/V; they isolate attention arithmetic but do not provide a full-FP32 model reference.

## Interpretation and next tests

Completed follow-up: [GEMMA4_LAYER_REPLAY_RESULTS.md](GEMMA4_LAYER_REPLAY_RESULTS.md). It contains FP32 decoder-layer controls and causality/window-isolation results; the replay recommendation below is retained as investigation history.

The fixed-input FP32 decoder replay has now been executed: [GEMMA4_LAYER_REPLAY_RESULTS.md](GEMMA4_LAYER_REPLAY_RESULTS.md). It passes the tested causality/window checks; native mask equivalence and full-model logprob agreement remain open.

- Repeated eager execution is the determinism control. Inspect the table rather than assuming a finite loss establishes parity. Small average NLL differences can conceal large individual-token changes relevant to RL importance ratios.
- In the sampled same-QKV checks, optimized attention has lower worst relative error against FP32 than BF16 eager. This is evidence that BF16 eager is not an exact reference; it does not by itself explain the full-model discrepancy or establish correctness at untested positions.
- Both optimized backends differ from BF16 eager on this fixture. Eager BF16 is not a definitive gold standard. Same-QKV FP32 comparisons help distinguish local attention arithmetic from accumulated model-level drift; they do not prove mask correctness at every position.
- Completed follow-up (see linked report): replay an early decoder layer on identical saved hidden states with FP32 weights/activations. Compare eager FP32, SDPA math FP32, and optimized BF16 against that reference. Repeat across global/sliding layers and the first layer where hidden-state error grows. A single full-FP32 31B model exceeds this GPU memory, while one-layer replay is feasible.
- Add explicit semantic invariants: perturb suffix tokens and verify earlier-position outputs do not change; compare an unpadded example with a padded batch; test unequal lengths and document boundaries separately. Preserve correct window semantics at 1,023/1,024/1,025 tokens.
- After those controls pass, evaluate multiple independent natural-language and SQL documents with held-out teacher-forced tokens. Report distributions of NLL/logprob/KL errors and investigate worst cases. A corpus-level loss improvement alone does not establish implementation equivalence.
- Only then extend length and validate backward/CP. Generation text is a secondary check because tiny numerical changes can change a greedy token and all subsequent outputs.

## Reproduction

```bash
.venv-bench/bin/python benchmarks/model_parity.py
.venv-bench/bin/python benchmarks/summarize_parity.py
```

Raw results, per-layer errors, input hashes, and fixture: [benchmark_results/model_parity](benchmark_results/model_parity/). Environment details: [GEMMA4_BENCHMARK_RESULTS.md](GEMMA4_BENCHMARK_RESULTS.md).
