# Gemma decoder-layer replay results

## Objective and scope

Enable Gemma 4 31B CP=2 with validated efficient attention for full-parameter NL2SQL RL in verl. The current training path is native FSDP2, Ulysses and FlexAttention; FFPA remains optional and gated on correctness. These historical single-GPU replays isolate numerical and masking behavior before distributed training. See [the current recipe](src/nl2sql_verl/README.md) and [chronological work log](WORK_LOG.md) for subsequent results.

## Method

Each backend replays captured decoder inputs from the same BF16 checkpoint. The reference upcasts those quantized weights and inputs to FP32; it is not a full-FP32 model reference or a higher-precision pretrained checkpoint. Attention, projected attention, and decoder outputs are compared. The stored measurements are descriptive; no new BF16 acceptance threshold is imposed.

The result schemas record different harness versions. This report preserves the supplied measurements and semantic checks without inventing missing reference rows, input hashes, or checks. Consult the corresponding execution log for mask-construction details.

## Maximum relative L2 error

| Backend | Cases | Attention | Projected attention | Decoder |
|---|---:|---:|---:|---:|
| eager | 12 | 0.00722447 | 0.00606696 | 0.00576028 |
| flex_attention | 12 | 0.00607094 | 0.00548878 | 0.00553399 |
| fp32_sdpa_math | 12 | 0.00000000 | 0.00000000 | 0.00000000 |
| replay_ffpa | 6 | 0.00607136 | 0.00548865 | 0.00559588 |
| sdpa | 12 | 0.00607170 | 0.00548868 | 0.00561072 |

## Semantic checks

Recorded replay cases: 54. Recorded semantic checks: 48. Semantic failures: 0.

Maximum recorded semantic difference: 0.00000000.

FP32 SDPA-math reference comparisons: 12; exact agreement at all measured stages: True.

## Decoder error by case

| Tokens | Layer | Backend | Relative L2 | Maximum absolute |
|---:|---:|---|---:|---:|
| 128 | 0 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 128 | 0 | eager | 0.00492755 | 0.70647430 |
| 128 | 0 | sdpa | 0.00461113 | 0.68390274 |
| 128 | 0 | flex_attention | 0.00460187 | 0.68390274 |
| 128 | 5 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 128 | 5 | eager | 0.00354921 | 1.11830139 |
| 128 | 5 | sdpa | 0.00353071 | 1.11830139 |
| 128 | 5 | flex_attention | 0.00352823 | 1.11830139 |
| 128 | 5 | replay_ffpa | 0.00353045 | 1.11830139 |
| 128 | 6 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 128 | 6 | eager | 0.00301867 | 1.33850098 |
| 128 | 6 | sdpa | 0.00297622 | 1.33850098 |
| 128 | 6 | flex_attention | 0.00297820 | 1.33850098 |
| 128 | 59 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 128 | 59 | eager | 0.00514818 | 0.13617992 |
| 128 | 59 | sdpa | 0.00514994 | 0.10711098 |
| 128 | 59 | flex_attention | 0.00512618 | 0.10711098 |
| 128 | 59 | replay_ffpa | 0.00521183 | 0.10711098 |
| 1024 | 0 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1024 | 0 | eager | 0.00433135 | 0.60680389 |
| 1024 | 0 | sdpa | 0.00397317 | 0.69442368 |
| 1024 | 0 | flex_attention | 0.00397037 | 0.69442368 |
| 1024 | 5 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1024 | 5 | eager | 0.00334102 | 1.34507751 |
| 1024 | 5 | sdpa | 0.00329672 | 1.34507751 |
| 1024 | 5 | flex_attention | 0.00330015 | 1.34507751 |
| 1024 | 5 | replay_ffpa | 0.00330332 | 1.34507751 |
| 1024 | 6 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1024 | 6 | eager | 0.00301413 | 1.42327881 |
| 1024 | 6 | sdpa | 0.00299533 | 1.42327881 |
| 1024 | 6 | flex_attention | 0.00299903 | 1.42327881 |
| 1024 | 59 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1024 | 59 | eager | 0.00535345 | 0.24178505 |
| 1024 | 59 | sdpa | 0.00509259 | 0.16791916 |
| 1024 | 59 | flex_attention | 0.00501602 | 0.16853142 |
| 1024 | 59 | replay_ffpa | 0.00506778 | 0.14378548 |
| 1025 | 0 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1025 | 0 | eager | 0.00474892 | 0.78073120 |
| 1025 | 0 | sdpa | 0.00439483 | 0.78073120 |
| 1025 | 0 | flex_attention | 0.00438880 | 0.78073120 |
| 1025 | 5 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1025 | 5 | eager | 0.00340129 | 1.29798889 |
| 1025 | 5 | sdpa | 0.00340684 | 1.23439026 |
| 1025 | 5 | flex_attention | 0.00339850 | 1.29798889 |
| 1025 | 5 | replay_ffpa | 0.00340606 | 1.23439026 |
| 1025 | 6 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1025 | 6 | eager | 0.00304106 | 1.53913879 |
| 1025 | 6 | sdpa | 0.00300902 | 1.39674377 |
| 1025 | 6 | flex_attention | 0.00300508 | 1.39674377 |
| 1025 | 59 | fp32_sdpa_math | 0.00000000 | 0.00000000 |
| 1025 | 59 | eager | 0.00576028 | 0.20132256 |
| 1025 | 59 | sdpa | 0.00561072 | 0.20132256 |
| 1025 | 59 | flex_attention | 0.00553399 | 0.20132256 |
| 1025 | 59 | replay_ffpa | 0.00559588 | 0.20132256 |

## Interpretation

Local replay does not resolve whole-model log-probability disagreement. Passing these controls does not validate every mask, cached decoding, packing, backward, or CP. Later native FSDP2/CP2 validation and the current H200 training results are recorded separately in WORK_LOG.md.

## Reproduction

```bash
python benchmarks/summarize_replay.py --input benchmark_results/layer_replay/results.json
```

Both summarize_replay.py and the older summarize_layer_replay.py entry point use this one renderer. Neither command reruns GPU experiments.
