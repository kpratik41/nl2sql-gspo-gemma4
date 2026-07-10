# NL2SQL Tool-Call Latency Study

## Purpose

Consolidate the latency, throughput, token/tool behavior, and BIRD execution accuracy results for tool-call inference on `outputs/old-dev-schema-tool-unpatched.jsonl`. Most completed runs use the first 200 examples; the TP=8/concurrency=128 run uses 400 examples and reports a 200-example-equivalent generation time.

## Source Reports

- `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/latency_study.md`
- `outputs/latency_study/gemma4_31b_tool_rl_latency/latency_study.md`

The base report contains the completed base, MTP, quantized, NVFP4, and E4B rows. The tool-call RL report currently has no completed runs.

## Method

- Backend: async vLLM
- Primary vLLM env: `nl2sql_vllm024` / vLLM `0.24.0`
- NVFP4 env: `nl2sql312` / vLLM `0.19.1`
- Input data: `outputs/old-dev-schema-tool-unpatched.jsonl`
- Gold/diff data: `data/bird_dev_data/raw/bird_dev_unpatched.json`
- Temperature: `0.0`
- Max prompt length: `34000`
- Max output tokens: `8000`
- vLLM max model length: `43500`
- Max tool rounds: `8`
- TTFT is measured from `engine.generate(...)` submission to the first streamed request output with text/token content.
- End-to-end latency includes all LLM rounds plus synchronous tool execution.
- `serving gen sec` excludes engine load, compilation, and CUDA graph capture when available; older rows fall back to total generation time.

## Concurrency Note

vLLM has an internal scheduler and request queue, but these runs still set `--vllm_async_concurrency` because the repo's async caller uses it as the application-level limit for in-flight examples. Lower concurrency can reduce user-visible latency, while higher concurrency can improve aggregate throughput until the system saturates.

## Base TP And Concurrency Sweep

Model: `google/gemma-4-31B-it`.

| run | TP | concurrency | max batched tok | EX acc | correct/total | total gen sec | serving gen sec | serving ex/s | avg e2e sec | TTFT p50 | TTFT p95 | e2e p50 | e2e p95 | avg decode sec | out tok/s serving | avg tool calls | pred executed | path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tp1_c16_n200 | 1 | 16 | default | 72.50% | 145/200 | 2299.85 | 2299.85 | 0.087 | 170.290 | 19.280 | 41.497 | 144.043 | 336.500 | 119.851 | 44.92 | 1.270 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp1_c16_n200` |
| tp2_c16_n200 | 2 | 16 | default | 71.50% | 143/200 | 1056.43 | 1056.43 | 0.189 | 73.015 | 2.434 | 18.093 | 55.486 | 163.326 | 66.259 | 93.26 | 1.270 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp2_c16_n200` |
| tp4_c16_n200 | 4 | 16 | default | 72.50% | 145/200 | 614.61 | 614.61 | 0.325 | 37.672 | 1.926 | 14.347 | 30.880 | 85.298 | 34.055 | 161.89 | 1.250 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp4_c16_n200` |
| tp8_c16_n200 | 8 | 16 | default | 72.00% | 144/200 | 1017.47 | 1017.47 | 0.197 | 30.488 | 1.631 | 12.149 | 24.512 | 67.689 | 27.419 | 98.02 | 1.285 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c16_n200` |
| tp8_c32_n200 | 8 | 32 | default | 72.50% | 145/200 | 437.53 | 437.53 | 0.457 | 53.415 | 1.710 | 37.033 | 43.133 | 119.363 | 46.587 | 225.65 | 1.275 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c32_n200` |
| tp8_c64_n200 | 8 | 64 | default | 72.50% | 145/200 | 551.56 | 551.56 | 0.363 | 134.719 | 13.135 | 87.213 | 114.019 | 292.167 | 105.409 | 180.13 | 1.280 | 198/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c64_n200` |
| tp8_c128_n400 | 8 | 128 | default | 70.75% | 283/400 | 1052.72 (526.36/200eq) | 1052.72 (526.36/200eq) | 0.380 | 268.078 | 54.094 | 161.690 | 255.810 | 487.733 | 145.214 | 174.28 | 1.225 | 397/400 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c128_n400` |

## MTP Comparison

Matched TP=8/concurrency=32 runs using the same first 200 examples.

| run | TP | concurrency | max batched tok | spec tok | accept rate | mean accept len | accepted/drafted tok | EX acc | correct/total | total gen sec | serving gen sec | serving ex/s | avg e2e sec | TTFT p50 | TTFT p95 | e2e p50 | e2e p95 | avg decode sec | out tok/s serving | avg tool calls | pred executed | path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tp8_c32_batched9984_n200 | 8 | 32 | 9984 | n/a | n/a | n/a | n/a/n/a | 71.50% | 143/200 | 486.98 | 373.89 | 0.535 | 52.687 | 1.556 | 35.514 | 41.370 | 116.838 | 46.696 | 263.91 | 1.315 | 196/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c32_batched9984_n200` |
| tp8_c32_mtp3_batched9984_n200 | 8 | 32 | 9984 | 3 | n/a | n/a | n/a/n/a | 71.50% | 143/200 | 459.04 | 336.90 | 0.594 | 48.922 | 1.775 | 41.259 | 37.456 | 115.138 | 41.270 | 288.24 | 1.250 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c32_mtp3_batched9984_n200` |
| tp8_c32_mtp3_batched9984_stats_n200 | 8 | 32 | 9984 | 3 | 87.2% | 3.616 | 82448/94533 | 72.00% | 144/200 | 400.17 | 327.43 | 0.611 | 47.660 | 1.804 | 36.249 | 38.637 | 102.479 | 40.540 | 301.11 | 1.230 | 198/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c32_mtp3_batched9984_stats_n200` |
| tp8_c32_mtp6_batched9984_stats_n200 | 8 | 32 | 9984 | 6 | 73.5% | 5.410 | 99173/134922 | 72.00% | 144/200 | 387.34 | 315.13 | 0.635 | 46.833 | 2.473 | 35.883 | 34.505 | 111.724 | 38.786 | 313.01 | 1.295 | 197/200 | `outputs/latency_study/gemma4_31b_it_base_vllm024_latency/runs/tp8_c32_mtp6_batched9984_stats_n200` |

## Quantized And Small Model Comparison

Base, Google W4A16, and Gemma 4 E4B IT used `nl2sql_vllm024` / vLLM `0.24.0`; NVIDIA NVFP4 succeeded with `nl2sql312` / vLLM `0.19.1` and `--quantization modelopt`. Gemma 4 E4B IT is not quantized, but is included as a smaller-model latency/accuracy reference.

| run | TP | concurrency | max batched tok | quantization | EX acc | correct/total | total gen sec | serving gen sec | serving ex/s | avg e2e sec | TTFT p50 | TTFT p95 | e2e p50 | e2e p95 | avg decode sec | out tok/s serving | avg tool calls | pred executed | path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01_base_gemma4_31b_it_tp4_c32_n200 | 4 | 32 | default | auto/default | 72.00% | 144/200 | 1123.01 | 568.10 | 0.352 | 84.717 | 2.106 | 44.461 | 66.166 | 186.850 | 75.709 | 169.94 | 1.220 | 200/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/01_base_gemma4_31b_it_tp4_c32_n200` |
| 02_google_gemma4_31b_it_qat_w4a16_ct_tp4_c32_n200 | 4 | 32 | default | auto/default | 72.50% | 145/200 | 595.34 | 452.53 | 0.442 | 67.835 | 2.202 | 46.983 | 49.162 | 178.479 | 58.784 | 225.51 | 1.310 | 197/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/02_google_gemma4_31b_it_qat_w4a16_ct_tp4_c32_n200` |
| 03c_nvidia_gemma4_31b_it_nvfp4_nl2sql312_tp4_c32_n200 | 4 | 32 | default | modelopt | 72.00% | 144/200 | 386.36 | 324.11 | 0.617 | 45.536 | 1.590 | 33.081 | 37.335 | 96.029 | 39.373 | 302.28 | 1.230 | 199/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/03c_nvidia_gemma4_31b_it_nvfp4_nl2sql312_tp4_c32_n200` |
| 04_nvidia_gemma4_31b_it_nvfp4_nl2sql312_tp4_c16_n200 | 4 | 16 | default | modelopt | 72.50% | 145/200 | 413.89 | 351.72 | 0.569 | 26.323 | 1.409 | 9.750 | 20.600 | 49.122 | 23.593 | 279.41 | 1.245 | 197/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/04_nvidia_gemma4_31b_it_nvfp4_nl2sql312_tp4_c16_n200` |
| 05_gemma4_e4b_it_tp1_c32_n200 | 1 | 32 | default | auto/default | 61.50% | 123/200 | 423.48 | 199.95 | 1.000 | 29.574 | 0.615 | 14.941 | 22.647 | 74.089 | 26.869 | 899.91 | 1.510 | 194/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/05_gemma4_e4b_it_tp1_c32_n200` |
| 05_gemma4_e4b_it_tp2_c32_n200 | 2 | 32 | default | auto/default | 62.00% | 124/200 | 261.61 | 164.51 | 1.216 | 24.114 | 0.595 | 16.216 | 18.676 | 53.215 | 21.255 | 1057.51 | 1.425 | 194/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/05_gemma4_e4b_it_tp2_c32_n200` |
| 05_gemma4_e4b_it_tp4_c32_n200 | 4 | 32 | default | auto/default | 59.50% | 119/200 | 269.24 | 170.78 | 1.171 | 25.727 | 0.634 | 16.759 | 18.730 | 74.463 | 22.775 | 1045.92 | 1.505 | 193/200 | `outputs/latency_study/gemma4_31b_quant_tp4_c32_latency/runs/05_gemma4_e4b_it_tp4_c32_n200` |

## Tool-Call RL Study Status

The report at `outputs/latency_study/gemma4_31b_tool_rl_latency/latency_study.md` currently has no completed runs. It records the intended RL latency-study setup: async vLLM, temperature 0.0, first 200 examples, concurrency 16 for initial TP comparison, max prompt length 34000, max output tokens 8000, max model length 43500, and max tool rounds 8.

## NVFP4 Run Notes

- `03_nvidia_gemma4_31b_it_nvfp4_tp4_c32_n200` used `nvidia/Gemma-4-31B-IT-NVFP4` with TP=4, concurrency=32, `--vllm_quantization modelopt`, and vLLM `load_format=auto`. vLLM detected the checkpoint as ModelOpt NVFP4, but engine initialization failed while tying `lm_head` to `embed_tokens`: `model_executor/layers/quantization/base_config.py` raises `NotImplementedError` from `tie_weights`.
- `03b_nvidia_gemma4_31b_it_nvfp4_modelopt_loadformat_tp4_c32_n200` retried the same model with explicit `--vllm_load_format modelopt`. That retry failed earlier because vLLM 0.24 reports `Load format modelopt is not supported`.
- `03c_nvidia_gemma4_31b_it_nvfp4_nl2sql312_tp4_c32_n200` succeeded in `nl2sql312` after installing ModelOpt plus CUDA nvcc/dev headers and exposing the conda CUDA target include path. This indicates the vLLM 0.24 failure is version-specific, while the old-env failures were local CUDA JIT header/path setup issues.
- `04_nvidia_gemma4_31b_it_nvfp4_nl2sql312_tp4_c16_n200` repeats the successful NVFP4 setup with concurrency 16. It keeps accuracy at 145/200, lowers average and tail end-to-end latency versus concurrency 32, and reduces serving throughput as expected from the lower request load.
- The NVFP4 checkpoint config has `tie_word_embeddings=True` and the weight index contains `model.language_model.embed_tokens.weight` but no separate `lm_head` tensor. vLLM 0.24 currently fails at that tied-weight path; vLLM 0.19.1 loads it successfully.

## Practical Readout

- For `google/gemma-4-31B-it`, TP=4/concurrency=16 is a strong latency point among the initial TP sweep, while TP=8/concurrency=32 improves throughput but raises average user-visible latency.
- TP=8 with very high concurrency, especially 64 or 128, worsens per-request latency substantially.
- MTP with captured stats improves serving generation time versus the non-MTP batched baseline, but the gain is modest for this tool-call NL2SQL workload.
- NVIDIA NVFP4 gives the best TP=4/concurrency=32 throughput among the 31B variants tested while preserving similar 200-example accuracy.
- Gemma 4 E4B IT is much faster than 31B but loses roughly 10 points of 200-example BIRD EX accuracy in this setup.
