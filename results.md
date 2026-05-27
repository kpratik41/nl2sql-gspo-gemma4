# Results

## Maskfix Checkpoint Pass@K: Old Dev Schema Tool

These runs evaluate checkpoints from the `maskfix` GRPO training run on `outputs/old-dev-schema-tool.jsonl`. Each checkpoint used 1 GPU, 16 sampled generations per example over all 1534 old-dev samples, and 24544 candidates per checkpoint.

### Run Metadata

| setting | value |
| --- | --- |
| training run | `outputs/training/train-6601-schema-bare-tool/gemma-4-E4B-it/grpo_deepspeed_p15500_c8000_g16_t1p2_bs4_ga8_lr2e-6_e4b_bare_lr2e6_maskfix_20260526_044450` |
| checkpoints summarized | `0`, `20`, `40`, `60`, `80`, `100`, `120` |
| input file | `outputs/old-dev-schema-tool.jsonl` |
| database dir | `/home/ec2-user/nl2sql-gspo-gemma4/databases/dev_databases` |
| difficulty file | `data/bird_dev_data/raw/dev_20251106.json` |
| examples | `1534` |
| generations per example | `16` |
| candidates per checkpoint | `24544` |
| temperature | `1.2` |
| top_p | `1.0` |
| max_new_tokens | `8000` |
| max_tool_rounds | `8` |
| vLLM tensor parallel size | `1` |
| GPUs per checkpoint run | `1` |
| vLLM max model length | `45000` from run naming `ctx45k` |
| vLLM async concurrency | `16` |
| model path template | training run path plus `checkpoint-{step}` |

Output folders:

| checkpoint | output folder |
| ---: | --- |
| 0 | `outputs/passk/maskfix_ckpt-0_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |
| 20 | `outputs/passk/maskfix_ckpt-20_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |
| 40 | `outputs/passk/maskfix_ckpt-40_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |
| 60 | `outputs/passk/maskfix_ckpt-60_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |
| 80 | `outputs/passk/maskfix_ckpt-80_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |
| 100 | `outputs/passk/maskfix_ckpt-100_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |
| 120 | `outputs/passk/maskfix_ckpt-120_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k` |

Each output folder contains `passk_summary.json`, `passk_summary.md`, `passk_candidates_raw.jsonl`, `passk_candidates.jsonl`, and `passk_per_example.jsonl`.

### Pass@K And Self-Consistency Results

The table uses estimated pass@k from `pass_at_k_estimated`; pass@1 is the same as candidate accuracy. Training entropy and grad norm come from `wandb/run-20260526_044814-gstnzv5s/files/output.log`, using rollout/logged train step `0` as checkpoint `0` as requested; this corresponds to the first trainer progress record, displayed as step 1 in the progress bar.

| checkpoint | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | SC option 1 | SC option 2 | temp0 acc | correct candidates | candidate acc | train entropy | grad norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `63.64%` | `69.71%` | `73.79%` | `77.05%` | `79.79%` | `67.86%` | `67.99%` | `65.12%` | `15619 / 24544` | `63.64%` | `0.4026` | `0.005927` |
| 20 | `63.73%` | `70.13%` | `74.01%` | `76.89%` | `79.27%` | `68.77%` | `68.90%` | `65.97%` | `15641 / 24544` | `63.73%` | `0.2822` | `0.04908` |
| 40 | `62.04%` | `67.88%` | `71.52%` | `74.47%` | `76.92%` | `66.30%` | `66.43%` | `62.45%` | `15226 / 24544` | `62.04%` | `0.1049` | `0.1286` |
| 60 | `65.41%` | `70.09%` | `73.04%` | `75.25%` | `77.12%` | `68.58%` | `68.77%` | `66.82%` | `16054 / 24544` | `65.41%` | `0.1079` | `0.1908` |
| 80 | `64.85%` | `68.77%` | `71.69%` | `73.84%` | `75.42%` | `67.28%` | `67.28%` | `64.93%` | `15917 / 24544` | `64.85%` | `0.09862` | `0.2043` |
| 100 | `66.22%` | `70.13%` | `73.11%` | `75.40%` | `77.25%` | `67.80%` | `67.93%` | `66.17%` | `16253 / 24544` | `66.22%` | `0.1573` | `0.02835` |
| 120 | `65.40%` | `69.55%` | `72.80%` | `75.50%` | `77.84%` | `67.47%` | `67.80%` | `66.36%` | `16052 / 24544` | `65.40%` | `0.1665` | `0.0832` |

Entropy drops sharply from checkpoint `0` to `40`, then stays low with small rebounds at `100` and `120`. Accuracy does not improve monotonically: pass@1 bottoms at `40`, peaks at `100`, then gives some back at `120`; pass@16 is actually highest at `0` and remains below the starting point through `120`. This suggests the model is becoming less exploratory while the single-sample candidate accuracy gets a modest, noisy gain, and the marginal benefit from sampling is smaller after training.

Self-consistency was computed by executing the 16 sampled SQLs for each prompt and clustering them by execution result. Option 1 selects from the largest valid, non-empty execution-result cluster among the 16 sampled generations; ties are broken by the checkpoint's temperature-0 SQL when that SQL is also valid and non-empty. Option 2 adds the checkpoint's temperature-0 SQL as a 17th candidate before clustering, then selects from the largest valid, non-empty cluster. The full checkpoint sweep, `0` through `120` in increments of `20`, was rerun after patching the evaluator to never choose clusters whose SQL fails execution or executes to an empty result set.

Self-consistency artifacts are in `outputs/analysis/maskfix_self_consistency`.

### Candidate And Tool Stats

`total tool calls` and `avg calls / candidate` come from `passk_summary.json`. Per-tool counts are parsed from `passk_candidates_raw.jsonl`, so they reflect tool-call-looking names in the raw model text.

| checkpoint | examples | candidates | pred executed | pred failed | total tool calls | avg calls / candidate | `sqlite_query` | `sqlite_peek` | `bm25_search_sqlite` | total time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1534` | `24544` | `23370` | `1174` | `31969` | `1.303` | `29991` | `1102` | `868` | `5.44 h` |
| 20 | `1534` | `24544` | `23385` | `1159` | `35310` | `1.439` | `33036` | `1340` | `913` | `4.18 h` |
| 40 | `1534` | `24544` | `23130` | `1414` | `96882` | `3.947` | `80820` | `8673` | `7017` | `5.21 h` |
| 60 | `1534` | `24544` | `23969` | `575` | `72767` | `2.965` | `68081` | `2792` | `1883` | `5.30 h` |
| 80 | `1534` | `24544` | `23709` | `835` | `70463` | `2.871` | `65198` | `2923` | `2348` | `4.80 h` |
| 100 | `1534` | `24544` | `24195` | `349` | `60399` | `2.461` | `55448` | `2710` | `2238` | `4.74 h` |
| 120 | `1534` | `24544` | `24167` | `377` | `63513` | `2.588` | `57918` | `2657` | `2942` | `5.63 h` |

### Approx Runtime

These are approximate wall-clock runtimes from `timing_seconds`, with each checkpoint run using 1 GPU.

| checkpoint | generation | evaluation | total |
| ---: | ---: | ---: | ---: |
| 0 | `5.33 h` | `0.11 h` | `5.44 h` |
| 20 | `4.06 h` | `0.12 h` | `4.18 h` |
| 40 | `5.10 h` | `0.10 h` | `5.21 h` |
| 60 | `5.19 h` | `0.11 h` | `5.30 h` |
| 80 | `4.67 h` | `0.13 h` | `4.80 h` |
| 100 | `4.60 h` | `0.13 h` | `4.74 h` |
| 120 | `5.48 h` | `0.15 h` | `5.63 h` |

### Stop Reasons And Token Stats

| checkpoint | finished | max tool rounds | context length exceeded | max new tokens | avg completion tokens | max prompt tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `24508` | `24` | `9` | `3` | `798.6` | `29692` |
| 20 | `24483` | `57` | `3` | `1` | `588.8` | `29692` |
| 40 | `23694` | `837` | `10` | `3` | `696.3` | `29692` |
| 60 | `24300` | `229` | `3` | `12` | `736.0` | `29692` |
| 80 | `23798` | `724` | `6` | `16` | `648.8` | `29692` |
| 100 | `24301` | `227` | `6` | `10` | `658.1` | `29692` |
| 120 | `24240` | `280` | `4` | `20` | `793.6` | `29692` |

## Old Dev 1534 Inference Comparison: Gemma 4 31B vs E4B

These are single-generation async vLLM inference runs over all 1534 old-dev samples with `temperature=0.0`, `top_p=1.0`, `max_new_tokens=8000`, `max_tool_rounds=8`, `eval_workers=16`, and `vllm_async_concurrency=16`.

For `google/gemma-4-31B-it`, the two tool-format runs used tensor parallel size 4 with `vllm_max_model_len=43000`; the two consensus-format reruns used tensor parallel size 1 with `vllm_max_model_len=45000`. For `google/gemma-4-E4B-it`, all runs used tensor parallel size 1; `old-dev-schema-consensus` was rerun with `vllm_max_model_len=45000`.

| model | input | accuracy | correct / total | pred executed | pred failed | pred missing SQL | total time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma-4-31B-it` | `old-dev-schema-tool` | `71.19%` | `1092 / 1534` | `1518` | `16` | `16` | `33.2 min` |
| `gemma-4-31B-it` | `old-dev-schema-bare-tool` | `69.17%` | `1061 / 1534` | `1501` | `33` | `29` | `20.6 min` |
| `gemma-4-31B-it` | `old-dev-schema-consensus` | `66.49%` | `1020 / 1534` | `1425` | `109` | `109` | `372.9 min` |
| `gemma-4-31B-it` | `old-dev-schema-bare-consensus` | `64.02%` | `982 / 1534` | `1409` | `125` | `125` | `205.5 min` |
| `gemma-4-E4B-it` | `old-dev-schema-tool` | `64.86%` | `995 / 1534` | `1495` | `39` | `33` | `22.3 min` |
| `gemma-4-E4B-it` | `old-dev-schema-bare-tool` | `63.43%` | `973 / 1534` | `1499` | `35` | `26` | `15.5 min` |
| `gemma-4-E4B-it` | `old-dev-schema-consensus` | `64.60%` | `991 / 1534` | `1480` | `54` | `42` | `42.2 min` |
| `gemma-4-E4B-it` | `old-dev-schema-bare-consensus` | `61.67%` | `946 / 1534` | `1467` | `67` | `53` | `33.5 min` |

## Old Dev Pass@K Summary

These pass@k runs used `temperature=1.2`, `top_p=1.0`, `num_generations=16`, `max_new_tokens=8000`, `max_tool_rounds=8`, and async vLLM over all 1534 old-dev samples. Each full run generated `24544` candidates.

| model | input | samples | candidate acc / pass@1 | pass@8 | pass@16 | avg tool calls / candidate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gemma-4-31B-it` | `old-dev-schema-tool` | `1534` | `68.68%` | `73.73%` | `74.97%` | `2.21` |
| `gemma-4-31B-it` | `old-dev-schema-bare-tool` | `1534` | `66.17%` | `71.88%` | `73.14%` | `2.51` |
| `gemma-4-E4B-it` | `old-dev-schema-tool` | `1534` | `63.60%` | `76.84%` | `78.94%` | `1.30` |
| `gemma-4-E4B-it` | `old-dev-schema-bare-tool` | `1534` | `61.58%` | `74.81%` | `77.51%` | `1.34` |
| `gemma-4-E4B-it` | `old-dev-schema-consensus` | `1534` | `60.77%` | `76.45%` | `78.68%` | `2.40` |
| `gemma-4-E4B-it` | `old-dev-schema-bare-consensus` | `1534` | `58.66%` | `73.69%` | `76.34%` | `2.42` |

## Old Dev Tool Pass@K Full Runs: Gemma 4 31B, Async vLLM, Temperature 1.2

These two runs evaluate `google/gemma-4-31B-it` on the full old-dev split with 16 sampled tool-calling generations per example. The generation artifacts were scored by executing predicted SQL against the BIRD dev databases and comparing execution results with the gold SQL.

The first `old-dev-schema-tool` scoring pass initially produced an invalid all-zero summary because the relative `databases/dev_databases` path did not exist from this checkout. A `databases` symlink was added to the real database root, `/home/ec2-user/nl2sql-gspo-gemma4/databases`, and the run was re-scored from `passk_candidates_raw.jsonl` using the existing `scripts/run_passk_bird.py` post-generation scoring functions with `eval_workers=16`. The numbers below are the corrected summaries.

Result folders:

```text
outputs/passk/gemma4_31b_old-dev-schema-tool_full1534_temp1p2_tp4_ctx45k
outputs/passk/gemma4_31b_old-dev-schema-bare-tool_full1534_temp1p2_tp4_ctx45k
```

### Run Configuration

| setting | value |
| --- | --- |
| model | `google/gemma-4-31B-it` |
| input files | `outputs/old-dev-schema-tool.jsonl`, `outputs/old-dev-schema-bare-tool.jsonl` |
| database dir | `databases/dev_databases` via symlink to `/home/ec2-user/nl2sql-gspo-gemma4/databases/dev_databases` |
| difficulty file | `data/bird_dev_data/raw/dev_20251106.json` |
| examples | `1534` |
| generations per example | `16` |
| candidates per run | `24544` |
| temperature | `1.2` |
| top_p | `1.0` |
| max_prompt_length | `35000` |
| max_new_tokens | `8000` |
| max_tool_rounds | `8` |
| vLLM tensor parallel size | `4` |
| vLLM max model length | `45000` |
| vLLM GPU memory utilization | `0.90` |
| vLLM async concurrency | `16` |
| eval workers | `16` |

### Pass@K Results

| input | candidate acc / pass@1 | prefix pass@1 | estimated pass@8 | prefix pass@8 | estimated pass@16 | prefix pass@16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `68.68%` | `68.58%` | `73.73%` | `73.60%` | `74.97%` | `74.97%` |
| `old-dev-schema-bare-tool` | `66.17%` | `66.62%` | `71.88%` | `72.16%` | `73.14%` | `73.14%` |

### Candidate And Execution Stats

| input | correct candidates | pred executed | pred failed | stop: finished | stop: max tool rounds | stop: max new tokens | total time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `16858 / 24544` | `24277` | `267` | `24413` | `104` | `27` | `43292.0s` |
| `old-dev-schema-bare-tool` | `16240 / 24544` | `24190` | `354` | `24349` | `169` | `26` | `42880.7s` |

### Tool Usage

| input | total tool calls | avg calls / candidate | `sqlite_query` | `sqlite_peek` | `bm25_search_sqlite` | other parsed names |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `54185` | `2.21` | `51979` | `1123` | `1068` | `15` |
| `old-dev-schema-bare-tool` | `61610` | `2.51` | `57825` | `2432` | `1319` | `35` |

### Tool Round Distribution

| tool rounds / candidate | `old-dev-schema-tool` | `old-dev-schema-bare-tool` |
| ---: | ---: | ---: |
| 0 | `6` | `4` |
| 1 | `1875` | `1637` |
| 2 | `17712` | `13522` |
| 3 | `3837` | `6725` |
| 4 | `726` | `1699` |
| 5 | `155` | `398` |
| 6 | `59` | `188` |
| 7 | `50` | `128` |
| 8 | `124` | `243` |

Compared with the E4B pass@k runs below, `gemma-4-31B-it` has much higher candidate accuracy, but much smaller gain from additional samples. On `old-dev-schema-tool`, E4B improves from `63.60%` pass@1 to `78.94%` pass@16, a `+15.35 pp` lift; 31B improves from `68.68%` to `74.97%`, a `+6.28 pp` lift. On the bare-tool data, E4B improves by `+15.93 pp`, while 31B improves by `+6.98 pp`.

A simple candidate-diversity check supports this: 31B produced fewer distinct SQL strings per example despite more tool calls. On `old-dev-schema-tool`, 31B averaged `3.37` unique SQL strings per 16 candidates, versus E4B at `6.31`. On `old-dev-schema-bare-tool`, 31B averaged `3.74`, versus E4B at `6.89`. The 31B model also had more examples where all 16 candidates were correct (`880` tool, `813` bare-tool), but more examples with zero correct candidates (`384` tool, `412` bare-tool) than E4B (`323` tool, `345` bare-tool). This pattern is consistent with higher single-sample accuracy but more correlated samples: when 31B knows the answer, it often repeats a correct family; when it misses, extra samples less often escape the same failure mode.

## Old Dev Gemma 4 E4B Inference: Async vLLM, Temperature 0.0, 1534 Samples

These runs evaluate one greedy/tool-calling generation per example over the full old-dev split. All four runs used `google/gemma-4-E4B-it`, async vLLM, `temperature=0.0`, `top_p=1.0`, `max_new_tokens=8000`, `max_tool_rounds=8`, `vllm_tensor_parallel_size=1`, and `vllm_async_concurrency=16`.

The `old-dev-schema-consensus` run initially failed at `vllm_max_model_len=43000` because the tool-loop prompt reached at least `43001` tokens. It was rerun successfully with `vllm_max_model_len=45000` while overwriting the original output folder.

Result folders:

```text
outputs/inference/dev/old-dev-schema-tool/google-gemma-4-E4B-it/vllm_async_tp1_dp1_c16_ctx43k_p34k_o8k_r8_temp0
outputs/inference/dev/old-dev-schema-bare-tool/google-gemma-4-E4B-it/vllm_async_tp1_dp1_c16_ctx43k_p34k_o8k_r8_temp0
outputs/inference/dev/old-dev-schema-consensus/google-gemma-4-E4B-it/vllm_async_tp1_dp1_c16_ctx43k_p34k_o8k_r8_temp0
outputs/inference/dev/old-dev-schema-bare-consensus/google-gemma-4-E4B-it/vllm_async_tp1_dp1_c16_ctx43k_p34k_o8k_r8_temp0
```

### Main Results

| input | accuracy | correct / total | pred executed | pred failed | pred missing SQL | total time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `64.86%` | `995 / 1534` | `1495` | `39` | `33` | `22.3 min` |
| `old-dev-schema-bare-tool` | `63.43%` | `973 / 1534` | `1499` | `35` | `26` | `15.5 min` |
| `old-dev-schema-consensus` | `64.60%` | `991 / 1534` | `1480` | `54` | `42` | `42.2 min` |
| `old-dev-schema-bare-consensus` | `61.67%` | `946 / 1534` | `1467` | `67` | `53` | `33.5 min` |

### By Difficulty

| input | simple | moderate | challenging |
| --- | ---: | ---: | ---: |
| `old-dev-schema-tool` | `64.07%` | `63.66%` | `70.13%` |
| `old-dev-schema-bare-tool` | `62.79%` | `63.21%` | `66.23%` |
| `old-dev-schema-consensus` | `63.26%` | `65.01%` | `68.83%` |
| `old-dev-schema-bare-consensus` | `60.47%` | `63.21%` | `63.20%` |

### Tool Usage

| input | avg calls / sample | `sqlite_query` | `sqlite_peek` | `bm25_search_sqlite` | `consensus_at_1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `1.289` | `1866` | `68` | `45` | `0` |
| `old-dev-schema-bare-tool` | `1.332` | `1940` | `59` | `44` | `0` |
| `old-dev-schema-consensus` | `2.734` | `1631` | `191` | `77` | `2297` |
| `old-dev-schema-bare-consensus` | `2.779` | `1635` | `262` | `85` | `2288` |

### Stop Reasons

| input | finished | max tool rounds | max new tokens |
| --- | ---: | ---: | ---: |
| `old-dev-schema-tool` | `1523` | `10` | `1` |
| `old-dev-schema-bare-tool` | `1525` | `9` | `0` |
| `old-dev-schema-consensus` | `1508` | `20` | `6` |
| `old-dev-schema-bare-consensus` | `1504` | `19` | `11` |

## Old Dev Tool Pass@K Smoke Runs: Gemma 4 31B, Async vLLM, Temperature 1.2, 50 Samples

These two runs use the updated Gemma-native tool-loop stopping behavior in `scripts/run_inference_bird.py`: stop token ids for `<tool_call|>`, `<|tool_response>`, `<turn|>`, and `<eos>`, plus a defensive fallback that keeps only the first parsed tool call in an assistant turn before executing the tool. Each run evaluates the first 50 examples with 16 generations per example, for 800 candidates per run.

Result folders:

```text
outputs/passk/old-dev-schema-tool_stopids_limit50_temp1p2
outputs/passk/old-dev-schema-bare-tool_stopids_limit50_temp1p2
```

### Run Configuration

| setting | value |
| --- | --- |
| model | `google/gemma-4-31B-it` |
| examples | `50` |
| generations per example | `16` |
| total candidates per run | `800` |
| temperature | `1.2` |
| top_p | `1.0` |
| max_new_tokens | `8000` |
| max_tool_rounds | `8` |
| vLLM tensor parallel size | `4` |
| vLLM async concurrency | `16` |

### Pass@K

| run | pass@1 | pass@2 | pass@3 | pass@4 | pass@5 | pass@8 | pass@12 | pass@16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `68.00%` | `69.80%` | `70.59%` | `71.09%` | `71.43%` | `71.90%` | `72.00%` | `72.00%` |
| `old-dev-schema-bare-tool` | `70.50%` | `72.72%` | `73.64%` | `74.20%` | `74.66%` | `75.80%` | `76.99%` | `78.00%` |

### Candidate And Tool Stats

| run | correct candidates | candidate accuracy | pred executed | pred failed | stop: finished | stop: max_tool_rounds | total tool calls | avg tool calls / generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `old-dev-schema-tool` | `544 / 800` | `68.00%` | `783` | `17` | `783` | `17` | `1105` | `1.381` |
| `old-dev-schema-bare-tool` | `564 / 800` | `70.50%` | `788` | `12` | `788` | `12` | `1099` | `1.374` |

### Tool Call Distribution

| run | `sqlite_query` | `sqlite_peek` | `bm25_search_sqlite` |
| --- | ---: | ---: | ---: |
| `old-dev-schema-tool` | `1046` | `38` | `21` |
| `old-dev-schema-bare-tool` | `1049` | `29` | `21` |

### Tool Round Distribution

| tool rounds / generation | `old-dev-schema-tool` | `old-dev-schema-bare-tool` |
| ---: | ---: | ---: |
| 1 | `679` | `666` |
| 2 | `65` | `78` |
| 3 | `23` | `21` |
| 4 | `8` | `11` |
| 5 | `1` | `7` |
| 6 | `1` | `0` |
| 7 | `0` | `1` |
| 8 | `23` | `16` |

### Tool-Loop Sanity Checks

| check | `old-dev-schema-tool` | `old-dev-schema-bare-tool` |
| --- | ---: | ---: |
| multiple tool calls before first tool response | `0` | `0` |
| multiple tool calls between inserted tool responses | `0` | `0` |
| continuation markers after call before response | `0` | `0` |
| tool calls without response | `0` | `0` |
| parse/tool error rows | `0` | `0` |
| `<|im_end|>` leaked into output | `0` | `0` |
| `<|turn|>` leaked into output | `0` | `0` |
| `<tool_call|>` / `<|tool_call>` leaked into output | `0` | `0` |
| `call:` count equals `<|tool_response>` count | `1105 = 1105` | `1099 = 1099` |

The generated transcripts show the desired tool workflow: one assistant tool call, executor-inserted tool response, then resumed model generation. This is strong output-level evidence that the hallucinated multi-tool-call-before-response issue is resolved for these runs. Direct vLLM finish/stop telemetry is not persisted in these outputs, so proving the exact raw vLLM stop reason would require adding explicit debug logging around `AsyncLLMEngine.generate`.

## BIRD Dev Schema-Tool Pass@K: Gemma 4 31B, Async vLLM, Temperature 0.8

Result folder:

```text
outputs/bird_dev_schema_tool_passk16_vllm_async_temp08_limit1534-0514
```

This run evaluated pass@k on the full BIRD dev set by sampling 16 independent tool-calling completions per example, executing each predicted SQL against the dev databases, and comparing execution results with the gold SQL.

### Run Configuration

| setting | value |
| --- | --- |
| model | `google/gemma-4-31B-it` |
| input file | `outputs/dev-20251106-schema-tool.jsonl` |
| database dir | `databases/dev_databases` |
| difficulty file | `data/bird_dev_data/raw/dev_20251106.json` |
| examples | `1534` |
| generations per example | `16` |
| total candidates | `24544` |
| temperature | `0.8` |
| top_p | `1.0` |
| max_new_tokens | `8000` |
| max_tool_rounds | `8` |
| vLLM tensor parallel size | `8` |
| vLLM async concurrency | `16` |

### Output Artifacts

| file | description |
| --- | --- |
| `passk_candidates_raw.jsonl` | raw generated candidates before SQL execution scoring |
| `passk_candidates.jsonl` | generated candidates with execution status, gold SQL, and correctness |
| `passk_per_example.jsonl` | per-question aggregate over the 16 candidates |
| `passk_summary.json` | machine-readable pass@k summary |
| `passk_summary.md` | human-readable pass@k summary |
| `passk_diagnostics.json` | detailed diagnostics over candidates, tool calls, tokens, and DBs |
| `passk_diagnostics.md` | compact diagnostics report |
| `all_wrong_analysis.jsonl` | per-example analysis for examples with zero correct candidates among 16 |
| `all_wrong_summary.md` | summary of the all-wrong examples |
| `idx*_candidates.jsonl` | extracted candidate subsets for selected failure-example IDs |

### Summary Metrics

| metric | value |
| --- | ---: |
| candidate accuracy | `69.76%` |
| correct candidates | `17121 / 24544` |
| estimated pass@1 | `69.76%` |
| prefix pass@1 | `70.21%` |
| estimated pass@16 | `75.68%` |
| prefix pass@16 | `75.68%` |
| examples with any correct candidate | `1161 / 1534` |
| examples with zero correct candidates | `373 / 1534` |
| examples with all 16 candidates correct | `919 / 1534` |
| predicted SQL executed | `24335 / 24544` |
| predicted SQL execution rate | `99.15%` |
| gold SQL executed | `24496 / 24544` |
| gold SQL execution rate | `99.80%` |
| total tool calls | `34663` |
| average tool calls per candidate | `1.41` |
| average tool rounds per candidate | `1.23` |

### Pass@K Table

| k | estimated pass@k | prefix pass@k |
| ---: | ---: | ---: |
| 1 | 69.76 | 70.21 |
| 2 | 72.07 | 72.43 |
| 3 | 73.00 | 73.27 |
| 4 | 73.55 | 73.60 |
| 5 | 73.93 | 73.73 |
| 6 | 74.23 | 74.12 |
| 7 | 74.47 | 74.38 |
| 8 | 74.67 | 74.51 |
| 9 | 74.84 | 74.64 |
| 10 | 75.00 | 74.84 |
| 11 | 75.14 | 74.97 |
| 12 | 75.27 | 75.16 |
| 13 | 75.38 | 75.36 |
| 14 | 75.49 | 75.49 |
| 15 | 75.59 | 75.62 |
| 16 | 75.68 | 75.68 |

### Candidate Execution And Stop Reasons

| item | count |
| --- | ---: |
| pred executed | `24335` |
| pred failed | `209` |
| gold executed | `24496` |
| gold failed | `48` |
| generation errors | `1` |
| stop: finished | `24303` |
| stop: max_tool_rounds | `213` |
| stop: max_new_tokens | `27` |
| stop: generation_error | `1` |

### Tool Usage

| metric | value |
| --- | ---: |
| total tool calls | `34663` |
| avg tool calls per candidate | `1.41` |
| total tool rounds | `30090` |
| avg tool rounds per candidate | `1.23` |
| per-example tool-call min | `16` |
| per-example tool-call p50 | `16` |
| per-example tool-call p90 | `34` |
| per-example tool-call p95 | `54` |
| per-example tool-call p99 | `113` |
| per-example tool-call max | `263` |

Most common tool orders:

| tool order | count |
| --- | ---: |
| `sqlite_query` | `21307` |
| `sqlite_query -> sqlite_query` | `1373` |
| `sqlite_query -> sqlite_query -> sqlite_query` | `269` |
| `sqlite_query -> bm25_search_sqlite -> sqlite_query` | `213` |
| `sqlite_query -> sqlite_query -> sqlite_query -> sqlite_query` | `140` |
| `sqlite_query -> bm25_search_sqlite -> sqlite_query -> sqlite_query` | `76` |
| `sqlite_query -> sqlite_peek -> sqlite_query` | `65` |
| `bm25_search_sqlite -> sqlite_query` | `59` |

### Token Statistics

| metric | prompt tokens | completion tokens |
| --- | ---: | ---: |
| min | `5878` | `0` |
| p50 | `13707` | `492` |
| p90 | `21067` | `1215` |
| p95 | `28920` | `1846` |
| p99 | `29172` | `3248` |
| max | `29464` | `8000` |
| mean | `14410.84` | `659.16` |

### Correct Candidates Per Example

| correct candidates among 16 | examples |
| ---: | ---: |
| 0 | `373` |
| 1 | `23` |
| 2 | `12` |
| 3 | `8` |
| 4 | `10` |
| 5 | `6` |
| 6 | `7` |
| 7 | `7` |
| 8 | `8` |
| 9 | `10` |
| 10 | `14` |
| 11 | `14` |
| 12 | `17` |
| 13 | `17` |
| 14 | `23` |
| 15 | `66` |
| 16 | `919` |

### DB-Level Diagnostics

| db_id | examples | candidate accuracy | pass@16 | pred execution rate | avg tool calls/candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `card_games` | 191 | 68.52 | 75.39 | 98.85 | 1.63 |
| `codebase_community` | 186 | 78.66 | 86.56 | 98.89 | 1.56 |
| `formula_1` | 174 | 70.01 | 75.29 | 99.03 | 1.38 |
| `thrombosis_prediction` | 163 | 67.98 | 75.46 | 99.04 | 1.40 |
| `student_club` | 158 | 83.39 | 87.97 | 99.53 | 1.40 |
| `toxicology` | 145 | 64.96 | 70.34 | 99.40 | 1.20 |
| `european_football_2` | 129 | 75.00 | 82.17 | 99.61 | 1.16 |
| `superhero` | 129 | 89.63 | 92.25 | 99.85 | 1.37 |
| `financial` | 106 | 31.25 | 35.85 | 99.35 | 1.33 |
| `california_schools` | 89 | 45.58 | 51.69 | 98.46 | 1.78 |
| `debit_card_specializing` | 64 | 75.39 | 81.25 | 98.14 | 1.20 |

Lowest-performing DBs by pass@16 were `financial` at `35.85%` and `california_schools` at `51.69%`. Highest-performing DBs were `superhero` at `92.25%`, `student_club` at `87.97%`, and `codebase_community` at `86.56%`.

### Prediction Error Themes

Top predicted-SQL execution errors:

| error | count |
| --- | ---: |
| `near "I": syntax error` | `105` |
| `near ";": syntax error` | `16` |
| `interrupted` | `13` |
| `near "call": syntax error` | `9` |
| `no such column: T1.UserId` | `9` |
| `ORDER BY clause should come after UNION ALL not before` | `9` |
| `You can only execute one statement at a time.` | `7` |
| `near "in": syntax error` | `6` |
| `near "convertedManaCost": syntax error` | `6` |
| `no such column: lt` | `4` |

### All-Wrong Analysis

`all_wrong_summary.md` analyzes examples where none of the 16 sampled candidates were correct.

| label | count |
| --- | ---: |
| all-wrong examples | `373` |
| all_pred_sql_executed_but_wrong | `348` |
| low_diversity_repeated_wrong_sql | `133` |
| high_diversity_no_correct_sql | `120` |
| some_pred_sql_execution_failed | `25` |
| sql_syntax_or_transcript_extraction_error | `16` |
| hit_max_tool_rounds | `12` |
| hit_max_new_tokens | `4` |
| gold_sql_execution_failed | `3` |
| schema_column_error | `2` |
| generation_error | `1` |

All-wrong examples by DB:

| db_id | all-wrong examples |
| --- | ---: |
| `financial` | `68` |
| `card_games` | `47` |
| `california_schools` | `43` |
| `formula_1` | `43` |
| `toxicology` | `43` |
| `thrombosis_prediction` | `40` |
| `codebase_community` | `25` |
| `european_football_2` | `23` |
| `student_club` | `19` |
| `debit_card_specializing` | `12` |
| `superhero` | `10` |

The all-wrong analysis also wrote a filtered source file reference:

```text
outputs/dev-20251106-schema-373.jsonl
```

### Timing

| phase | seconds | approximate |
| --- | ---: | ---: |
| generation | `25421.76` | `7.06 h` |
| evaluation | `508.17` | `8.47 min` |
| total | `25932.12` | `7.20 h` |

### Notes

- `candidate_accuracy` treats every sampled candidate independently.
- `estimated pass@k` uses the standard combinatorial pass@k estimate from the 16 sampled candidates for each example.
- `prefix pass@k` checks whether at least one of the first `k` candidates by sample id was correct.
- At `k=16`, estimated and prefix pass@k match because both use all 16 candidates.
- This run used the full `schema-tool` dev file, not the `schema-bare-tool` file. The prompt-token distribution confirms much larger prompts: p50 prompt length was `13707` tokens and p95 was `28920` tokens.
- The gap between candidate accuracy (`69.76%`) and pass@16 (`75.68%`) shows about `5.93` points of recoverable headroom if a reranker or verifier can select the correct candidate from the sampled set.

## GRPO E4B Bare-Tool Checkpoint Sweep: Checkpoints 0-150

This sweep evaluates one async vLLM tool-calling generation per example over all 1534 old-dev samples for checkpoints `0, 10, ..., 150` from the GRPO E4B bare-tool training run.

Result root:

```text
outputs/training/train-6601-schema-bare-tool/gemma-4-E4B-it/grpo_deepspeed_p15500_c8000_g16_t1p2_bs4_ga8_lr2e-6_e4b_bare_lr2e6_20260524_130108
```

Each checkpoint folder contains `eval_summary.json`, `eval_summary.md`, `run_report.md`, `per_example_report.csv`, `eval_results.jsonl`, `prediction_details.jsonl`, and `predict_dev.json`.

### Run Configuration

| setting | value |
| --- | --- |
| base checkpoint-0 model | `google/gemma-4-E4B-it` |
| input file | `outputs/old-dev-schema-tool.jsonl` |
| examples | `1534` |
| inference backend | `vllm_async` |
| max prompt length | `35000` |
| max new tokens | `8000` |
| max tool rounds | `8` |
| eval workers | `16` |
| vLLM tensor parallel size | `1` |
| vLLM data parallel size | `1` |
| vLLM async concurrency | `16` |
| vLLM max model length | `48000` |

### Checkpoint Summary

| ckpt | correct | EX acc | finished | max rounds | tool calls | `sqlite_query` | `sqlite_peek` | `bm25_search_sqlite` | avg calls | avg toks | pred missing | pred failed | both exec | total sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `1003` | `65.38%` | `1526` | `8` | `2005` | `1899` | `65` | `41` | `1.307` | `710.0` | `22` | `25` | `1508` | `2644.8` |
| `10` | `996` | `64.93%` | `1528` | `6` | `2024` | `1918` | `58` | `48` | `1.319` | `732.8` | `18` | `22` | `1511` | `2117.6` |
| `20` | `985` | `64.21%` | `1522` | `12` | `2046` | `1920` | `78` | `48` | `1.334` | `735.1` | `26` | `31` | `1502` | `2047.6` |
| `30` | `995` | `64.86%` | `1526` | `7` | `2024` | `1939` | `56` | `29` | `1.319` | `732.2` | `22` | `27` | `1506` | `2137.4` |
| `40` | `1003` | `65.38%` | `1529` | `5` | `1991` | `1906` | `54` | `31` | `1.298` | `722.6` | `22` | `25` | `1508` | `2095.6` |
| `50` | `996` | `64.93%` | `1527` | `6` | `1995` | `1914` | `54` | `27` | `1.301` | `730.5` | `25` | `33` | `1500` | `2039.0` |
| `60` | `989` | `64.47%` | `1527` | `7` | `1995` | `1906` | `63` | `26` | `1.301` | `730.0` | `21` | `28` | `1505` | `2045.8` |
| `70` | `992` | `64.67%` | `1528` | `6` | `1979` | `1898` | `52` | `29` | `1.290` | `728.7` | `25` | `30` | `1503` | `2125.2` |
| `80` | `998` | `65.06%` | `1527` | `7` | `2039` | `1943` | `63` | `33` | `1.329` | `751.6` | `23` | `28` | `1505` | `2085.2` |
| `90` | `1007` | `65.65%` | `1531` | `3` | `1966` | `1893` | `56` | `17` | `1.282` | `741.8` | `19` | `24` | `1509` | `2061.7` |
| `100` | `1000` | `65.19%` | `1527` | `6` | `2013` | `1913` | `68` | `32` | `1.312` | `753.0` | `18` | `28` | `1505` | `2097.1` |
| `110` | `999` | `65.12%` | `1531` | `3` | `1958` | `1881` | `51` | `26` | `1.276` | `744.1` | `24` | `33` | `1500` | `2074.9` |
| `120` | `988` | `64.41%` | `1531` | `3` | `2005` | `1917` | `63` | `25` | `1.307` | `762.1` | `15` | `23` | `1510` | `2134.5` |
| `130` | `998` | `65.06%` | `1529` | `5` | `1989` | `1899` | `60` | `30` | `1.297` | `747.9` | `20` | `27` | `1506` | `2122.8` |
| `140` | `989` | `64.47%` | `1526` | `7` | `2019` | `1935` | `59` | `25` | `1.316` | `768.0` | `23` | `31` | `1502` | `2136.8` |
| `150` | `992` | `64.67%` | `1526` | `8` | `2004` | `1936` | `48` | `20` | `1.306` | `762.0` | `18` | `24` | `1509` | `2125.8` |

### Run Health

- No checkpoint run appears to have failed globally: every checkpoint generated all `1534` examples and had `0` filtered examples.
- Generation stop reasons were only `finished` and `max_tool_rounds`; no `max_new_tokens` or other stop reason appeared in these checkpoint summaries.
- `checkpoint-90` had the best overall EX accuracy at `65.65%` (`1007 / 1534`) and only `3` max-tool-round stops.
- `checkpoint-20` was the weakest checkpoint by accuracy at `64.21%` (`985 / 1534`), with the most tool calls (`2046`), most max-tool-round stops (`12`), and high predicted-SQL failure count (`31`).
- `checkpoint-120` had the best SQL extraction/execution health by missing predicted SQL (`15`) and both-SQL-executed count (`1510`), but its EX accuracy was only `64.41%`.
- `checkpoint-140` was the most verbose by average completion tokens (`768.0`) without an accuracy gain.
- `checkpoint-150` finished below checkpoint 0 overall: `64.67%` vs `65.38%`, despite higher average completion tokens (`762.0` vs `710.0`).

### Error Themes

The non-OK per-example statuses are dominated by predicted-SQL issues rather than generation-level run failures:

- `pred_error: empty sql` was the most common error at every checkpoint, ranging from `15` at checkpoint 120 to `26` at checkpoint 20.
- Other recurring predicted-SQL failures included missing columns or tables, syntax errors, and a few interrupted predicted-SQL executions.
- A recurring `gold_error: interrupted` appears once in every checkpoint run.

### Recommendation

Use `checkpoint-90` as the preferred checkpoint from this sweep. It has the best overall EX accuracy, strong run health, low max-tool-round count, and relatively low tool-call volume. The final checkpoint, `checkpoint-150`, is not better than checkpoint 0 or checkpoint 90 on overall EX accuracy.
