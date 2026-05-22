# Results

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
