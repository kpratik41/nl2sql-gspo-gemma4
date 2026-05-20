# Results

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
