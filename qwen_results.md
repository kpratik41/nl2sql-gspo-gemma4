# Qwen Results

## Qwen3.6-35B-A3B - BIRD Dev Temp 0

Run:

- Model: `Qwen/Qwen3.6-35B-A3B`
- Data: `outputs/old-dev-schema-tool-unpatched.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-35B-A3B/full1534_tp2_shards4_temp0_openai_tool_qwen_native_required_retryfix`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall BIRD EX:

- Correct: `1028 / 1534`
- Accuracy: `67.01%`

SQL execution:

- Pred SQL extracted: `1528 / 1534`
- Pred SQL missing: `6`
- Pred SQL executed: `1525 / 1534`
- Extracted SQL execution failures: `3`
- Both pred and gold executed: `1524`

Tool usage:

- Total tool calls: `3958`
- Avg tool calls overall: `2.58`
- Avg tool calls on EX-correct examples: `2.22`
- Avg tool calls on EX-incorrect examples: `3.30`
- Tool counts: `sqlite_query=2975`, `sqlite_peek=509`, `bm25_search_sqlite=474`
- Rejected parsed tool calls: `110`, all parsed as invalid `scratch_pad`
- Forced-final examples: `86`
- Empty-tool retries: `271`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 383 | 251 | 65.54% | 2.50 |
| `shard_1` | 384 | 263 | 68.49% | 2.47 |
| `shard_2` | 383 | 261 | 68.15% | 2.71 |
| `shard_3` | 384 | 253 | 65.89% | 2.65 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 672 | 925 | 72.65% |
| Moderate | 273 | 464 | 58.84% |
| Challenging | 83 | 145 | 57.24% |

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `california_schools` | 57 | 89 | 64.04% |
| `card_games` | 116 | 191 | 60.73% |
| `codebase_community` | 133 | 186 | 71.51% |
| `debit_card_specializing` | 42 | 64 | 65.62% |
| `european_football_2` | 88 | 129 | 68.22% |
| `financial` | 69 | 106 | 65.09% |
| `formula_1` | 105 | 174 | 60.34% |
| `student_club` | 130 | 158 | 82.28% |
| `superhero` | 112 | 129 | 86.82% |
| `thrombosis_prediction` | 81 | 163 | 49.69% |
| `toxicology` | 95 | 145 | 65.52% |

## Qwen3.6-27B Dense - BIRD Dev Temp 0

Run:

- Model: `Qwen/Qwen3.6-27B`
- Data: `outputs/old-dev-schema-tool-unpatched.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-27B/full1534_tp2_shards4_temp0_openai_tool_qwen_native_required_retryfix`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall BIRD EX:

- Correct: `1063 / 1534`
- Accuracy: `69.30%`

SQL execution:

- Pred SQL extracted: `1525 / 1534`
- Pred SQL missing: `9`
- Pred SQL executed: `1524 / 1534`
- Extracted SQL execution failures: `1`
- Total missing or execution-failed pred SQL: `10`
- Both pred and gold executed: `1523`

Tool usage:

- Total tool calls: `4438`
- Avg tool calls overall: `2.89`
- Avg tool calls on EX-correct examples: `2.55`
- Avg tool calls on EX-incorrect examples: `3.67`
- Avg tool calls where pred SQL executed: `2.87`
- Avg tool calls where pred SQL was missing or did not execute: `6.30`
- Tool counts: `sqlite_query=2707`, `bm25_search_sqlite=1292`, `sqlite_peek=439`
- Rejected parsed tool calls: `3`
- Forced-final examples: `71`
- Empty-tool retries: `206`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 383 | 260 | 67.89% | 2.76 |
| `shard_1` | 384 | 267 | 69.53% | 2.84 |
| `shard_2` | 383 | 275 | 71.80% | 2.86 |
| `shard_3` | 384 | 261 | 67.97% | 3.12 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 683 | 925 | 73.84% |
| Moderate | 296 | 464 | 63.79% |
| Challenging | 84 | 145 | 57.93% |

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `california_schools` | 56 | 89 | 62.92% |
| `card_games` | 122 | 191 | 63.87% |
| `codebase_community` | 129 | 186 | 69.35% |
| `debit_card_specializing` | 41 | 64 | 64.06% |
| `european_football_2` | 91 | 129 | 70.54% |
| `financial` | 72 | 106 | 67.92% |
| `formula_1` | 111 | 174 | 63.79% |
| `student_club` | 131 | 158 | 82.91% |
| `superhero` | 118 | 129 | 91.47% |
| `thrombosis_prediction` | 90 | 163 | 55.21% |
| `toxicology` | 102 | 145 | 70.34% |

## California Schools Comparison

### Qwen3.6-27B Dense - California Schools Temp 0

Run:

- Model: `Qwen/Qwen3.6-27B`
- Data: `outputs/old-dev-schema-tool-unpatched-california_schools.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-27B/california_schools_tp2_shards4_temp0_openai_tool_qwen_native_required_retryfix`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall California-schools EX:

- Correct: `56 / 89`
- Accuracy: `62.92%`

SQL execution:

- Pred SQL extracted: `88 / 89`
- Pred SQL missing: `1`
- Pred SQL executed: `88 / 89`
- Extracted SQL execution failures: `0`

Tool usage:

- Total tool calls: `331`
- Avg tool calls overall: `3.72`
- Avg tool calls on EX-correct examples: `3.14`
- Avg tool calls on EX-incorrect examples: `4.70`
- Tool counts: `sqlite_query=211`, `bm25_search_sqlite=79`, `sqlite_peek=41`
- Rejected parsed tool calls: `0`
- Forced-final examples: `8`
- Empty-tool retries: `2`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 22 | 18 | 81.82% | 3.14 |
| `shard_1` | 22 | 9 | 40.91% | 4.59 |
| `shard_2` | 22 | 14 | 63.64% | 3.45 |
| `shard_3` | 23 | 15 | 65.22% | 3.70 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 37 | 54 | 68.52% |
| Moderate | 18 | 30 | 60.00% |
| Challenging | 1 | 5 | 20.00% |

### 35B MoE vs 27B Dense on California Schools

| Model | Correct | Rows | Accuracy | Avg tools | Avg tools on correct | Pred SQL executed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.6-35B-A3B` | 57 | 89 | 64.04% | 3.28 | 2.82 | 89 / 89 |
| `Qwen3.6-27B` | 56 | 89 | 62.92% | 3.72 | 3.14 | 88 / 89 |

Takeaway: on `california_schools`, the 27B dense model is within `1` example of the 35B-A3B MoE result, but uses more tool calls on average and has one missing pred SQL.
