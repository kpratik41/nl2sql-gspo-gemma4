# Thinking Machine Eval

## Eval datasets used with `google/gemma-4-31B-it`

Three BIRD-style tool-calling eval files, all run with `scripts/run_inference_bird_async_sharded.py`, `TP=2`, `SHARDS=4`, `temperature=0.0`, `top_p=1.0`, `max_tool_rounds=8`:

| # | Input file | Rows | Diff JSON (`--diff_json_path`) | Has `difficulty`? | Status |
| --- | --- | ---: | --- | --- | --- |
| 1 | `outputs/arcwise_plat_sql-schema-tool.jsonl` | 498 | `data/revisql/raw/arcwise_plat_sql.json` | No (all "unknown") | Done |
| 2 | `outputs/arcwise_plat_full-schema-tool.jsonl` | 498 | `data/revisql/raw/arcwise_plat_full.json` | No (all "unknown") | Done |
| 3 | `outputs/mini_dev_sqlite-schema-tool.jsonl` | 500 | `data/bird_minidev_data/raw/mini_dev_sqlite.json` | Yes | Done |

## Summary across all 3 datasets

| Dataset | Rows | Correct | Overall EX |
| --- | ---: | ---: | ---: |
| `arcwise_plat_sql` | 498 | 427 | 85.74% |
| `arcwise_plat_full` | 498 | 443 | 88.96% |
| `mini_dev_sqlite` (official BIRD mini-dev) | 500 | 357 | 71.40% |

By database (accuracy %, blank = db not present in that dataset):

| Database | `arcwise_plat_sql` | `arcwise_plat_full` | `mini_dev_sqlite` |
| --- | ---: | ---: | ---: |
| `student_club` | 97.92 | 95.83 | 87.50 |
| `card_games` | 90.38 | 88.46 | 71.15 |
| `formula_1` | 87.88 | 92.42 | 71.21 |
| `codebase_community` | 87.76 | 91.84 | 69.39 |
| `superhero` | 86.54 | 90.38 | 84.62 |
| `european_football_2` | 86.27 | 92.16 | 72.55 |
| `toxicology` | 85.00 | 90.00 | 65.00 |
| `financial` | 83.33 | 86.67 | 59.38 |
| `thrombosis_prediction` | 82.00 | 90.00 | 64.00 |
| `debit_card_specializing` | 80.00 | 83.33 | 70.00 |
| `california_schools` | 63.33 | 63.33 | 60.00 |

Notes:

- `mini_dev_sqlite` is markedly harder for the model across almost every DB than the arcwise sets, despite overlapping `db_id` coverage — consistent with it being the official/independent BIRD mini-dev split rather than the arcwise-generated variants.
- `california_schools` is the weakest DB in all three datasets (60-63% range).
- `arcwise_plat_full` outperforms `arcwise_plat_sql` on 9 of 11 shared DBs; only `california_schools` (tied) and `card_games`/`student_club` (`arcwise_plat_sql` slightly higher) buck that trend.

## google/gemma-4-31B-it - arcwise_plat_sql BIRD EX

Run:

- Model: `google/gemma-4-31B-it`
- Data: `outputs/arcwise_plat_sql-schema-tool.jsonl` (498 rows)
- Diff JSON: `data/revisql/raw/arcwise_plat_sql.json` (explicit `--diff_json_path`; no `difficulty` field, so difficulty breakdown is all "unknown")
- Output: `outputs/inference/arcwise_plat_sql/gemma-4-31B-it/vllm_async_tp2_dp4_ctx43k_p34k_o8k_r8_20260828_031400`
- Inference: async vLLM sharded (`scripts/run_inference_bird_async_sharded.py`), `TP=2`, `SHARDS=4`, `temperature=0.0`, `top_p=1.0`
- Tool calling: `max_tool_rounds=8`

Overall BIRD EX:

- **Correct: 427 / 498**
- **Accuracy: 85.74%**

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `student_club` | 47 | 48 | 97.92% |
| `card_games` | 47 | 52 | 90.38% |
| `formula_1` | 58 | 66 | 87.88% |
| `codebase_community` | 43 | 49 | 87.76% |
| `superhero` | 45 | 52 | 86.54% |
| `european_football_2` | 44 | 51 | 86.27% |
| `toxicology` | 34 | 40 | 85.00% |
| `financial` | 25 | 30 | 83.33% |
| `thrombosis_prediction` | 41 | 50 | 82.00% |
| `debit_card_specializing` | 24 | 30 | 80.00% |
| `california_schools` | 19 | 30 | 63.33% |

Weakest DB: `california_schools` at 63.33% (19/30).

SQL execution:

- Pred SQL extracted: `497 / 498`, missing `1`
- Gold SQL extracted: `498 / 498`, missing `0`
- Pred SQL executed: `496 / 498`, execution failures `2`
- Gold SQL executed: `497 / 498`, execution failures `1`
- Both pred and gold executed: `495`

Generation stats:

- Tool calls total: `555` (avg `1.114`/example)
- Tool counts: `sqlite_query=552`, `sqlite_peek=1`, `bm25_search_sqlite=2`
- Stop reasons: `finished=497`, `max_tool_rounds=1`
- Completion tokens total: `221978`, avg `445.74`/example
- Max prompt tokens: `27228`

Timing:

- Generation: `711.00s`
- Evaluation: `63.37s`
- Total: `774.88s`

## google/gemma-4-31B-it - arcwise_plat_full BIRD EX

Run:

- Model: `google/gemma-4-31B-it`
- Data: `outputs/arcwise_plat_full-schema-tool.jsonl` (498 rows)
- Diff JSON: `data/revisql/raw/arcwise_plat_full.json` (explicit `--diff_json_path`; no `difficulty` field, so difficulty breakdown is all "unknown")
- Output: `outputs/inference/arcwise_plat_full/gemma-4-31B-it/vllm_async_tp2_dp4_ctx43k_p34k_o8k_r8_temp0_20260828_043843`
- Inference: async vLLM sharded (`scripts/run_inference_bird_async_sharded.py`), `TP=2`, `SHARDS=4`, `temperature=0.0`, `top_p=1.0`
- Tool calling: `max_tool_rounds=8`

Overall BIRD EX:

- **Correct: 443 / 498**
- **Accuracy: 88.96%**

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `student_club` | 46 | 48 | 95.83% |
| `formula_1` | 61 | 66 | 92.42% |
| `european_football_2` | 47 | 51 | 92.16% |
| `codebase_community` | 45 | 49 | 91.84% |
| `superhero` | 47 | 52 | 90.38% |
| `thrombosis_prediction` | 45 | 50 | 90.00% |
| `toxicology` | 36 | 40 | 90.00% |
| `card_games` | 46 | 52 | 88.46% |
| `financial` | 26 | 30 | 86.67% |
| `debit_card_specializing` | 25 | 30 | 83.33% |
| `california_schools` | 19 | 30 | 63.33% |

Weakest DB: `california_schools` at 63.33% (19/30) — same as `arcwise_plat_sql`.

SQL execution:

- Pred SQL extracted: `497 / 498`, missing `1`
- Gold SQL extracted: `498 / 498`, missing `0`
- Pred SQL executed: `496 / 498`, execution failures `2`
- Gold SQL executed: `497 / 498`, execution failures `1`
- Both pred and gold executed: `495`

Generation stats:

- Tool calls total: `545` (avg `1.094`/example)
- Tool counts: `sqlite_query=541`, `bm25_search_sqlite=4`
- Stop reasons: `finished=497`, `max_tool_rounds=1`
- Completion tokens total: `216942`, avg `435.63`/example
- Max prompt tokens: `27253`

Timing:

- Generation: `1066.09s`
- Evaluation: `63.29s`
- Total: `1129.89s`

## google/gemma-4-31B-it - mini_dev_sqlite BIRD EX

Run:

- Model: `google/gemma-4-31B-it`
- Data: `outputs/mini_dev_sqlite-schema-tool.jsonl` (500 rows)
- Diff JSON: `data/bird_minidev_data/raw/mini_dev_sqlite.json` (explicit `--diff_json_path`; has `difficulty` field)
- Output: `outputs/inference/mini_dev_sqlite/gemma-4-31B-it/vllm_async_tp2_dp4_ctx43k_p34k_o8k_r8_temp0_20260828_044318`
- Inference: async vLLM sharded (`scripts/run_inference_bird_async_sharded.py`), `TP=2`, `SHARDS=4`, `temperature=0.0`, `top_p=1.0`
- Tool calling: `max_tool_rounds=8`

Overall BIRD EX:

- **Correct: 357 / 500**
- **Accuracy: 71.40%**

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| simple | 120 | 148 | 81.08% |
| moderate | 176 | 250 | 70.40% |
| challenging | 61 | 102 | 59.80% |

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `student_club` | 42 | 48 | 87.50% |
| `superhero` | 44 | 52 | 84.62% |
| `european_football_2` | 37 | 51 | 72.55% |
| `formula_1` | 47 | 66 | 71.21% |
| `card_games` | 37 | 52 | 71.15% |
| `debit_card_specializing` | 21 | 30 | 70.00% |
| `codebase_community` | 34 | 49 | 69.39% |
| `toxicology` | 26 | 40 | 65.00% |
| `thrombosis_prediction` | 32 | 50 | 64.00% |
| `california_schools` | 18 | 30 | 60.00% |
| `financial` | 19 | 32 | 59.38% |

Weakest DB: `financial` at 59.38% (19/32).

SQL execution:

- Pred SQL extracted: `497 / 500`, missing `3`
- Gold SQL extracted: `500 / 500`, missing `0`
- Pred SQL executed: `497 / 500`, execution failures `3`
- Gold SQL executed: `499 / 500`, execution failures `1`
- Both pred and gold executed: `496`

Generation stats:

- Tool calls total: `608` (avg `1.216`/example)
- Tool counts: `sqlite_query=600`, `sqlite_peek=6`, `bm25_search_sqlite=2`
- Stop reasons: `finished=497`, `max_tool_rounds=3`
- Completion tokens total: `253911`, avg `507.82`/example
- Max prompt tokens: `27272`

Timing:

- Generation: `573.67s`
- Evaluation: `62.71s`
- Total: `636.90s`

