# Thinking Machine Eval

## Eval datasets used with `google/gemma-4-31B-it`

Three BIRD-style tool-calling eval files, all run with `scripts/run_inference_bird_async_sharded.py`, `TP=2`, `SHARDS=4`, `temperature=0.0`, `top_p=1.0`, `max_tool_rounds=8`:

| # | Input file | Rows | Diff JSON (`--diff_json_path`) | Has `difficulty`? | Status |
| --- | --- | ---: | --- | --- | --- |
| 1 | `outputs/arcwise_plat_sql-schema-tool.jsonl` | 498 | `data/revisql/raw/arcwise_plat_sql.json` | No (all "unknown") | Done |
| 2 | `outputs/arcwise_plat_full-schema-tool.jsonl` | 498 | `data/revisql/raw/arcwise_plat_full.json` | No (all "unknown") | Done |
| 3 | `outputs/mini_dev_sqlite-schema-tool.jsonl` | 500 | `data/bird_minidev_data/raw/mini_dev_sqlite.json` | Yes | Running (auto-started after run 2 freed GPUs) |

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

