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


---

# Qwen3.8-27B on the same three eval sets

Companion experiment to the `google/gemma-4-31B-it` runs above, on branch
`qwen-3p8-thinky`. Same three question sets, same gold, same few-shot
demonstrations, same generation settings — the only deliberate differences are
the model and the tool-call dialect baked into the system prompt.

## Data files generated

The Qwen eval files are **not** rebuilt from raw. `scripts/qwen/build_qwen_eval_data.py`
takes the Gemma tool-format file and swaps the system message wholesale, from
`build_tool_system_prompt("default")` to `build_tool_system_prompt("default_qwen")`
(14912 -> 13937 chars). The Gemma prompt teaches `call:name{...}` and forbids XML;
the Qwen template teaches Qwen's native `<tool_call>` XML and adds the
text-affinity guidance inserted by `_to_qwen_template` in `prompts.py`.

Everything else in each row — user message (schema + 3 few-shot demos), `tools`,
`db_id`, `gold_sql`, `evidence`, `question` — is left untouched. The script
asserts the input system prompt is byte-exact before converting, so a template
drift fails loudly instead of shipping a half-converted prompt.

| # | Qwen input file (generated) | Rows | Built from (Gemma file) | Diff JSON (`--diff_json_path`) |
| --- | --- | ---: | --- | --- |
| 1 | `outputs/qwen-arcwise_plat_sql-schema-tool.jsonl` | 498 | `outputs/arcwise_plat_sql-schema-tool.jsonl` | `data/revisql/raw/arcwise_plat_sql.json` |
| 2 | `outputs/qwen-arcwise_plat_full-schema-tool.jsonl` | 498 | `outputs/arcwise_plat_full-schema-tool.jsonl` | `data/revisql/raw/arcwise_plat_full.json` |
| 3 | `outputs/qwen-mini_dev_sqlite-schema-tool.jsonl` | 500 | `outputs/mini_dev_sqlite-schema-tool.jsonl` | `data/bird_minidev_data/raw/mini_dev_sqlite.json` |

Commands used:

```bash
python scripts/qwen/build_qwen_eval_data.py \
  --input  outputs/arcwise_plat_sql-schema-tool.jsonl \
  --output outputs/qwen-arcwise_plat_sql-schema-tool.jsonl --overwrite
python scripts/qwen/build_qwen_eval_data.py \
  --input  outputs/arcwise_plat_full-schema-tool.jsonl \
  --output outputs/qwen-arcwise_plat_full-schema-tool.jsonl --overwrite
python scripts/qwen/build_qwen_eval_data.py \
  --input  outputs/mini_dev_sqlite-schema-tool.jsonl \
  --output outputs/qwen-mini_dev_sqlite-schema-tool.jsonl --overwrite
```

### Verification of the generated files

- **Only the system prompt moved.** Row-by-row over all fields and every message:
  1496/1496 rows swapped, **0 non-system diffs**. `gold_sql`, `question`,
  `evidence`, `tools` and the user message are byte-identical to the Gemma files.
- **Tool signatures unchanged.** The baked `tools` array equals this branch's
  `get_tool_definitions()` exactly, and `tool_catalog_compact()` appears verbatim
  in the converted prompt. `gen_tools.py` differs across branches only in
  sync-vs-async wrapping (`async def` directly instead of `_to_async` wrappers);
  parameter names, defaults and required-lists are identical:
  `bm25_search_sqlite(db_id, table, column, query, top_k, where)`,
  `sqlite_peek(db_id, table, columns, limit, where)`,
  `sqlite_query(db_id, sql, max_return_rows)`.
- **Few-shot provenance** (3 demos/row, verified by membership, not assumption):

  | Dataset | Demos | In ReViSQL BIRD-Platinum train | In train-6601 | Demos from an evaluated DB |
  | --- | ---: | ---: | ---: | --- |
  | `qwen-arcwise_plat_sql` | 1494 | **1494 (100%)** | 801 | none |
  | `qwen-arcwise_plat_full` | 1494 | **1494 (100%)** | 798 | none |
  | `qwen-mini_dev_sqlite` | 1500 | 221 | **1497 (99.8%)** | none |

  The two corpora share 1160 questions, which is why the off-diagonal counts are
  non-zero. Plat-SQL/Plat-Full draw from the data ReViSQL released
  (`data/revisql/raw/bird-verified-train.jsonl`, 2064 rows); Mini-Dev draws from
  `data/bird_train_data/raw/train-6601.jsonl`. No demonstration comes from any of
  the 11 evaluated databases in any of the three files.

## Run configuration

Runner: `scripts/qwen/run_qwen38_thinking_machine_evals.sh` — the three datasets
in sequence, each waiting for all 8 GPUs to go idle first.

- Model: `Qwen/Qwen3.8-27B` (local snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`)
- Inference: `scripts/run_inference_bird_qwen_async.py`, in-process async vLLM engine
- `TP=2`, `SHARDS=4` (GPU groups `0,1` `2,3` `4,5` `6,7`), concurrency 16/shard
- `temperature=0.0`, `top_p=1.0`, `top_k=20`, `max_tool_rounds=8`
- `max_new_tokens=8000`, `max_prompt_length=34000`, `vllm_max_model_len=43000`
- `--no_prompt_rewrite` (the files already carry the Qwen dialect; the runtime
  rewrite is for the OpenAI-server path and would strip the XML examples)
- `--tool_choice_policy required_first`, `--empty_tool_retries 1`
- Thinking **off** (runner default), matching the Gemma baseline
- `NL2SQL_TOOL_LOOP_GUARD=1` — applies only at temperature 0

Differences from the Gemma runs, for the record: `top_k=20` (Qwen's own
generation config; the Gemma runs set no `top_k`), the Qwen system-prompt dialect,
and `tool_choice_policy`/`empty_tool_retries`, which have no Gemma equivalent.

Attention backends are auto-selected, nothing forced. Confirmed from the worker log:

```
Using FlashInfer GDN prefill kernel        <- the 48 linear-attention layers
Using FLASH_ATTN attention backend out of potential backends: [...]
Using FlashAttention version 3             <- the 16 full-attention layers
```

`.venv/bin` must be on `PATH`: the FlashInfer GDN kernel is ninja-JIT-compiled,
and without it EngineCore dies with every sample a `generation_error`, 0 tool
calls, 0% accuracy — while the process still exits 0. The runner therefore judges
each dataset on `eval_summary.json`, not on the exit code.

## Results

| Dataset | Rows | Correct | Overall EX | gemma-4-31B-it | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `arcwise_plat_sql` | 498 | _pending_ | _pending_ | 85.74% | |
| `arcwise_plat_full` | 498 | _pending_ | _pending_ | 88.96% | |
| `mini_dev_sqlite` | 500 | _pending_ | _pending_ | 71.40% | |

Run in progress; this table and the per-dataset sections below are filled in as
each dataset completes.
