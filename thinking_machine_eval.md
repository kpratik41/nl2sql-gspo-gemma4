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
| `arcwise_plat_sql` | 498 | 417 | 83.73% | 85.74% | -2.01 |
| `arcwise_plat_full` | 498 | 436 | 87.55% | 88.96% | -1.41 |
| `mini_dev_sqlite` | 500 | 356 | 71.20% | 71.40% | -0.20 |

All three complete. Qwen3.8-27B lands slightly below gemma-4-31B-it on every
set, and the gap narrows as the gold gets cleaner and the set gets harder:
-2.01 on Plat-SQL, -1.41 on Plat-Full, -0.20 on Mini-Dev. On the uncorrected
Mini-Dev split the two models are within a single question of each other.

### Combined accuracy and input-file sizes

| Dataset | Rows | gemma-4-31B-it EX | Qwen3.8-27B EX | Delta | Gemma input file | Qwen input file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arcwise_plat_sql` | 498 | 85.74% (427) | 83.73% (417) | -2.01 | 45,320,494 B | 44,339,434 B |
| `arcwise_plat_full` | 498 | 88.96% (443) | 87.55% (436) | -1.41 | 45,323,660 B | 44,342,600 B |
| `mini_dev_sqlite` | 500 | 71.40% (357) | 71.20% (356) | -0.20 | 45,238,725 B | 44,253,725 B |

### Input file integrity, gemma vs qwen

Checked because a 1-2 point deficit could plausibly be a data-generation fault
rather than a model difference. It is not.

| Dataset | Delta (bytes) | Delta per row | Rows |
| --- | ---: | ---: | ---: |
| `arcwise_plat_sql` | -981,060 | -1,970 | 498 |
| `arcwise_plat_full` | -981,060 | -1,970 | 498 |
| `mini_dev_sqlite` | -985,000 | -1,970 | 500 |

The Qwen files are ~2.2% smaller, which is exactly the expected amount: the Qwen
system prompt is 975 chars shorter (13937 vs 14912) and appears twice per row, in
both `prompt` and `messages` (-1950), plus ~20 bytes of JSON escaping for the XML
tool-call examples.

The decisive check is the distribution, not the total: across all 1496 rows there
is **exactly one distinct delta value, 1970**. A generation fault -- a truncated
schema, a dropped few-shot, a mangled row -- would produce deltas that vary from
row to row. Zero user-message mismatches and zero gold-SQL mismatches.

In tokens the Qwen input is 5% *shorter*, not larger: mean 10063 vs 10590 over the
first 120 rows of Plat-Full, each file measured under its own model's tokenizer.
So the 47470 `max_prompt_tokens` seen on the Qwen runs is not the input file; it
is the tool loop appending query results, which is also why only Qwen hit
`context_length_exceeded`.

### Why Qwen scores lower

Question-by-question on Plat-Full, the same 498 rows:

| | Count |
| --- | ---: |
| Both correct | 419 |
| Gemma only (Qwen lost) | 24 |
| Qwen only (Qwen won) | 17 |
| Neither correct | 38 |
| Net | -7 |

Three things rule out a data fault. Disagreement runs **both ways** -- Qwen wins
17 questions Gemma misses, whereas corrupted input fails in one direction only.
The 24 losses spread over 9 databases with at most 4 in any one, rather than
clustering as malformed rows would. And 21 of the 24 executed successfully and
simply returned the wrong rows -- semantic errors, not parse or prompt failures;
only 3 produced empty SQL, and only 3 rows in 498 had no extractable prediction.

What differs is behaviour: Qwen works the tool loop harder for a slightly worse
result -- 1.37-1.57 tool calls per example against Gemma's 1.11, and 463-517
completion tokens against 446 -- and hits the round cap 6-12 times per dataset
where Gemma hit it once. It is also a 27B model against a 31B.

`california_schools` is the one shared failure mode: 16/30 for Qwen on both
arcwise sets, 60.00% for both models on Mini-Dev, and untouched by the
Plat-SQL -> Plat-Full correction that lifted nearly every other database.

## Qwen3.8-27B - arcwise_plat_sql BIRD EX

- Data: `outputs/qwen-arcwise_plat_sql-schema-tool.jsonl` (498 rows)
- Diff JSON: `data/revisql/raw/arcwise_plat_sql.json` (no `difficulty` field, so the
  breakdown is all "unknown")
- Output: `outputs/inference/arcwise_plat_sql/Qwen3.8-27B/vllm_async_tp2_dp4_ctx43k_p34k_o8k_r8_temp0_20260828_051521`
- Model read from the HF cache on EBS (this run only; runs 2 and 3 read from the
  NVMe copy instead)

Overall BIRD EX:

- **Correct: 417 / 498**
- **Accuracy: 83.73%**  (gemma-4-31B-it: 85.74%, -2.01 points)

By database, against the Gemma run on the same 498 questions:

| Database | Qwen correct | Rows | Qwen EX | gemma-4-31B-it EX | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `student_club` | 47 | 48 | 97.92% | 97.92% | 0.00 |
| `card_games` | 49 | 52 | 94.23% | 90.38% | +3.85 |
| `european_football_2` | 45 | 51 | 88.24% | 86.27% | +1.97 |
| `toxicology` | 35 | 40 | 87.50% | 85.00% | +2.50 |
| `thrombosis_prediction` | 42 | 50 | 84.00% | 82.00% | +2.00 |
| `codebase_community` | 41 | 49 | 83.67% | 87.76% | -4.09 |
| `formula_1` | 55 | 66 | 83.33% | 87.88% | -4.55 |
| `superhero` | 43 | 52 | 82.69% | 86.54% | -3.85 |
| `financial` | 23 | 30 | 76.67% | 83.33% | -6.66 |
| `debit_card_specializing` | 21 | 30 | 70.00% | 80.00% | -10.00 |
| `california_schools` | 16 | 30 | 53.33% | 63.33% | -10.00 |

`california_schools` is the weakest DB for both models, and Qwen is 10 points
worse on it than Gemma. `debit_card_specializing` shows the same 10-point gap.
Qwen is ahead on 4 of 11 DBs and level on a 5th, so the 2-point overall deficit
is concentrated in a handful of databases rather than spread evenly.

SQL execution:

- Pred SQL extracted: `496 / 498`, missing `2`
- Gold SQL extracted: `498 / 498`, missing `0`
- Pred SQL executed: `496 / 498`, execution failures `2`
- Gold SQL executed: `497 / 498`, execution failures `1`
- Both pred and gold executed: `495`

Generation stats:

- Tool calls total: `707` (avg `1.420`/example, against Gemma's `1.114`)
- Tool counts: `sqlite_query=687`, `sqlite_peek=9`, `bm25_search_sqlite=11`
- Stop reasons: `finished=487`, `forced_final_at_cap=10`, `context_length_exceeded=1`
- Rejected tool calls: `0`
- Completion tokens total: `237785`, avg `477.48`/example (Gemma: `445.74`)
- Max prompt tokens: `47470` (Gemma: `27228`)

Timing:

- Generation: `1026.39s` (includes engine startup: weight load, torch.compile,
  and the FlashInfer GDN ninja JIT)
- Evaluation: `66.66s`
- Total: `1093.58s`

Note on the timing comparison: this figure is **not** comparable to the Gemma
`711.00s` generation number as a model-speed measurement. Roughly 12 of the 18
minutes here were engine startup -- weights loading off EBS at 22-37s per shard
with all 4 shards reading the same checkpoint concurrently, plus a 56s
torch.compile and the FlashInfer JIT. Actual token generation began at 05:29:34
and finished at 05:33:35, about 4 minutes. Runs 2 and 3 load from a copy on the
instance-store NVMe instead, which removes most of that startup cost.

## Qwen3.8-27B - arcwise_plat_full BIRD EX

- Data: `outputs/qwen-arcwise_plat_full-schema-tool.jsonl` (498 rows)
- Diff JSON: `data/revisql/raw/arcwise_plat_full.json` (no `difficulty` field)
- Output: `outputs/inference/arcwise_plat_full/Qwen3.8-27B/vllm_async_tp2_dp4_ctx43k_p34k_o8k_r8_temp0_20260828_053508`
- Model read from `/opt/dlami/nvme/models/Qwen3.8-27B` (instance-store NVMe)

Overall BIRD EX:

- **Correct: 436 / 498**
- **Accuracy: 87.55%**  (gemma-4-31B-it: 88.96%, -1.41 points)

By database, against the Gemma run on the same 498 questions:

| Database | Qwen correct | Rows | Qwen EX | gemma-4-31B-it EX | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `student_club` | 47 | 48 | 97.92% | 95.83% | +2.09 |
| `toxicology` | 37 | 40 | 92.50% | 90.00% | +2.50 |
| `formula_1` | 61 | 66 | 92.42% | 92.42% | 0.00 |
| `european_football_2` | 47 | 51 | 92.16% | 92.16% | 0.00 |
| `codebase_community` | 45 | 49 | 91.84% | 91.84% | 0.00 |
| `card_games` | 47 | 52 | 90.38% | 88.46% | +1.92 |
| `superhero` | 45 | 52 | 86.54% | 90.38% | -3.84 |
| `thrombosis_prediction` | 43 | 50 | 86.00% | 90.00% | -4.00 |
| `financial` | 25 | 30 | 83.33% | 86.67% | -3.34 |
| `debit_card_specializing` | 23 | 30 | 76.67% | 83.33% | -6.66 |
| `california_schools` | 16 | 30 | 53.33% | 63.33% | -10.00 |

The two models are exactly level on 3 of 11 databases and Qwen is ahead on 3
more, so again the deficit is concentrated: `california_schools` alone accounts
for 3 of the 10-question gap. Qwen scores 16/30 on `california_schools` on both
arcwise sets -- correcting the questions and evidence moved it not at all, while
it lifted Gemma by 0 points there too. That database is the shared failure mode,
not a model-specific one.

SQL execution:

- Pred SQL extracted: `495 / 498`, missing `3`
- Gold SQL extracted: `498 / 498`, missing `0`
- Pred SQL executed: `495 / 498`, execution failures `3`
- Gold SQL executed: `497 / 498`, execution failures `1`
- Both pred and gold executed: `494`

Generation stats:

- Tool calls total: `680` (avg `1.365`/example)
- Tool counts: `sqlite_query=662`, `sqlite_peek=5`, `bm25_search_sqlite=13`
- Stop reasons: `finished=491`, `forced_final_at_cap=6`, `context_length_exceeded=1`
- Rejected tool calls: `0`
- Completion tokens total: `230523`, avg `462.90`/example
- Max prompt tokens: `47470`

Timing:

- Generation: `345.32s`
- Evaluation: `66.48s`
- Total: `412.58s`

This is the first run to load from the NVMe copy, and the difference is the
whole story of the earlier timing caveat: the checkpoint load went from about
8 minutes to **7 seconds** (18/18 shards at 2.84 it/s), cutting total wall clock
from `1093.58s` to `412.58s` for the same 498 rows. The `345.32s` here is close
to a real generation measurement; the `1026.39s` on the Plat-SQL run was not.

## Qwen3.8-27B - mini_dev_sqlite BIRD EX

- Data: `outputs/qwen-mini_dev_sqlite-schema-tool.jsonl` (500 rows)
- Diff JSON: `data/bird_minidev_data/raw/mini_dev_sqlite.json` (has `difficulty`)
- Output: `outputs/inference/mini_dev_sqlite/Qwen3.8-27B/vllm_async_tp2_dp4_ctx43k_p34k_o8k_r8_temp0_20260828_053508`
- Model read from `/opt/dlami/nvme/models/Qwen3.8-27B`

Overall BIRD EX:

- **Correct: 356 / 500**
- **Accuracy: 71.20%**  (gemma-4-31B-it: 71.40%, -0.20 points -- one question)

By difficulty, the only one of the three sets carrying labels:

| Difficulty | Qwen correct | Rows | Qwen EX | gemma-4-31B-it EX | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `simple` | 122 | 148 | 82.43% | 81.08% | +1.35 |
| `moderate` | 176 | 250 | 70.40% | 70.40% | 0.00 |
| `challenging` | 58 | 102 | 56.86% | 59.80% | -2.94 |

The models are **identical on moderate** (176/250 each) and Qwen is ahead on
simple. The entire overall deficit sits in `challenging`, where Qwen loses 3
questions. That is the cleanest signal across all three eval sets: the two models
are equivalent on routine queries and diverge only at the hard end.

By database:

| Database | Qwen correct | Rows | Qwen EX | gemma-4-31B-it EX | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `superhero` | 46 | 52 | 88.46% | 84.62% | +3.84 |
| `student_club` | 41 | 48 | 85.42% | 87.50% | -2.08 |
| `card_games` | 39 | 52 | 75.00% | 71.15% | +3.85 |
| `debit_card_specializing` | 22 | 30 | 73.33% | 70.00% | +3.33 |
| `european_football_2` | 37 | 51 | 72.55% | 72.55% | 0.00 |
| `formula_1` | 46 | 66 | 69.70% | 71.21% | -1.51 |
| `codebase_community` | 33 | 49 | 67.35% | 69.39% | -2.04 |
| `toxicology` | 25 | 40 | 62.50% | 65.00% | -2.50 |
| `financial` | 20 | 32 | 62.50% | 59.38% | +3.12 |
| `california_schools` | 18 | 30 | 60.00% | 60.00% | 0.00 |
| `thrombosis_prediction` | 29 | 50 | 58.00% | 64.00% | -6.00 |

Qwen is ahead on 4 databases, level on 2, behind on 5. Note `california_schools`
at 60.00% for both models -- identical here as well, and *higher* than either
model manages on the arcwise sets (53.33% Qwen / 63.33% Gemma). Mini-Dev has 30
`california_schools` rows like the arcwise sets but different gold for many of
them, so the arcwise correction appears to have made that database harder, not
easier, for both models.

SQL execution:

- Pred SQL extracted: `500 / 500`, missing `0`
- Gold SQL extracted: `500 / 500`, missing `0`
- Pred SQL executed: `499 / 500`, execution failures `1`
- Gold SQL executed: `499 / 500`, execution failures `1`
- Both pred and gold executed: `498`

Generation stats:

- Tool calls total: `784` (avg `1.568`/example, against Gemma's `1.216`)
- Tool counts: `sqlite_query=759`, `sqlite_peek=15`, `bm25_search_sqlite=10`
- Stop reasons: `finished=488`, `forced_final_at_cap=12`
- Rejected tool calls: `0`
- Completion tokens total: `258672`, avg `517.34`/example (Gemma: `507.82`)
- Max prompt tokens: `28382`

Timing:

- Generation: `390.75s`
- Evaluation: `62.99s`
- Total: `454.42s`

## Cross-model summary

| Dataset | Rows | gemma-4-31B-it | Qwen3.8-27B | Delta |
| --- | ---: | ---: | ---: | ---: |
| `arcwise_plat_sql` | 498 | 85.74% | 83.73% | -2.01 |
| `arcwise_plat_full` | 498 | 88.96% | 87.55% | -1.41 |
| `mini_dev_sqlite` | 500 | 71.40% | 71.20% | -0.20 |

Observations:

- **Both models rank the three sets identically**: Plat-Full > Plat-SQL >
  Mini-Dev. The arcwise gold correction is worth about +3 points to Gemma and
  +3.8 to Qwen, and the uncorrected Mini-Dev split is ~16 points harder for both.
  The eval sets behave consistently across two very different models, which is
  the main thing these runs were meant to establish.
- **Qwen is behind everywhere, but never by much**, and the gap shrinks as the
  gold gets cleaner and the questions get harder. On Mini-Dev the two are within
  one question.
- **Qwen spends more to get there.** Tool calls per example run 1.37-1.57 against
  Gemma's 1.11-1.22 on the same questions, and Qwen hits the round cap 6-12 times
  per set where Gemma hits it 1-3 times. Qwen also pushed a 47470-token prompt on
  both arcwise sets against Gemma's ~27200, so its tool results are accumulating
  far more context.
- **`california_schools` is a shared, correction-resistant failure.** Both models
  score exactly 60.00% on it in Mini-Dev and both are flat between Plat-SQL and
  Plat-Full. It is the weakest database in 5 of the 6 runs.
- Zero generation errors and zero rejected tool calls in all three Qwen runs.

Caveat on comparability: these are two different models on the same data, not a
controlled ablation. Qwen runs with `top_k=20` (its own generation config; the
Gemma runs set no `top_k`), a different system-prompt dialect, and
`tool_choice_policy=required_first` with `empty_tool_retries=1`, which have no
Gemma equivalent. The questions, gold, few-shot demonstrations, temperature,
round budget and token limits are identical.

---

# Qwen3.8-27B with thinking enabled

Same three eval files, same model, same NVMe weights. Thinking is generated per
round and dropped from the history (`preserve_thinking` off), because the shipped
chat template re-renders historical reasoning and `qwen38_eval_plan.md` records
that it emits empty `<think></think>` blocks when it is preserved -- drift that
compounds over a tool loop up to 8 rounds deep.

Two settings had to move with it, and they are a confound worth stating rather
than burying: `max_new_tokens` 8000 -> 16000 and `max_model_len` 43000 -> 65536.
An earlier thinking sweep at 8000 truncated reasoning mid-thought and scored the
fragment as an answer, so it was discarded and is not recorded here. 16000 does
not fit under a 43000 context alongside a 34000-token prompt, so the context had
to rise too. Thinking-off numbers were **not** re-run: they stay at 8000/43000,
where they remain directly comparable to the Gemma baseline.

## Results

| Dataset | Rows | Qwen thinking ON | Qwen thinking OFF | gemma-4-31B-it | ON vs OFF | ON vs Gemma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arcwise_plat_sql` | 498 | **86.35%** (430) | 83.73% (417) | 85.74% (427) | **+2.62** | **+0.61** |
| `arcwise_plat_full` | 498 | **89.56%** (446) | 87.55% (436) | 88.96% (443) | **+2.01** | **+0.60** |
| `mini_dev_sqlite` | 500 | 70.20% (351) | 71.20% (356) | 71.40% (357) | **-1.00** | -1.20 |

Thinking is not uniformly good. It is worth about +2 points on both arcwise sets,
enough to turn a deficit against Gemma into a small lead on each. On the
uncorrected Mini-Dev split it **costs** a point.

That split is the interesting part. The arcwise sets have expert-corrected gold;
Mini-Dev does not, and 163 of its 498 shared golds differ from the corrected
Plat-SQL versions. Longer reasoning converges on the defensible reading of an
ambiguous question, which is rewarded where the gold was fixed and penalised
where the gold still encodes the original, less defensible one. The by-difficulty
split on Mini-Dev supports this: thinking loses ground on `simple` (82.43 ->
81.08) and `moderate` (70.40 -> 68.00) while *gaining* on `challenging`
(56.86 -> 59.80). It helps where the work is genuinely hard and hurts where the
question was merely under-specified.

## Cost

| | Thinking OFF | Thinking ON |
| --- | ---: | ---: |
| avg completion tokens, Plat-SQL | 477 | 1520 |
| avg completion tokens, Plat-Full | 463 | 1291 |
| avg completion tokens, Mini-Dev | 517 | 1822 |
| avg tool calls/example | 1.37-1.57 | 1.44-1.61 |

Roughly 3x the generated tokens for +2 points on corrected gold and -1 on
uncorrected. Tool-call counts barely move, so the extra tokens are reasoning,
not extra database work.

## Truncation, and why 16k was not simply "enough"

| Dataset | `max_new_tokens` hit | at 8000 | max prompt tokens |
| --- | ---: | ---: | ---: |
| `arcwise_plat_sql` | 8 | 9 | 41417 |
| `arcwise_plat_full` | 4 | -- | 35454 |
| `mini_dev_sqlite` | 9 | -- | 38117 |

Doubling the budget moved Plat-SQL truncations from 9 to 8. These are not
samples that were slightly short of room; they are runaway reasoning loops that
a larger budget will not rescue, so raising the ceiling again is not worth the
GPU time. Prompt lengths all landed well under the 65536 ceiling, and the
`context_length_exceeded` that appeared once in the 43000-context thinking-off
runs did not recur.

Mini-Dev also produced 1 `generation_error` and 11 rows with no extractable
prediction, against 3 on Plat-Full -- the only run in the suite with a
generation error.

## Per-database, thinking on

| Database | Plat-SQL | Plat-Full | Mini-Dev |
| --- | ---: | ---: | ---: |
| `student_club` | -- | 93.75 | 83.33 |
| `european_football_2` | -- | 94.12 | 74.51 |
| `codebase_community` | -- | 93.88 | 65.31 |
| `financial` | -- | 93.33 | 59.38 |
| `toxicology` | -- | 92.50 | 67.50 |
| `formula_1` | -- | 92.42 | 71.21 |
| `card_games` | -- | 92.31 | 65.38 |
| `superhero` | -- | 88.46 | 86.54 |
| `thrombosis_prediction` | -- | 88.00 | 62.00 |
| `debit_card_specializing` | -- | 80.00 | 70.00 |
| `california_schools` | -- | 63.33 | 56.67 |

`california_schools` reaches 63.33% on Plat-Full with thinking -- which is
exactly gemma-4-31B-it's score there, and exactly Qwen's own thinking-off score.
Across both models, both arcwise sets, corrected and uncorrected gold, and
thinking on and off, that database lands in the same 56-63% band every time. It
is the one failure mode nothing tried so far has moved.

## Note on the Gemma baseline

Verified rather than assumed: the Gemma runs had thinking **off**. Its chat
template sets `enable_thinking = enable_thinking | default(false)`, and
`scripts/run_inference_bird_async_sharded.py` never passes the kwarg.

The two templates default in opposite directions, which is a trap for any future
script. Gemma defaults to false; Qwen's is
`enable_thinking is undefined or enable_thinking is true`, i.e. **on**. The Qwen
runner passes the value explicitly, so the thinking-off runs really were off --
confirmed empirically by 477 average completion tokens against 1520 with
thinking on. A script that merely omits the kwarg would silently get thinking-on
for Qwen and thinking-off for Gemma, with nothing to flag it.

gemma-4-31B-it does support thinking (6 `enable_thinking` and 3
`preserve_thinking` usages in its template), so a thinking-on Gemma run is
possible and would be the fair both-models-at-their-best comparison. It has not
been run: the Gemma runner on this branch has no thinking flag to pass.

---

# Qwen3.8-Flash-Next on arcwise_plat_full

A third model on the primary eval set only. `Qwen4ExpForConditionalGeneration`
(model_type `qwen4_exp`), a 180B MoE with 512 experts and 10 active per token,
hybrid attention over 48 layers, ~360GB of BF16 weights in 131 shards.

## Setup, and why it differs from the 27B runs

Served by vLLM in Docker at `tp=8` on one shard, rather than the in-process
async engine used for every 27B result. That was forced, not chosen: the vLLM
recipe for this model requires 0.29.0+ and states that PyPI installation is not
supported for it. PyPI's newest is 0.28.0 and the newest nightly wheel is
0.28.1rc1.dev43, so there is no wheel to install into a venv. The recipe image
`vllm/vllm-openai:qwen38-flash-next` carries a dev build
(`0.1.dev20073+g8e685d198`, transformers 5.15.1) that does register the
architecture, and it isolates the run completely from the vllm 0.19.1
environment producing the 27B numbers.

- Data: `outputs/qwen-arcwise_plat_full-schema-tool.jsonl`, **unchanged from the
  27B runs**. Verified rather than assumed: Flash-Next ships a
  `chat_template.jinja` byte-identical to Qwen3.8-27B's, the same
  `Qwen2Tokenizer` with the same 33 added tokens and the same eos/pad, and the
  same `<tool_call>` dialect.
- Diff JSON: `data/revisql/raw/arcwise_plat_full.json`
- Output: `outputs/inference/arcwise_plat_full/Qwen3.8-Flash-Next/vllm_server_tp8_ctx43k_o8k_r8_qwen3_xml_temp0_20260828_064525`
- `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`, util 0.96,
  weights read from the NVMe copy.

The server path means vLLM parses tool calls out of raw text, which the
in-process runner did itself. A mismatched parser extracts nothing and reads as
a plausible low accuracy rather than an error, so the run was gated on a
20-example smoke test first: it returned 30 tool calls, confirming `qwen3_xml`,
and the full 498 then ran against the same server.

One parameter could not be matched: the server client has no `--top_k`, so the
27B runs' `top_k=20` was not passed. At temperature 0 this is inert -- decoding
is greedy argmax -- so the runs stay comparable.

## Results

- **Correct: 438 / 498**
- **Accuracy: 87.95%**

| Model | Config | Plat-Full EX |
| --- | --- | ---: |
| Qwen3.8-27B | thinking on | **89.56%** |
| gemma-4-31B-it | thinking off | 88.96% |
| **Qwen3.8-Flash-Next** | **thinking off** | **87.95%** |
| Qwen3.8-27B | thinking off | 87.55% |

Flash-Next beats the 27B at the same setting by 0.40, but trails Gemma by 1.01
and the 27B with thinking by 1.61. A 180B MoE returning 0.4 points over a 27B
dense model is a modest result for the parameter count on this task.

| Database | Correct | Rows | Flash-Next | 27B (off) | gemma-4-31B-it |
| --- | ---: | ---: | ---: | ---: | ---: |
| `student_club` | 46 | 48 | 95.83% | 97.92% | 95.83% |
| `codebase_community` | 46 | 49 | 93.88% | 91.84% | 91.84% |
| `financial` | 28 | 30 | 93.33% | 83.33% | 86.67% |
| `formula_1` | 60 | 66 | 90.91% | 92.42% | 92.42% |
| `european_football_2` | 46 | 51 | 90.20% | 92.16% | 92.16% |
| `toxicology` | 36 | 40 | 90.00% | 92.50% | 90.00% |
| `superhero` | 45 | 52 | 86.54% | 86.54% | 90.38% |
| `thrombosis_prediction` | 43 | 50 | 86.00% | 86.00% | 90.00% |
| `card_games` | 44 | 52 | 84.62% | 90.38% | 88.46% |
| `debit_card_specializing` | 25 | 30 | 83.33% | 76.67% | 83.33% |
| `california_schools` | 19 | 30 | 63.33% | 63.33% | 63.33% |

`california_schools` is 19/30 again. Three models, four configurations, corrected
and uncorrected gold, thinking on and off -- every one lands on exactly 19/30 or
its 16/30 neighbour. Nothing tried so far has moved that database.

## Behaviour: a much more active agent

| | 27B (off) | Flash-Next (off) |
| --- | ---: | ---: |
| Tool calls total | 680 | **1175** |
| Avg calls/example | 1.365 | **2.359** |
| `sqlite_query` | 701 | 1116 |
| `sqlite_peek` | 4 | **30** |
| `bm25_search_sqlite` | 10 | **29** |
| Avg completion tokens | 463 | 576 |
| Pred SQL extracted | 494/498 | **498/498** |
| Pred SQL execution failures | 4 | **0** |

Flash-Next issues 73% more tool calls and leans far harder on schema
exploration -- `sqlite_peek` 7.5x, `bm25_search_sqlite` 3x. It is also the only
run in the entire suite with perfect SQL hygiene: 498/498 extracted, zero
execution failures, where every other run had between 2 and 11. Its SQL is more
reliably well-formed; it simply picks the wrong query slightly more often than
Gemma.

**A caveat on the accuracy.** Stop reasons were
`{'stop': 435, 'forced_final': 62, 'tool_calls': 1}` -- 62 examples, 12.4%, were
forced to finalise, against 6 `forced_final_at_cap` for the 27B. That is the
tool loop being cut off mid-investigation, which is what a model wanting ~2.4
calls does against a round budget tuned for one wanting ~1.4. Some part of the
1-point gap to Gemma is plausibly truncated investigation rather than worse
reasoning, and a run at `max_tool_rounds=12` would separate the two.

## Timing

- Server startup: `243s` (360GB across tp=8 from NVMe)
- Generation: `689.22s`
- Evaluation: `121.49s`
- Total: `811.27s`

The 20-example smoke took 39.15s, which scaled naively to 16.2 min for 498 --
about 50% over the 11.5 min actually taken. At concurrency 16 a 20-example
sample is one full wave plus a 4-example tail, so ramp-up and drain dominate it;
the steady-state rate observed 3 minutes into the full run projected 10.6 min
and was close.
