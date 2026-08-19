# SFT / RFT Data Pipeline — Path A: Privileged-Teacher Rejection Sampling

End-to-end description of how the Gemma-4-31B SFT/RFT warm-start dataset is
built and how the SFT run is launched, so the process can be reproduced in
another environment.

## 2026-08-12 SFT checkpoint selection results

Run directory:
`outputs/sft/gemma4_31b_rft_sft_run2`

Dev data:
`outputs/old-dev-schema-tool-unpatched.jsonl`

Temp-0 inference used async vLLM with `tp=2`, one shard per checkpoint,
`max_prompt_length=34000`, `max_new_tokens=8000`, `max_tool_rounds=8`, and
`vllm_max_model_len=43000`.

Pass@16 used async vLLM with `temperature=1.2`, `top_p=1.0`,
`num_generations=16`, `tp=2`, `num_shards=4`, `vllm_async_concurrency=32`,
`max_prompt_length=30000`, `max_new_tokens=8000`, `max_tool_rounds=8`, and
`vllm_max_model_len=43000`. Note: the merged pass@k summary files record
merge-script defaults in their `run_config`; the shard logs and queue logs
contain the true checkpoint, TP, shard, and concurrency settings.

| SFT ckpt | temp-0 EX | temp-0 correct | pass@1 est. | pass@16 | pass@16 candidate correct | pass@16 pred exec failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 73.27% | 1124/1534 | 45.11% | 83.44% | 11071/24544 | 9337 |
| 20 | 73.01% | 1120/1534 | 66.92% | 83.05% | 16424/24544 | 1353 |
| 50 | 73.08% | 1121/1534 | 68.97% | 84.75% | 16929/24544 | 677 |

Interpretation: checkpoint 50 is the preferred RL warm start from these
measurements. Its greedy temp-0 EX is tied with checkpoints 10/20, but its
sampled pass@16 is best and it has the fewest non-executing sampled candidates.
Checkpoint 10 looks strong at temp 0 but brittle under `temperature=1.2`
sampling, which is less attractive for RL rollouts.

This is **not** a simple keep/drop filter. It is a four-stage rejection-sampling
(STaR / RFT) pipeline that:

1. Finds the hardest train examples (the "all-wrong band") via pass@16.
2. Generates verified agentic **teacher** trajectories on those examples, with
   the gold SQL injected as a privileged, non-verbalized internal reference.
3. Rejection-filters those trajectories (execution-verified against gold, plus
   no leakage) and rewrites them into hint-free **student** records.
4. Runs masked multi-turn SFT with assistant-only loss on the resulting records.

A colleague's original implementation of stages A2–A4 lives on a
`feature/teacher_privelege_info` branch that is **not available in this
environment** and is not on this remote. The script paths and flag names below
are therefore a **specification to build against**, not files to check out —
they describe the intended interface, taken from the original pipeline
description. Everything under "Reproduction status" tracks what actually exists
here.

---

## Stage A0 — Corrected schema type labels (2026-08-17)

Everything below consumes a schema-built training file. That file was rebuilt on
2026-08-17 because the type label rendered into each prompt was wrong for ~5% of
columns. Read this before regenerating any data or comparing against older runs.

### Why — SQLite is dynamically typed

`schema_build.py::classify_column` picked the label by **sniffing sampled
values**: any column whose first 20 values all parsed as floats was labelled
`NUMERIC`. BIRD declares 160 train and 37 dev columns as `TEXT` in the SQLite DDL
(and in its own `train_tables.json` / `dev_tables.json`) while storing
numeric-looking strings — `'0.25'`, `'4200'`, `'nan'`, `'01100170109835'`. The
pipeline overrode a correct schema with a wrong label.

That matters because of TEXT affinity: every text value sorts **above** every
number, so

```sql
WHERE Sentiment_Polarity > 0    -- column is TEXT holding '1.0', '-0.5', 'nan'
```

is not a numeric comparison. It matches nearly every row, returns a plausible
answer, and never raises. Nothing downstream catches it — not execution, not the
reward. Telling the model the column is `NUMERIC` invites exactly that query.

A second, independent bug: the `LIMIT 20` sample read was gated behind
`if include_stats:`, so `--no-stats` silently disabled **date** inference too.
Bare builds lost `DATE` on 18 train and 6 dev columns declared `TEXT`.

### What — three commits on `consensus-sft`

| commit | change |
| --- | --- |
| `7008461` | `classify_column` consults `typeof()`, not sniffed values (cherry-picked from `qwen-3p8` `5f016d8`) |
| `d52f532` | `--workers` for parallel per-database introspection |
| `15e1fc9` | pass@16 launcher path fixes; `BASE_OUT` derived from the input filename |

`classify_column` now layers signals by what each is authoritative for:

1. **Declared `DATE`/`TIME` keyword** — first, because SQLite has no date storage
   class, so `typeof()` would flatten all date columns to `TEXT`.
2. **Date-looking samples** — catches date columns declared `TEXT`.
3. **`typeof()` storage class** — the truthful answer for everything else, and
   why it outranks sniffing.
4. **Declared numeric keyword** — fallback only for empty / all-NULL columns
   where `typeof()` has nothing to report (38 columns).

Value sniffing no longer decides `NUMERIC`. Sampling is no longer gated on
`include_stats`.

Note a second-order effect on stats builds: `kind` also selects which statistics
are computed, so a column moving `NUMERIC` -> `TEXT` swaps `Min`/`Avg`/`Max` for
`Top values`. That is the intent — `Avg: 2.9e13` over a zero-padded school ID was
meaningless — but dev prompts change by more than one token per affected column.

### Measured impact

Verified independently against every column in both database sets; the numbers
match `qwen-3p8`'s `scripts/data_generation/PIPELINE_CHANGES.md` exactly.

| measurement | value |
| --- | ---: |
| columns scanned (train + dev) | `4365` |
| mixed storage-class columns | `0` |
| empty / all-NULL (no `typeof()`) | `38` |
| train: sniffed `NUMERIC` -> truly `TEXT` | `160` |
| train: `--no-stats` `TEXT` -> should be `DATE` | `18` |
| dev: sniffed `NUMERIC` -> truly `TEXT` | `37` |
| dev: `--no-stats` `TEXT` -> should be `DATE` | `6` |

The regenerated train file shows exactly `160` `NUMERIC` -> `TEXT` transitions
and nothing else across all 69 databases; dev shows exactly `37`. A single
well-defined transition per file is the sign the change does one thing.

The old `train-6601-schema-bare-tool.jsonl` predates the `include_stats` gate, so
it already carried correct `DATE` labels and needed only the `NUMERIC` fix. The
old `old-dev-schema-bare-tool.jsonl` was built after the gate and lost both.

### Defective golds — 27 train rows dropped

`find_text_affinity_defects.py` executes each gold against a
`CAST(... AS REAL)`-corrected variant and keeps the rows whose answers differ. It
covers the three unsafe numeric contexts — comparison vs a literal, `MIN`/`MAX`,
`ORDER BY` — and leaves `AVG`/`SUM`/arithmetic alone because SQLite coerces text
for those. `'1.0'` is normalized against `1.0` so a formatting change is not read
as a behaviour change.

`27 / 6601` train rows and `6 / 1534` dev rows fail. The same 27 indices are
found before and after the type fix, which is the expected control: gold SQL and
the databases never changed, only the rendered labels.

Worst cases: `college_completion.grad_150` returns `1870` rows as written and `0`
when corrected; `hockey.ENG` returns `1511` vs `7`; `movielens.rating` `33` vs
`0`.

**Golds are never rewritten** — that would be inventing ground truth. Defective
rows are dropped from **train** only. Dev stays at `1534` rows; filtering an
evaluation set would bias the benchmark.

Full record of what was dropped and why, including each gold and its corrected
variant: `outputs/train-typefix-affinity-defects.json`.

### Reproduction

```bash
# train: schema -> tool format -> drop 27 affinity defects -> 6574 rows
.venv/bin/python scripts/data_generation/schema_build.py --split train --n-examples -1 \
  --output outputs/train-6601-schema-bare-typefix.jsonl \
  --messages-only --no-fewshots --no-stats --no-nullability --no-comments --workers 24
.venv/bin/python scripts/data_generation/build_tool_dataset.py \
  --input  outputs/train-6601-schema-bare-typefix.jsonl \
  --output outputs/train-6601-schema-bare-tool-typefix.jsonl --prompt-template default
.venv/bin/python scripts/data_generation/find_text_affinity_defects.py \
  --input  outputs/train-6601-schema-bare-tool-typefix.jsonl \
  --report outputs/train-typefix-affinity-defects.json \
  --output outputs/train-6574-schema-bare-tool-typefix.jsonl --split train

# dev: WITH stats/examples/comments/nullability, no filtering
.venv/bin/python scripts/data_generation/schema_build.py --split dev --n-examples -1 \
  --input-file     data/bird_dev_data/raw/bird_dev-unpatched-few-shot.json \
  --database-dir   databases/dev_databases \
  --meanings-file  data/bird_dev_data/raw/column_meaning.json \
  --output outputs/old-dev-schema-unpatched-typefix.jsonl --messages-only --workers 16
.venv/bin/python scripts/data_generation/build_tool_dataset.py \
  --input  outputs/old-dev-schema-unpatched-typefix.jsonl \
  --output outputs/old-dev-schema-tool-unpatched-typefix.jsonl --prompt-template default
```

The flag sets are not guesses: each was confirmed by regenerating a few rows and
diffing against the pre-existing file. Train reproduces
`train-6601-schema-bare.jsonl` byte-for-byte; dev reproduces
`old-dev-schema-unpatched.jsonl` byte-for-byte apart from the corrected type
labels and their dependent stats.

Timings with `--workers`: train `143 s` (was ~35 min serial), dev `29 s`, the
train defect scan `589 s`. Past ~24 workers there is nothing to gain — the
critical path is a single database (`bike_share_1`, ~100 s) that cannot be split.

### The files this produced

| file | rows | use |
| --- | ---: | --- |
| `outputs/train-6574-schema-bare-tool-typefix.jsonl` | `6574` | **training / pass@16 input** |
| `outputs/old-dev-schema-tool-unpatched-typefix.jsonl` | `1534` | **evaluation input** |
| `outputs/train-typefix-affinity-defects.json` | `27` | audit trail for the dropped rows |

Validation: dev is `-0.06%` in size vs the original, `1534` rows preserved, and
`gold_sql` / `question` / `db_id` / system prompt / `tools` identical in every
row — only the schema block moved.

Older `outputs/*.jsonl` are left untouched. They carry the pre-fix labels, so do
not mix them with the files above.

### Pinning — `source_idx` is a line number

The tool JSONL carries **no `source_idx` field**; every stage derives it from
`enumerate()` over its input file. Dropping 27 rows renumbers everything after
line `193`. So whichever file A1 pass@16 runs on must also be passed to:

- `scripts/teacher/run_a2_greedy.sh` (`INPUT_FILE=`)
- `scripts/teacher/run_a2b_sampled.sh` (`INPUT_FILE=`)
- `scripts/teacher/run_a3a_selftrace.sh` (`INPUT_FILE=`)
- `scripts/teacher/build_rft_from_traces.py` (`--train_file`)

All four still default to `outputs/train-6601-schema-bare-tool.jsonl`. Mixing
files makes `target_idx_all_wrong.txt` address different questions, silently, with
no error raised.

Related: `data/bird_train_data/raw/train.json` has `9428` entries, **no
`difficulty` field at all**, and does not align positionally with the 6601 file.
Train pass@k difficulty breakdowns have therefore always been `unknown`; the
`DIFF_JSON` argument is inert for the train split. Dev is unaffected.

### Not applied

`qwen-3p8` also adds text-affinity guidance to the system prompt (its "Change 4"),
warning that a TEXT column may hold numeric-looking values and that comparisons on
it use string ordering. It is applied inside `_to_qwen_template`, so it reaches
the Qwen side only. Porting it here means inserting at the
`- Use sqlite_peek for date columns when the stored format is uncertain.` anchor
in `prompts.py` (lines `144` and `361`) and regenerating every `outputs/*-tool*`
file, since the system prompt is baked into them. Deliberately deferred: it would
change the dev prompts that existing SFT/eval numbers were measured against.

---

## Stage A1 — Find the all-wrong band

**Goal.** Identify train examples the current model cannot solve even with 16
sampled attempts. Those are the examples worth teaching.

### A1.1 Run pass@16 over the training file

Sample `num_generations=16` candidates per training example, execute each
predicted SQL against the training databases, and score with BIRD row-set
equality. Produces `passk_candidates.jsonl` and `passk_per_example.jsonl`.

For a warm-start dataset the "current model" is the **base** model, since SFT
happens before any RL.

```bash
bash scripts/run_passk16_train6601.sh
```

### A1.2 Analyze failures

```bash
python scripts/analyze_passk_all_wrong.py \
  --passk-dir    outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp1_shards8/merged \
  --source-jsonl outputs/train-6601-schema-bare-tool.jsonl \
  --tool-jsonl   outputs/train-6601-schema-bare-tool.jsonl \
  --database-dir databases/train_databases \
  --output-jsonl outputs/train-6601-all-wrong.jsonl \
  --execute-sql --workers 32
```

`--execute-sql` is required for the shape/row-count labels
(`output_column_count_mismatch`, `row_count_mismatch`, `pred_empty_gold_nonempty`,
`same_shape_but_wrong_values`) and for `gold_sql_execution_failed`. `--workers`
parallelises per-example analysis across threads; each analysis opens its own
read-only SQLite connection, so it is safe, and it turns a serial multi-minute
run into seconds.

It reads `passk_candidates.jsonl` and `passk_per_example.jsonl`, selects
examples where **every one of the 16 candidates failed**, and classifies the
failure with heuristic labels:

| label | meaning |
| --- | --- |
| `gold_sql_execution_failed` | gold SQL itself does not execute |
| `all_pred_sql_execution_failed` | every predicted SQL errored |
| `all_pred_sql_executed_but_wrong` | all ran, none matched gold rows |
| `output_column_count_mismatch` | wrong number of output columns |
| `row_count_mismatch` | right shape, wrong row count |
| `pred_empty_gold_nonempty` | predictions returned nothing |
| `same_shape_but_wrong_values` | shape matches, values differ |
| `strong_consensus_wrong_result` | model confidently agrees on a wrong answer |
| `low_diversity_repeated_wrong_sql` | same wrong SQL repeated |
| `high_diversity_no_correct_sql` | many distinct attempts, none correct |
| `hit_max_tool_rounds` | ran out of tool rounds |
| `hit_max_new_tokens` | ran out of generation budget |

Outputs:

- `all_wrong_analysis.jsonl`
- `all_wrong_summary.md`
- filtered JSONL at `--output-jsonl`

### A1.3 Produce `TARGET_IDX_FILE`

```bash
python scripts/teacher/make_target_idx.py \
  --passk-dir  outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp1_shards8/merged \
  --output-dir outputs/teacher
```

Splits the run into the three bands and writes two id files (one `source_idx`
per line, ascending):

| file | band | Stage A2 hint strategy |
| --- | --- | --- |
| `target_idx_all_wrong.txt` | all-wrong, minus `gold_sql_execution_failed` | `full_sql` |
| `target_idx_selftrace.txt` | all "sometimes" + all-correct capped `≤12`/db | `none` |

Examples the model **already solves** are *not* gold-conditioned; they are
harvested as hint-free self traces to anchor the SFT distribution. The
all-correct cap keeps easy, over-represented databases from dominating that
anchor set. Also writes `target_idx_summary.json` with the counts and the exact
cap effect.

Useful flags: `--all-correct-cap-per-db` (`-1` disables the cap),
`--keep-gold-failures`, `--seed` (the per-db all-correct sample is seeded, so the
selection is reproducible).

---

## Stage A2 — Gold-conditioned teacher traces

Runs on a single `tp=8` async vLLM engine. For every targeted training example:

### A2.1 Load the standard training row

Load the original system/user prompt. The gold SQL exists only as a top-level
metadata field, never in the prompt.

### A2.2 Inject the privileged hint

```python
teacher_hint.inject_privileged_hint(strategy="full_sql")
```

Teacher-only prompt modifications:

- prepend a privileged self-check instruction
- append the gold SQL inside

```text
<internal_reference_do_not_reveal>
...
</internal_reference_do_not_reveal>
```

The teacher is instructed to solve independently and to **never** quote,
mention, paraphrase, or reveal the reference.

**This hint is never saved into training data.**

### A2.3 Agentic tool loop

Uses the same inference loop as evaluation: draft SQL → `sqlite_query` verify →
final answer. Captures a structured multi-turn transcript.

| setting | value |
| --- | --- |
| `temperature` | `0` |
| `top_p` | `1` |
| `num_samples` | `1` |
| `max_new_tokens` | `8000` |
| `max_tool_rounds` | `8` |
| `max_prompt_length` | `34000` |
| `vllm_max_model_len` | `43000` |
| `tensor_parallel_size` | `8` |

### A2.4 Verify against gold

Extract the final SQL and require:

- safe read-only SQL
- predicted rows equal gold rows under BIRD row-set equality

Gold rows are cached.

### A2.5 Leakage detection — hard vs soft

Every assistant turn is scanned for leak signals, and the normalized gold SQL is
checked for appearing anywhere outside `<sql_code>` blocks. Leaks are **split
into two classes**, and only hard leaks reject a trace:

| class | signals | action |
| --- | --- | --- |
| **HARD** | near-copy of gold SQL; gold-exclusive tables/columns; explicit "gold" mentions | **dropped** |
| **SOFT** | incidental mentions of expected output / result / rows | **kept** |

This is the rule to implement. A simpler "any leak → drop" rule (matching on
`reference`, `gold`, `provided`, `expected`, `ground truth` without
classification) was tried first and discarded too many usable traces — in the
original run it held coverage to ~`299` ids, versus ~`429` once soft leaks were
retained, with no measurable leakage on dev.

### A2.6 Copy detection

Flags — recorded but **not** hard failures:

- `copy_first_call`
- `zero_verify_copy`
- `near_copy` (>0.95 similarity)

**A trace is kept iff it is verified and has no leakage.** Copy flags are
recorded but do not automatically reject a trace.

Outputs:

- `teacher_traces.jsonl`
- `teacher_summary.json`
- `skipped_prompts.jsonl`

Summary statistics: `n_targets`, `n_samples_total`, `n_verified_samples`,
`n_kept_samples`, `n_leaked_samples`, `target_coverage_rate`,
`copy_rate_over_kept`, `elapsed_s`.

With `--hint_strategy none` the student's own prompt is run without hints to
harvest self traces. The all-wrong band uses `full_sql`.

### A2 is run twice

The settings table above describes the greedy first pass. A second sampled pass
increases coverage of the all-wrong band, and the two are unioned:

| run | mode | samples/idx | temperature |
| --- | --- | ---: | ---: |
| A2 | greedy | `1` | `0.0` |
| A2b | pass@8 | `8` | `0.7` |

### A3a — hint-free self-traces (sometimes / all-correct bands)

Same generator with `HINT_STRATEGY=none`:

| setting | value |
| --- | ---: |
| `num_samples` | `2` |
| `temperature` | `0.7` |

The all-correct band is **capped to ≤12 ids per database** so that easy,
over-represented databases do not dominate the anchor set. Keep the best
verified trace per id.

---

## Stage A3 — Build hint-free student RFT records

```bash
python scripts/teacher/build_rft_from_traces.py
```

**Inputs:** one or more `teacher_traces.jsonl`, plus the original bare-tool
train file.

For each example with at least one kept trace, choose the best traces ranked by:

1. non-copy
2. fewer tool rounds
3. earliest sample

**We use `--max_per_idx=1`**, matching the original. The strategy is *generate
wide, keep the single best*: sample `8` candidates per example so that a
verified trace is found for as many ids as possible, then let the ranking pick
one. More candidates improves both coverage and selection quality — with up to
8 verified traces to choose from, the "non-copy, fewest tool rounds" ranking has
real choice, whereas at 2 samples it usually has none.

One record per id also keeps the dataset balanced across examples rather than
letting easily-solved ids contribute twice.

Build a student record where:

- **Prompt:** the original hint-free system/user prompt
- **Messages:** hint-free prompt + teacher transcript
- **Metadata:** `db_id`, `gold_sql`, `evidence`, `question`, `tools`,
  `source_idx`, `teacher_final_sql`, `teacher_tool_rounds`, `copy_flags`

Option `--drop_copy_first_call` produces a reasoning-only dataset by excluding
transcription traces.

**Outputs:** the RFT JSONL and `rft_build_summary.json`.

Leakage is monitored using dev pass@k performance.

---

## Stage A4 — Masked multi-turn SFT

Pipeline:

1. Render each RFT row using the tokenizer chat template.
2. Supervise loss on **assistant turns only**.
3. System, user and tool tokens receive label `-100`.
4. Assistant spans are detected via longest-common-prefix against
   `add_generation_prompt=True`.
5. Rows exceeding `max_seq_len` are dropped.
6. Rows with no supervised assistant tokens are dropped.

```bash
MODEL_PATH=/.../gemma-4-31B-it
DEEPSPEED_CONFIG=configs/ds_zero3_bf16_no_scheduler.json

LEARNING_RATE=1e-5
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

MAX_SEQ_LEN=20480

WARMUP_RATIO=0.03
lr_scheduler_type=cosine
SAVE_STEPS=20
SAVE_TOTAL_LIMIT=8
```

---

## Reproduction status in this environment

Tracked against the `consensus` branch of this checkout.

| stage | status | notes |
| --- | --- | --- |
| A1.1 pass@16 over train | **done** | base `google/gemma-4-31B-it`, 6601 examples x 16 = 105616 candidates, temp 1.2, async vLLM, tp=1 x 8 shards, 4 h 56 min. See "Training-Data Pass@16: Base Gemma 4 31B" in `results.md`. |
| A1.2 analyze all-wrong | **done** | `1233` all-wrong examples analyzed with `--execute-sql --workers 32`. Outputs in the merged pass@k dir plus `outputs/train-6601-all-wrong.jsonl`. |
| A1.3 `TARGET_IDX_FILE` | **done** | `scripts/teacher/make_target_idx.py` wrote the teacher band (`1223`) and self-trace band (`1496`) to `outputs/teacher/`. |
| A2 teacher traces (greedy) | **done** | `scripts/teacher/gen_teacher_traces.py` + `teacher_hint.py`. `431` kept ids from `1223` targets = `35.24%` coverage, 12.6 min on tp=8. Output `outputs/teacher/a2_greedy/`. |
| A2b sampled teacher pass | **done** | `--num_samples 8 --temperature 0.7`, tp=1 x 8 shards, 40 min. `538` kept ids; union with A2 = `541` (`44.24%`). Output `outputs/teacher/a2b_sampled/merged/`. |
| A3a self-traces | **done** | `--hint_strategy none --num_samples 8 --temperature 1.2`, tp=2 x 4 shards, 59 min. `1673 / 1754` ids covered (`95.4%`). Output `outputs/teacher/a3a_selftrace/merged/`. |
| A3 build RFT records | **done** | `scripts/teacher/build_rft_from_traces.py`, `--max_per_idx=1`. `2214` records at `outputs/teacher/rft/train_rft_31b.jsonl`. Final gate passed. |
| A4 masked SFT | **done** | `scripts/teacher/train_sft.py` + `sft_masking.py`, 70 steps / 2 epochs on 8 GPUs, 3 h 22 min. 7 checkpoints at `outputs/sft/gemma4_31b_rft_sft/`. |
| A5 dev evaluation | **done** | temp-0 over `old-dev-schema-tool-unpatched.jsonl`, all 7 checkpoints, tp=1 one per GPU, 2 h 37 min wall clock. Best `72.62%` (ckpt-20 and ckpt-70). |

Nothing in A2–A4 exists yet and none of it is arriving from elsewhere — each
stage has to be written in this repo before it can run. The file names in the
stage sections above are the intended targets to create.

## SFT Data generation - Run 2

Run 2 uses the fresh base-model pass@16 run:

```text
outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp2_shards4/merged
```

This run used `TP=2`, `NUM_SHARDS=4`, `num_generations=16`,
`temperature=1.2`, and `top_p=1.0` over all `6601` training examples.

### Run 2 pass@16 bands

| band | definition | examples | planned SFT treatment |
| --- | --- | ---: | --- |
| all-wrong | `0 / 16` correct | `1228` | privileged teacher traces through A2/A2b |
| heterogeneous / mixed | `1-15 / 16` correct | `697` | select one correct pass@16 student trace |
| all-correct | `16 / 16` correct | `4676` | select `2x` the mixed count from correct pass@16 traces |

For the all-correct band, `2 * 697 = 1394`, so Run 2 will use:

| student source | records |
| --- | ---: |
| mixed band: one correct trace per example | `697` selected, `691` after strict phrase filter |
| all-correct band: sampled/capped correct traces | `1394` |
| **pass@16-derived student traces total** | **`2091` selected, `2085` after strict phrase filter** |

This replaces the need to rerun A3a self-trace generation. The pass@16
candidate file already contains non-privileged student generations with
`prediction_text`, tool calls, tool responses, final SQL, and BIRD correctness
labels. We should convert selected correct candidates into SFT records, masking
tool responses so the model is not trained to generate tool outputs.

Selection rule:

1. Keep all `697` heterogeneous examples, selecting one correct candidate per
   `idx`.
2. Keep `1394` examples from the `4676` all-correct band, selecting one correct
   candidate per `idx`.
3. Prefer cleaner candidates when multiple correct candidates are available:
   fewer tool rounds, successful `stop_reason=finished`, no malformed tool
   transcript, shorter completion if still tied.

### Run 2 all-wrong teacher data

`scripts/teacher/make_target_idx.py` dropped the gold-SQL execution failures
from the all-wrong band:

| quantity | count | file |
| --- | ---: | --- |
| all-wrong examples | `1228` | pass@16 per-example summary |
| gold SQL execution failures dropped | `8` | `outputs/teacher/target_idx_summary.json` |
| teachable all-wrong targets | `1220` | `outputs/teacher/target_idx_all_wrong.txt` |

A2 greedy teacher generation has completed:

```text
outputs/teacher/a2_greedy_tp2_shards4/merged/teacher_traces.jsonl
```

| A2 greedy metric | value |
| --- | ---: |
| targets | `1220` |
| generated samples | `1220` |
| verified samples | `442` |
| kept samples / unique ids | `430` |
| hard-leak samples | `20` |
| target coverage | `35.25%` |

Leakage audit: `0` kept rows have hard leaks and `0` kept rows are unverified.

A2b completed on A2-uncovered all-wrong ids:

```text
outputs/teacher/target_idx_all_wrong_a2_uncovered.txt
outputs/teacher/a2b_uncovered_tp2_shards4/
```

| A2b setting | value |
| --- | --- |
| `TP` | `2` |
| `NUM_SHARDS` | `4` |
| `NUM_SAMPLES` | `8` |
| `TEMPERATURE` | `0.7` |
| `TOP_P` | `1.0` |
| `HINT` | `full_sql` |
| targets | `790` |
| total sampled rollouts | `6320` |

Shard split:

| shard | targets | sampled rollouts |
| ---: | ---: | ---: |
| `0` | `205` | `1640` |
| `1` | `206` | `1648` |
| `2` | `192` | `1536` |
| `3` | `187` | `1496` |

A2b result:

| A2b metric | value |
| --- | ---: |
| generated samples | `6320` |
| verified samples | `370` |
| kept samples | `283` |
| kept unique ids | `109` |
| hard-leak samples | `181` |
| target coverage over A2-uncovered ids | `13.8%` |
| copy rate over kept | `55.83%` |

Leakage audit: `0` kept rows have hard leaks and `0` kept rows are unverified.

Because A2b targets only A2-uncovered ids, the final teacher unique-id count is:

```text
teacher_unique = 430 + 109 = 539
```

After the strict phrase cleanup described below, `458` teacher records remain
for SFT (`385` from A2 and `73` from A2b).

### Run 2 planned SFT composition

| source | records / unique ids |
| --- | ---: |
| pass@16 mixed student traces | `697` selected, `691` after strict phrase filter |
| pass@16 all-correct student traces | `1394` |
| A2 greedy all-wrong teacher traces | `430` selected, `385` after strict phrase filter |
| A2b sampled all-wrong teacher traces | `109` selected, `73` after strict phrase filter |
| **final Run 2 SFT size** | **`2543` after strict phrase filter** |

This composition intentionally avoids letting the `4676` all-correct examples
dominate the SFT set while still keeping a strong hint-free student anchor:
all mixed examples are included, and all-correct contributes exactly twice the
mixed count.

### Run 2 generated SFT files

Builder:

```text
scripts/teacher/build_run2_sft_dataset.py
```

Inputs:

| input | path |
| --- | --- |
| train source rows | `outputs/train-6601-schema-bare-tool.jsonl` |
| pass@16 per-example bands | `outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp2_shards4/merged/passk_per_example.jsonl` |
| pass@16 candidates | `outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp2_shards4/merged/passk_candidates.jsonl` |
| A2 greedy teacher traces | `outputs/teacher/a2_greedy_tp2_shards4/merged/teacher_traces.jsonl` |
| A2b sampled teacher traces | `outputs/teacher/a2b_uncovered_tp2_shards4/merged/teacher_traces.jsonl` |
| all-wrong teacher targets | `outputs/teacher/target_idx_all_wrong.txt` |
| A2-uncovered teacher targets | `outputs/teacher/target_idx_all_wrong_a2_uncovered.txt` |
| target summary | `outputs/teacher/target_idx_summary.json` |

Outputs, all preserved separately:

| output | records | path |
| --- | ---: | --- |
| mixed pass@16 student records | `691` | `outputs/teacher/rft_run2/run2_mixed_pass16_records.jsonl` |
| all-correct pass@16 student records | `1394` | `outputs/teacher/rft_run2/run2_all_correct_pass16_records.jsonl` |
| all pass@16 student records | `2085` | `outputs/teacher/rft_run2/run2_student_pass16_records.jsonl` |
| all-wrong teacher records (A2 + A2b) | `458` | `outputs/teacher/rft_run2/run2_teacher_records.jsonl` |
| combined, sorted by source group | `2543` | `outputs/teacher/rft_run2/train_rft_31b_run2.sorted.jsonl` |
| **combined, shuffled for SFT** | **`2543`** | **`outputs/teacher/rft_run2/train_rft_31b_run2.shuffled.jsonl`** |
| build summary | n/a | `outputs/teacher/rft_run2/run2_build_summary.json` |
| strict phrase removals | `87` | `outputs/teacher/rft_run2/run2_strict_phrase_removed_records.json` |


Strict phrase cleanup removed `87` selected records before writing the final SFT files:

| removed band | records removed |
| --- | ---: |
| mixed pass@16 | `6` |
| A2 teacher | `45` |
| A2b teacher | `36` |
| all-correct pass@16 | `0` |

The raw source artifacts are preserved unchanged; only the Run 2 derived JSONLs under
`outputs/teacher/rft_run2/` were rewritten after filtering.

The final training file should be the shuffled file:

```text
outputs/teacher/rft_run2/train_rft_31b_run2.shuffled.jsonl
```

Hugging Face `Trainer` shuffles the training dataloader by default, but Run 2
also writes a deterministic pre-shuffled JSONL (`shuffle_seed=0`). That keeps
the file safe for any sequential/streaming reader and avoids source-group
ordering effects before the trainer-level shuffle.

Final data gate on the shuffled file:

| check | result |
| --- | ---: |
| records | `2543` |
| distinct ids | `2543` |
| records with `internal_reference` | `0` |
| privileged teacher records with hard leaks | `0` |
| strict forbidden-phrase records | `0` |
| empty-message records | `0` |
| max tokens after chat-template rendering | `13720` |
| records over `MAX_SEQ_LEN=20480` | `0` |
| records with no supervised assistant tokens | `0` |
| masking boundary failures | `0` |
| tool-call reasoning fields starting with literal `thought` | `0` |
| repeated leading `thought` labels in final assistant content | `0` |
| extracted final SQL exactly matching `teacher_final_sql` | `2543 / 2543` |

Run 2 also normalizes decoded Gemma thought-channel artifacts during SFT
assembly:

- Structured assistant `reasoning` on tool-call turns strips leading decoded
  `thought` labels, because the Gemma chat template renders the thought channel
  itself.
- The upstream tool-loop transcript helper now applies the same cleanup before
  storing `reasoning`, so future pass@k/teacher trace generation does not
  preserve decoded channel labels in structured tool-call turns.
- Plain final assistant `content` keeps a single leading `thought` label when it
  appears after tool responses, matching current RL inference `prediction_text`,
  but repeated labels such as `thought\nthought\n...` are collapsed to one.
- The loaded `google/gemma-4-31B-it` tokenizer resolves to the cached main
  snapshot `842da3794eaa0b77d5f08bae87a17459d91ff475` for chat-template
  rendering.

Use this SFT command shape for Run 2:

```bash
TRAIN_FILE=outputs/teacher/rft_run2/train_rft_31b_run2.shuffled.jsonl \
OUT=outputs/sft/gemma4_31b_rft_sft_run2 \
MODEL=/home/ec2-user/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475 \
PY=/home/ec2-user/miniconda3/envs/nl2sql312/bin/python \
bash scripts/teacher/run_a4_sft.sh
```

### A1.1 result — the band sizes

From the completed base-model pass@16
(`outputs/passk/train6601_bare_tool_gemma4-31b-it_temp1p2_tp1_shards8`):

| band | examples | share | Stage A2 treatment |
| --- | ---: | ---: | --- |
| all-wrong (`0/16`) | `1233` | `18.68%` | gold-conditioned teacher traces, `--hint_strategy full_sql` |
| heterogeneous (`1-15/16`) | `689` | `10.44%` | hint-free self traces, `--hint_strategy none` |
| all-correct (`16/16`) | `4679` | `70.88%` | hint-free self traces, `--hint_strategy none` |

So the teacher stage targets **1233** examples and the anchor set is **5368**
examples.

### A1.2 result — failure labels over the 1233 all-wrong examples

Labels are not mutually exclusive; one example can carry several.

| label | examples |
| --- | ---: |
| `all_pred_sql_executed_but_wrong` | `1081` |
| `same_shape_but_wrong_values` | `649` |
| `low_diversity_repeated_wrong_sql` | `450` |
| `row_count_mismatch` | `327` |
| `high_diversity_no_correct_sql` | `271` |
| `output_column_count_mismatch` | `245` |
| `some_pred_sql_execution_failed` | `137` |
| `hit_max_tool_rounds` | `103` |
| `all_pred_sql_execution_failed` | `15` |
| `gold_sql_execution_failed` | `10` |
| `hit_max_new_tokens` | `9` |
| `sql_syntax_or_transcript_extraction_error` | `8` |

The dominant failure mode is **semantic, not syntactic**: `1081 / 1233` produce
SQL that runs cleanly but returns the wrong rows, and only `15` fail to execute
at all. `649` return the right *shape* with wrong values. That is the profile
teacher traces are meant to fix — the model can write valid SQL, it picks the
wrong joins, filters, or aggregations.

`low_diversity_repeated_wrong_sql` (`450`) is the sampling-collapse signature
again: on more than a third of these the model emits essentially the same wrong
query 16 times, so extra sampling cannot rescue them.

### A1.3 result — target bands

`scripts/teacher/make_target_idx.py` over the merged pass@16:

| band | ids | file | Stage A2 hint strategy |
| --- | ---: | --- | --- |
| teacher (all-wrong, teachable) | `1223` | `outputs/teacher/target_idx_all_wrong.txt` | `full_sql` |
| self-trace anchor | `1754` | `outputs/teacher/target_idx_selftrace.txt` | `none` |

Teacher band = `1233` all-wrong minus the `10` `gold_sql_execution_failed` ids
(gold does not execute, so verification against gold could never pass; teaching
them is impossible by construction). Dropped ids:
`1126, 1144, 1496, 1500, 1522, 2198, 2203, 4371, 4567, 4569`.

Self-trace band = **all** `689` sometimes ids (`num_correct` `1`-`15`, taken
whole, no filtering) + `1065` all-correct ids (`16/16`) after a per-database cap.

**The cap is a per-database quota on example count, not a pass@k threshold.**
There are `4679` all-correct examples against only `689` heterogeneous ones, so
without a cap the anchor set would be ~`5368` and dominated by whichever
databases happen to contain the most easy questions, skewing the SFT
distribution by database rather than by task.

We use **`--all-correct-cap-per-db 16`** (the original used `12`):

| cap | all-correct kept | self-trace band |
| ---: | ---: | ---: |
| `12` | `807` | `1496` |
| **`16`** | **`1065`** | **`1754`** |
| `-1` (off) | `4679` | `5368` |

At `16`, `1065` of `4679` all-correct examples are kept and `3614` dropped
across `69` databases. Raising the cap only changes the self-trace band; the
teacher band is byte-identical, so the completed A2/A2b runs remain valid.

The two id files are disjoint, sorted, unique, and every id was verified to sit
in the band it claims. Counts and the exact cap effect are recorded in
`outputs/teacher/target_idx_summary.json`.

### A2 result — greedy teacher traces

`bash scripts/teacher/run_a2_greedy.sh` — base `google/gemma-4-31B-it`, tp=8,
`full_sql` hint, 1 greedy sample per target, 12.6 min.

| metric | value |
| --- | ---: |
| targets | `1223` |
| samples generated | `1223` |
| verified (rows match gold) | `441` (`36.1%`) |
| hard-leaked | `19` |
| **kept (verified, no hard leak)** | **`431`** |
| **target coverage** | **`35.24%`** |
| soft-leak-only samples (kept) | `300` |
| copy rate over kept | `60.8%` |
| mean tool rounds over kept | `1.45` (zero-round: `0`) |

**Our greedy pass alone matches the reference run's greedy + sampled union**
(`431` / `35.24%` here versus `299` / `24.8%` greedy and `429` / `35.6%` after
A2b there). A2b should push us above that.

Why samples failed verification:

| reason | samples |
| --- | ---: |
| `result_mismatch` | `748` |
| `no_final_sql` | `31` |
| `pred_execution_failed` | `2` |
| `unsafe_sql` | `1` |

Even with the gold SQL visible, the teacher fails to reproduce gold rows on
`64%` of these prompts. That is a useful sanity signal: the hint is being used
as a reference to check against rather than blindly transcribed, and these are
genuinely the hardest examples in the set.

All `19` hard leaks were explicit verbal mentions — "the reference query"
(`13`), "ground truth" (`7`), plus one each of `internal_reference`,
`do_not_reveal`, and "gold query". No gold SQL was found verbalized in prose.

#### Copy flags matter for A3

| flag | kept traces | share |
| --- | ---: | ---: |
| `near_copy` (final SQL ≈ gold) | `260` | `60.3%` |
| `copy_first_call` (first tool SQL ≈ gold) | `220` | `51.0%` |
| no copy flag at all | `169` | `39.2%` |

Copy flags never reject a trace, but they change what Stage A3 can build:

| A3 option | records available |
| --- | ---: |
| keep everything | `431` |
| `--drop_copy_first_call` | `211` |
| fully copy-free only | `169` |

A `60.8%` copy rate means the teacher often writes gold on the first attempt and
verifies it, which is a weaker reasoning demonstration than deriving the query.
Since `zero_verify_copy` never fired (every kept trace ran at least one tool
round), even the copied ones show execute-and-check behaviour. The choice of
subset is deferred to A3.

### A2b result — sampled teacher traces, and the union

`bash scripts/teacher/run_a2b_sampled.sh` — `--num_samples 8 --temperature 0.7`,
**tp=1 x 8 shards**, 40 min (`06:04 -> 06:44 UTC`), 9784 samples.

| metric | A2 greedy | A2b sampled |
| --- | ---: | ---: |
| samples | `1223` | `9784` |
| verified | `441` | `3541` |
| hard-leaked | `19` | `285` |
| kept samples | `431` | `3366` |
| **kept unique ids** | **`431`** | **`538`** |
| coverage of 1223 | `35.24%` | `43.99%` |
| copy rate over kept | `60.8%` | `60.7%` |

**Union coverage: `541 / 1223` = `44.24%`.**

| overlap | ids |
| --- | ---: |
| found by both | `428` |
| new from A2b only | `110` |
| found by greedy only | `3` |

A2b adds `110` ids the greedy pass could not solve, but the two agree almost
entirely — only `3` ids were greedy-exclusive. Sampling at temperature `0.7`
therefore buys breadth (new ids) rather than replacing greedy, which is why the
original pipeline unions them rather than choosing one.

Our union (`541`, `44.24%`) exceeds the reference run's post-recovery figure
(`429`, `35.6%` of its 1204 band) by roughly `9` points.

Sharding note: `tp=1 x 8 shards` ran 9784 samples in 40 min. Shards received
`141`-`165` targets each and finished within seconds of one another, so the
`source_idx % num_shards` split is well balanced for this workload.

#### Copy profile is stable across both runs

| run | kept | copy-free | `copy_first_call` |
| --- | ---: | ---: | ---: |
| A2 greedy | `431` | `169` (`39.2%`) | `220` (`51.0%`) |
| A2b sampled | `3366` | `1322` (`39.3%`) | `1668` (`49.6%`) |

Sampling at `0.7` did **not** reduce the copy rate — both runs sit at ~`39%`
copy-free. The expectation that higher temperature would diverge more from gold
did not hold; the teacher tends to write gold on its first attempt regardless of
sampling temperature.

#### What this leaves for A3

Over the `541` union ids:

| quantity | value |
| --- | ---: |
| ids with at least one copy-free trace | `269` (`49.7%`) |
| ids with at least two kept traces | `500` |
| records available at `--max_per_idx=2` | `1041` |

So `--max_per_idx=2` yields about `1041` teacher records, and restricting to
copy-free traces would cap the teacher band at `269` ids. That trade — roughly
`4x` more records versus demonstrably-reasoned ones — is the main decision at A3.

### A3a result — hint-free self-traces (anchor band)

`HINT=none NUM_SAMPLES=8 TEMPERATURE=1.2 TP=2 NUM_SHARDS=4 bash scripts/teacher/run_a2b_sampled.sh`
— 59 min (`14:21 -> 15:21 UTC`), 14032 generations over 1754 anchor ids.

| metric | value |
| --- | ---: |
| targets | `1754` |
| samples generated | `14032` |
| verified | `11722` |
| hard-leaked | `48` |
| kept samples | `11703` |
| **kept unique ids** | **`1673`** |
| **coverage** | **`95.38%`** |
| copy rate over kept | `26.9%` |

#### Predicted versus measured

Because A3a samples at temperature `1.2` — the temperature the pass@16 bands
were measured at — per-sample success should equal `num_correct/16`, making
coverage from `n` samples predictable as `1-(1-num_correct/16)^n`. This was
stated before the run:

| sub-band | predicted | measured |
| --- | ---: | ---: |
| heterogeneous (`1-15/16`) | `620/689` (`90%`) | `608/689` (`88%`) |
| all-correct (`16/16`) | `1065/1065` (`100%`) | `1065/1065` (`100%`) |
| **total** | **`1685/1754` (`96%`)** | **`1673/1754` (`95.4%`)** |

The model matched theory to within `12` ids overall and `2` percentage points on
the hardest sub-band, and the all-correct band verified perfectly. That is a
strong check that the generation loop, the verification path and the band
definitions are all mutually consistent — a materially different result would
have implied a bug rather than a modelling surprise.

#### Self-traces are far less copy-prone than teacher traces

| run | kept samples | copy-free |
| --- | ---: | ---: |
| A2 greedy (hinted) | `431` | `39.2%` |
| A2b sampled (hinted) | `3366` | `39.3%` |
| **A3a self (unhinted)** | **`11703`** | **`73.2%`** |

Copy flags compare the produced SQL against gold, so a high rate under
`full_sql` means the teacher was transcribing the reference. With no reference
in the prompt, A3a's `73.2%` copy-free rate is the honest baseline — the
residual `26.8%` are simply cases where the model independently writes SQL that
matches gold closely, which is what solving the question correctly looks like.

#### `--max_per_idx=1` now has real choice

| quantity | value |
| --- | ---: |
| covered ids | `1673` |
| ids with `>=2` verified traces | `1607` |
| mean verified traces per covered id | `7.0` |
| ids where a copy-free trace is available | `1351` |

Generating 8 samples and keeping one means the ranking selects from `7.0`
verified candidates on average, and can pick a copy-free trace for `1351` of
`1673` ids. At the original `2` samples most covered ids would have had exactly
one verified trace and the ranking would have had nothing to rank.

### A3 result — assembled RFT dataset

```bash
python scripts/teacher/build_rft_from_traces.py \
  --traces outputs/teacher/a2_greedy/teacher_traces.jsonl \
           outputs/teacher/a2b_sampled/merged/teacher_traces.jsonl \
           outputs/teacher/a3a_selftrace/merged/teacher_traces.jsonl \
  --output outputs/teacher/rft/train_rft_31b.jsonl \
  --max_per_idx 1
```

Input was `15500` kept traces over `2214` distinct ids (mean `7.0` per id, max
`9`). Selecting one per id removes that redundancy — without it, `1233` ids
would appear 8 times and `107` ids once.

| metric | value |
| --- | ---: |
| **records** | **`2214`** |
| distinct ids | `2214` (one record per question) |
| teacher band (A2 u A2b) | `541` |
| self band (A3a) | `1673` |
| copy-free records | `1620` (`73.2%`) |
| mean tool rounds | `1.41` |
| records with `>=1` tool round | `2214` (all) |
| records ending in an assistant turn | `2214` (all) |

Final gate — both must be zero before the file is written:

| check | result |
| --- | ---: |
| records whose messages contain `internal_reference` | `0` |
| records with a hard leak on re-scan | `0` |

Rendered token length (400-record sample, Gemma chat template with tools):

| min | p50 | p90 | p95 | max |
| ---: | ---: | ---: | ---: | ---: |
| `4676` | `5510` | `6404` | `6819` | `9110` |

**No record exceeds `max_seq_len=20480`** — the longest is `9110` tokens, so
the A4 length filter will drop nothing.

Composition versus the reference run (`~1490` records): ours is `2214`, with
`541` teacher recoveries against their `~429`.

### A4 result — masked multi-turn SFT

`bash scripts/teacher/run_a4_sft.sh` — base `google/gemma-4-31B-it`, 8 GPUs,
DeepSpeed ZeRO-3 bf16 with CPU optimizer offload, 3 h 22 min.

| setting | value |
| --- | --- |
| model | `google/gemma-4-31B-it` (base; SFT is the warm start *before* RL) |
| train file | `outputs/teacher/rft/train_rft_31b.jsonl` (`2214` records) |
| deepspeed | `configs/ds_zero3_bf16_no_scheduler.json` |
| learning rate | `1e-5` |
| epochs | `2` |
| per-device batch / grad accum | `1` / `8` |
| effective batch | `1 x 8 GPUs x 8 = 64` |
| **total steps** | **`70`** (`ceil(2214/64) = 35` per epoch) |
| max_seq_len | `20480` |
| warmup ratio / scheduler | `0.03` / `cosine` |
| save_steps / save_total_limit | `10` / `8` |
| output | `outputs/sft/gemma4_31b_rft_sft` |

No new DeepSpeed config was needed. `ds_zero3_bf16_no_scheduler.json` differs
from the RL config only by dropping the `WarmupDecayLR` block, which is exactly
what lets HF Trainer's `lr_scheduler_type=cosine` and `warmup_ratio` drive the
schedule.

#### Dataset gate

All `2214` records were usable: `0` dropped for exceeding `max_seq_len`, `0` with
zero supervised tokens, `0` masking boundary failures. **Nothing is truncated** —
the trainer drops over-length records and reports the count.

Full-dataset sequence length is `4676` min / `5594` p50 / `16335` max, so
`max_seq_len=20480` is required; an earlier estimate of `9110` max came from a
400-record sample and would have justified lowering it, wrongly.

#### Loss and gradient trace

| step | 1 | 3 | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | `.358` | `.273` | `.238` | `.166` | `.161` | `.134` | `.117` | `.106` | `.116` | `.117` |
| grad_norm | `41.6` | `31.2` | `3522` | `5.78` | `8.79` | `4.18` | `2.46` | `11.2` | `21.5` | `17.2` |

Final `train_loss` `0.1442`.

Starting loss is low (`0.358`) because `1673` of `2214` records are the model's
own self-traces, which it already assigns high probability; the learning signal
is concentrated in the `541` teacher recoveries.

A single gradient spike to `3522` at step 5 resolved on its own — norms fell to
`250`, then `25.6`, then single digits by step 10. `gradient_clipping: 1.0` in
the DeepSpeed config bounds every update regardless, and no NaN/inf/overflow or
skipped step was logged. The most likely cause is batch composition: supervised
tokens per record span `156` to `6701` (a `43x` range, `15` records above
`3000`), and each micro-batch is a single record.

#### Checkpoints

Seven checkpoints at `59 GB` each (`~413 GB` total) plus the final model in the
run root:

`checkpoint-10`, `-20`, `-30`, `-40`, `-50`, `-60`, `-70`

Loss plateaus after roughly step 40 while the cosine schedule decays to zero, so
the final checkpoint is **not** automatically the best. A5 selects by dev
accuracy — the reference run peaked at its `ckpt-60` of ~80 and declined after.

### A5 result — dev evaluation across checkpoints

`bash scripts/teacher/run_a5_sweep.sh` — temp-0, `old-dev-schema-tool-unpatched.jsonl`
(1534 examples), async vLLM, **tp=1 with one checkpoint per GPU** so all seven
ran concurrently. Wall clock 2 h 37 min for the whole sweep. Results are written
inside each checkpoint folder under
`temp0_olddev_schema_tool_unpatched_vllm_async_tp1/`.

| checkpoint | accuracy | correct | simple | moderate | challenging |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | `71.45%` | `1096/1534` | `76.86` | `65.30` | `56.55` |
| **20** | **`72.62%`** | `1114/1534` | `77.30` | `66.59` | `62.07` |
| 30 | `71.97%` | `1104/1534` | `76.43` | `66.16` | `62.07` |
| 40 | `72.03%` | `1105/1534` | `77.19` | `65.52` | `60.00` |
| 50 | `72.10%` | `1106/1534` | `77.41` | `65.95` | `57.93` |
| 60 | `72.56%` | `1113/1534` | `77.84` | `65.95` | `60.00` |
| **70** | **`72.62%`** | `1114/1534` | `77.73` | `66.16` | `60.69` |

#### Comparison on the identical eval file

| model | dev temp-0 |
| --- | ---: |
| base `gemma-4-31B-it` on `old-dev-schema-bare-tool` | `69.17%` |
| base `gemma-4-31B-it` on `old-dev-schema-tool` | `71.19%` |
| **SFT best (ckpt-20 / ckpt-70)** | **`72.62%`** |
| RL `checkpoint-90` (same file, our rerun) | `73.73%` |

SFT improves on the base model but **does not reach the RL checkpoint**:
`72.62%` versus `73.73%`, a gap of `1.11 pp` (`17` examples). Against the base
model the gain is `+1.43 pp` over the tool-format baseline, or `+3.45 pp` over
bare-tool, which is the format the SFT data was actually built from.

#### The curve is flat

Accuracy moves within `71.45%`-`72.62%` across all seven checkpoints — a spread
of `1.17 pp`, or `18` examples out of 1534. After checkpoint 20 there is no
trend: `71.97`, `72.03`, `72.10`, `72.56`, `72.62`. Training loss fell from
`0.358` to `0.117` over the same range while dev accuracy did not move, so the
later steps were fitting the training set without generalising.

This differs from the reference run, which reported a rise to a clear peak at
its ckpt-60 (`72.69%`) followed by a decline. We see neither a strong peak nor
degradation. Two checkpoints tie at `72.62%`; **ckpt-20 is the better pick** —
identical accuracy at less than a third of the training, and further from any
memorisation risk.

Generation health is clean at every checkpoint: `1496`-`1522` of 1534 finished
normally, `11`-`38` hit `max_new_tokens`, and only one run hit a tool-round cap.
Tool calling was not degraded by SFT.

### A5b result — pass@16 on the SFT checkpoints

Temperature `1.2`, async vLLM, **tp=2 x 4 shards**, `24544` candidates each,
same `old-dev-schema-tool-unpatched.jsonl` used for the temp-0 sweep. Results
live inside each checkpoint folder under
`passk16_olddev_schema_tool_unpatched_temp1p2_tp2_shards4/`. About 1 h 45 min
per checkpoint; ckpt-70 was chained to start automatically once ckpt-20 released
the GPUs.

| model | pass@1 | pass@2 | pass@4 | pass@8 | **pass@16** | lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base `gemma-4-31B-it` | `68.68%` | — | — | `73.73%` | `74.97%` | `+6.28` |
| RL `checkpoint-90` | `73.68%` | — | — | — | `76.79%` | `+3.11` |
| **SFT ckpt-20** | `68.03%` | `75.00%` | `78.91%` | `81.50%` | **`83.44%`** | **`+15.41`** |
| **SFT ckpt-70** | `69.07%` | `76.00%` | `80.04%` | `82.83%` | **`84.88%`** | **`+15.81`** |

Both SFT checkpoints beat the RL checkpoint on pass@16 by `6.7`-`8.1 pp` and the
base model by `8.5`-`9.9 pp`, **despite lower pass@1 than RL**. Temp-0 accuracy
and pass@16 rank the models in opposite orders.

#### SFT reverses the RL entropy collapse

| model | mean distinct SQL / 16 | all-16-identical |
| --- | ---: | ---: |
| base `gemma-4-31B-it` | `3.37` | — |
| RL `checkpoint-90` | `2.02` | `49.0%` |
| SFT ckpt-20 | **`8.93`** | `0.3%` |
| SFT ckpt-70 | `7.77` | `2.2%` |

RL drove sampling diversity down from `3.37` to `2.02` distinct queries per 16
samples, with half of all dev prompts emitting sixteen byte-identical answers.
SFT on multi-turn reasoning traces raises it to `7.77`-`8.93`, with almost no
prompt collapsing to a single answer. This is the same phenomenon recorded in
`results.md` ("Maskfix Checkpoint Pass@K", "Beta schedule"), running in reverse.

Correct-candidate distribution over the 1534 dev examples:

| bucket | RL ckpt-90 | SFT ckpt-20 | SFT ckpt-70 |
| --- | ---: | ---: | ---: |
| all-wrong (`0/16`) | `23.2%` | `16.6%` | `15.1%` |
| mixed (`1-15`) | `7.2%` | **`49.5%`** | **`48.7%`** |
| all-correct (`16/16`) | `69.6%` | `33.9%` | `36.2%` |

The RL policy is bimodal — `93%` of prompts sit at an extreme. Both SFT
checkpoints put roughly half the dev set in the mixed band.

**This matters directly for the next RL run.** DAPO dynamic sampling keeps only
heterogeneous groups, and at `checkpoint-90` just `7.2%` of dev groups qualified,
which is the `dapo/selection_fill_rate` collapse logged in `results.md`
(`0.8333` at step 0 falling to `0.08333` at step 80). Warm-starting RL from an
SFT checkpoint with `~49%` heterogeneous groups should raise the fill rate by
roughly `7x` and make `K=16` oversampling far less wasteful.

#### ckpt-20 versus ckpt-70

| metric | ckpt-20 | ckpt-70 |
| --- | ---: | ---: |
| dev temp-0 | `72.62%` | `72.62%` |
| pass@1 | `68.03%` | `69.07%` |
| pass@16 | `83.44%` | **`84.88%`** |
| mean distinct SQL/16 | **`8.93`** | `7.77` |
| all-wrong share | `16.6%` | **`15.1%`** |

The two tie exactly on temp-0, but ckpt-70 is `1.44 pp` better on pass@16 with a
smaller all-wrong band, while ckpt-20 keeps more sampling diversity. The extra
50 training steps did **not** collapse diversity as over-training would predict
— `7.77` distinct SQL per 16 is still far above the base model's `3.37`.

Pick by intended use: **ckpt-70** for best sampling/reranking headroom, or
**ckpt-20** for maximum group heterogeneity as an RL warm start at a third of the
training cost. Both are equivalent for greedy decoding.

For reference, the original run reported `81.29%` pass@16 with `69.10%` per
candidate on its ckpt-60; ours reach `83.44%` and `84.88%`.

#### Implementation note — the identifier leak heuristic is disabled

A first version of `detect_leaks` flagged gold tables/columns appearing in
assistant prose. On an 8-example smoke test it rejected **8/8 traces, all false
positives**: the model writes its own candidate SQL as `CandidateSQL=...` inside
`<scratch_pad>` and in `call:sqlite_query{...}` payloads, and since the full
schema is in the prompt, naming gold's tables is simply what solving the
question looks like. Two fixes followed:

1. `strip_sql_regions` now removes `<sql_code>` blocks, tool calls, tool
   responses, `CandidateSQL=` lines and bare `SELECT`/`WITH` statements before
   scanning prose.
2. The identifier heuristic is **off by default** (`enable_identifier_heuristic`)
   and should stay off for schema-in-prompt formats.

Hard-leak detection now relies on explicit verbal mentions plus gold SQL
appearing in genuine prose. Running the smoke traces through the fixed detector
gives `0` hard leaks while still catching "gold query" and `internal_reference`
echoes.

### What has to be built here

The `feature/teacher_privelege_info` branch is not present locally and not on
`origin` (`git ls-remote --heads origin | grep teacher` returns nothing), and is
not expected to become available. Stages A2–A4 are therefore new work in this
repo.

Useful building blocks that already exist and should be reused rather than
rewritten:

| need | existing code |
| --- | --- |
| agentic tool loop (draft SQL → `sqlite_query` → final answer) | `generate_one_with_vllm_async_tool_loop` in `scripts/run_inference_bird.py` |
| async vLLM engine setup, sharding, prompt-length filtering | `scripts/run_inference_bird.py`, `scripts/run_passk_bird.py` |
| BIRD row-set equality + gold row caching | `bird_execute_sql`, `bird_get_gold_rows`, `bird_result_match` in `src/nl2sql_gspo/sql_utils.py` |
| safe read-only SQL check | `is_safe_readonly_sql` in `src/nl2sql_gspo/sql_utils.py` |
| final-SQL extraction from a transcript | `extract_sql` in `src/nl2sql_gspo/sql_utils.py` |
| tool definitions / catalog | `src/nl2sql_gspo/tool_calling.py`, `gen_tools.py` |
| tool execution during inference | `src/nl2sql_gspo/inference_tool_executor.py` |
| DeepSpeed ZeRO-3 bf16 config for A4 | `configs/ds_zero3_bf16_no_scheduler.json` |

The genuinely new pieces are: privileged-hint injection, leakage detection, copy
detection, best-trace selection, and assistant-only label masking.

---

## Reference numbers from the original implementation

These are the colleague's numbers from the original run. **Our own results are
the authoritative ones and live in the per-stage "result" sections above** —
this section is kept only as a comparison baseline.

### Superseded by our measurements

Every data-pipeline figure below has been replaced by a measured value:

| stage | reference | **ours** | see |
| --- | ---: | ---: | --- |
| all-wrong band | `1204` | **`1233`** (`1223` teachable) | "A1.2 result" |
| A2 greedy coverage | `299` (`24.8%`) | **`431`** (`35.24%`) | "A2 result" |
| A2 u A2b coverage | `429` (`35.6%`) | **`541`** (`44.24%`) | "A2b result" |
| self-trace targets | `~1490` | **`1754`** | "A1.3 result" |
| self-trace coverage | not reported | **`1673`** (`95.4%`) | "A3a result" |
| SFT records | `~1490` | **`2214`** | "A3 result" |
| — teacher band | `~429` | **`541`** | |
| — self band | `~1060` + few hundred | **`1673`** | |
| SFT steps | `~80` | **`70`** | "A4 result" |
| best dev temp-0 | `72.69%` (ckpt-60) | **`72.62%`** (ckpt-20/70) | "A5 result" |
| best pass@16 | `81.29%` (ckpt-60) | **`84.88%`** (ckpt-70) | "A5b result" |

### A5 — dev evaluation (their numbers; ours pending)

| checkpoint | accuracy |
| --- | ---: |
| ckpt-20 | `70.80%` |
| ckpt-40 | `70.66%` |
| **ckpt-60** | **`72.69%` (best)** |
| ckpt-80 | `72.43%` |

Pass@16 on checkpoint-60 — 8 shards, `24544` candidates, temperature `1.2`:

| metric | value |
| --- | ---: |
| pass@16 | `81.29%` |
| per-candidate @1 | `69.10%` |
| gain over previous RL/base ceiling (~`77%`) | ~`+4.4 pp` |

Their evaluation input is not recorded, which matters: base 31B scores `71.19%`
on `old-dev-schema-tool` but `69.17%` on `old-dev-schema-bare-tool`
(`results.md`), so the apparent SFT gain swings between ~`+1.5 pp` and
~`+3.5 pp` depending on which file they used. Our A5 sweep fixes the input
explicitly at `outputs/old-dev-schema-tool-unpatched.jsonl`, which is also what
`outputs/checkpoint-90` (the RL model, `73.73%`) was measured on — so our SFT
and RL numbers are directly comparable to each other.

---

## Deliberate divergences from the original run

Decided, not accidental. Our numbers will differ from the reference numbers for
these reasons.

**1. `max_per_idx=1`, same as the original — but generate far wider.** Rather
than raising records-per-id to offset our smaller sometimes band, we raise the
*sample count* and keep the single best trace. Every kept trace still records
its `copy_flags` in the record metadata, so a copy-free variant of the dataset
can be filtered later without re-running any generation.

**1b. All-correct cap `16` instead of `12`.** Widens the anchor band from `1496`
to `1754` ids (`1065` all-correct instead of `807`), giving more of the model's
own correct behaviour to hold the SFT distribution in place.

**1c. A3a runs `8` samples at temperature `1.2`, not `2` at `0.7`.**
Temperature `1.2` is the temperature the pass@16 bands were *measured* at, so
per-sample success equals the known `num_correct/16` rather than an unmeasured
shift. Raising samples from `2` to `8` lifts predicted anchor coverage from
`89%` to `96%` of the 1754 ids (heterogeneous `71% -> 90%`), at the cost of
`3508 -> 14032` generations.

Verification is unchanged and still required at `hint_strategy=none`. Leak
detection is inert there — with no privileged reference in the prompt there is
nothing to leak — but *verification* is about label correctness, not leakage:
on the heterogeneous band the unaided model is frequently wrong, and unverified
traces would teach it its own errors on exactly the questions it is worst at.

**2. Our sometimes band is smaller and we accept it.** Ours is **`689`**
heterogeneous examples; the original drew ~`1060` records from its sometimes
band, so its band must have been larger — probably a different sampling
temperature or model revision in its pass@16. We are **not** re-running A1.1 to
match. The SFT composition will be weighted more toward teacher recoveries and
less toward student self-traces, and that is fine.

**3. pass@16 on train, not dev.** The reference A5 pass@16 is the dev split
(`24544 = 1534 x 16`). Ours is the train split (`105616 = 6601 x 16`), which is
correct for this purpose: A1 exists to select *training* examples for reasoning
traces, and SFT is done on the train file. Do not compare our `81.32%` train
pass@16 against their `81.29%` dev pass@16 — the near-identical values are a
coincidence between two different measurements.

## Open questions

Not blocking, but unresolved.

**1. The reference A3 composition does not sum to its own total.**
The record count is given as `1490`, but the rows are `429` + `~1060` + "few
hundred" ≈ `1790+`. Either the `~1060` figure already includes all-correct
records, or the true total exceeds `1490`. `1490` also appears twice — as A3a
*ids targeted* and as A3 *records produced* — which is suspicious, since not
every targeted id yields a verified trace. Affects only how we read the
reference numbers, not what we do.

**2. Our all-wrong band is 1233; theirs is 1204 (+29).** Expected variation
across pass@16 runs at temperature 1.2, plus the analyzer drops
`gold_sql_execution_failed` rows. Our A2 target list will not be identical.

**3. Baseline for the "+4.4 pp" claim.** Their ~`77%` ceiling matches the RL
checkpoint's dev pass@16 in `results.md`, not the base model's `74.97%`. State
explicitly which baseline any future gain is measured against.

**4. SFT best (72.69%) vs our RL checkpoint (73.73%).** Their best SFT
checkpoint on dev temp-0 is slightly *below* the `73.73%` we measured for
`checkpoint-90`. The comparison may be unsound if the eval file differs
(bare-tool vs tool, patched vs unpatched). Base 31B is `71.19%` on the tool
split and `69.17%` on bare-tool (`results.md`), so the baseline choice swings the
apparent SFT gain between ~`+1.5 pp` and ~`+3.5 pp`. Confirm the eval input
before concluding anything about SFT vs RL.

### Files this pipeline will produce

None of these exist yet; each is created by the stage that owns it.

| stage | produces |
| --- | --- |
| A1.2 | `all_wrong_analysis.jsonl`, `all_wrong_summary.md`, `outputs/train-6601-all-wrong.jsonl` |
| A1.3 | `TARGET_IDX_FILE` (list of all-wrong `source_idx` values) |
| A2 | `teacher_traces.jsonl`, `teacher_summary.json`, `skipped_prompts.jsonl` |
| A3 | RFT JSONL, `rft_build_summary.json` |
| A4 | SFT checkpoints under the run output dir |
