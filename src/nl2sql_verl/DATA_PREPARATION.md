# Preparing the reference NL2SQL datasets

For future H200 runs, use the versioned native dataset below. The original
conversion remains available to reproduce the prompts used by the retired run.

## Versioned native Gemma dataset

`outputs/gemma4_native_v1/data/` contains 6,601 train rows, the same 640 validation
rows in the same order, and all 1,534 dev rows. Questions, evidence, gold SQL,
train ordering, eval few-shots, statistics and descriptions are preserved. Only
the system tool protocol and implicit foreign-key targets inside the schema
block change. Parent parquets remain in `outputs/cp1_recipe/data/`.

```bash
.venv-verl/bin/python -m src.nl2sql_verl.prepare_native_data \
  --model /opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
.venv-verl/bin/python -m src.nl2sql_verl.validate_native_data
```

Preparation refuses an existing output directory. Validation independently
checks corrected targets against SQLite's declared primary keys, every gold
and source row, and all user text outside the schema. The manifest records
parent parquet hashes, source/tokenizer hashes, database schema hashes,
corrected-row counts and prompt lengths. `foreign_key_corrections.json` lists
the actual old/new references.

SQLite permits `REFERENCES parent` without naming columns; that targets the
parent primary key. The old renderer emitted `.None`. The versioned builder
resolves single and composite primary keys in declared order and fails if a
target cannot be resolved. Corrections affect 2,068 train, 82 validation and
193 full-dev prompts. No unresolved `.None` schema references remain. No rows
are dropped or truncated. Maximum prompt lengths are 10,911 train, 29,618
validation and 29,775 full dev, within the 30,720-token prompt budget.

The `gemma4_native_v1` profile retains the task/SQL strategy, replaces the
incompatible bare-call instructions, and serializes every demonstration with
`<|tool_call>...<tool_call|>` and Gemma string delimiters. Its model-specific
serializer lives in `gemma_native_profile.py`; semantic tool schemas stay
shared. Other model profiles can provide different instructions and parsers
without changing SQLite behavior. The legacy `gemma4` profile is preserved.

Versioned rows route to `nl2sql_native_tool_agent` and `tools_native.yaml`.
These add actual execution/error counters, generation traces and exact final
SQL verification telemetry. The seven task reward channels and weights are
unchanged. The cumulative 4,096-token response budget includes masked tool
observations and formatting; each generation receives only its remaining budget.

The native reward boundary requires completed final-answer SQL. An unfinished
trajectory cannot earn execution/result/linking rewards from a fenced draft
earlier in its reasoning. This stricter native-only guard preserves the shared
legacy scorer for the retired run. `rescore_native_validation.py` can recheck
saved model generations after a terminal-only scorer change without generating
new model outputs; it preserves original results and records both source hashes.

Validation saves `fixed_validation_sample.json`: 128 train and 128 validation
examples, covering every database, the eight longest prompts in each split,
and seeded random fill. `validate_native_rollouts.py` runs this fixed sample
on an isolated TP4 engine and records raw token/stop traces, actual tool events,
format/execution/result scores, error recovery, caps and verified final SQL.

The future H200 recipe selects this version. Native-profile inference tests
alone do not certify full two-node training integration.

The spare-node integration harness is `run_native_smoke.py`; it explicitly
refuses any Ray cluster except the isolated eight-GPU node-2 cluster on port
6381. The actor capacity harness is `tests/h200_native_capacity.py`, using the
longest corrected train prompt plus 4,096 response loss tokens for two Adam
updates. After collecting these and both fixed-sample reports under
`outputs/h200_checks/native_v1/`, reconcile the evidence with:

```bash
.venv-verl/bin/python -m src.nl2sql_verl.validate_native_readiness
```

Its report distinguishes passed data/tool/one-node checks from the remaining
full two-node native-profile smoke. The old full-run readiness artifact is
not overwritten or relabeled. Raw model-validation results are in `base/` and
`step5/`; future native training also saves `<step>.tools.jsonl` beside the
ordinary rollout JSONL and logs actual tool counts to metrics/W&B.

## Original reference conversion

The entry point is `src/nl2sql_verl/prepare_reference_data.py`. It uses
`data/raw/train-6601.jsonl` as the training corpus and reward-SQL authority,
and preserves the prepared prompts from these supplied references:

- Train: `outputs/train-6601-schema-bare-tool.jsonl` (6,601 rows).
- Eval: `outputs/old-dev-schema-tool-unpatched.jsonl` (1,534 rows).

These references are local, ignored artifacts and must be staged with the raw
source to reproduce this preparation. This command converts and verifies the
existing prompts; it does not recompute schemas, column profiles, descriptions,
or BM25 retrieval. Source hashes in the output manifest identify exactly which
artifacts were used. Training rows follow the raw source order, which differs
from the train reference order; `extra_info.reference_index` records the match.

```bash
python3 -m venv .venv-data
.venv-data/bin/pip install -r src/nl2sql_verl/data-requirements.txt
.venv-data/bin/python -m src.nl2sql_verl.prepare_reference_data \
  --model /home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --eval-size 640
.venv-data/bin/python -m src.nl2sql_verl.validate_reference_data
```

Existing output parquet files are not overwritten. For another preparation,
pass a fresh `--output-dir`. Preparation uses the tokenizer on CPU and does not
start training, model inference, or SQL-answer evaluation.

## Prompt contents

| Component | Train reference | Eval reference |
|---|---|---|
| Question, evidence/hint, database ID | Yes | Yes |
| Table/column names, type labels, primary/foreign keys | Yes | Yes |
| Column descriptions | No | Yes, where supplied |
| Nullability labels | No | Yes |
| Row counts, null/non-null and distinct counts | No | Yes |
| Type-dependent min/average/max, frequent/sample values | No | Yes |
| Retrieved question/evidence/SQL examples | None | Three per prompt, from train |
| Three available tool declarations | Yes | Yes |
| Precomputed assistant/tool trajectories | None | None |
| Target gold SQL in generation prompt | No | No |

The system prompt also contains generic SQL/tool-use demonstrations in both
splits. Those are separate from retrieved, task-specific few-shot examples.
The eval reference is intentionally preserved as supplied, including its legacy
column descriptions/type labels. No "typefix" or text cleanup is substituted.

## Outputs and verl fields

Default output directory: `outputs/cp1_recipe/data/`.

- `train.parquet`: all 6,601 rows; duplicates retained.
- `validation.parquet`: fixed 640-row sample, used by the current recipe.
- `dev_full.parquet`: all 1,534 eval rows in reference order.
- `manifest.json`: input/code/tokenizer hashes, counts, subset indices,
  duplicate counts and exact tool-aware prompt length summaries.
- `validation_report.json`: saved-parquet parity with every reference row,
  raw training gold-SQL checks, database separation and few-shot provenance.
- `*_prompt_lengths.json`: per-row prompt token counts.

The 640-row subset uses Python's seeded sampler with seed `20261026`, retaining
reference order within the selected subset. Its indices are saved explicitly.
This is a new fixed subset; the previous machine's selection is not reproduced.
Use `--eval-size 0` to use all dev rows for `validation.parquet` too.

Parquet stores un-tokenized chat messages and structured metadata:

- `prompt`: system and user messages, without assistant answers/tool results.
- `reward_model.ground_truth`: JSON containing database ID and target SQL,
  used by the reward scorer rather than appended to the prompt.
- `extra_info`: source question/evidence, index, database ID, reference index,
  split, tool profile, selected tool names and per-tool database binding.
- `data_source`, `ability`, `agent_name`: verl routing fields.

Tool function schemas are supplied through the shared tool contract/runtime
configuration, rather than duplicated in a parquet `tools` column. Preparation
counts tokens with those schemas passed to the model's chat template. Actual
assistant tool calls and SQLite responses are generated during rollout.

No length filtering or truncation occurs. The manifest reports which rows
exceed `--max-prompt-length` (default 11,264); it does not certify the training
configuration or GPU capacity. Preserve the data and adjust/validate context
settings before launching a run if the manifest reports overlength prompts.

The current Gemma tokenizer measured train prompts at 4,405–10,791 tokens,
and full-dev prompts at 6,227–29,695 tokens. In the selected 640-row subset,
507 prompts exceed 11,264 tokens (maximum 29,538). Full dev has 1,193 such
prompts. These are prompt lengths including the actual tool schema/chat
template, before adding the generation budget. The existing 13,312-token
rollout context cannot cover the supplied enriched eval prompts.

## Tool modularity and other models

`tool_profiles.py` separates three concerns:

1. Semantic function declarations come from
   `consensus/tool_calling.py:get_tool_definitions` and match `configs/tools.yaml`.
2. The `gemma4` profile supplies the exact reference system instructions.
   A `custom` profile takes a complete `--system-prompt-file`, allowing another
   model's instructions without copying Gemma's `call:tool{...}` syntax.
3. The selected tokenizer's `apply_chat_template(..., tools=...)` serializes
   messages and tool declarations for length checks. The dataset builder does
   not write model-specific call tokens.

For example, a future model can use `--tool-profile custom --system-prompt-file
path/to/model_system_prompt.txt --model path/to/that/tokenizer`. The replacement
prompt must retain the task's final-answer/schema-linking requirements if using
the current rewards. Its verl rollout parser, tool format setting, model engine
and vLLM support must also be configured and tested separately. The current
runtime parser remains `nl2sql_gemma4`; custom parquet preparation alone does
not add Nemotron runtime support.

`prepare_data.py` remains the existing generic parquet normalizer.
`prepare_cp1_data.py` depends on the old machine's intermediate parquet files
and is not the entry point for these supplied JSONL references.

For generation of new schema/retrieval references from scratch, the historical
source is commit `588e1378777c64fa1a48d2548efccd51edd29270` on `consensus`:
`scripts/data_generation/schema_build.py`, `few_shot_bm25.py` and
`build_tool_dataset.py`. Those stages are not rerun by this converter.
