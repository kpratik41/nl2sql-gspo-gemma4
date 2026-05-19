# Copilot Instructions for NL2SQL RL Training

## Project Overview

This repository trains and evaluates a Gemma 4 31B instruction model on BIRD-style NL2SQL with reinforcement learning, tool-call rollouts, SQLite execution rewards, vLLM rollout serving, and post-training BIRD execution-accuracy evaluation.

The repository name still contains `gspo`, but the current standard GRPO training path no longer enables GSPO sequence-level importance sampling. Do not re-add `importance_sampling_level="sequence"` unless the user explicitly asks to restore GSPO. The active GRPO path is a TRL `GRPOTrainer` subclass with DAPO-style loss settings and DAPO-style dynamic sampling.

The target task is NL2SQL: given a natural-language question, schema, optional hint/evidence, and optional tool access, generate valid SQLite SQL inside the required final-answer XML shape.

Primary model target:

- `google/gemma-4-31B-it`
- Text-only training and inference in this repo
- Gemma 4 may expose multimodal modules; freeze non-text/multimodal modules if falling back to a multimodal model class

Primary training components:

- TRL `GRPOTrainer` via `DynamicSamplingGRPOTrainer`
- Optional TRL experimental `AsyncGRPOTrainer`
- vLLM server-mode rollouts
- DeepSpeed ZeRO-3 for the default GRPO path
- FSDP for the optional AsyncGRPO path
- BIRD-style SQLite execution rewards
- Native Gemma/OpenAI-style tool schemas and callable tool rollouts

## Source Of Truth

Prefer the actual launcher and Python entrypoints over older docs:

- Training launcher: `scripts/launch_train.sh`
- Training entrypoint: `src/nl2sql_gspo/train_gspo_nl2sql.py`
- Custom trainer: `src/nl2sql_gspo/dynamic_sampling_trainer.py`
- vLLM launcher: `scripts/launch_vllm.sh`
- Inference launcher: `scripts/launch_inference.sh`
- Inference/evaluation: `scripts/run_inference_bird.py`
- Rewards: `src/nl2sql_gspo/rewards.py`
- SQL/database helpers: `src/nl2sql_gspo/sql_utils.py`
- Tool definitions/wrappers: `src/nl2sql_gspo/tool_calling.py`
- Data normalization: `src/nl2sql_gspo/data.py`
- Model/tokenizer loading: `src/nl2sql_gspo/model_utils.py`

`configs/train_gspo_gemma4.yaml` currently has zero lines and is not the active config. Do not treat it as authoritative.

## Repository Layout

Expected top-level structure:

```text
nl2sql-gspo-gemma4/
├── .github/copilot-instructions.md
├── configs/
│   ├── ds_zero3_bf16.json
│   ├── fsdp_gemma4_bf16.json
│   └── train_gspo_gemma4.yaml      # currently empty
├── data/
│   ├── bird_train_data/raw/
│   └── bird_dev_data/raw/
├── databases/
│   ├── train_databases/
│   └── dev_databases/
├── gemma-4-31b-it-local/
├── scripts/
│   ├── launch_train.sh
│   ├── launch_vllm.sh
│   ├── launch_inference.sh
│   ├── run_inference_bird.py
│   ├── run_passk_bird.py
│   ├── run_self_consistency_bird.py
│   ├── generate_failure_instructions.py
│   ├── analyze_passk_all_wrong.py
│   ├── probe_train_heterogeneity.py
│   ├── smoke_test_rewards.py
│   ├── check_cluster_health.py
│   ├── what_if_rewards.py
│   └── data_generation/
├── src/nl2sql_gspo/
├── tests/test_core.py
├── gen_tools.py
├── prompts.py
├── requirements.txt
├── outputs/
└── logs/
```

Generated `outputs/` and `logs/` are part of normal workflows but should not be treated as source code. `temp.py` and `temp.jsonl` exist in the workspace; inspect before relying on them.

## Current Training Launcher Defaults

`scripts/launch_train.sh` is the active recipe. It starts 6 training processes on GPUs `0,1,2,3,4,5`; `scripts/launch_vllm.sh` starts the rollout server on GPUs `6,7`.

Current default training launch values:

```text
MODEL_NAME=google/gemma-4-31B-it
OUTPUT_DIR=outputs/gemma4_31b_gspo_bird
TRAIN_FILE=outputs/train-6601-schema-tool.jsonl
EVAL_FILE=outputs/dev-20251106-schema-tool.jsonl
DATABASE_DIR=databases
TRAIN_LIMIT=-1
EVAL_LIMIT=-1
TRAINER_BACKEND=grpo
DISTRIBUTED_BACKEND=deepspeed
DEEPSPEED_CONFIG=configs/ds_zero3_bf16.json
FSDP="full_shard auto_wrap"
FSDP_CONFIG=configs/fsdp_gemma4_bf16.json
```

Rollout and sampling defaults:

```text
MAX_PROMPT_LENGTH=13500
MAX_COMPLETION_LENGTH=8000
NUM_GENERATIONS=16
TEMPERATURE=1.2
TOP_P=0.95
ENABLE_TOOL_ROLLOUTS=1
ASYNC_MAX_TOOL_CALLING_ITERATIONS=8
VLLM_SERVER_BASE_URL=http://127.0.0.1:8000
VLLM_GROUP_PORT=29600
```

Optimization defaults:

```text
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=5e-7
NUM_TRAIN_EPOCHS=1
MAX_STEPS=-1
WARMUP_RATIO=0.03
BF16=true
```

Objective and clipping defaults:

```text
loss_type=dapo
scale_rewards=batch
beta=0.0
epsilon=0.2
epsilon_high=0.28
num_iterations=1
steps_per_generation=None unless STEPS_PER_GENERATION is set
mask_truncated_completions=false
```

DAPO dynamic-sampling defaults:

```text
ENABLE_DYNAMIC_SAMPLING=1
DYNAMIC_SAMPLING_MIN_STD=1e-6
DAPO_MAX_ROUNDS=1
DAPO_OVERSAMPLE_FACTOR=16
DYNAMIC_SAMPLING_REWARD_NAME=result_reward
```

Reward-shaping defaults:

```text
EXEC_TIMEOUT_S=60
REWARD_WORKERS=1
LENGTH_PENALTY_MAX=8000
LENGTH_PENALTY_BUFFER=512
REWARD_WEIGHTS=0.2,0.5,2.0,0.5,0.5,0.1,0.1
```

Logging and checkpoint defaults:

```text
WANDB_PROJECT=gemma4-31b-bird-gspo
REPORT_TO=wandb
RUN_NAME=gemma4-31b-gspo-bird-<timestamp>
LOGGING_DIR=outputs/gemma4_31b_gspo_bird/tb/<timestamp>
LOGGING_STEPS=10
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=10
SAVE_ONLY_MODEL=1
SAVE_LATEST_FULL_CHECKPOINT=1
LATEST_FULL_CHECKPOINT_DIR_NAME=latest-full-checkpoint
EVAL_STEPS=10
EVAL_ON_START=0
LOG_COMPLETIONS=0
NUM_COMPLETIONS_TO_PRINT=0
DAPO_DEBUG_ROLLOUTS=0
TOOL_LOOP_DEBUG=0
```

The launcher writes terminal logs to `logs/train_<timestamp>.log`.

## GRPO, DAPO, And GSPO Status

The default path uses `TRAINER_BACKEND=grpo`, which instantiates `DynamicSamplingGRPOTrainer` with TRL `GRPOConfig`.

Current GRPO path:

- Uses `loss_type="dapo"`
- Uses `scale_rewards="batch"`
- Uses asymmetric clipping through `epsilon=0.2`, `epsilon_high=0.28`
- Uses DAPO-style dynamic sampling in the custom trainer
- Uses vLLM server-mode rollouts
- Does not set `importance_sampling_level="sequence"`

Do not describe the current GRPO path as GSPO-enabled. The old setting:

```python
importance_sampling_level="sequence"
```

has intentionally been removed from `src/nl2sql_gspo/train_gspo_nl2sql.py`.

The repo may still log TRL sampling/importance-correction metrics because the custom trainer and TRL vLLM integration have their own rollout correction logic. That is not the same as explicitly configuring GSPO sequence-level importance sampling.

## DAPO Dynamic Sampling

`DynamicSamplingGRPOTrainer` implements DAPO-style dynamic sampling on top of TRL `GRPOTrainer`.

Core behavior:

- Standard generation/scoring is run on prompt groups.
- Each prompt group contains `num_generations` completions.
- Heterogeneity is checked per prompt group.
- By default, heterogeneity uses `result_reward` because `DYNAMIC_SAMPLING_REWARD_NAME=result_reward`.
- A group is heterogeneous when intra-group reward std is at least `dynamic_sampling_min_std`.
- With `dapo_oversample_factor > 1`, the trainer uses single-shot oversampling.
- With `dapo_oversample_factor == 1`, the trainer uses iterative oversample-and-replace up to `dapo_max_rounds`.
- Non-heterogeneous padding groups can have `completion_mask` zeroed so they contribute no policy gradient.
- `num_items_in_batch` is recomputed from the final completion mask.
- If all final groups are zero-masked, the trainer can skip the expensive policy loss path and return zero loss.

Current default single-shot volume with 6 trainer ranks, `per_device_train_batch_size=1`, `num_generations=16`, and `dapo_oversample_factor=16`:

```text
6 ranks * 1 group/rank * 16 generations/group * 16 oversample = 1536 candidate completions per generation step
```

If `PER_DEVICE_TRAIN_BATCH_SIZE=2`, this doubles to `3072` candidate completions per generation step.

Logged DAPO/debug metrics include:

- `dapo/rounds_used`
- `dapo/groups_attempted`
- `dapo/groups_heterogeneous`
- `dapo/groups_kept`
- `dapo/groups_padded`
- `dapo/heterogeneity_rate`
- `dapo/selection_fill_rate`
- `[dapo] step=...` summary lines
- `[rollout-debug] ... truncated=... completion_tokens ...`

Use logs to inspect truncation:

```bash
rg -n "rollout-debug|truncated=|overlong=|length_penalty_reward" logs/train_*.log
```

## Prompt Length And Context Budget

Training filters prompts before training/eval because current TRL no longer carries `max_prompt_length` inside `GRPOConfig`. Filtering uses:

```python
tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, tools=tools)
```

Always pass row-level `tools` while counting tool-dataset prompt length. Gemma tokenizers can return objects such as `BatchEncoding`; count actual `input_ids` or encoding IDs, not `len(BatchEncoding)`.

Current default:

```text
MAX_PROMPT_LENGTH=13500
MAX_COMPLETION_LENGTH=8000
VLLM_MAX_MODEL_LEN=24576
```

With `VLLM_MAX_MODEL_LEN=24576` and `MAX_COMPLETION_LENGTH=8000`, a prompt cap above roughly `16576` leaves less than 8000 tokens of completion headroom.

Recent measured rendered tool-template prompt lengths for current generated files:

```text
outputs/train-6601-schema-tool.jsonl:
  total=6601
  p50=11619
  p75=14321
  p90=21479
  p95=79031
  p99=79217
  max=79648

outputs/dev-20251106-schema-tool.jsonl:
  total=1534
  p50=14784
  p75=17531
  p90=22976
  p95=29400
  p99=29634
  max=29945
```

Measured filter counts:

```text
MAX_PROMPT_LENGTH=9000:
  train dropped 4287/6601, kept 2314
  eval  dropped 1196/1534, kept 338

MAX_PROMPT_LENGTH=13500:
  train dropped 1900/6601, kept 4701
  eval  dropped 792/1534, kept 742

MAX_PROMPT_LENGTH=15500:
  train dropped 1488/6601, kept 5113
  eval  dropped 590/1534, kept 944

MAX_PROMPT_LENGTH=16576:
  train dropped 1152/6601, kept 5449
  eval  dropped 462/1534, kept 1072

MAX_PROMPT_LENGTH=18000:
  train dropped 1101/6601, kept 5500
  eval  dropped 393/1534, kept 1141

MAX_PROMPT_LENGTH=20000:
  train dropped 1090/6601, kept 5511
  eval  dropped 320/1534, kept 1214
```

To filter nothing in the current train/eval files, the prompt cap would need at least `79648`, which is not compatible with the current vLLM context configuration. Prefer schema compaction or dataset regeneration for huge outliers rather than simply raising the cap.

## vLLM Server Workflow

Start vLLM before training:

```bash
bash scripts/launch_vllm.sh
```

Then start training:

```bash
bash scripts/launch_train.sh
```

Default `scripts/launch_vllm.sh` values:

```text
CUDA_VISIBLE_DEVICES=6,7
MODEL_NAME=google/gemma-4-31B-it
VLLM_SERVER_KIND=trl
VLLM_TENSOR_PARALLEL_SIZE=2
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_MAX_MODEL_LEN=24576
dtype=bfloat16
port=8000
```

For the default GRPO path, the vLLM launcher uses:

```bash
python -m nl2sql_gspo.vllm_serve_compat
```

This wrapper delegates to TRL's vLLM server entrypoint and adds weight-sync diagnostics around `WeightSyncWorkerExtension.update_named_param`.

For AsyncGRPO, start raw vLLM with:

```bash
VLLM_SERVER_KIND=async_grpo bash scripts/launch_vllm.sh
```

That path sets `VLLM_SERVER_DEV_MODE=1` and uses:

```text
--logprobs-mode processed_logprobs
--weight-transfer-config '{"backend":"nccl"}'
```

For Gemma 4, leave `VLLM_ATTENTION_BACKEND` unset unless intentionally debugging backend selection. vLLM 0.19.x has model-aware logic for Gemma 4; forcing `TORCH_SDPA` can bypass safeguards and break this model family.

Do not switch training to colocated vLLM unless explicitly requested. Server mode is the expected setup because it separates rollout memory from trainer memory.

## Distributed Training And Memory

Default GRPO training uses:

- 6 training GPUs
- DeepSpeed ZeRO-3
- bf16
- gradient checkpointing enabled
- vLLM rollouts on separate GPUs
- CPU optimizer offload

`configs/ds_zero3_bf16.json`:

- `bf16.enabled=true`
- `zero_optimization.stage=3`
- `offload_optimizer.device=cpu`
- `offload_optimizer.pin_memory=true`
- `gradient_clipping=1.0`
- AdamW with `lr`, `betas`, and `weight_decay` set to `auto`
- WarmupDecayLR with step counts set to `auto`

No explicit parameter offload is configured. Model parameters are sharded by ZeRO-3 but not configured for CPU parameter offload.

FSDP config in `configs/fsdp_gemma4_bf16.json`:

- wraps `Gemma4TextDecoderLayer`
- `activation_checkpointing=true`
- `use_orig_params=true`
- `sync_module_states=true`
- `forward_prefetch=false`
- `limit_all_gathers=true`

AsyncGRPO defaults to FSDP when `TRAINER_BACKEND=async_grpo`; the launcher picks FSDP automatically unless `DISTRIBUTED_BACKEND` is explicitly set.

Memory and sequence length:

- KV cache during rollout generation scales roughly linearly with sequence length.
- Efficient attention kernels make attention memory much better than naive quadratic attention, but attention compute is still very sequence-length sensitive.
- Any fallback path that materializes full attention matrices can become quadratic in memory.
- Long prompts plus tool loops can push total prompt+completion+tool context near `VLLM_MAX_MODEL_LEN`.

## AsyncGRPO Backend

The training script supports:

```text
TRAINER_BACKEND=async_grpo
DISTRIBUTED_BACKEND=fsdp
```

AsyncGRPO behavior:

- Imports `trl.experimental.async_grpo`
- Uses `AsyncGRPOConfig` and `AsyncGRPOTrainer`
- Requires FSDP or no distributed backend; DeepSpeed is rejected
- Loads tokenizer but passes the model name/path to the trainer instead of loading a local model object first
- Does not accept `eval_dataset`; eval file is loaded for normalization/filtering checks but online eval is skipped
- Ignores `top_p` in the current TRL API path
- Does not use custom dynamic sampling or `num_iterations`
- Does not use `save_latest_full_checkpoint`
- Applies reward weights by wrapping reward functions, because AsyncGRPO sums reward functions directly
- Uses sync wrappers for tool functions because the experimental rollout worker expects synchronous callables

Use:

```bash
VLLM_SERVER_KIND=async_grpo bash scripts/launch_vllm.sh
TRAINER_BACKEND=async_grpo DISTRIBUTED_BACKEND=fsdp bash scripts/launch_train.sh
```

Treat AsyncGRPO as experimental; TRL APIs may drift.

## Dataset Formats

The code supports normalized chat/tool records and raw BIRD-style records.

Raw BIRD-style examples can contain:

```json
{
  "db_id": "database_name",
  "question": "...",
  "evidence": "...",
  "SQL": "SELECT ..."
}
```

Schema-built/tool examples normally contain:

```json
{
  "db_id": "...",
  "gold_sql": "SELECT ...",
  "evidence": "...",
  "question": "...",
  "tools": [...],
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

`normalize_record` outputs:

```json
{
  "prompt": ["system/user only"],
  "messages": ["system/user only"],
  "db_id": "...",
  "gold_sql": "...",
  "evidence": "...",
  "question": "...",
  "tools": [...]
}
```

Rules:

- Assistant messages are excluded from the prompt.
- Assistant content can be used as `gold_sql` if no top-level gold field exists.
- Preserve uppercase BIRD `SQL` as `gold_sql`.
- Recover `db_id` from `<db_id>...</db_id>` or `<database_schema>\n`db_id`` in legacy prompts.
- Recover evidence from `<hint>...</hint>` in legacy prompts.
- Keep `remove_unused_columns=False` in trainer configs because rewards need `db_id`, `gold_sql`, `evidence`, `messages`, and sometimes `tools`.

## Data Generation

`scripts/data_generation/few_shot_bm25.py` generates BM25-based few-shot artifacts for train/dev. Train examples exclude retrieved few-shot examples from the same `db_id`.

`scripts/data_generation/schema_build.py`:

- Accepts JSONL and JSON array inputs
- Supports train/dev split defaults
- Reads DBs from `databases/<split>_databases`
- Reads split-specific column meanings
- Emits chat-format JSONL
- Can emit messages-only rows
- Includes top-level `db_id`, `gold_sql`, `evidence`, and `question` when not in messages-only mode
- Can include or omit few-shot examples
- Can include or omit stats
- Can include or omit nullability labels
- Renders row/column counts and per-column stats/examples
- Uses top-occurring concrete values as examples
- Truncates long rendered example values with `...`

`scripts/data_generation/build_tool_dataset.py` converts schema-built rows into tool-calling rows:

- Replaces the system prompt with `prompts.py::SYSTEM_PROMPT_TEMPLATES`
- Injects `{TOOL_CATALOG_COMPACT}` from `tool_catalog_compact()`
- Attaches native tool schemas from `get_tool_definitions()`
- Emits `db_id`, `gold_sql`, `evidence`, `question`, `tools`, `prompt`, and `messages`
- Keeps prompt/messages to system+user only
- Extracts gold SQL with the shared SQL extractor
- Raises if required fields are missing

Current tool dataset names used by training:

- `outputs/train-6601-schema-tool.jsonl`
- `outputs/dev-20251106-schema-tool.jsonl`

Other generated variants may exist, such as bare schema and bare-tool files, but the launcher defaults to the full `*-schema-tool.jsonl` files.

## Database Layout

The expected BIRD DB layout is:

```text
databases/
├── train_databases/
│   └── <db_id>/
│       ├── <db_id>.sqlite
│       └── database_description/
└── dev_databases/
    └── <db_id>/
        ├── <db_id>.sqlite
        └── database_description/
```

`get_database_path` supports:

- `database_dir/<db_id>/<db_id>.sqlite`
- `database_dir/<db_id>/<db_id>.db`
- `database_dir/<db_id>.sqlite`
- `database_dir/<db_id>.db`
- `database_dir/train_databases/<db_id>/<db_id>.sqlite`
- `database_dir/train_databases/<db_id>/<db_id>.db`
- `database_dir/dev_databases/<db_id>/<db_id>.sqlite`
- `database_dir/dev_databases/<db_id>/<db_id>.db`

Training uses `--database_dir databases`, so both train and dev DBs can be resolved.

## Tools

Tool declarations live in `src/nl2sql_gspo/tool_calling.py`; implementations live in `gen_tools.py`.

The intended tool set is exactly:

- `bm25_search_sqlite(db_id, table, column, query, top_k=10, where=None)`
- `sqlite_peek(db_id, table, columns, limit=10, where=None)`
- `sqlite_query(db_id, sql, max_return_rows=100)`

`gen_tools.py` may contain additional functions such as `consensus_at_1`, but they are not part of the current tool dataset or GRPO tool catalog.

Tool env setup:

- `configure_tool_db_roots(database_dir, extra_roots)` populates `BIRD_DB_ROOTS` if it is not already set.
- It includes `database_dir`, split subdirs, and default `databases` roots.
- Standard GRPO uses `get_grpo_tool_functions()`.
- AsyncGRPO uses `get_sync_grpo_tool_functions()` because its tool worker currently expects synchronous callables.

Tool prompt behavior:

- The prompt is draft-first and verification-oriented.
- The model should normally draft candidate SQL, verify with `sqlite_query`, then use `sqlite_peek` or `bm25_search_sqlite` when execution or semantic checks reveal uncertainty.
- The prompt requires `ExpectedOutputColumns=[...]` before calling `sqlite_query`.
- Successful `sqlite_query` responses include a `column_coverage_reminder` to reduce under-projection.
- Numeric scale checks should use schema ranges or `sqlite_peek` when values may be 0-1 fractions, 0-100 percentages, or counts.
- Predicate source fidelity matters: when similar columns exist across joined tables, choose the column whose table semantics match the question.

Reward extraction must not treat scratchpad `CandidateSQL` or SQL embedded inside `call:sqlite_query{...}` as the final answer. Unfinished tool-call rollouts should not receive execution/result reward.

## Output Format Contract

The current strict reward format expects:

```xml
<scratch_pad>...</scratch_pad>
<relevant_tables>...</relevant_tables>
<relevant_columns>...</relevant_columns>
<final_answer>
<sql_code>SELECT ...</sql_code>
</final_answer>
```

`format_reward` requires:

- non-empty `<scratch_pad>`
- non-empty `<relevant_tables>`
- non-empty `<relevant_columns>`
- final SQL inside `<final_answer><sql_code>...</sql_code></final_answer>`
- clean SQL code without nested scratch/final tags, tool calls, or code fences

## Rewards

`make_nl2sql_rewards` returns exactly seven reward functions in this order:

```text
format_reward
execution_reward
result_reward
table_linking_reward
column_linking_reward
nonnull_reward
length_penalty_reward
```

Default weights, in the same order:

```text
0.2, 0.5, 2.0, 0.5, 0.5, 0.1, 0.1
```

Reward meanings:

- `format_reward`: binary strict XML/output-shape reward.
- `execution_reward`: binary reward for predicted SQL executing successfully.
- `result_reward`: binary BIRD execution-accuracy reward using raw-row set equality.
- `table_linking_reward`: binary reward for predicted table set equaling gold table set.
- `column_linking_reward`: continuous Jaccard reward over predicted vs gold column sets.
- `nonnull_reward`: binary reward for execution success plus at least one non-null returned cell.
- `length_penalty_reward`: DAPO Soft Overlong Punishment, always `<= 0`.

BIRD result semantics:

```text
set(predicted_rows) == set(gold_rows)
```

Do not normalize strings, floats, whitespace, or row order for `result_reward`. It must stay aligned with standalone BIRD dev evaluation.

Execution details:

- Predicted SQL execution is cached per process by `(db_id, extracted_sql)`.
- Cache size is capped at `8192`.
- Gold execution is cached by `(db_id, gold_sql)`.
- `REWARD_WORKERS` controls per-rank thread parallelism for DB-backed rewards.
- `REWARD_WORKERS=1` means serial SQL reward execution per rank.
- Higher values parallelize reward execution but can stress SQLite/filesystem resources.
- `EXEC_TIMEOUT_S` is used for predicted and gold SQL reward execution.
- A hard timeout wrapper adds a 15 second buffer and can leak a stuck worker thread so the rank can continue to the next collective.

Length penalty:

- `LENGTH_PENALTY_MAX` should usually match `MAX_COMPLETION_LENGTH`.
- Current launcher defaults both to `8000`.
- `LENGTH_PENALTY_BUFFER=512`, so the penalty begins at `7488` tokens by default.
- With a tokenizer available, length is counted with tokenizer tokens; otherwise it falls back to word count.

## SQL Safety And Extraction

SQL helpers live in `src/nl2sql_gspo/sql_utils.py`.

Important extraction behavior:

- Prefer SQL inside the final-answer contract.
- Fall back to tagged SQL, fenced SQL, or raw SQL only when appropriate.
- If tool-call syntax exists but no final-answer SQL exists, return empty SQL.
- This prevents rewarding draft SQL or tool-call SQL arguments.

Readonly safety:

- Allowed SQL must contain `SELECT` or `WITH`.
- Disallowed keywords include `DROP`, `ALTER`, `TRUNCATE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`, `REINDEX`, `CREATE`, `INSERT`, `UPDATE`, and `DELETE`.
- SQLite connections are opened read-only where possible.

## Model And Tokenizer Loading

`model_utils.py` behavior:

- `load_tokenizer` resolves tokenizer files from local checkpoints or base model metadata.
- If a local model checkpoint lacks tokenizer files, it may fall back to `_name_or_path`, `name_or_path`, or `model_name_or_path` from `config.json`.
- If tokenizer fast loading fails, it retries with the slow tokenizer.
- If pad token is absent, it uses EOS as pad.
- If no chat template is present, it tries the `-it` instruct sibling.
- If still absent, it installs a plain text fallback chat template.
- `load_model_and_tokenizer` first tries `AutoModelForCausalLM` with bf16 and `attn_implementation="sdpa"`.
- If CausalLM fails and `AutoModelForImageTextToText` exists, it falls back and freezes multimodal modules.
- `model.config.use_cache=False` for training.
- Inference loading sets `use_cache=True` and tries SDPA, then eager, then default, then multimodal fallback.

## Checkpointing And Resume

There are two checkpoint modes:

- Rotating model-only checkpoints under `OUTPUT_DIR/checkpoint-*`
- Stable full restart checkpoint under `OUTPUT_DIR/latest-full-checkpoint`

Default launcher:

```text
SAVE_ONLY_MODEL=1
SAVE_TOTAL_LIMIT=10
SAVE_LATEST_FULL_CHECKPOINT=1
```

Model-only continuation:

```bash
MODEL_NAME=outputs/gemma4_31b_gspo_bird/checkpoint-100 \
OUTPUT_DIR=outputs/gemma4_31b_gspo_bird_restart \
bash scripts/launch_train.sh
```

Exact full resume:

```bash
RESUME_FROM_CHECKPOINT=outputs/gemma4_31b_gspo_bird/latest-full-checkpoint \
bash scripts/launch_train.sh
```

The launcher rejects using a model-only checkpoint as `RESUME_FROM_CHECKPOINT` because it lacks optimizer/scheduler/RNG/DeepSpeed state. For model-only continuation, set `MODEL_NAME` instead.

## Standalone Inference And Evaluation

Run inference after training or on a separate node:

```bash
bash scripts/launch_inference.sh
```

Default inference launcher values:

```text
INFERENCE_BACKEND=transformers
MODEL_PATH=outputs/gemma4_31b_gspo_bird
INPUT_FILE=outputs/dev-20251106-schema.jsonl
DATABASE_DIR=databases/dev_databases
DIFF_JSON_PATH=data/bird_dev_data/raw/dev_20251106.json
OUTPUT_DIR=outputs/bird_dev_inference_<timestamp>
NUM_EXAMPLES=-1
MAX_PROMPT_LENGTH=30000
MAX_NEW_TOKENS=4096
MAX_TOOL_ROUNDS=8
TEMPERATURE=0.0
TOP_P=1.0
EVAL_TIMEOUT=60
EVAL_WORKERS=16
```

Supported inference backends:

- `transformers`
- `vllm`
- `vllm_async`

Transformers backend:

- Defaults to `TRANSFORMERS_DEVICE_MAP=none`.
- Defaults `TRANSFORMERS_DATA_PARALLEL_SIZE=0`, meaning one worker per visible GPU.
- Intended for models that fit on a single GPU per worker.
- Use `TRANSFORMERS_DEVICE_MAP=auto` to shard one model across visible GPUs.

vLLM backend:

- Defaults to `VLLM_TENSOR_PARALLEL_SIZE=4`.
- Defaults to `VLLM_DATA_PARALLEL_SIZE=2`.
- Uses explicit worker processes for local data parallelism.
- Do not rely on single-process `LLM(data_parallel_size=...)` for vLLM 0.19.x.
- `vllm_async` also accepts `VLLM_ASYNC_CONCURRENCY`.

Inference filtering:

- Normalizes rows with `normalize_record`.
- Uses `prompt` when present.
- For legacy `messages`, strips assistant turns to avoid leaking gold SQL.
- Renders with row-level `tools`.
- Filters rows over `MAX_PROMPT_LENGTH` rather than truncating.
- Writes filtered rows to `filtered_examples.jsonl`.

Inference outputs include:

- `predict_dev.json`
- `prediction_details.jsonl`
- `filtered_examples.jsonl`
- `eval_results.jsonl`
- `eval_summary.json`
- `eval_summary.md`
- `eval_summary_by_difficulty.csv`
- `eval_summary_by_db.csv`

Local BIRD evaluation:

- Executes predicted and gold SQL.
- Compares raw row sets.
- Reports simple/moderate/challenging/total accuracy using dev difficulty JSON.
- Includes extraction and execution counts.
- Includes predicted-side and gold-side error text.

## Tool Inference

Tool inference is agentic, not one-shot.

`scripts/run_inference_bird.py`:

- Generates until a tool boundary.
- Parses Gemma-style `call:name{...}` tool calls.
- Executes calls through `inference_tool_executor.py` / `gen_tools.py`.
- Appends assistant tool calls plus tool responses.
- Re-renders the chat template.
- Continues until no tool calls remain or `max_tool_rounds` is reached.

This matters because Gemma 4 generation can stop at tool-response boundary tokens. A first tool call is not a final answer; the runner must feed tool results back and resume generation.

## Pass@k, Self-Consistency, And Analysis Scripts

Additional scripts:

- `scripts/run_passk_bird.py`: pass@k evaluation over sampled candidates.
- `scripts/run_self_consistency_bird.py`: generates multiple candidates, executes them, ignores empty result sets, then majority-votes over raw result sets.
- `scripts/generate_failure_instructions.py`: mines failure heuristics from heterogeneous train prompts.
- `scripts/analyze_passk_all_wrong.py`: analyzes pass@k all-wrong cases.
- `scripts/probe_train_heterogeneity.py`: probes reward heterogeneity on train data.
- `scripts/what_if_rewards.py`: reward what-if analysis.
- `scripts/smoke_test_rewards.py`: quick reward sanity checks.
- `scripts/inspect_jsonl.py`: inspect generated JSONL data.
- `scripts/check_cluster_health.py`: GPU/NCCL/topology health check.

Self-consistency rules:

- Vote signatures are unordered raw result sets.
- Empty execution results are ignored.
- Ties break by earliest sample index and then shorter SQL.
- Moderate temperatures such as `0.5-0.8` often help diversity; greedy generation can reduce voting value.

## Monitoring

Training logs:

- Terminal log: `logs/train_<timestamp>.log`
- vLLM log: `logs/vllm_<timestamp>.log`
- TensorBoard dir: `outputs/gemma4_31b_gspo_bird/tb/<timestamp>` when enabled
- W&B enabled by default via `WANDB_PROJECT` and `REPORT_TO=wandb`

Useful log searches:

```bash
rg -n "dapo\\]|rollout-debug|truncated=|completion_tokens|length_penalty_reward" logs/train_*.log
rg -n "dropped .*prompt >" logs/train_*.log
rg -n "weight sync|update_named_param|exception|timeout" logs/vllm_*.log
```

Common training metrics:

- reward means/stds per reward function
- aggregate reward and reward std
- `frac_reward_zero_std`
- DAPO attempted/heterogeneous/kept/padded groups
- completion lengths and clipped ratios
- tool call frequency and failures
- entropy
- clip ratios
- grad norm
- learning rate
- step time
- token count

## Tests

Run:

```bash
python -m unittest tests/test_core.py
```

Tests cover:

- raw BIRD record normalization
- schema-built message normalization
- prompt filtering with tokenizer objects that return `BatchEncoding`-like wrappers
- CLI parsing for training args
- SQL extraction, readonly safety, database path resolution
- BIRD raw-row set semantics
- reward cache behavior
- length penalty behavior
- nonnull reward behavior
- strict format reward behavior
- inference prompt preparation
- inference device grouping
- schema rendering options
- self-consistency voting
- dynamic sampling helper functions

When changing behavior, update or add focused tests in `tests/test_core.py`.

## Editing Guidance

Prefer narrow edits that respect existing module boundaries:

- Data format handling: `src/nl2sql_gspo/data.py`
- Model/tokenizer loading: `src/nl2sql_gspo/model_utils.py`
- Reward logic: `src/nl2sql_gspo/rewards.py`
- SQL safety/execution/extraction: `src/nl2sql_gspo/sql_utils.py`
- Tool definitions/wrappers: `src/nl2sql_gspo/tool_calling.py`
- Training CLI/config: `src/nl2sql_gspo/train_gspo_nl2sql.py`
- Dynamic sampling/trainer internals: `src/nl2sql_gspo/dynamic_sampling_trainer.py`
- vLLM server behavior: `src/nl2sql_gspo/vllm_serve_compat.py` and `scripts/launch_vllm.sh`
- Inference and evaluation: `scripts/run_inference_bird.py` and `scripts/launch_inference.sh`
- Dataset generation: `scripts/data_generation/`
- Distributed defaults: `scripts/launch_train.sh`, `scripts/launch_vllm.sh`, `configs/*.json`, and `train_gspo_nl2sql.py`

Do not silently change BIRD result semantics. Keep training reward and standalone evaluator aligned.

Do not reward SQL from scratchpads or tool calls as final SQL.

Do not include assistant gold SQL in training/inference prompts.

Do not treat README values as current without checking launchers; README may drift.

After meaningful workflow/config changes, update this file and consider updating `README.md`.

## Common Commands

Start rollout server:

```bash
bash scripts/launch_vllm.sh
```

Start training:

```bash
bash scripts/launch_train.sh
```

Short smoke training:

```bash
TRAIN_LIMIT=256 EVAL_LIMIT=128 MAX_STEPS=2 bash scripts/launch_train.sh
```

Override LR:

```bash
LEARNING_RATE=1e-6 bash scripts/launch_train.sh
```

Exact resume:

```bash
RESUME_FROM_CHECKPOINT=outputs/gemma4_31b_gspo_bird/latest-full-checkpoint bash scripts/launch_train.sh
```

Model-only continuation:

```bash
MODEL_NAME=outputs/gemma4_31b_gspo_bird/checkpoint-100 \
OUTPUT_DIR=outputs/gemma4_31b_gspo_bird_continue \
bash scripts/launch_train.sh
```

Post-training inference:

```bash
bash scripts/launch_inference.sh
```

Small inference smoke:

```bash
NUM_EXAMPLES=2 bash scripts/launch_inference.sh
```

vLLM inference:

```bash
INFERENCE_BACKEND=vllm NUM_EXAMPLES=2 bash scripts/launch_inference.sh
```

Cluster health check:

```bash
python scripts/check_cluster_health.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/check_cluster_health.py
```

Run tests:

```bash
python -m unittest tests/test_core.py
```
