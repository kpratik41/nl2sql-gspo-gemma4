# Copilot Instructions for NL2SQL GSPO Training Project

## Project Overview

This repository implements reinforcement learning training for a large language model on an NL2SQL task using:

- TRL `GRPOTrainer`
- GSPO-style sequence-level importance sampling
- vLLM server mode for rollouts
- DeepSpeed ZeRO-3 for distributed training
- BIRD-style NL2SQL train/dev data
- SQLite execution-based rewards

The current target model is a Gemma 4 31B (Instruction-tuned) multimodal model. For this project, only text input and text output are used. Vision/audio/multimodal modules should be frozen if present. The language model decoder should remain trainable unless explicitly switching to LoRA/adapters.

The task is NL2SQL: given a natural language question, schema, and optional evidence/hint, generate valid SQLite SQL.

---

## Repository Structure

Expected structure:

```text
nl2sql-gspo-gemma4/
├── .github/
│   └── copilot-instructions.md
├── CLAUDE.md
├── configs/
│   └── ds_zero3_bf16.json
├── scripts/
│   ├── launch_vllm.sh
│   ├── launch_train.sh
│   ├── launch_inference.sh
│   ├── check_cluster_health.py
│   ├── generate_failure_instructions.py
│   ├── run_inference_bird.py
│   ├── smoke_test_rewards.py
│   ├── inspect_jsonl.py
│   └── data_generation/
│       ├── schema_build.py
│       └── few_shot_bm25.py
├── src/
│   └── nl2sql_gspo/
│       ├── __init__.py
│       ├── train_gspo_nl2sql.py
│       ├── data.py
│       ├── model_utils.py
│       ├── rewards.py
│       ├── sql_utils.py
│       └── schema_utils.py
├── tests/
│   └── test_core.py
├── data/
│   ├── bird_train_data/
│   │   ├── raw/
│   │   └── processed/
│   └── bird_dev_data/
│       ├── raw/
│       └── processed/
├── databases/
│   ├── train_databases/
│   └── dev_databases/
├── outputs/
└── logs/

Current top-level processed files such as `data/bird_train.jsonl` and `data/bird_dev.jsonl` may not exist yet. Raw split-specific folders are the current source of truth.

Data Layout

The data/ folder holds raw and processed dataset artifacts.

data/bird_train_data/raw/ and data/bird_dev_data/raw/ contain original downloaded BIRD or Hugging Face source files.
data/bird_train_data/processed/ and data/bird_dev_data/processed/ contain cleaned or converted files used for training/evaluation.

Raw train sources in this workspace are commonly JSONL, while raw dev sources may be JSON arrays.

Generated few-shot artifacts may also be stored alongside the raw split files, for example `data/bird_train_data/raw/train-6601-few-shot.jsonl` and `data/bird_dev_data/raw/dev_20251106-few-shot.json`.

The databases/ folder holds SQLite databases.

The current workspace contains 69 train databases under `databases/train_databases` and 11 dev databases under `databases/dev_databases`.

Expected database layout:

databases/
├── train_databases/
│   └── <db_id>/
│       ├── <db_id>.sqlite
│       └── database_description/
└── dev_databases/
    └── <db_id>/
        ├── <db_id>.sqlite
        └── database_description/

Training should usually point to:

--database_dir databases/train_databases

Evaluation/dev should usually point to:

--database_dir databases/dev_databases

Do not assume all databases live directly under databases/. The code supports split-based database folders and may also be passed the top-level databases/ directory.

Dataset Format

Training data is expected to be JSONL, and the code should handle both normalized chat-style records and raw BIRD-style records.

Chat/harmony-style input:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "SELECT ..."}
  ],
  "db_id": "database_name",
  "gold_sql": "SELECT ...",
  "evidence": "optional external knowledge or hint"
}

Raw BIRD-style input also appears in this workspace:

{
  "db_id": "database_name",
  "question": "...",
  "evidence": "optional external knowledge or hint",
  "SQL": "SELECT ..."
}

Utility scripts under `scripts/data_generation/` may need to support both raw JSONL and JSON-array dataset files.

The `scripts/data_generation/few_shot_bm25.py` utility generates both inference and training few-shot files. For dev/inference data, each example receives top-k few-shot examples retrieved from the train file. For train data, each example receives top-k few-shot examples retrieved from the train file while excluding examples from the same `db_id`.

The `scripts/data_generation/schema_build.py` utility supports both train and dev splits. It should be able to consume JSONL or JSON inputs, default to the split-specific few-shot file and column-meaning file in `data/bird_<split>_data/raw/`, read schemas from `databases/<split>_databases/`, and emit chat-format JSONL under `outputs/`. It also renders compact schema statistics such as table row/column counts and per-column null/distinct/min/max summaries. `Examples:` should show the top occurring concrete values for numeric, date, and text columns rather than summary stats, and any rendered example longer than 50 characters should be truncated with `...`. The renderer also supports `--no-nullability` to omit `Nullable` / `Not Null` labels when a leaner schema prompt is needed. When invoked with `--no-fewshots`, it should omit the few-shot preamble entirely so the user prompt starts at `<question>` rather than leaving a placeholder block.

Schema-built JSONL should include top-level `db_id`, `gold_sql`, `evidence`, and `question` fields in addition to `messages`. The shared normalizer also recovers `db_id` and `evidence` from older message-only schema-built prompts that embed `<db_id>` and `<hint>` tags.

The training prompt should contain only system/user messages. The assistant message should be used as the gold SQL target for rewards, not included in the prompt.
When normalizing raw BIRD data, preserve uppercase `SQL` fields as `gold_sql`.

The normalized dataset should expose these fields:

{
    "prompt": [...],
    "messages": [...],
    "db_id": "...",
    "gold_sql": "...",
    "evidence": "..."
}

Keep remove_unused_columns=False in TRL config because reward functions need db_id, gold_sql, evidence, and messages.

Training Method

This project uses TRL GRPOTrainer, but configured for GSPO-style training by setting:

importance_sampling_level="sequence"

This is important. Do not accidentally remove this setting.

The current intended setup is:
Training GPUs: 0,1,2,3,4,5
vLLM GPUs:     6,7
Training:      DeepSpeed ZeRO-3
Rollouts:      vLLM server mode
Precision:     bf16

The vLLM server should be started separately before training.

vLLM Server Mode

The training script expects vLLM to be running in server mode.

Typical command:
bash scripts/launch_vllm.sh

Then training:
bash scripts/launch_train.sh

To resume from a saved checkpoint through the launcher workflow, set `RESUME_FROM_CHECKPOINT` before running `bash scripts/launch_train.sh`.

The training script uses:
use_vllm=True
vllm_mode="server"
vllm_server_base_url="http://127.0.0.1:8000"

The training script also accepts an optional `--resume_from_checkpoint` argument for explicit checkpoint resume.

The current launcher recipe trains from `outputs/train-6601-schema-filtered.jsonl` and evaluates on `outputs/dev-20251106-schema-256.jsonl`.
It currently uses `num_generations=16`, launcher default `gradient_accumulation_steps=16` (override via `GRADIENT_ACCUMULATION_STEPS`), launcher default `max_prompt_length=16000` (override via `MAX_PROMPT_LENGTH`), and `max_completion_length=4096` with vLLM server mode (vLLM `max_model_len=24576`), and the launcher defaults to `google/gemma-4-31B-it`.
The training launcher also supports `MAX_STEPS` for short smoke runs while keeping the normal epoch-based recipe as the default.
The training launcher accepts `LEARNING_RATE` or `LR` env overrides for one-off runs without editing the script.
The training launcher defaults to model-only rotating checkpoints (`SAVE_ONLY_MODEL=1`, `SAVE_TOTAL_LIMIT=3`), which write HF model weights/config and trainer state while skipping DeepSpeed optimizer/scheduler/scaler/RNG state.
The launcher also defaults `SAVE_LATEST_FULL_CHECKPOINT=1`, which refreshes `OUTPUT_DIR/latest-full-checkpoint` on every save with a full DeepSpeed resume checkpoint containing optimizer/scheduler/RNG state.
For model-only checkpoints, continue training by setting `MODEL_NAME` to the checkpoint directory and leaving `RESUME_FROM_CHECKPOINT` unset. For exact resume, set `RESUME_FROM_CHECKPOINT` to `OUTPUT_DIR/latest-full-checkpoint`.
The vLLM launcher routes through `python -m nl2sql_gspo.vllm_serve_compat`, a local wrapper around TRL's server entrypoint.
The local `nl2sql_gspo.vllm_serve_compat` wrapper also installs a weight-sync diagnostic around TRL's `WeightSyncWorkerExtension.update_named_param`, so server logs include the parameter name, dtype, shape, approximate GiB size, and elapsed time for each incoming weight update, plus the same metadata on exceptions/timeouts.
The repository also includes `scripts/check_cluster_health.py`, a local health-check utility that reports GPU inventory, `nvidia-smi` topology, peer-access status, NCCL-related env vars, and runs a small NCCL all-reduce/broadcast/all-gather smoke test across the currently visible GPUs.
For `google/gemma-4-31B-it`, leave `VLLM_ATTENTION_BACKEND` unset by default. vLLM `0.19.x+` contains a Gemma 4 config hook that detects heterogeneous `head_dim` / `global_head_dim` and forces `TRITON_ATTN` when `global_head_dim > 256`; explicitly setting `VLLM_ATTENTION_BACKEND=TORCH_SDPA` bypasses that safeguard and breaks this model family.
The training launcher skips eval-on-start by default; set `EVAL_ON_START=1` to run a pre-training dev baseline. It also supports optional `--train_limit` / `--eval_limit` row caps for smoke runs.

Do not switch to colocated vLLM unless explicitly requested. Server mode is preferred for this project because it separates rollout memory from training memory.

Model Loading Rules

The model is Gemma 4 31B or an internal equivalent.

Because Gemma 4 may be multimodal, model loading should:

Try AutoModelForCausalLM first for text-only training.
If that fails, try a multimodal class such as AutoModelForImageTextToText.
Freeze non-text modules if present.

Reward Functions

Reward functions live in: src/nl2sql_gspo/rewards.py
Supporting SQL/database utilities live in: src/nl2sql_gspo/sql_utils.py
Supporting schema utilities live in: src/nl2sql_gspo/schema_utils.py
Dynamic-sampling trainer subclass lives in: src/nl2sql_gspo/dynamic_sampling_trainer.py

Current reward functions are:

- `format_reward` (binary): strict `<scratch_pad>...</scratch_pad><final_answer><sql_code>...</sql_code></final_answer>` regex match
- `execution_reward` (binary): predicted SQL executes without error
- `result_reward` (binary, BIRD EX): uses official BIRD semantics — `set(predicted_rows) == set(gold_rows)` on RAW rows (no normalization), with a per-query timeout. Implemented via `bird_execute_sql` + `bird_get_gold_rows` (with a per-process gold cache keyed by `(db_id, gold_sql)`) + `bird_result_match` in `src/nl2sql_gspo/sql_utils.py`.
- `table_linking_reward` (binary): predicted table set == gold table set
- `column_linking_reward` (continuous): Jaccard of pred vs gold column sets
- `nonnull_reward` (binary, small): predicted SQL executes AND returns at least one non-null cell
- `length_penalty_reward` (continuous, ≤ 0): DAPO §3.4 Soft Overlong Punishment. 0 when `len ≤ L_max - L_cache`, linear ramp to -1 within the buffer, saturates at -1 beyond `L_max`. Length is measured in tokens via the trainer's tokenizer when available; falls back to `len(text.split())` otherwise.

Default reward weights: `[0.2, 0.5, 2.0, 0.5, 0.5, 0.1, 0.1]` (max positive weighted reward = 3.8; length penalty contributes ≤ 0). The old `schema_linking_reward`, `ngram_reward`, and `evidence_utilization_reward` have been removed.

Reward execution timeouts are configurable via `--exec_timeout_s` (default 60s for both predicted and gold SQL — more permissive than BIRD's 30s default to avoid penalising slow reference queries during training). The Soft Overlong Punishment knobs are `--length_penalty_max` (default 4096, should match `--max_completion_length`) and `--length_penalty_buffer` (default 512, ≈ 12.5% of L_max per the DAPO recipe).

Do NOT change `result_reward` away from BIRD set semantics on raw rows; it must stay aligned with the dev-set evaluator in `scripts/run_inference_bird.py` and the official `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`.

The training script also supports configurable monitoring backends via `--report_to`, `--reward_workers` for per-rank thread parallelism in DB-backed rewards such as `result_reward`, and writes TensorBoard logs to `--logging_dir` when `tensorboard` is included. The launchers timestamp terminal logs under `logs/`, default W&B run names, and default TensorBoard directories.

The trainer is `DynamicSamplingGRPOTrainer` from `src/nl2sql_gspo/dynamic_sampling_trainer.py` — a subclass of TRL's `GRPOTrainer` implementing DAPO §3.2 dynamic sampling in two modes: **single-shot oversample** (preferred, enabled when `--dapo_oversample_factor > 1`) and **iterative oversample-and-replace** (when `--dapo_oversample_factor == 1`). In single-shot mode each rank generates `K * target_local_groups` groups in ONE rollout round (round 1 uses dataloader prompts; remaining `(K-1) * target_local_groups` come from a shared shuffled backup queue over `train_dataset`, consumed from the tail in all-rank blocks and reshuffled only after exhaustion), computes candidate rewards, then keeps the first `target_local_groups` heterogeneous groups; if fewer are found, the remaining slots are filled with **random non-heterogeneous groups whose `completion_mask` is zeroed** (no gradient contribution). Candidate reward filtering happens before the expensive policy old-logprob / vLLM importance-correction pass, so trainer-side logprobs are computed only for the final kept/padded groups rather than all `K` candidates. If every final group is zero-masked, the trainer marks the batch with `_skip_policy_loss` and returns a scalar zero loss after `_prepare_inputs`, avoiding a wasteful DeepSpeed forward/backward on an all-zero batch. In iterative mode, each call runs up to `--dapo_max_rounds` rollout rounds, keeping heterogeneous groups across rounds; rank-shape is kept uniform via `accelerator.gather(need_local).max()` which causes already-filled ranks to do redundant rollouts, so the single-shot path is recommended unless the oversample factor would be too large. Heterogeneity = intra-group reward std `>= --dynamic_sampling_min_std`; by default the `--dynamic_sampling_reward_name` reward (default `result_reward`) is used. If the budget is exhausted the buffer is padded with zero-masked groups so output shape matches what TRL expects. After concatenation, `num_items_in_batch` is recomputed from the final `completion_mask`. Logged metrics: `dapo/rounds_used`, `dapo/groups_attempted`, `dapo/groups_heterogeneous`, `dapo/groups_kept`, `dapo/groups_padded`, `dapo/heterogeneity_rate`, `dapo/selection_fill_rate`. The trainer prints a one-line config summary on rank 0 at startup and a `[dapo] step=N ...` line after each generation step.

The training script also supports `--num_iterations` (PPO mu, launcher default `1`), `--steps_per_generation`, `--mask_truncated_completions`, `--vllm_group_port` (launcher default `29600` to avoid ephemeral-port collisions during vLLM weight updates), `--reward_workers` for per-rank thread parallelism in SQL execution rewards, and optional `--log_completions` sample-table printing. The launcher defaults `MASK_TRUNCATED_COMPLETIONS=0`; prompt length is handled by filtering rows before training/eval, and completion-side overlong masking is opt-in. The prompt-length filter must count actual tokenizer `input_ids` because Gemma tokenizers may return a `BatchEncoding` from `apply_chat_template(..., tokenize=True)` whose object length is not the token count. If enabled, `--mask_truncated_completions` masks only completions that reach `max_completion_length` without EOS/PAD; short vLLM completions that omit EOS are not treated as overlong. The launcher exposes the full DAPO knob set via env vars: `NUM_ITERATIONS`, `STEPS_PER_GENERATION`, `ENABLE_DYNAMIC_SAMPLING`, `DYNAMIC_SAMPLING_MIN_STD`, `DAPO_MAX_ROUNDS`, `DAPO_OVERSAMPLE_FACTOR`, `DYNAMIC_SAMPLING_REWARD_NAME`, `MASK_TRUNCATED_COMPLETIONS`, `EXEC_TIMEOUT_S`, `REWARD_WORKERS`, `LENGTH_PENALTY_MAX`, `LENGTH_PENALTY_BUFFER`, `REWARD_WEIGHTS`, `VLLM_GROUP_PORT`, `LOG_COMPLETIONS`, `NUM_COMPLETIONS_TO_PRINT`, `SAVE_STEPS`, `EVAL_STEPS`, `LOGGING_STEPS`, `EVAL_LIMIT`, `GRADIENT_ACCUMULATION_STEPS`. Launcher defaults: `SAVE_STEPS=25`, `EVAL_STEPS=25`, `LOG_COMPLETIONS=0`, `EVAL_LIMIT=64`, `DAPO_MAX_ROUNDS=1`, `DAPO_OVERSAMPLE_FACTOR=8`, `DYNAMIC_SAMPLING_REWARD_NAME=result_reward`, `NUM_ITERATIONS=1`, `GRADIENT_ACCUMULATION_STEPS=16`, `REWARD_WORKERS=1`. Eval uses the same `num_generations` as training. With `gradient_accumulation_steps=16, num_iterations=1, per_device_train_batch_size=1` and 6 trainer ranks the DAPO target is `G=6` heterogeneous prompt-groups per optimizer step (1 group per rank × 6 ranks); single-shot oversample with `K=8` generates `K * G * num_generations = 8 * 6 * 16 = 768` completions per step in one round.

Tests

Focused unit tests live in: tests/test_core.py

These tests currently cover:

- BIRD raw record normalization
- DAPO truncation masking only for completions that reach `max_completion_length`
- schema-built message normalization for `db_id` and `evidence`
- inference prompt preparation for normalized and legacy schema-built rows
- prompt-length filtering for tokenizers that return `BatchEncoding` objects
- SQL extraction and readonly checks
- Database path resolution for split-based database folders

Documentation Maintenance

After any relevant code change, review this file for drift.

Update `.github/copilot-instructions.md` when a code change affects repo structure, workflow, assumptions, supported behavior, or tests.

Also update `README.md` when setup, usage, or run instructions change.


When Editing Code

When making code changes, prefer minimal targeted edits.

If changing database path logic, update only sql_utils.py.

If changing data format handling, update only data.py.

If changing model loading/freezing, update only model_utils.py.

If changing reward logic, update only rewards.py, sql_utils.py, or schema_utils.py.

Inference and Evaluation

Standalone dev-set inference should run outside the training loop. In the current single-node recipe, training uses GPUs `0-5` and the vLLM server uses GPUs `6-7`, so there are no spare GPUs for local checkpoint inference while training is active. The intended workflow is to run inference after training finishes on the same 8-GPU node, unless a second node is available.

The repository now includes:

- `scripts/run_inference_bird.py`: generates SQL from a trained checkpoint on schema-augmented dev data and evaluates it locally.
- `scripts/launch_inference.sh`: launcher for post-training or separate-node inference.
- `scripts/run_passk_bird.py`: generates `k` sampled candidates per example and reports pass@k from prefixes of the same sampled set.
- `scripts/run_self_consistency_bird.py`: generates `k` sampled candidates per example, ignores candidates that execute to empty result sets, then majority-votes over the remaining raw execution-result sets to produce one self-consistent SQL choice per example.
- `scripts/generate_failure_instructions.py`: samples schema-built training prompts, runs sampled vLLM generations, filters to heterogeneous prompts using BIRD EX accuracy, mines common failure heuristics from wrong generations, and writes candidate instruction rules to Markdown/JSON artifacts.

Standalone inference supports a backend switch via `--inference_backend transformers|vllm`. The launcher mirrors this through `INFERENCE_BACKEND` and forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, and `VLLM_MAX_MODEL_LEN`.

For self-consistency runs, keep `--num_generations` explicit and treat `--temperature` as a search knob rather than reusing the greedy-ish pass@k default blindly. For NL2SQL, a moderate temperature such as `0.5-0.8` is the recommended starting range because it adds enough diversity for useful voting without collapsing execution validity.

The launcher also forwards `TRANSFORMERS_DEVICE_MAP` and `TRANSFORMERS_DATA_PARALLEL_SIZE` for the local `transformers` backend.

The inference launcher now appends a `YYYYMMDD_HHMMSS` suffix to `OUTPUT_DIR` by default for both backends. Set `APPEND_OUTPUT_TIMESTAMP=0` to disable this, or set `OUTPUT_TIMESTAMP` explicitly to control the suffix.

Standalone inference defaults to `max_prompt_length=30000` and `max_new_tokens=4096`, and filters out prompts longer than that limit instead of truncating them. Filtered prompts are printed during the run and written to `filtered_examples.jsonl` in the output directory.

The local BIRD evaluator defaults to `eval_timeout=60` seconds per example and `eval_workers=16` concurrent evaluation workers.

For the local `transformers` backend, standalone inference now defaults to explicit multi-process data parallel with `device_map` disabled. This is intended for models that fit on a single GPU. To shard one model across the visible GPUs instead, opt in with `TRANSFORMERS_DEVICE_MAP=auto`.

For the local vLLM backend, standalone inference defaults to `vllm_tensor_parallel_size=4` and `vllm_data_parallel_size=2` for 8-GPU single-node runs.

For local standalone inference, data parallel should be implemented through explicit multi-process worker orchestration. Do not rely on single-process `LLM(data_parallel_size=...)` with vLLM 0.19.x.

The inference workflow writes JSON, Markdown, and CSV summaries for dev-set execution accuracy, including per-difficulty and per-database breakdowns.

Per-example evaluation results should include predicted-side and gold-side execution flags and error text when SQL execution fails. The JSON summary should also include extraction and execution counts such as predicted SQL extracted, predicted SQL executed, gold SQL executed, and both SQL executed.

The local BIRD execution-accuracy scorer should follow the official dev-set semantics from `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`:

- execute predicted SQL and gold SQL on the target SQLite database
- compare results using `set(predicted_rows) == set(gold_rows)`
- report simple/moderate/challenging/total accuracy using the `difficulty` field from the dev JSON
- standalone inference post-processing should use the shared `bird_execute_sql` / `bird_result_match` helpers directly for both predicted and gold SQL evaluation; avoid the legacy spawned pair-executor path because it can drop results as `no result` under concurrent load

If changing distributed training settings, update:
scripts/launch_train.sh
configs/ds_zero3_bf16.json
src/nl2sql_gspo/train_gspo_nl2sql.py

