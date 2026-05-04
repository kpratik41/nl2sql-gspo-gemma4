# NL2SQL GSPO Gemma4

## Training

Start the vLLM server first:

```bash
bash scripts/launch_vllm.sh
```

Then launch training:

```bash
bash scripts/launch_train.sh
```

The launcher now trains from the generated schema-augmented training file and evaluates on the generated dev schema file:

- `outputs/train-6601-schema-filtered.jsonl`
- `outputs/dev-20251106-schema.jsonl`

The current training launcher recipe uses `num_generations=16`, `gradient_accumulation_steps=16`, `max_prompt_length=16000`, and `max_completion_length=4096` with vLLM server-mode rollouts (vLLM `max_model_len=24576`). Override the prompt filter with `MAX_PROMPT_LENGTH` and gradient accumulation with `GRADIENT_ACCUMULATION_STEPS` when launching. The default base model in the launchers is `google/gemma-4-31B-it`.

The training launcher defaults to model-only checkpoints (`SAVE_ONLY_MODEL=1`), which writes HF model weights/config and trainer state but skips DeepSpeed optimizer/scheduler/scaler/RNG state. Set `SAVE_ONLY_MODEL=0` only when you need a full optimizer-state checkpoint for exact training resume.

For model-only checkpoints, restart from the saved weights by setting `MODEL_NAME` to the checkpoint path and leaving `RESUME_FROM_CHECKPOINT` unset. Use a fresh `OUTPUT_DIR` if you want to preserve the previous checkpoint directory.

The vLLM launcher goes through a local compatibility wrapper at `python -m nl2sql_gspo.vllm_serve_compat` so TRL `0.29.1` can still serve against local vLLM `0.19.x`. This wrapper strips the unsupported `truncate_prompt_tokens` field before constructing vLLM sampling parameters.

The training launcher skips dev evaluation before the first optimizer step by default. Set `EVAL_ON_START=1` to opt into the pre-training dev baseline.

The reward stack is now binary-heavy with a single continuous shaping signal. `result_reward` uses the **official BIRD execution-accuracy semantics** (`set(predicted_rows) == set(gold_rows)` on raw rows, with a per-query timeout):

- `format_reward` (binary, 0/1): exact `<scratch_pad>...</scratch_pad><final_answer><sql_code>...</sql_code></final_answer>` shape
- `execution_reward` (binary): predicted SQL executes without error
- `result_reward` (binary, BIRD EX): set-equality on raw rows vs gold
- `table_linking_reward` (binary): predicted table set == gold table set
- `column_linking_reward` (continuous): Jaccard(pred_columns, gold_columns)
- `nonnull_reward` (binary, small): predicted SQL executes AND returns at least one non-null cell
- `length_penalty_reward` (continuous, ≤ 0): DAPO §3.4 Soft Overlong Punishment — 0 when length ≤ `L_max - L_cache`, linear ramp to -1 inside the buffer, saturates at -1 beyond `L_max`

Default weights: `0.2, 0.5, 2.0, 0.5, 0.5, 0.1, 0.1` (max positive weighted reward = 3.8; `length_penalty_reward` only contributes ≤ 0). `result_reward` dominates so the policy is pulled toward execution-equivalent answers; the other rewards keep advantages non-flat whenever `result_reward` collapses to all-0 or all-1 inside a group; the length penalty discourages truncation noise.

To resume training from a saved checkpoint:

```bash
RESUME_FROM_CHECKPOINT=outputs/gemma4_31b_gspo_bird/checkpoint-100 bash scripts/launch_train.sh
```

### DAPO-style controls

The trainer is `DynamicSamplingGRPOTrainer` (a thin subclass of TRL's `GRPOTrainer`) and supports:

- `--num_iterations` (PPO mu, default `1` in the launcher): optimizer passes per generation buffer, with GSPO sequence-level importance sampling and `epsilon=0.2 / epsilon_high=0.28` clipping.
- `--enable_dynamic_sampling` (default on in the launcher) + `--dapo_oversample_factor` (default `8`): DAPO §3.2 single-shot oversample-and-replace. Each rank generates `K * target_local_groups` prompt groups in one rollout round, drawing backup prompts from a shared shuffled tail queue over `train_dataset`, then keeps the first heterogeneous groups whose reward std passes `--dynamic_sampling_min_std`. Candidate rewards are filtered before policy old-logprob correction, so the expensive trainer-side logprob pass is run only for kept/padded groups. Any remaining slots are filled with non-heterogeneous groups whose `completion_mask` is zeroed. If every final slot is zero-masked, the trainer returns a zero policy loss after input preparation instead of running a full DeepSpeed forward/backward.
- `--mask_truncated_completions` (default off in the launcher): optional DAPO overlong masking for rollouts that reach `max_completion_length` without EOS/PAD. The normal NL2SQL launcher path filters over-length prompts up front and does not apply completion-side overlong masking unless `MASK_TRUNCATED_COMPLETIONS=1` is set.
- `--length_penalty_max` / `--length_penalty_buffer` (defaults `4096` / `512`): DAPO §3.4 Soft Overlong Punishment. Adds a 7th continuous reward function in `[-1, 0]`: 0 when length ≤ `L_max - L_cache`, linear ramp to -1 within the buffer, saturated at -1 beyond `L_max`. Length is measured in tokens via the trainer's tokenizer.
- `--exec_timeout_s` (default `60`): per-query SQL execution timeout for both predicted and gold SQL during reward computation. More permissive than BIRD's 30s default to avoid penalising slow gold queries during training.
- `--steps_per_generation`: optional override for generation cadence.

Launcher env vars: `NUM_ITERATIONS`, `ENABLE_DYNAMIC_SAMPLING` (`0`/`1`), `DYNAMIC_SAMPLING_MIN_STD`, `DAPO_MAX_ROUNDS`, `DAPO_OVERSAMPLE_FACTOR`, `DYNAMIC_SAMPLING_REWARD_NAME`, `MASK_TRUNCATED_COMPLETIONS` (`0`/`1`), `STEPS_PER_GENERATION`, `EVAL_ON_START` (`0`/`1`), `EXEC_TIMEOUT_S`, `LENGTH_PENALTY_MAX`, `LENGTH_PENALTY_BUFFER`, `REWARD_WEIGHTS`, `VLLM_GROUP_PORT`, `SAVE_STEPS`, `EVAL_STEPS`, `LOGGING_STEPS`, `LOG_COMPLETIONS`, `NUM_COMPLETIONS_TO_PRINT`, `EVAL_LIMIT`. Logging defaults: `SAVE_STEPS=25`, `EVAL_STEPS=25`, `LOGGING_STEPS=5`, `LOG_COMPLETIONS=0`, `EVAL_LIMIT=64`, `MASK_TRUNCATED_COMPLETIONS=0`; `DAPO_OVERSAMPLE_FACTOR` defaults to `8`. `VLLM_GROUP_PORT` defaults to `29600` to avoid OS ephemeral-port collisions during vLLM weight updates.

To limit training or evaluation to a subset of rows for smoke runs:

```bash
TRAIN_LIMIT=256 EVAL_LIMIT=128 bash scripts/launch_train.sh
```

Set either value to `-1` to use the full file.

Run dev-set inference and BIRD execution-accuracy scoring after training finishes on the same 8-GPU node, or on a different node if you later have one available. In the current 6-train-GPU + 2-vLLM-GPU recipe, the training job already occupies the full node while training is active, so post-training inference is the intended workflow.

```bash
bash scripts/launch_inference.sh
```

This launcher loads the checkpoint from `outputs/gemma4_31b_gspo_bird`, generates SQL for `outputs/dev-20251106-schema.jsonl`, writes official-style `predict_dev.json`, and computes BIRD-style execution accuracy against `databases/dev_databases` with difficulty breakdown from `data/bird_dev_data/raw/dev_20251106.json`.

Standalone inference supports `--inference_backend transformers|vllm`. The shell launcher mirrors this through `INFERENCE_BACKEND`, and also forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `TRANSFORMERS_DEVICE_MAP`, `TRANSFORMERS_DATA_PARALLEL_SIZE`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, and `VLLM_MAX_MODEL_LEN`.

The launcher now appends a `YYYYMMDD_HHMMSS` timestamp suffix to `OUTPUT_DIR` by default for both backends. Set `APPEND_OUTPUT_TIMESTAMP=0` to keep the directory name unchanged, or set `OUTPUT_TIMESTAMP` explicitly to control the suffix.

Standalone inference defaults to `max_prompt_length=30000` and `max_new_tokens=4096`, and filters out over-length prompts instead of truncating them. Filtered samples are printed during the run and also written to `filtered_examples.jsonl`.

Standalone inference now uses the normalized `prompt` field when it is present. For older schema-built files that only contain `messages`, it strips any `assistant` turns before rendering the generation prompt so gold SQL is not leaked into the input context.

The local BIRD evaluator now defaults to `eval_timeout=120` seconds per example and `eval_workers=16` concurrent evaluation workers.

The `transformers` backend now defaults to explicit multi-process data parallel instead of `device_map=auto`. By default it starts one worker per visible GPU and each worker loads its own model replica, so this mode is intended for models that fit on a single GPU. To shard one model across the visible GPUs instead, set `TRANSFORMERS_DEVICE_MAP=auto`.

Inference now fails fast if normalized rows are missing required metadata such as `db_id` or `gold_sql`, instead of silently running evaluation against the wrong database.

For the local vLLM backend, the defaults are `tensor_parallel_size=4` and `data_parallel_size=2`, which is intended for an 8-GPU single-node run.

In this repository, local vLLM data parallel is implemented with explicit worker processes, each running a tensor-parallel vLLM engine on its own GPU group. This avoids the unsupported single-process `LLM(data_parallel_size=...)` path in vLLM 0.19.x.

For a quick smoke run, set `NUM_EXAMPLES=1` before launching inference.

Example launcher invocations:

```bash
INFERENCE_BACKEND=transformers NUM_EXAMPLES=2 bash scripts/launch_inference.sh
INFERENCE_BACKEND=transformers TRANSFORMERS_DEVICE_MAP=auto MODEL_PATH=google/gemma-4-31B NUM_EXAMPLES=2 bash scripts/launch_inference.sh
INFERENCE_BACKEND=vllm MODEL_PATH=google/gemma-4-E4B-it NUM_EXAMPLES=2 bash scripts/launch_inference.sh
```

Inference outputs are written under the timestamped output directory selected by the launcher, for example `outputs/bird_dev_inference_20260501_071530/`:

- `predict_dev.json`: official BIRD prediction format (`SQL\t----- bird -----\tdb_id`)
- `prediction_details.jsonl`: decoded completions and extracted SQL
- `filtered_examples.jsonl`: prompts skipped because they exceeded `max_prompt_length`
- `eval_results.jsonl`: per-example execution results, including predicted-side and gold-side execution flags and error text
- `eval_summary.json`: simple/moderate/challenging/total EX accuracy plus extraction and execution counts
- `eval_summary.md`: summary tables by difficulty and by database
- `eval_summary_by_difficulty.csv`: CSV summary by difficulty
- `eval_summary_by_db.csv`: CSV summary by database

The local EX scorer intentionally follows the official BIRD dev evaluation semantics from `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`: it executes predicted and gold SQL on SQLite and checks whether `set(pred_rows) == set(gold_rows)`.

## Monitoring

The trainer logs online RL metrics such as reward means, reward std, DAPO candidate heterogeneity counts, selected/padded group counts, KL, entropy, clipping ratios, completion lengths, and the trainer loss. With `--eval_file` enabled, evaluation runs every `eval_steps` and emits `eval_*` metrics through the same reporting backend. Set `EVAL_ON_START=1` only when you want an additional pre-training baseline. Prompt/completion sample tables are disabled in the terminal by default; set `LOG_COMPLETIONS=1` to print them.

The launcher defaults to Weights & Biases because it exports `WANDB_PROJECT` and defaults `REPORT_TO=wandb`. Console logs, W&B run names, and TensorBoard logging directories are timestamped by default. On AWS you can also enable TensorBoard event files by setting `REPORT_TO=wandb,tensorboard` before launching training.

```bash
REPORT_TO=wandb,tensorboard RUN_NAME=nl2sql-gspo-aws bash scripts/launch_train.sh
```

Terminal logs are written under `logs/train_<timestamp>.log` and `logs/vllm_<timestamp>.log`. TensorBoard logs are written under `outputs/gemma4_31b_gspo_bird/tb/<timestamp>` by default and can be forwarded from a head node or synced to shared storage for remote monitoring.

## Data Generation

Generate BM25-based few-shot files for both train and dev splits:

```bash
c:/Users/kprat/OneDrive/Desktop/Project/nl2sql-gspo-gemma4/.venv/Scripts/python.exe scripts/data_generation/few_shot_bm25.py --top-n 5
```

This writes:

- `data/bird_train_data/raw/train-6601-few-shot.jsonl`
- `data/bird_dev_data/raw/dev_20251106-few-shot.json`

The train few-shot file excludes examples from the same `db_id` as the source record.

Generate schema-augmented chat-format data for training or inference:

```bash
c:/Users/kprat/OneDrive/Desktop/Project/nl2sql-gspo-gemma4/.venv/Scripts/python.exe scripts/data_generation/schema_build.py --split train --n-examples -1 --output outputs/train-6601-schema.jsonl
```

```bash
c:/Users/kprat/OneDrive/Desktop/Project/nl2sql-gspo-gemma4/.venv/Scripts/python.exe scripts/data_generation/schema_build.py --split dev --n-examples -1 --output outputs/dev-20251106-schema.jsonl
```

The schema builder now:

- resolves train/dev defaults from the repository layout
- accepts both JSONL and JSON input files
- writes top-level `db_id`, `gold_sql`, `evidence`, and `question` fields alongside `messages`
- injects per-column meanings from the split-specific column meaning file
- renders table row/column counts and per-column stats such as null count, distinct count, min/max, and top text values
- logs progress on the first sample, every 50 samples by default, and the last sample

Older schema-built files that only contain `messages` are still accepted: the shared loader recovers `db_id` from `<db_id>...</db_id>` and `evidence` from `<hint>...</hint>` inside the prompt.

At inference time, those older message-only rows are also converted back to a generation prompt using only the `system` and `user` turns.
