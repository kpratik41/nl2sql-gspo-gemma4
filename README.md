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

Override the learning rate for one-off runs with `LEARNING_RATE` or `LR`:

```bash
MODEL_NAME=outputs/gemma4_31b_gspo_bird/checkpoint-70 \
OUTPUT_DIR=outputs/gemma4_31b_gspo_bird_ckpt70_lr1e6_<timestamp> \
LEARNING_RATE=1e-6 \
bash scripts/launch_train.sh
```

The launcher now trains from the generated schema-augmented training file and evaluates on the generated dev schema file:

- `outputs/train-6601-schema-filtered.jsonl`
- `outputs/dev-20251106-schema-256.jsonl`

The current training launcher recipe uses `num_generations=16`, `gradient_accumulation_steps=16`, `max_prompt_length=16000`, and `max_completion_length=4096` with vLLM server-mode rollouts (vLLM `max_model_len=24576`). Override the prompt filter with `MAX_PROMPT_LENGTH`, gradient accumulation with `GRADIENT_ACCUMULATION_STEPS`, short smoke-run length with `MAX_STEPS`, and DB-backed reward parallelism with `REWARD_WORKERS` when launching. The default base model in the launchers is `google/gemma-4-31B-it`, and the default eval file is `outputs/dev-20251106-schema-256.jsonl`.

The training launcher defaults to model-only checkpoints (`SAVE_ONLY_MODEL=1`) and keeps the rotating `checkpoint-*` folders trimmed to `SAVE_TOTAL_LIMIT=3`. Those rotating checkpoints write HF model weights/config and trainer state but skip DeepSpeed optimizer/scheduler/scaler/RNG state.

In addition, the launcher now defaults `SAVE_LATEST_FULL_CHECKPOINT=1`, which refreshes `OUTPUT_DIR/latest-full-checkpoint` on every save with a full DeepSpeed resume checkpoint containing optimizer/scheduler/RNG state. Use `RESUME_FROM_CHECKPOINT=.../latest-full-checkpoint` for an exact restart, or set `MODEL_NAME` to one of the rotating model-only `checkpoint-*` directories and leave `RESUME_FROM_CHECKPOINT` unset for model-only continuation.

The vLLM launcher goes through a local wrapper at `python -m nl2sql_gspo.vllm_serve_compat`, which delegates to TRL's shipped vLLM server entrypoint from the local repo package.

For `google/gemma-4-31B-it`, leave `VLLM_ATTENTION_BACKEND` unset unless you are intentionally overriding vLLM's model-aware backend selection. In vLLM `0.19.x+`, Gemma 4 detects heterogeneous `head_dim` / `global_head_dim` and forces `TRITON_ATTN` when `global_head_dim=512`; setting `VLLM_ATTENTION_BACKEND=TORCH_SDPA` bypasses that safeguard and fails for this model.

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

To resume training from the most recent full restartable checkpoint:

```bash
RESUME_FROM_CHECKPOINT=outputs/gemma4_31b_gspo_bird/latest-full-checkpoint bash scripts/launch_train.sh
```

To continue from a model-only checkpoint instead:

```bash
MODEL_NAME=outputs/gemma4_31b_gspo_bird/checkpoint-100 \
OUTPUT_DIR=outputs/gemma4_31b_gspo_bird_restart \
bash scripts/launch_train.sh
```

### DAPO-style controls

The trainer is `DynamicSamplingGRPOTrainer` (a thin subclass of TRL's `GRPOTrainer`) and supports:

- `--num_iterations` (PPO mu, default `1` in the launcher): optimizer passes per generation buffer, with GSPO sequence-level importance sampling and `epsilon=0.2 / epsilon_high=0.28` clipping.
- `--enable_dynamic_sampling` (default on in the launcher) + `--dapo_oversample_factor` (default `8`): DAPO §3.2 single-shot oversample-and-replace. Each rank generates `K * target_local_groups` prompt groups in one rollout round, drawing backup prompts from a shared shuffled tail queue over `train_dataset`, then keeps the first heterogeneous groups whose reward std passes `--dynamic_sampling_min_std`. Candidate rewards are filtered before policy old-logprob correction, so the expensive trainer-side logprob pass is run only for kept/padded groups. Any remaining slots are filled with non-heterogeneous groups whose `completion_mask` is zeroed. If every final slot is zero-masked, the trainer returns a zero policy loss after input preparation instead of running a full DeepSpeed forward/backward.
- `--mask_truncated_completions` (default off in the launcher): optional DAPO overlong masking for rollouts that reach `max_completion_length` without EOS/PAD. The normal NL2SQL launcher path filters over-length prompts up front and does not apply completion-side overlong masking unless `MASK_TRUNCATED_COMPLETIONS=1` is set.
- `--length_penalty_max` / `--length_penalty_buffer` (defaults `4096` / `512`): DAPO §3.4 Soft Overlong Punishment. Adds a 7th continuous reward function in `[-1, 0]`: 0 when length ≤ `L_max - L_cache`, linear ramp to -1 within the buffer, saturated at -1 beyond `L_max`. Length is measured in tokens via the trainer's tokenizer.
- `--exec_timeout_s` (default `60`): per-query SQL execution timeout for both predicted and gold SQL during reward computation. More permissive than BIRD's 30s default to avoid penalising slow gold queries during training.
- `--steps_per_generation`: optional override for generation cadence.

Launcher env vars: `NUM_ITERATIONS`, `ENABLE_DYNAMIC_SAMPLING` (`0`/`1`), `DYNAMIC_SAMPLING_MIN_STD`, `DAPO_MAX_ROUNDS`, `DAPO_OVERSAMPLE_FACTOR`, `DYNAMIC_SAMPLING_REWARD_NAME`, `MASK_TRUNCATED_COMPLETIONS` (`0`/`1`), `STEPS_PER_GENERATION`, `EVAL_ON_START` (`0`/`1`), `EXEC_TIMEOUT_S`, `REWARD_WORKERS`, `LENGTH_PENALTY_MAX`, `LENGTH_PENALTY_BUFFER`, `REWARD_WEIGHTS`, `VLLM_GROUP_PORT`, `SAVE_STEPS`, `EVAL_STEPS`, `LOGGING_STEPS`, `LOG_COMPLETIONS`, `NUM_COMPLETIONS_TO_PRINT`, `EVAL_LIMIT`. Logging defaults: `SAVE_STEPS=25`, `EVAL_STEPS=25`, `LOGGING_STEPS=5`, `LOG_COMPLETIONS=0`, `EVAL_LIMIT=64`, `MASK_TRUNCATED_COMPLETIONS=0`; `DAPO_OVERSAMPLE_FACTOR` defaults to `8`. `REWARD_WORKERS=1` keeps training reward execution serial per rank, while higher values parallelize DB-backed rewards like `result_reward`. Eval uses the same `num_generations` setting as training. `VLLM_GROUP_PORT` defaults to `29600` to avoid OS ephemeral-port collisions during vLLM weight updates.

Use `scripts/check_cluster_health.py` before a long run when you want a quick local sanity check for GPU visibility, peer access, `nvidia-smi` topology, and a real NCCL collective smoke test across the currently visible GPUs:

```bash
.conda/nl2sql312/bin/python scripts/check_cluster_health.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .conda/nl2sql312/bin/python scripts/check_cluster_health.py
```

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

Standalone inference supports `--inference_backend vllm|vllm_async`. The shell launcher mirrors this through `INFERENCE_BACKEND`, and also forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, and `VLLM_ASYNC_CONCURRENCY`.

When `OUTPUT_DIR` is not set, the launcher creates a descriptive output directory under `outputs/inference/<split>/<input_stem>/<model_tag>/`. The final run folder includes backend, tensor/data parallel settings, async concurrency when applicable, context/prompt/output limits, tool-round budget, and a `YYYYMMDD_HHMMSS` timestamp suffix. If you set `OUTPUT_DIR` yourself, the launcher keeps that path and only appends the timestamp suffix by default. Set `APPEND_OUTPUT_TIMESTAMP=0` to keep the directory name unchanged, or set `OUTPUT_TIMESTAMP` explicitly to control the suffix.

Standalone inference defaults to `max_prompt_length=34000`, `max_new_tokens=8000`, and `vllm_max_model_len=43000`, and filters out over-length prompts instead of truncating them. Filtered samples are printed during the run and also written to `filtered_examples.jsonl`.

Standalone inference now uses the normalized `prompt` field when it is present. For older schema-built files that only contain `messages`, it strips any `assistant` turns before rendering the generation prompt so gold SQL is not leaked into the input context.

The local BIRD evaluator now defaults to `eval_timeout=60` seconds per example and `eval_workers=16` concurrent evaluation workers.

Inference now fails fast if normalized rows are missing required metadata such as `db_id` or `gold_sql`, instead of silently running evaluation against the wrong database.

For the local vLLM backend, the defaults are `tensor_parallel_size=2` and `data_parallel_size=4`, which is intended for an 8-GPU single-node run.

For the async vLLM backend, the defaults are `tensor_parallel_size=8` and `data_parallel_size=1`. This path is intended primarily for tool-calling data and also accepts `VLLM_ASYNC_CONCURRENCY`.

In this repository, local vLLM data parallel is implemented with explicit worker processes, each running a tensor-parallel vLLM engine on its own GPU group. This avoids the unsupported single-process `LLM(data_parallel_size=...)` path in vLLM 0.19.x.

For a quick smoke run, set `NUM_EXAMPLES=1` before launching inference.

Example launcher invocations:

```bash
INFERENCE_BACKEND=vllm MODEL_PATH=google/gemma-4-E4B-it NUM_EXAMPLES=2 bash scripts/launch_inference.sh
INFERENCE_BACKEND=vllm_async INPUT_FILE=outputs/dev-20251106-schema-tool.jsonl MODEL_PATH=google/gemma-4-E4B-it NUM_EXAMPLES=2 bash scripts/launch_inference.sh
```

Inference outputs are written under the timestamped output directory selected by the launcher, for example:

```text
outputs/inference/dev/dev-20251106-schema/gemma4_31b_gspo_bird/vllm_tp2_dp4_ctx43k_p34k_o8k_r8_20260520_190000/
outputs/inference/dev/dev-20251106-schema-tool/gemma4_31b_gspo_bird/vllm_async_tp8_dp1_c8_ctx43k_p34k_o8k_r8_20260520_190000/
```

- `predict_dev.json`: official BIRD prediction format (`SQL\t----- bird -----\tdb_id`)
- `prediction_details.jsonl`: decoded completions and extracted SQL
- `filtered_examples.jsonl`: prompts skipped because they exceeded `max_prompt_length`
- `eval_results.jsonl`: per-example execution results, including predicted-side and gold-side execution flags and error text
- `eval_summary.json`: simple/moderate/challenging/total EX accuracy plus extraction and execution counts
- `eval_summary.md`: summary tables by difficulty and by database
- `eval_summary_by_difficulty.csv`: CSV summary by difficulty
- `eval_summary_by_db.csv`: CSV summary by database

The local EX scorer intentionally follows the official BIRD dev evaluation semantics from `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`: it executes predicted and gold SQL on SQLite and checks whether `set(pred_rows) == set(gold_rows)`.

The repository also includes `scripts/run_passk_bird.py` for pass@k evaluation and `scripts/run_self_consistency_bird.py` for self-consistency evaluation. The self-consistency script generates `k` candidates per example, executes them on SQLite, discards candidates whose execution result is empty, and then majority-votes over the remaining raw result sets. Ties break by earliest sample index and then shorter SQL. Recommended starting settings are `--num_generations 16` and `--temperature 0.7`; for NL2SQL, lower temperatures such as `0.1` usually improve top-1 accuracy but often reduce vote diversity, so self-consistency tends to benefit from a moderate temperature instead of the near-greedy setting used for pass@k.

For train-set failure analysis, `scripts/generate_failure_instructions.py` samples prompts from the schema-built training file, runs `n` sampled vLLM generations per prompt, scores every candidate with BIRD EX accuracy, keeps only heterogeneous prompts (neither all-correct nor all-wrong), mines common mistake heuristics from the wrong generations, and writes prompt-ready instruction candidates to `failure_instruction_rules.md`.

Example failure-instruction mining run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/generate_failure_instructions.py \
	--model_name_or_path outputs/gemma4_31b_gspo_bird/checkpoint-70 \
	--input_file outputs/train-6601-schema-filtered.jsonl \
	--database_dir databases/train_databases \
	--output_dir outputs/train_failure_instructions_ckpt70 \
	--sample_size 1000 \
	--num_generations 16 \
	--temperature 0.8 \
	--top_p 0.95 \
	--vllm_tensor_parallel_size 4 \
	--vllm_data_parallel_size 2 \
	--overwrite
```

Example self-consistency run:

```bash
python scripts/run_self_consistency_bird.py \
	--model_name_or_path outputs/gemma4_31b_gspo_bird/checkpoint-30 \
	--input_file outputs/dev-20251106-schema.jsonl \
	--database_dir databases/dev_databases \
	--diff_json_path data/bird_dev_data/raw/dev_20251106.json \
	--output_dir outputs/self_consistency_temp07/checkpoint-30 \
	--num_generations 16 \
	--temperature 0.7 \
	--eval_timeout 60 \
	--eval_workers 8 \
	--vllm_tensor_parallel_size 4 \
	--vllm_data_parallel_size 2 \
	--overwrite
```

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
