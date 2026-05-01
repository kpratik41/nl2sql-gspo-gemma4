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

The current training launcher recipe uses `num_generations=16`, `max_prompt_length=16384`, and `max_completion_length=4096` with vLLM server-mode rollouts. The default base model in the launchers is `google/gemma-4-31B`.

The trainer also runs dev evaluation before the first optimizer step via `eval_on_start`, which is useful for capturing a true pre-training baseline on the eval split.

The reward stack is currently weighted to make execution correctness dominate soft shaping signals:

- `format_reward`: `0.25`
- `execution_reward`: `1.0`
- `result_reward`: `2.5`
- `schema_linking_reward`: `0.5`
- `ngram_reward`: `0.5`
- `evidence_utilization_reward`: `0.25`

To resume training from a saved checkpoint:

```bash
RESUME_FROM_CHECKPOINT=outputs/gemma4_31b_gspo_bird/checkpoint-100 bash scripts/launch_train.sh
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

Standalone inference supports `--inference_backend transformers|vllm`. The shell launcher mirrors this through `INFERENCE_BACKEND`, and also forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, and `VLLM_MAX_MODEL_LEN`.

The launcher now appends a `YYYYMMDD_HHMMSS` timestamp suffix to `OUTPUT_DIR` by default for both backends. Set `APPEND_OUTPUT_TIMESTAMP=0` to keep the directory name unchanged, or set `OUTPUT_TIMESTAMP` explicitly to control the suffix.

Standalone inference defaults to `max_prompt_length=30000` and `max_new_tokens=4096`, and filters out over-length prompts instead of truncating them. Filtered samples are printed during the run and also written to `filtered_examples.jsonl`.

The `transformers` backend already loads a single model with `device_map=auto`, so when `CUDA_VISIBLE_DEVICES` exposes multiple GPUs it can shard that one model across all visible GPUs.

Inference now fails fast if normalized rows are missing required metadata such as `db_id` or `gold_sql`, instead of silently running evaluation against the wrong database.

For the local vLLM backend, the defaults are `tensor_parallel_size=4` and `data_parallel_size=2`, which is intended for an 8-GPU single-node run.

In this repository, local vLLM data parallel is implemented with explicit worker processes, each running a tensor-parallel vLLM engine on its own GPU group. This avoids the unsupported single-process `LLM(data_parallel_size=...)` path in vLLM 0.19.x.

For a quick smoke run, set `NUM_EXAMPLES=1` before launching inference.

Example launcher invocations:

```bash
INFERENCE_BACKEND=transformers NUM_EXAMPLES=2 bash scripts/launch_inference.sh
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

The trainer logs online RL metrics such as reward means, reward std, KL, entropy, clipping ratios, completion lengths, and the trainer loss. With `--eval_file` enabled, evaluation runs once before training starts and then every `eval_steps`, emitting `eval_*` metrics through the same reporting backend.

The launcher defaults to Weights & Biases because it exports `WANDB_PROJECT`. On AWS you can also enable TensorBoard event files by setting `REPORT_TO=wandb,tensorboard` before launching training.

```bash
REPORT_TO=wandb,tensorboard RUN_NAME=nl2sql-gspo-aws bash scripts/launch_train.sh
```

TensorBoard logs are written under `outputs/gemma4_31b_gspo_bird/tb` by default and can be forwarded from a head node or synced to shared storage for remote monitoring.

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
