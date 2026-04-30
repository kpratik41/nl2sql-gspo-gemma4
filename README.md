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

The current launcher recipe uses `num_generations=16`, `max_prompt_length=16384`, and `max_completion_length=4096` with vLLM server-mode rollouts.

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

Run dev-set inference and BIRD execution-accuracy scoring after training finishes on the same 8-GPU node, or on a different node if you later have one available. In the current 6-train-GPU + 2-vLLM-GPU recipe, the training job already occupies the full node while training is active, so post-training inference is the intended workflow.

```bash
bash scripts/launch_inference.sh
```

This launcher loads the checkpoint from `outputs/gemma4_31b_gspo_bird`, generates SQL for `outputs/dev-20251106-schema.jsonl`, writes official-style `predict_dev.json`, and computes BIRD-style execution accuracy against `databases/dev_databases` with difficulty breakdown from `data/bird_dev_data/raw/dev_20251106.json`.

Inference outputs are written under `outputs/bird_dev_inference/`:

- `predict_dev.json`: official BIRD prediction format (`SQL\t----- bird -----\tdb_id`)
- `prediction_details.jsonl`: decoded completions and extracted SQL
- `eval_results.jsonl`: per-example execution results
- `eval_summary.json`: simple/moderate/challenging/total EX accuracy
- `eval_summary.md`: summary tables by difficulty and by database

The local EX scorer intentionally follows the official BIRD dev evaluation semantics from `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`: it executes predicted and gold SQL on SQLite and checks whether `set(pred_rows) == set(gold_rows)`.

## Monitoring

The trainer logs online RL metrics such as reward means, reward std, KL, entropy, clipping ratios, completion lengths, and the trainer loss. With `--eval_file` enabled, evaluation runs on the dev split every `eval_steps` and emits `eval_*` metrics through the same reporting backend.

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
- injects per-column meanings from the split-specific column meaning file
- renders table row/column counts and per-column stats such as null count, distinct count, min/max, and top text values
- logs progress on the first sample, every 50 samples by default, and the last sample
