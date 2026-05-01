# Copilot Instructions for NL2SQL GSPO Training Project

## Project Overview

This repository implements reinforcement learning training for a large language model on an NL2SQL task using:

- TRL `GRPOTrainer`
- GSPO-style sequence-level importance sampling
- vLLM server mode for rollouts
- DeepSpeed ZeRO-3 for distributed training
- BIRD-style NL2SQL train/dev data
- SQLite execution-based rewards

The current target model is a Gemma 4 31B multimodal model. For this project, only text input and text output are used. Vision/audio/multimodal modules should be frozen if present. The language model decoder should remain trainable unless explicitly switching to LoRA/adapters.

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

The `scripts/data_generation/schema_build.py` utility supports both train and dev splits. It should be able to consume JSONL or JSON inputs, default to the split-specific few-shot file and column-meaning file in `data/bird_<split>_data/raw/`, read schemas from `databases/<split>_databases/`, and emit chat-format JSONL under `outputs/`. It also renders compact schema statistics such as table row/column counts and per-column null/distinct/min/max summaries.

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

The current launcher recipe trains from `outputs/train-6601-schema-filtered.jsonl` and evaluates on `outputs/dev-20251106-schema.jsonl`.
It currently uses `num_generations=16`, `max_prompt_length=16384`, and `max_completion_length=4096` with vLLM server mode, and the launcher defaults to `google/gemma-4-31B`.
It also runs `eval_on_start` for a pre-training dev baseline and supports optional `--train_limit` / `--eval_limit` row caps for smoke runs.

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

Current reward functions include:

format_reward
execution_reward
result_reward
schema_linking_reward
ngram_reward
evidence_utilization_reward

The current training recipe reweights these rewards to prioritize exact execution correctness:

- format: `0.25`
- execution: `1.0`
- result: `2.5`
- schema_linking: `0.5`
- ngram: `0.5`
- evidence_utilization: `0.25`

The training script also supports configurable monitoring backends via `--report_to` and writes TensorBoard logs to `--logging_dir` when `tensorboard` is included.

Tests

Focused unit tests live in: tests/test_core.py

These tests currently cover:

- BIRD raw record normalization
- schema-built message normalization for `db_id` and `evidence`
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

Standalone inference supports a backend switch via `--inference_backend transformers|vllm`. The launcher mirrors this through `INFERENCE_BACKEND` and forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, and `VLLM_MAX_MODEL_LEN`.

The inference launcher now appends a `YYYYMMDD_HHMMSS` suffix to `OUTPUT_DIR` by default for both backends. Set `APPEND_OUTPUT_TIMESTAMP=0` to disable this, or set `OUTPUT_TIMESTAMP` explicitly to control the suffix.

Standalone inference defaults to `max_prompt_length=30000` and `max_new_tokens=4096`, and filters out prompts longer than that limit instead of truncating them. Filtered prompts are printed during the run and written to `filtered_examples.jsonl` in the output directory.

For the local `transformers` backend, inference model loading should continue to use `device_map="auto"` so a single model can be sharded across all visible GPUs.

For the local vLLM backend, standalone inference defaults to `vllm_tensor_parallel_size=4` and `vllm_data_parallel_size=2` for 8-GPU single-node runs.

For local standalone inference, data parallel should be implemented through explicit multi-process worker orchestration. Do not rely on single-process `LLM(data_parallel_size=...)` with vLLM 0.19.x.

The inference workflow writes JSON, Markdown, and CSV summaries for dev-set execution accuracy, including per-difficulty and per-database breakdowns.

Per-example evaluation results should include predicted-side and gold-side execution flags and error text when SQL execution fails. The JSON summary should also include extraction and execution counts such as predicted SQL extracted, predicted SQL executed, gold SQL executed, and both SQL executed.

The local BIRD execution-accuracy scorer should follow the official dev-set semantics from `AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py`:

- execute predicted SQL and gold SQL on the target SQLite database
- compare results using `set(predicted_rows) == set(gold_rows)`
- report simple/moderate/challenging/total accuracy using the `difficulty` field from the dev JSON

If changing distributed training settings, update:
scripts/launch_train.sh
configs/ds_zero3_bf16.json
src/nl2sql_gspo/train_gspo_nl2sql.py

