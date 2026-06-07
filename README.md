# NL2SQL Gemma Inference

This repository contains inference and evaluation utilities for NL2SQL on BIRD-style SQLite datasets. It supports standard generation, native Gemma tool-calling loops, pass@k evaluation, and execution-result self-consistency.

## Setup

```bash
pip install -r requirements.txt
```

The launchers expect `PYTHONPATH` to include `src`; `scripts/launch_inference.sh` sets that automatically.

## Run Inference

```bash
bash scripts/launch_inference.sh
```

The launcher uses async vLLM, loads `outputs/gemma4_31b_gspo_bird`, reads `outputs/bird_dev-schema.jsonl`, writes official-style `predict_dev.json`, and computes BIRD-style execution accuracy against `databases/dev_databases`.

Common overrides:

```bash
MODEL_PATH=google/gemma-4-E4B-it NUM_EXAMPLES=2 bash scripts/launch_inference.sh
INPUT_FILE=outputs/bird_dev-schema-tool.jsonl MODEL_PATH=google/gemma-4-E4B-it NUM_EXAMPLES=2 bash scripts/launch_inference.sh
```

Standalone inference is async-vLLM-only. The shell launcher forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `SHARD_INDEX`, `NUM_SHARDS`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, and `VLLM_ASYNC_CONCURRENCY`.

When `OUTPUT_DIR` is not set, the launcher creates a descriptive output directory under:

```text
outputs/inference/<split>/<input_stem>/<model_tag>/
```

The final run folder includes backend, tensor-parallel size, async concurrency, context/prompt/output limits, tool-round budget, and a timestamp suffix. Set `APPEND_OUTPUT_TIMESTAMP=0` to keep an explicit `OUTPUT_DIR` unchanged.

### Native Sharding

Inference and pass@k both support process-level sharding with original example indices preserved. Each shard keeps rows where `source_idx % NUM_SHARDS == SHARD_INDEX`; when `NUM_SHARDS > 1`, standalone inference and the launcher append a directory name like `shard-00000-of-00008` under `OUTPUT_DIR`.

Run temperature-0 async vLLM inference with tensor parallel size 1 across 8 one-GPU shards:

```bash
for shard in $(seq 0 7); do
  SHARD_INDEX="${shard}" \
  NUM_SHARDS=8 \
  INFERENCE_CUDA_VISIBLE_DEVICES="${shard}" \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  INPUT_FILE=outputs/old-dev-schema-tool.jsonl \
  OUTPUT_DIR=outputs/inference/old-dev-schema-tool/temp0_async_tp1_shards8 \
  APPEND_OUTPUT_TIMESTAMP=0 \
  bash scripts/launch_inference.sh &
done
wait
```

Merge completed inference shards:

```bash
python scripts/run_inference_bird.py \
  --input_file outputs/old-dev-schema-tool.jsonl \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/bird_dev.json \
  --output_dir outputs/inference/old-dev-schema-tool/temp0_async_tp1_shards8_merged \
  --merge_shard_dirs outputs/inference/old-dev-schema-tool/temp0_async_tp1_shards8/shard-* \
  --overwrite
```

## Outputs

- `predict_dev.json`: official BIRD prediction format (`SQL\t----- bird -----\tdb_id`)
- `prediction_details.jsonl`: decoded completions and extracted SQL
- `filtered_examples.jsonl`: prompts skipped because they exceeded `max_prompt_length`
- `eval_results.jsonl`: per-example execution results
- `eval_summary.json`: simple/moderate/challenging/total EX accuracy
- `eval_summary.md`: summary tables by difficulty and by database
- `eval_summary_by_difficulty.csv`: CSV summary by difficulty
- `eval_summary_by_db.csv`: CSV summary by database

The local EX scorer follows the official BIRD dev evaluation semantics: it executes predicted and gold SQL on SQLite and checks raw row-set equality.

## Tool Calling

Tool-aware rows include a top-level `tools` list and prompt messages. Inference executes Gemma-style tool calls through `src/nl2sql_gspo/inference_tool_executor.py`, using async functions in `gen_tools.py`.

The tool environment searches databases through `BIRD_DB_ROOTS`; `scripts/run_inference_bird.py` configures this from `--database_dir`.

To build tool-aware inference rows from schema-built rows:

```bash
python scripts/data_generation/build_tool_dataset.py \
  --input outputs/bird_dev-schema.jsonl \
  --output outputs/bird_dev-schema-tool.jsonl
```

Use `--prompt-template consensus` to include the consensus tool definition.

## Data Preparation

Generate BM25-based few-shot files:

```bash
python scripts/data_generation/few_shot_bm25.py --top-n 5
```

Generate schema-augmented chat-format rows for inference:

```bash
python scripts/data_generation/schema_build.py \
  --split dev \
  --n-examples -1 \
  --output outputs/bird_dev-schema.jsonl
```

The schema builder writes top-level `db_id`, `gold_sql`, `evidence`, and `question` fields alongside `messages`, injects per-column meanings, and renders table/column statistics for prompting. Older message-only rows are still accepted by the shared loader.

## Pass@k And Self-Consistency

Run pass@k evaluation:

```bash
python scripts/run_passk_bird.py \
  --model_name_or_path outputs/gemma4_31b_gspo_bird/checkpoint-80 \
  --input_file outputs/bird_dev-schema-tool.jsonl \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/bird_dev.json \
  --output_dir outputs/passk/checkpoint-80 \
  --num_generations 16 \
  --temperature 0.7 \
  --overwrite
```

Pass@k uses the same sharding flags:

```bash
python scripts/run_passk_bird.py \
  --model_name_or_path outputs/gemma4_31b_gspo_bird/checkpoint-80 \
  --input_file outputs/bird_dev-schema-tool.jsonl \
  --output_dir outputs/passk/checkpoint-80_shards8 \
  --num_generations 16 \
  --shard_index 0 \
  --num_shards 8 \
  --overwrite
```

Merge pass@k shards with `--merge_shard_dirs outputs/passk/checkpoint-80_shards8/shard-*`.

Run self-consistency evaluation:

```bash
python scripts/run_self_consistency_bird.py \
  --passk_candidates_path outputs/passk/checkpoint-80/passk_candidates.jsonl \
  --database_dir databases/dev_databases \
  --output_dir outputs/self_consistency/checkpoint-80 \
  --overwrite
```

Self-consistency consumes the sampled candidates written by pass@k, executes them on SQLite, discards empty execution results, and majority-votes over raw result sets. It does not use temperature-0 inference outputs. Ties break by earliest sample index and then shorter SQL.
