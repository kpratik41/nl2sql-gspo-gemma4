# NL2SQL Gemma Inference

This repository contains inference and evaluation utilities for NL2SQL on BIRD-style SQLite datasets. It supports standard generation, native Gemma tool-calling loops, pass@k evaluation, and execution-result self-consistency.

## Setup

```bash
pip install -r requirements.txt
```

The launchers expect `PYTHONPATH` to include `src`; `scripts/launch_inference.sh` sets that automatically.

## Resources Required

**Requested allocation: 2, 4, or 8 x H200 (80GB+). If H200 is unavailable, H100
works equally well** — the pipeline only needs >= 80 GB per card and scales with
whatever count is allocated. Any of the three configurations below runs the full
1534-question evaluation without changes to the code.

The model is 31B in bf16, roughly 62 GB of weights. It is therefore run with
**tensor-parallel size 2** so the weights plus a 53k-token KV cache fit
comfortably on 80 GB cards. Remaining GPUs are used for data-parallel shards,
which is faster than widening tensor parallelism for this workload.

| GPUs | tp | shards | full pipeline wall clock |
| ---: | ---: | ---: | --- |
| 8 | 2 | 4 | **~12 h (measured)** |
| 4 | 2 | 2 | ~24 h (extrapolated) |
| 2 | 2 | 1 | ~48 h (extrapolated) |

**8 GPUs is strongly preferred.** The submitted configuration is pass@16 with
self-consistency: 16 samples for each of the 1534 questions, 24,544 tool-using
rollouts in total. Our measured 8-GPU run took 42,482 s of generation plus 334 s
of execution scoring, about 11.9 hours end to end. The 4- and 2-GPU rows are
linear extrapolations over shard count and have not been timed directly.

The self-consistency voting stage that follows generation is CPU-only and takes
a few minutes.

For reference, a single temperature-0 pass over the same 1534 questions takes
**~47 min on 8 GPUs** (measured: 2741 s generation + 78 s scoring). That is the
relevant number only if a temperature-0 configuration is submitted instead.

Disk: the checkpoint is ~62 GB. Peak host RAM stays under 32 GB per shard.

## Required Input Files

**This pipeline requires `column_meaning.json`.** The schema builder injects the
per-column descriptions inline into every prompt, so the file must be present
for the test split at:

```text
data/bird_test_data/raw/column_meaning.json
```

`scripts/run_bird_test_pipeline.sh` checks for it during preflight and stops
with a clear error if it is missing. If a column has no entry the builder simply
omits that comment rather than failing, so a partial file is tolerated — but
prompts will be weaker.

The other required test inputs are `test.json`, `test_tables.json`, and the
`test_databases/` SQLite directory. Note that our development results were
produced with a `column_meaning.json` that is byte-identical to the TA-SQL
reference file named in the submission guidelines, so dev and test prompts are
built from the same description source.

## Reproducing The Submission (One Command)

This is the path to run for the test set. It goes from the raw `test.json` the
BIRD team provides to the final prediction file, and every stage is skipped when
its output already exists, so rerunning after any failure continues where it
stopped.

```bash
MODEL_PATH=outputs/gemma-best-rl \
bash scripts/run_bird_test_pipeline.sh
```

Stages: few-shot retrieval -> schema build -> tool-row build -> pass@16
generation (auto-sharded across the available GPUs) -> self-consistency vote ->
a verification gate that fails if any question id is missing or malformed.

Final artifact:

```text
outputs/bird_test_pipeline/self_consistency/predict_test.json
```

one entry per question id in official BIRD format, `SQL\t----- bird -----\tdb_id`.

Useful overrides (all optional):

```bash
MODEL_PATH=outputs/gemma-best-rl \
NUM_GENERATIONS=16 \
TEMPERATURE=1.2 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
NUM_SHARDS=4 \
RUN_ROOT=outputs/bird_test_pipeline \
bash scripts/run_bird_test_pipeline.sh
```

`NUM_SHARDS` defaults to `GPU_COUNT / VLLM_TENSOR_PARALLEL_SIZE`, so on 8 GPUs it
runs 4 shards of tensor-parallel-2 without being set. To reproduce our
development numbers instead, set `SPLIT=dev`.

The sections below document the individual stages for anyone who wants to run
them separately.

## Run Inference

```bash
bash scripts/launch_inference.sh
```

The launcher uses async vLLM, loads `outputs/gemma-best-rl`, reads `outputs/bird_dev-schema.jsonl`, writes official-style `predict_dev.json`, and computes BIRD-style execution accuracy against `databases/dev_databases`.

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

Inference and pass@k both support process-level sharding with original example indices preserved. Each shard keeps rows where `source_idx % NUM_SHARDS == SHARD_INDEX`; when `NUM_SHARDS > 1`, standalone inference and the launcher append a directory name like `shard-00000-of-00004` under `OUTPUT_DIR`.

A 31B model in bf16 does not fit on one 80 GB card at this context length, so
each shard uses tensor-parallel 2. On 8 GPUs that is 4 shards; on 4 GPUs use 2
shards, and on 2 GPUs a single shard with `NUM_SHARDS=1`.

Run temperature-0 async vLLM inference as 4 shards of tensor-parallel 2:

```bash
for shard in 0 1 2 3; do
  gpus="$((shard * 2)),$((shard * 2 + 1))"
  SHARD_INDEX="${shard}" \
  NUM_SHARDS=4 \
  INFERENCE_CUDA_VISIBLE_DEVICES="${gpus}" \
  VLLM_TENSOR_PARALLEL_SIZE=2 \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  INPUT_FILE=outputs/old-dev-schema-tool.jsonl \
  OUTPUT_DIR=outputs/inference/old-dev-schema-tool/temp0_async_tp2_shards4 \
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
  --output_dir outputs/inference/old-dev-schema-tool/temp0_async_tp2_shards4_merged \
  --merge_shard_dirs outputs/inference/old-dev-schema-tool/temp0_async_tp2_shards4/shard-* \
  --overwrite
```

## Outputs

- `predict_dev.json`: official BIRD prediction format (`SQL\t----- bird -----\tdb_id`)
- `prediction_details.jsonl`: decoded completions and extracted SQL
- `filtered_examples.jsonl`: prompts that exceeded `max_prompt_length`
- `generation_progress.jsonl`: per-example results appended as they finish; used to resume
- `eval_results.jsonl`: per-example execution results
- `eval_summary.json`: simple/moderate/challenging/total EX accuracy
- `eval_summary.md`: summary tables by difficulty and by database
- `eval_summary_by_difficulty.csv`: CSV summary by difficulty
- `eval_summary_by_db.csv`: CSV summary by database

The local EX scorer follows the official BIRD dev evaluation semantics: it executes predicted and gold SQL on SQLite and checks raw row-set equality.

The predictions file always contains one entry per input row. An example whose
prompt exceeds `max_prompt_length` cannot be generated, so it is recorded in
`filtered_examples.jsonl`, given `stop_reason=prompt_too_long` in
`prediction_details.jsonl`, and written to the predictions file with the
`--fallback_sql` value (default `SELECT 1`). It therefore scores as incorrect
rather than leaving a hole in the file. Default limits are
`--max_prompt_length 44000` and `--vllm_max_model_len 53000`; the latter must
stay above the former, since the tool loop appends each tool response to the
running context.

## Logging And Restarting Mid-Run

Every run writes a full log and is checkpointed per example, so an interrupted
evaluation restarts from the example it reached rather than from the beginning.
No work already generated is repeated and no tokens are spent twice.

**Logs written for every run**

- `pipeline.log` in the run root: `scripts/run_bird_test_pipeline.sh` tees every
  stage's stdout and stderr here, so one file carries the whole run. Each
  generation shard additionally writes `<passk_dir>-shard<N>.log`, and a failing
  shard is named explicitly in the pipeline log.
- `generation_progress.jsonl`: one JSON line per completed example, appended and
  fsynced as it finishes. This is the restart checkpoint.
- `run_report.md` and `per_example_report.csv`: per-run summary and per-example
  breakdown written by `scripts/run_inference_bird.py`.
- `prediction_details.jsonl`: the decoded completion, extracted SQL, tool-call
  count and `stop_reason` for every example, which is what to inspect when an
  output looks wrong.
- `filtered_examples.jsonl`: prompts that exceeded the length limit, so an
  abnormal output can immediately be attributed to truncation rather than to a
  model failure.
- `eval_summary.md` / `eval_summary.json`: accuracy by difficulty and by database
  (dev only; skipped on test, which has no gold SQL).

**To restart after any failure, rerun the identical command.** The run reads
`generation_progress.jsonl`, skips finished examples, and continues. If every
example is already present, the model is not even loaded and the run proceeds
straight to scoring. A progress file torn by a hard kill is handled: the partial
final line is discarded and that single example is regenerated.

Generation is checkpointed per example. Every completed example is appended to
`generation_progress.jsonl` and fsynced before the next one starts, so a crash,
an OOM kill, or a node restart loses at most the in-flight examples.

Resuming requires the rerun to land in the **same output directory**. The
launcher appends a timestamp to `OUTPUT_DIR` by default, which would start a
fresh run every time, so pin the directory for any long job:

```bash
OUTPUT_DIR=outputs/inference/bird_test_run \
APPEND_OUTPUT_TIMESTAMP=0 \
bash scripts/launch_inference.sh
```

Rerun that identical command after a failure and it resumes. Invoking
`scripts/run_inference_bird.py` directly needs no extra flags, since
`--output_dir` is already explicit there.

The run reads `generation_progress.jsonl`, skips everything already generated,
and continues from the first incomplete example. Only the remaining prompts are
sent to the model; if every example is already present the engine is not loaded
at all and the run goes straight to evaluation. A progress file truncated
mid-write by a hard kill is handled — the torn line is discarded and that one
example is regenerated.

Pass `--no_resume` (or `NO_RESUME=1`) to discard prior progress and regenerate
everything from scratch.

## Tool Calling

Tool-aware rows include a top-level `tools` list and prompt messages. Inference executes Gemma-style tool calls through `src/nl2sql_gspo/inference_tool_executor.py`, using async functions in `gen_tools.py`.

The tool environment searches databases through `BIRD_DB_ROOTS`; `scripts/run_inference_bird.py` configures this from `--database_dir`.

To build tool-aware inference rows from schema-built rows:

```bash
python scripts/data_generation/build_tool_dataset.py \
  --input outputs/bird_dev-schema.jsonl \
  --output outputs/bird_dev-schema-tool.jsonl
```

## Data Preparation

Generate BM25-based few-shot files:

```bash
python scripts/data_generation/few_shot_bm25.py --top-n 3
```

Generate schema-augmented chat-format rows for inference:

```bash
python scripts/data_generation/schema_build.py \
  --split dev \
  --n-examples -1 \
  --output outputs/bird_dev-schema.jsonl
```

The schema builder writes top-level `db_id`, `gold_sql`, `evidence`, and `question` fields alongside `messages`, injects per-column meanings, and renders table/column statistics for prompting. Older message-only rows are still accepted by the shared loader.

For leaderboard-style BIRD test inputs, build the prompt JSONL first and keep it as the reproducible inference artifact. Test rows do not contain gold SQL, so local EX accuracy is skipped; inference still writes official-format predictions and prediction-execution sanity reports.

**Retrieve few-shot demonstrations first.** `schema_build.py` reads a
`few_shot_examples` field from its input; pointing it at a raw `test.json`
silently produces prompts with no demonstrations at all, which do not match the
prompt format the model was validated on. Demonstrations are retrieved from the
training pool only, never from the split under evaluation:

```bash
python scripts/data_generation/few_shot_bm25.py \
  --reference-input data/bird_train_data/raw/train-6601.jsonl \
  --dev-input data/bird_test_data/raw/test.json \
  --dev-output data/bird_test_data/raw/test-few-shot.json \
  --top-n 3
```

Then build the schema-augmented rows from that file:

```bash
python scripts/data_generation/schema_build.py \
  --split test \
  --input-file data/bird_test_data/raw/test-few-shot.json \
  --database-dir databases/test_databases \
  --meanings-file data/bird_test_data/raw/column_meaning.json \
  --n-examples -1 \
  --output outputs/bird_test-schema.jsonl

python scripts/data_generation/build_tool_dataset.py \
  --input outputs/bird_test-schema.jsonl \
  --output outputs/bird_test-schema-tool.jsonl
```

Then run inference against `outputs/bird_test-schema-tool.jsonl` and `databases/test_databases`. Use `--predictions_filename predict_test.json` only if the submission instructions require that filename; the JSON values remain in official BIRD format (`SQL\t----- bird -----\tdb_id`).

## Pass@k And Self-Consistency

Run pass@k evaluation:

```bash
python scripts/run_passk_bird.py \
  --model_name_or_path outputs/gemma-best-rl \
  --input_file outputs/bird_dev-schema-tool.jsonl \
  --database_dir databases/dev_databases \
  --diff_json_path data/bird_dev_data/raw/bird_dev.json \
  --output_dir outputs/passk/gemma-best-rl \
  --num_generations 16 \
  --temperature 1.2 \
  --overwrite
```

Pass@k uses the same sharding flags:

```bash
python scripts/run_passk_bird.py \
  --model_name_or_path outputs/gemma-best-rl \
  --input_file outputs/bird_dev-schema-tool.jsonl \
  --output_dir outputs/passk/gemma-best-rl_shards4 \
  --num_generations 16 \
  --shard_index 0 \
  --num_shards 4 \
  --vllm_tensor_parallel_size 2 \
  --overwrite
```

Merge pass@k shards with `--merge_shard_dirs outputs/passk/gemma-best-rl_shards4/shard-*`.

Run self-consistency evaluation:

```bash
python scripts/run_self_consistency_bird.py \
  --passk_candidates_path outputs/passk/gemma-best-rl/passk_candidates.jsonl \
  --database_dir databases/dev_databases \
  --output_dir outputs/self_consistency/gemma-best-rl \
  --overwrite
```

Self-consistency consumes the sampled candidates written by pass@k, executes them on SQLite, discards empty execution results, and majority-votes over raw result sets. It does not use temperature-0 inference outputs. Ties break by earliest sample index and then shorter SQL.

Pass `--input_file` (the same JSONL pass@k generated from) so the predictions
file is checked for coverage. Any question id left without a usable candidate is
filled with `--fallback_sql` and reported, instead of being dropped from the
submitted file:

```bash
python scripts/run_self_consistency_bird.py \
  --passk_candidates_path outputs/passk/gemma-best-rl/passk_candidates.jsonl \
  --input_file outputs/bird_dev-schema-tool.jsonl \
  --database_dir databases/dev_databases \
  --output_dir outputs/self_consistency/gemma-best-rl \
  --overwrite
```

### Resume And Coverage For Pass@k

Pass@k is checkpointed per candidate, not per example. Each `(example, sample)`
pair is appended to `generation_progress.jsonl` and fsynced as it completes, so
a 16-sample run killed partway resumes at the exact candidate it reached rather
than regenerating finished work. Rerun the identical command into the same
`--output_dir` to resume, or pass `--no_resume` to start over. If every
candidate is already present the engine is not loaded at all.

Examples whose prompt exceeds `--max_prompt_length` cannot be generated. Rather
than dropping out of the candidate pool — which would remove them from the
self-consistency predictions file entirely — they receive a full set of
`--num_generations` fallback candidates carrying `--fallback_sql` and
`stop_reason=prompt_too_long`. Pass@k denominators therefore stay correct and
those examples score as wrong instead of vanishing.

Pass@k defaults match the temperature-0 path: `--max_prompt_length 44000` and
`--vllm_max_model_len 53000`.

On unlabeled test inputs, pass@k and self-consistency still run, but accuracy/pass@k fields are reported as `n/a`. Self-consistency writes an official prediction file in its output directory.
