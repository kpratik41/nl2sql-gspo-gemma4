# NL2SQL Gemma — BIRD Test-Set Submission

Inference code for our BIRD leaderboard submission. **Two checkpoints are
submitted**; please score them in the order given in Step 6 and Step 7.

Follow the steps below in order. Everything after Step 8 is reference material.

---

## Step 1 — Create the environment

Python 3.12 with CUDA-capable GPUs. Create a fresh environment so the pinned
versions below are the ones actually used:

```bash
conda create -n nl2sql python=3.12 -y
conda activate nl2sql
pip install -r requirements.txt
```

or with venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Keep this environment active for every step that follows.** The pipeline calls
`python`, so an inactive environment fails with `python: command not found`.

Verify:

```bash
python -c "import torch, vllm, transformers; print(torch.cuda.device_count(), 'GPUs visible')"
```

---

## Step 2 — Set the Hugging Face token

The two model repositories are private. Export the read token supplied with our
submission email:

```bash
export HF_TOKEN=<token supplied with this submission>
```

or store it once:

```bash
hf auth login          # paste the same token when prompted
```

The token grants read access to both repositories and nothing else. Without it
the run stops at the download step with a 401/403 rather than part-way through
generation.

---

## Step 3 — Place the test questions and schema

Copy these three files from the BIRD test release into
`data/bird_test_data/raw/`:

```text
data/bird_test_data/raw/test.json              questions; the "SQL" field is empty
data/bird_test_data/raw/test_tables.json       schema description
data/bird_test_data/raw/column_meaning.json    per-column descriptions (REQUIRED)
```

**`column_meaning.json` is not optional.** The schema builder inlines those
descriptions into every prompt, and the models were tuned on prompts containing
them; without it accuracy drops. Our dev results used a `column_meaning.json`
byte-identical to the TA-SQL reference file named in the submission guidelines.

The directory ships empty apart from a README naming these files.

---

## Step 4 — Place the test databases

Put the SQLite databases at the repository root, one directory per `db_id`:

```text
databases/test_databases/<db_id>/<db_id>.sqlite
```

A symlink is fine if the databases live elsewhere:

```bash
ln -s /path/to/test_databases databases/test_databases
```

Steps 3 and 4 are both checked during preflight. A missing file stops the run
immediately with a message naming it, before any weights are loaded.

---

## Step 5 — Download the model weights

The pipeline does this automatically in a **single process** before it shards,
so you can skip straight to Step 6. Run it ahead of time if you would rather
separate the download from the evaluation:

```bash
hf download pratikkakkar/gemma-4-31b-it-bird-sft-rl
hf download pratikkakkar/gemma-4-31b-it-bird-rl
```

Each model is ~58 GB, so allow **~120 GB free in `~/.cache/huggingface`** for
both, plus space for outputs.

> **Why this matters:** generation runs N shards in parallel and each resolves
> the model independently. Against a cold cache that means N processes reaching
> for the hub at once — they serialise on a file lock, and the transfer has been
> seen to stall in `hf_xet`'s finalize step under that contention. The pipeline
> therefore downloads once, serially, and only then starts the shards, by which
> point every shard is a cache hit. If a download stalls anyway, retry with
> `HF_HUB_DISABLE_XET=1`.

---

## Step 6 — Run the primary model (SFT + RL)

**Please evaluate this model first.** It is the stronger of the two.

Each model takes about **3.5 hours on 8 GPUs**, so run it detached — otherwise a
dropped SSH connection kills the run. `tmux` is preinstalled on Ubuntu Server and
Amazon Linux; a `nohup` fallback for machines without it is at the end of this
step.

Start a detached session and launch the run inside it:

```bash
tmux new -s bird

# inside the session:
conda activate nl2sql                      # or: source .venv/bin/activate
export HF_TOKEN=<token supplied with this submission>
MODEL_PATH=pratikkakkar/gemma-4-31b-it-bird-sft-rl \
RUN_ROOT=outputs/bird_test_sft_rl \
  bash run.sh 2>&1 | tee run_sft_rl.log
```

Then detach with **`Ctrl-b`** followed by **`d`**. The run keeps going. To come
back:

```bash
tmux ls                    # list sessions
tmux attach -t bird        # reattach
```

Progress can also be followed without attaching:

```bash
tail -f outputs/bird_test_sft_rl/pipeline.log
```

During generation the pipeline prints a combined progress line every 5 minutes,
so you can tell at a glance that the run is alive:

```text
[2026-...] [pipeline]   generating [elapsed 1h12m30s]: shard0=1250/6144 shard1=1300/6144 shard2=1210/6128 shard3=1180/6128
```

A shard shows `loading` until its engine has finished bringing ~62 GB of weights
onto the GPUs, which takes a few minutes. Set `PROGRESS_INTERVAL=60` for a line
every minute instead.

Full per-shard detail, including vLLM's own output, goes to
`outputs/bird_test_sft_rl/passk-shard<N>.log`.

If `tmux` is unavailable, `nohup` needs no package:

```bash
nohup env MODEL_PATH=pratikkakkar/gemma-4-31b-it-bird-sft-rl \
          RUN_ROOT=outputs/bird_test_sft_rl \
          bash run.sh > run_sft_rl.log 2>&1 &
```

This step produces:

```text
outputs/bird_test_sft_rl/self_consistency/predict_test.json
```

## Step 7 — Run the second model (RL only)

Same shape as Step 6, in a detached session, with a different model and a
different `RUN_ROOT`:

```bash
tmux new -s bird2

# inside the session:
conda activate nl2sql                      # or: source .venv/bin/activate
export HF_TOKEN=<token supplied with this submission>
MODEL_PATH=pratikkakkar/gemma-4-31b-it-bird-rl \
RUN_ROOT=outputs/bird_test_rl_only \
  bash run.sh 2>&1 | tee run_rl_only.log
```

Produces:

```text
outputs/bird_test_rl_only/self_consistency/predict_test.json
```

`RUN_ROOT` differs between Step 6 and Step 7 so the two sets of predictions do
not overwrite each other. `MODEL_PATH` has no default and must be passed on
every run, so it is always unambiguous which checkpoint produced a given file.

Run the two models one after the other, not at the same time: each uses every
visible GPU.

---

## Step 8 — Collect the predictions

Each run writes the file to score at `<RUN_ROOT>/self_consistency/predict_test.json`,
in the official format (`SQL\t----- bird -----\tdb_id`), one entry per question id.

Alongside it:

```text
<RUN_ROOT>/run_manifest.tsv    what each stage produced
<RUN_ROOT>/pipeline.log        full log of the run
```

A verification gate runs before the pipeline reports success and fails if any
question id is missing or malformed, so a short or corrupt file is never
reported as complete.

**The run is resumable.** Re-running the same command after an interruption
skips finished stages and picks generation up from the candidate it reached; no
work already done is repeated.

---

## What we measured

BIRD dev (1534 questions), execution accuracy, produced by this same code:

| order | model | Hugging Face repository | temp 0 | self-consistency |
| :---: | --- | --- | ---: | ---: |
| **1st** | **SFT + RL** *(primary)* | `pratikkakkar/gemma-4-31b-it-bird-sft-rl` | **73.53%** | **74.58%** |
| 2nd | RL only | `pratikkakkar/gemma-4-31b-it-bird-rl` | 73.34% | 73.73% |

Self-consistency over 16 samples at temperature 1.2 is the submitted
configuration. The dev predictions behind those numbers ship in `predictions/`
so they can be checked without re-running generation — see `## Scoring` below.

---

## Resources Required

**Requested allocation: 2, 4, or 8 x H200. If H200 is unavailable, H100
works but all testing has been done on H200**. Below numbers on the full
1534-question evaluation without changes to the code.

The model is 31B in bf16, roughly 62 GB of weights. It is therefore run with
**tensor-parallel size 2**.

| GPUs | tp | shards | full pipeline wall clock |
| ---: | ---: | ---: | --- |
| 8 | 2 | 4 | **2 h 44 min (measured)** |
| 4 | 2 | 2 | ~5 h 20 min (extrapolated) |
| 2 | 2 | 1 | ~10 h 30 min (extrapolated) |

**8 GPUs is preferred.** The submitted configuration is pass@16 with
self-consistency: 16 samples for each of the 1534 questions, 24,544 tool-using
rollouts in total.

## How The Pipeline Runs

`run.sh` wraps `scripts/run_bird_test_pipeline.sh`. Stages, in order:

```text
0  few-shot retrieval      BM25 over the TRAIN pool only, never the split being scored
1  schema build            per-question schema block + column descriptions
2  tool-row build          wraps each row in the tool-calling system prompt
3a weight prefetch         single process, before any sharding
3b pass@16 generation      sharded across the available GPUs, temperature 1.2
4  self-consistency vote   majority over execution result sets
5  verification gate       every question id present and correctly formatted
```

Each stage is skipped when its output already exists **and** its `done.flag` was
written, so rerunning after a failure continues where it stopped rather than
repeating finished work.

Tensor-parallel size is fixed at 2 — the 31B model in bf16 is ~62 GB and does
not fit one 80 GB card at this context length. Remaining GPUs are used for data
parallelism: 8 GPUs give 4 shards, 2 GPUs give 1 shard.

To run from local weights instead of downloading, point `MODEL_PATH` at a
directory; the prefetch step detects this and skips the hub entirely:

```bash
MODEL_PATH=/path/to/local/weights RUN_ROOT=outputs/bird_test_local bash run.sh
```

## Repository Map

```text
.
├── run.sh                                  entry point; wraps the pipeline
├── prompts.py                              system prompt (mission, rules, tool syntax, memory)
├── gen_tools.py                            tool implementations exposed to the model
├── requirements.txt                        pinned inference dependencies
├── predictions/                            dev predictions from both submitted models
│   ├── dev_predictions_sft_rl.json         SFT+RL, self-consistency, 74.58%
│   └── dev_predictions_rl_only.json        RL only, self-consistency, 73.73%
├── scripts/
│   ├── run_bird_test_pipeline.sh           the pipeline: few-shot -> schema -> tools -> prefetch -> pass@k -> vote -> verify
│   ├── run_passk_bird.py                   pass@k sampling with the tool loop; shards across GPU pairs
│   ├── run_self_consistency_bird.py        majority vote over execution result sets
│   ├── run_inference_bird.py               single-pass inference and the async vLLM tool loop
│   ├── eval_bird_ex.py                     standalone BIRD EX scorer (see "Scoring")
│   ├── launch_inference.sh                 convenience launcher; sets PYTHONPATH
│   └── data_generation/
│       ├── few_shot_bm25.py                BM25 retrieval of demonstrations from the TRAIN pool
│       ├── schema_build.py                 builds the schema block and column descriptions
│       └── build_tool_dataset.py           wraps schema rows in the tool-calling system prompt
└── src/nl2sql_gspo/
    ├── tool_calling.py                     tool catalog, native call parsing, dispatch
    ├── inference_tool_executor.py          executes tool calls against the SQLite databases
    ├── schema_utils.py                     schema introspection and M-schema rendering
    ├── sql_utils.py                        SQL extraction, safety checks, BIRD execution + row-set match
    ├── model_utils.py                      model/tokenizer loading, chat templating
    └── data.py                             dataset loading helpers
```

## Configuration

Every setting has a validated default; override by exporting the variable before
the command in Step 6 or Step 7. Nothing here needs changing for a standard
submission run — `MODEL_PATH` and `RUN_ROOT` are the two you set explicitly.

| variable | default | governs |
| --- | --- | --- |
| `MODEL_PATH` | **none — required** | which checkpoint to run; the run fails if unset (Steps 6-7) |
| `SPLIT` | `test` | which `data/bird_<split>_data` and `databases/<split>_databases` to use |
| `RUN_ROOT` | `outputs/bird_test_pipeline` | where this run's artifacts go; **set per model** so runs do not collide |
| `NUM_GENERATIONS` | `16` | samples per question for pass@k; `1` gives a single greedy pass |
| `TEMPERATURE` | `1.2` | sampling temperature; the value every validated pass@16 run used |
| `NUM_EXAMPLES` | `-1` | limit questions processed; `-1` is all |
| `VLLM_TENSOR_PARALLEL_SIZE` | `2` | fixed at 2; 31B bf16 does not fit one 80 GB card at this context |
| `NUM_SHARDS` | GPUs / tp | independent engines, one shard of questions each |
| `EVAL_TIMEOUT` | `30` | scoring timeout — matches the official BIRD `--meta_time_out` |
| `TOOL_TIMEOUT` | `60` | tool calls during generation; deliberately not tied to scoring |
| `MAX_PROMPT_LENGTH` | `44000` | prompts above this are filtered and get `FALLBACK_SQL` |
| `VLLM_MAX_MODEL_LEN` | `53000` | must exceed `MAX_PROMPT_LENGTH`; the tool loop grows context |
| `FALLBACK_SQL` | `SELECT 1` | written for a question that could not be generated |
| `FEWSHOT_TOP_N` | `3` | demonstrations per question; matches the prompt the model was tuned on |
| `PROGRESS_INTERVAL` | `300` | seconds between generation progress lines |
| `EVAL_WORKERS` | `8` | threads used to execute candidates when scoring and voting |

## Smoke Test

`scripts/launch_inference.sh` runs a single greedy pass. **It is not the
submitted configuration** -- it exists so you can confirm the environment,
model download and tool loop all work on a couple of examples before committing
the machine to the full run.

Smoke test on 2 examples:

```bash
MODEL_PATH=pratikkakkar/gemma-4-31b-it-bird-sft-rl \
NUM_EXAMPLES=2 \
bash scripts/launch_inference.sh
```

It writes an official-format `predict_dev.json` and computes BIRD-style
execution accuracy. Sharding is handled the same way as the pipeline: set
`NUM_SHARDS` and `VLLM_TENSOR_PARALLEL_SIZE`, and run one process per shard with
`SHARD_INDEX` and `INFERENCE_CUDA_VISIBLE_DEVICES` set. Prefer
`scripts/run_bird_test_pipeline.sh`, which does that fan-out and the merge for
you.

The launcher forwards `MAX_PROMPT_LENGTH`, `MAX_NEW_TOKENS`, `MAX_TOOL_ROUNDS`,
`TEMPERATURE`, `TOP_P`, `EVAL_TIMEOUT`, `TOOL_TIMEOUT`, `SHARD_INDEX`,
`NUM_SHARDS`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`,
`VLLM_MAX_MODEL_LEN`, and `VLLM_ASYNC_CONCURRENCY`.

When `OUTPUT_DIR` is not set, a descriptive directory is created under
`outputs/inference/<split>/<input_stem>/<model_tag>/` with a timestamp suffix.
Set `APPEND_OUTPUT_TIMESTAMP=0` to keep an explicit `OUTPUT_DIR` unchanged, which
is required for resume to work.

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
- `run_manifest.tsv`: one row per completed stage — stage, artifact path, row
  count, status, timestamp. Written by the pipeline as each stage finishes, so
  it doubles as a record of what was produced and a quick coverage check.
- `done.flag`: written into a stage directory only after that stage returns
  successfully. A stage is treated as complete only when its flag exists —
  checking output-file size alone is not enough, because a process killed
  mid-write leaves a non-empty but truncated file that would otherwise be
  silently reused. Run directories created before this marker existed have no
  flag, so rerunning against one redoes the stage rather than reusing it.

The local EX scorer follows the official BIRD dev evaluation semantics: it
executes predicted and gold SQL on SQLite and checks raw row-set equality, with
a **30-second per-query timeout matching the official evaluator's
`--meta_time_out` default**, so a query that would time out on the BIRD harness
also times out here rather than scoring correct locally and wrong there.

Tool calls made *during* generation use a separate, more generous
`--tool_timeout`. That budget covers the model exploring the database, not the
graded query, so it is deliberately not tied to the scoring timeout.

| flag | default | governs |
| --- | ---: | --- |
| `--eval_timeout` | 30 s | scoring — matches the official BIRD `--meta_time_out` |
| `--tool_timeout` | 60 s | tool calls during generation |

Both are exposed by the launchers as `EVAL_TIMEOUT` and `TOOL_TIMEOUT`.

The predictions file always contains one entry per input row. An example whose
prompt exceeds `max_prompt_length` cannot be generated, so it is recorded in
`filtered_examples.jsonl`, given `stop_reason=prompt_too_long` in
`prediction_details.jsonl`, and written to the predictions file with the
`--fallback_sql` value (default `SELECT 1`). It therefore scores as incorrect
rather than leaving a hole in the file. Default limits are
`--max_prompt_length 44000` and `--vllm_max_model_len 53000`; the latter must
stay above the former, since the tool loop appends each tool response to the
running context.

## Scoring

`scripts/eval_bird_ex.py` scores a predictions file against gold SQL. It reuses
the same execution helpers the pipeline uses (`bird_execute_sql`,
`bird_result_match` in `src/nl2sql_gspo/sql_utils.py`), so a number produced by
this script is directly comparable to the pipeline's own eval output.

```bash
python scripts/eval_bird_ex.py \
  --predictions outputs/bird_test_sft_rl/self_consistency/predict_test.json \
  --gold data/bird_dev_data/raw/bird_dev.json \
  --database_dir databases/dev_databases
```

It prints overall EX plus breakdowns by difficulty and by database, and accepts
`--output_json` to write per-question results.

**The two timeouts are different and must not be conflated.** `--meta_time_out`
(default **30 s**) caps the *graded* query, matching the official BIRD
evaluator. Tool calls issued by the model *during generation* use a separate,
more generous **60 s** budget (`TOOL_TIMEOUT`); that covers the model exploring
the database, not the query being scored.

Two of the 1534 BIRD dev gold queries do not finish inside 30 s. Under BIRD
semantics they are unscorable and count as wrong for every system, so the
practical ceiling on dev is 1532, not 1534. A scorer without a real per-query
timeout will report roughly two questions more than the official harness would.

Reported dev numbers in this repository were produced with the command above,
against `data/bird_dev_data/raw/bird_dev.json` and `databases/dev_databases`,
at the default 30 s `--meta_time_out`.

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

**That said, if a run is interrupted we recommend restarting the pipeline from
the beginning rather than resuming.** Delete the run directory and launch the
same command again:

```bash
rm -rf outputs/bird_test_sft_rl
MODEL_PATH=pratikkakkar/gemma-4-31b-it-bird-sft-rl \
RUN_ROOT=outputs/bird_test_sft_rl \
  bash run.sh
```

Resume is there so a long run is not lost to a transient failure, but a clean
restart is the safest way to be certain no samples were dropped by whatever
interrupted the run — a dropped sample means fewer candidates in the
self-consistency vote for that question, and therefore a small accuracy cost.
Given the accuracy of the submitted numbers matters more than the extra hours, we
would rather you restart from scratch if anything goes wrong mid-run.

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

Retrieve BM25 few-shot demonstrations from the training pool. Three
demonstrations per question, which is what every validated prompt carries:

```bash
python scripts/data_generation/few_shot_bm25.py \
  --reference-input data/bird_train_data/raw/train-6601.jsonl \
  --dev-input data/bird_dev_data/raw/bird_dev.json \
  --dev-output data/bird_dev_data/raw/dev-few-shot.json \
  --top-n 3
```

Generate schema-augmented chat-format rows, then tool-aware rows:

```bash
python scripts/data_generation/schema_build.py \
  --split dev \
  --n-examples -1 \
  --output outputs/bird_dev-schema.jsonl

python scripts/data_generation/build_tool_dataset.py \
  --input outputs/bird_dev-schema.jsonl \
  --output outputs/bird_dev-schema-tool.jsonl
```

Our reported development results were produced from
`outputs/old-dev-schema-tool-unpatched.jsonl`, which is the tool-aware dev file
built by this same chain.

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
  --model_name_or_path <org>/<repo> \
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
  --model_name_or_path <org>/<repo> \
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
