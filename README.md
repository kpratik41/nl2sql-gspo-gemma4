# BIRD Test-Set Submission — Qwen3.8-27B, tool-calling NL2SQL

Self-consistency over 16 tool-calling rollouts. For each question the model is
given the database schema with per-column statistics and BM25-retrieved
few-shot examples, then generates 16 candidate solutions at temperature 1.2.
Each rollout may call `sqlite_query`, `sqlite_peek` and `bm25_search_sqlite`
against the database to verify its own SQL before answering. The 16 candidates
are clustered by the result set their SQL returns, and the largest cluster wins.

On the BIRD **dev** set this scores **72.10%** execution accuracy, against
71.38% for a single greedy sample. Both numbers come from scoring the exported
prediction file directly.

A greedy pass is also run and used *only* to break ties between equally-sized
clusters: 72.10% with it against 71.97% without. That is 2 questions, so the
stage is optional -- set `USE_TEMP0=0` to skip it and save an inference pass.

---

## 1. Environment

Ubuntu, CUDA 12.2+, Python 3.12. Built and verified on 8×H200; the pipeline
needs **2 GPUs with ≥80 GB each** (H100 80G or H200).

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # activate; do not call .venv/bin/python directly, see below
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is pinned to the exact versions this was verified against.
The important ones:

| package | version | note |
|---|---|---|
| torch | 2.10.0+cu128 | cu128 wheels; works on a CUDA 12.2/12.3 host |
| vllm | 0.19.1 | inference engine |
| transformers | 5.0.0 | Qwen3.8 support |
| flashinfer-python | 0.5.4 | linear-attention kernels |
| ninja | any | **required** — see below |
| rank_bm25 | 0.2.2 | few-shot retrieval |

**`ninja` must be on `PATH`, not merely installed.** Qwen3.8 has 48
linear-attention layers whose FlashInfer GDN prefill kernel is JIT-compiled at
runtime. If `ninja` is not on `PATH` the JIT dies with
`[Errno 2] No such file or directory: 'ninja'`, every generation returns an
error, **and the process still exits 0**. Activating the venv puts it on `PATH`;
`scripts/bird_test/run_bird_test.sh` also exports it explicitly. If you see a
run finish suspiciously fast with empty SQL, check for this first.

`flash-attn` is **not** required and is not installed. At Qwen3.8's head_dim of
256, torch SDPA dispatches to cuDNN's fused kernel, which measured ~2× faster
than FlashAttention on H200. vLLM selects FlashAttention 3 internally for the
16 full-attention layers.

## 2. Inputs

```
data/bird_test_data/raw/test.json           # questions; "SQL" is "" throughout
data/bird_test_data/raw/test_tables.json
data/bird_test_data/raw/column_meaning.json # REQUIRED (see §6)
databases/test_databases/<db_id>/<db_id>.sqlite
data/bird_train_data/raw/train-6601.jsonl   # few-shot retrieval pool (included)
```

## 3. Run

```bash
export MODEL_PATH=<hugging-face-repo-id-or-local-path>
bash scripts/bird_test/run_bird_test.sh
```

Output: **`outputs/bird_test/predict_test.json`**, in BIRD's expected format:

```json
{"0": "SELECT ...\t----- bird -----\tdb_id", "1": "..."}
```

Every stage is **resumable** — a completed stage is detected and skipped, so a
failure in selection never repeats the hours of GPU time spent generating. Each
stage writes its own log under `logs/bird_test/`.

| stage | output | approx. time (2 GPUs) |
|---|---|---|
| 1. few-shot retrieval | `data/bird_test_data/raw/test-few-shot.json` | ~2 min, CPU |
| 2. schema build | `outputs/bird_test/test-schema.jsonl` | ~10 min, CPU |
| 3. tool prompts | `outputs/bird_test/test-schema-tool.jsonl` | ~2 min, CPU |
| 4. pass@16 generation | `outputs/bird_test/passk16/merged/` | **dominant cost** |
| 5. temperature-0 pass | `outputs/bird_test/temp0/predict_dev.json` | ~1/16 of stage 4 |
| 6. selection | `outputs/bird_test/predict_test.json` | ~10 min, CPU |

Stage 4 is the one that matters for scheduling. On dev (1534 questions × 16
candidates = 24544 generations) it took ~63 minutes across 8 engines. With 2
GPUs at `TP=2` that is a **single** engine, so budget roughly 8 hours per 1500
questions and scale by test-set size.

To use more GPUs, nothing needs editing — shards are computed as
`GPU_COUNT / TP` at runtime:

```bash
TP=2 bash scripts/bird_test/run_bird_test.sh     # 8 GPUs -> 4 shards, ~4x faster
```

## 4. Configuration

Every value below is an environment variable override.

| variable | default | meaning |
|---|---|---|
| `MODEL_PATH` | *(required)* | HF repo id or local directory |
| `NUM_GENERATIONS` | 16 | candidates per question |
| `TEMPERATURE` | 1.2 | sampling temperature |
| `TOP_P` / `TOP_K` | 1.0 / 20 | |
| `TP` | 2 | tensor-parallel size |
| `SHARDS` | `GPU_COUNT / TP` | independent engines |
| `CONCURRENCY` | 16 | in-flight requests per shard |
| `MAX_TOOL_ROUNDS` | 8 | tool calls per rollout before a final answer is forced |
| `MAX_PROMPT_LENGTH` | 34000 | conversation cap incl. accumulated tool output |
| `MAX_NEW_TOKENS` | 8000 | per generation |
| `EVAL_TIMEOUT` | 60 | seconds per SQL execution |
| `USE_TEMP0` | 1 | run the greedy tie-breaking pass (stage 5) |

Thinking mode is **off** (`enable_thinking: false`). Qwen3.8 otherwise opens a
`<think>` block on every turn, which consumes the token budget without
improving accuracy here.

## 5. Error handling and restart

BIRD asks that a failure not require starting over. Accordingly:

- Each of the six stages is skipped when its output already exists, so re-running
  the entry point resumes rather than restarts.
- Generation writes `passk_candidates_raw.jsonl` **before** any post-processing;
  if a later step fails, the generations survive.
- Each shard writes to its own directory and its own log; a single failed shard
  can be re-run alone by setting `SHARDS`/`shard_index` to match.
- SQL execution is bounded two ways: a per-query timeout **and** a SQLite
  progress-handler interrupt. `sqlite3`'s `connect(timeout=)` bounds only lock
  waits, not query execution, so a runaway query on a large test database would
  otherwise hang indefinitely.
- The greedy pass enables a repeat-tool-call guard: with an identical context a
  greedy decoder re-emits an identical tool call and burns its whole round
  budget. On dev, 70 of 1534 greedy rollouts hit the round cap and 64 of those
  had re-issued a byte-identical query; the guard returns "already ran this"
  instead of re-executing, cutting capped rollouts to 50. It is deliberately
  **off** for stage 4, where sampling breaks the loop on its own.
- Selection never emits a blank query: if all 16 candidates for a question fail
  or return nothing, the first candidate that produced any SQL is used. On dev
  this kept blank output at **0.00%**, well inside BIRD's 5% abnormal-output
  threshold.

## 6. Notes for the evaluation team

- **`column_meaning.json` is required.** It supplies the per-column comments
  embedded in the prompt.
- **No ground truth is used anywhere.** `test.json` ships `"SQL": ""` and
  nothing in this pipeline reads that field. Candidate selection is computed
  purely from agreement among the model's own execution results.
- **No network access is needed or attempted at inference time.** There are no
  third-party API calls, no telemetry, and no experiment tracking. Set
  `HF_HUB_OFFLINE=1` if `MODEL_PATH` is a local directory.
- **Databases are only read.** The tools issue read-only `SELECT`/`WITH … SELECT`
  statements against the local SQLite files; nothing is written, copied or
  transmitted.
- **Few-shot examples come only from the BIRD train split.** Verified
  programmatically: the builder aborts if any test database appears in the
  retrieval pool, so no test question is ever used as a demonstration for
  another.

## 7. Layout

```
scripts/bird_test/
  run_bird_test.sh          entry point, all six stages
  build_test_few_shots.py   BM25 retrieval from the train pool
  select_and_export.py      self-consistency selection -> predict_test.json
scripts/data_generation/
  schema_build.py           schema + stats + column comments
  build_tool_dataset.py     tool-format prompts
scripts/
  run_passk_bird_qwen_async.py    pass@k generation (in-process vLLM engine)
  run_inference_bird_qwen_async.py  single-rollout loop, imported by the above
src/nl2sql_gspo/
  inference_tool_executor.py  tool dispatch
  tool_calling.py             tool definitions
  sql_utils.py, schema_utils.py
gen_tools.py                  sqlite_query / sqlite_peek / bm25_search_sqlite
prompts.py                    system prompt
```
