# Nemotron Lightning on BIRD

Temperature-0, BF16 tool-assisted inference with NVIDIA Nemotron 3.5 Lightning
30B-A3B on the **old 1,534-question BIRD dev set**. Preparation and inference default
to the **89 `california_schools` questions**; full-dev runs require an explicit
input selection. Both baseline and extended-statistics full runs are complete.

## Latest extended-statistics result

The updated full run processed all 1,534 records and scored **827/1,404 eligible
questions (58.90%)**, with **91.53% coverage**. All 129 European-football prompts
exceeded the context limit before generation. California question 85 was filtered
after its history grew to 38,871 input tokens, leaving insufficient room for the
full 8,192-token output allowance. No prompts or history were trimmed.

On these same 1,404 questions, the previous full run scored **824/1,404 (58.69%)**:
a net gain of three answers, with 75 gains and 72 losses. Counting filtered
questions as incorrect gives **827/1,534 (53.91%)**. The eligible score must not be
compared directly with the baseline's full-coverage 59.13% score. Both statistics
and tool argument handling changed; one run does not isolate their effects.

There were 1,393 final SQL answers, 11 missing answers, four final-SQL execution
errors, and one gold-query timeout (question 701). Inference took 1,421.0 seconds
(23m 41s), and evaluation took 198.2 seconds (3m 18s). The run made 3,777 tool calls
with 95 tool errors. Every preflight token count matched server usage.

The standalone extended-statistics California run scored **42/89 (47.19%)** with
no filtering, versus 48/89 (53.93%) for the earlier fixed-tools baseline. In the
updated full run, California scored **46/88 eligible (52.27%)**, with one filtered
question. Batched temperature-zero inference did not reproduce the standalone
result.

The committed [results summary](results/extended_stats_summary.json) includes
per-database scores, coverage, comparison counts, settings, and provenance hashes.
Full traces, predictions, source snapshots, and logs remain local under
`outputs/bird_dev1534_extended_stats_temp0/`. Its `exports/` directory contains
`full_run_stats.txt` and a self-contained `example_q66_full_trace.jsonl` with one
updated input, all tool definitions, and the recorded model/tool responses.

## Initial California result

The completed California run produced **47/89 correct (52.81% execution
accuracy)**. All 89 questions returned parseable final SQL; one final query failed
execution. No samples were filtered, and no prompts/history were trimmed.
The longest actual input including tool history was 19,649 tokens. Every preflight
token count matched vLLM's reported input count. Shards handled 45 and 44 questions;
inference took 126 seconds after server warmup. Full results and the audit are in
`outputs/california_schools_temp0/evaluation.json` and `run_report.json`.

## Full old-dev result

The full **1,534-question** run with **fixed tools and the baseline prompt** scored
**907/1,534 (59.13% execution accuracy)**. It uses temperature 0, three training
BM25 examples, cached M-Schema statistics, eight tool rounds, and TP4 across each
of two shards with context 43,000. All questions were newly inferred in this run.

| Database | Correct / samples | EX |
| --- | --- | --- |
| california_schools | 41 / 89 | 46.07% |
| financial | 65 / 106 | 61.32% |
| toxicology | 85 / 145 | 58.62% |
| card_games | 103 / 191 | 53.93% |
| codebase_community | 124 / 186 | 66.67% |
| superhero | 97 / 129 | 75.19% |
| formula_1 | 86 / 174 | 49.43% |
| european_football_2 | 83 / 129 | 64.34% |
| thrombosis_prediction | 73 / 163 | 44.79% |
| student_club | 116 / 158 | 73.42% |
| debit_card_specializing | 34 / 64 | 53.12% |

There were 1,519 parseable final answers, 15 missing final answers, 13 final-SQL
execution errors, and one gold-query timeout (`codebase_community`, question 701).
The unchanged evaluator uses a 60-second timeout and counts that gold error as
incorrect in the full 1,534 denominator. No questions were filtered or prompts
trimmed. The largest actual input was 32,807 tokens, and every preflight token
count matched server usage. Each shard handled 767 questions.
The one gold query also timed out in a separate 180-second diagnostic retry;
`gold_timeout_recheck.json` records it, and the original evaluation is unchanged.

Inference took 1,553.7 seconds (25m 54s); evaluation took 260.2 seconds (4m 20s).
Both replicas were stopped afterward and all GPU memory was released.
California's prepared records were verified identical to the standalone run,
but its score changed from 48/89 to 41/89. These runs therefore do not establish
a reproducible accuracy gain from the tool fix; runtime/batch variation remains
uncontrolled.

Artifacts are under `outputs/bird_dev1534_fixed_tools_temp0/`: `predict.json`,
`evaluation.json`, `run_report.json` (per-database and difficulty metrics),
`inference_audit.json`, and every question's trace. `reproducibility.json` and
`source_snapshot/` retain model/data/source hashes, hardware and package versions,
and the implementation used. Server and inference/evaluation logs are also saved.

## Data and model

- Model cache: `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9`.
  All 14 indexed BF16 shards are present (65,827,374,264 bytes in total).
- Dev JSON is freshly downloaded from the official
  [June 2024 archive](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip),
  member `dev_20240627/dev.json`. HTTP range requests extract only JSON;
  database archives are never downloaded. This is **not** `dev_20251106` or mini-dev.
- Few-shot source: a fresh download of the official
  [BIRD23 filtered train set](https://huggingface.co/datasets/birdsql/bird23-train-filtered),
  pinned to revision `4068469807b255fcfc0816bdd520946fe460d256`, with 6,601 examples.
- Database roots: `/home/ec2-user/nl2sql-gspo-gemma4/databases/{dev,train}_databases`.
  Set `BIRD_DB_ROOTS` to colon-separated roots to override. Preparation verifies
  the databases for both splits exist. Few-shot retrieval needs training JSON;
  it does not execute the training SQL.
- `data/raw/provenance.json` records URLs, archive metadata, download time, counts,
  and SHA-256 hashes. Dev gold SQL stays in the raw evaluation data and is never
  inserted into inference records or tool responses.

## Prompts and tools

Prompts use [M-Schema](https://github.com/XGenerationLab/M-Schema) structure:
database identifier, tables, fields, declared types, keys, descriptions, and
foreign keys. Each column includes exact full-table NULL/non-NULL and distinct
counts, min/max, numeric mean, and three most frequent values with counts.
Numeric means exclude text storage values; min/max retain SQLite's native
ordering. Long values are visibly truncated only for prompt display.

An optional `--stats-profile extended` adds numeric median, Q25/Q75, population
standard deviation, numeric/storage counts, NULL fraction, and explicit mode
frequency, fraction, and number of tied modes. The numeric statistics exclude
NULL and TEXT values rather than coercing numeric-looking strings to numbers.
Quartiles use linear interpolation at `(n - 1) * p`; mode covers non-NULL values
and shows a deterministic representative when tied. Full-table summaries are
context, not substitutes for question-specific filtered aggregates.

These additional statistics have a separate versioned `cache/schema_extended/`
cache. Baseline caches, fingerprints, and prepared prompts are preserved.
Extended-profile `sqlite_peek` also reports these statistics and computes/caches
them for its actual `where` filter; filtered and unfiltered profiles do not share
results. The runner reads the profile from the prepared record automatically.

```bash
.venv/bin/python -m bird_inference.prepare --db all --stats-profile extended
```

This produces `data/prepared/all.stats-extended.jsonl` with 1,534 records, the same
baseline system prompt and three BM25 examples, and added schema statistics.
The completed run is summarized above. Its initial prompts span
3,492–49,889 tokens. All 129 European-football prompts exceed the current
43,000-token context (even before reserving 8,192 output tokens). The existing
runner filters these records without trimming them. A full-coverage comparison needs
an explicitly chosen context/prompt design; do not treat a filtered run as
comparable to the original 1,534-question score.

Self-contained one-record model inputs are available as
`data/prepared/example_q66.baseline.input.jsonl` and
`data/prepared/example_q66.extended.input.jsonl`. Each contains an API payload in
`request` (messages, all tools, generation/template settings) and the native
template text in `rendered_prompt`. There is no target gold SQL or model output.
The example grows from 12,928 to 21,214 input tokens. These export envelopes are
for inspection or extracting `request`; use the full prepared input file above
with `bird_inference.run`.

Statistics are stored under `cache/schema/`, protected by file locks and written
atomically. Keys include cache version, resolved DB path, inode, size, timestamps,
SQLite header change counter, WAL state, and description-file metadata. Repeated
questions do not recompute stats. Delete a database's cached files to force a
rebuild. Use stable database files during a run; inference checks fingerprints
against prepared prompts before starting.

Exactly three training examples are selected per dev question by BM25Okapi over
`question + evidence`, with deterministic ties by training row index. Retrieval
never searches dev answers. Train and dev database sets are checked as disjoint.

Tool behavior follows `origin/consensus` at
`588e1378777c64fa1a48d2548efccd51edd29270` (`gen_tools.py`):

| Tool | Purpose |
| --- | --- |
| `sqlite_query` | Execute actual read-only candidate SQL; return columns, rows and truncation status. Display limits never change the SQL. |
| `sqlite_peek` | Inspect distinct samples, types and profiles, optionally filtered with `where`. Reuse cached exact profiles. |
| `bm_25_search` | Search stored column values with BM25 plus substring/word matches; deduplicate the combined results. |

The consensus search tool was named `bm25_search_sqlite`; this implementation
exposes the requested `bm_25_search` name. Compared with consensus, profiles are
exact and cached rather than randomly sampled; SQL is executed without silent
rewrites. Peek/search accept exact schema names, case differences, and balanced
SQL identifier quotes; genuinely unknown names return valid names for repair.
They do not guess a different column through fuzzy matching. Tools are restricted to the current
question's database. Value search caches up to 100,000 distinct values and reports
whether its corpus was truncated. These bounds do not truncate evaluation results.

The full-run audit also found 46 peek requests with a string rather than an array
for `columns`. Peek now accepts unambiguous JSON-array strings, comma-separated
names, and an explicit `*`, validating all resolved names against the schema.
An exact column name containing a comma takes precedence over splitting. This
normalizes 34 of those historical calls across 29 questions; 15 of those questions
were incorrect. This is a validation repair, not a measured accuracy gain. Neither
SQL strings nor the evaluator are rewritten. Details are in
`outputs/stats_improvement_audit/tool_argument_audit.json`.

The prepared JSONL stores `messages` and all three JSON-schema `tools`. Passing
both to the model's native chat template inserts tool definitions and its XML
calling instructions into the actual model prompt. There is no Gemma-specific
`call:tool{...}` syntax. Model tool requests are executed locally, and real results
are returned as `role: tool` messages before the next generation.

## Tool-calling research

NVIDIA's [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
documents `--enable-auto-tool-choice --tool-call-parser qwen3_coder
--reasoning-parser nemotron_v3`, standard function schemas in the `tools` request
field, and `chat_template_kwargs: {force_nonempty_content: true}` for coding agents.
Its native template renders calls as `<tool_call><function=NAME>` with nested
`<parameter=NAME>` values. The application consumes parsed `message.tool_calls`,
dispatches them, appends results, and requests the next assistant turn.

The [NVIDIA NIM guide](https://docs.nvidia.com/nim/large-language-models/2.0.10/get-started/advanced/get-started-nemotron-3.5-lightning.html)
also documents these parsers, reasoning budgets and total generation token caps.
NVIDIA's [agent scaffolding examples](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/OpenScaffoldingResources/README.md)
provide additional integration examples. This run uses direct local vLLM,
thinking enabled, temperature 0, top-p 1, seed 0, and no speculative decoding.
Greedy decoding does not guarantee bitwise reproducibility across runtime or batch changes.

## Earlier accuracy experiments

The recorded baseline remains **47/89 (52.81%)**. A score of 58.43% requires
52 correct answers, five more. In an offline diagnostic of the failed results,
seven matched the gold result after column permutation, and five others matched
after projecting away extra columns. These are diagnostics, not corrected scores:
the evaluator still compares positional columns and has not been changed.
Six peek calls across five questions failed on quote-wrapped valid names; three
of those questions ultimately failed. The identifier fix now accepts those calls.
The completed experiments below measure its observed effect alongside possible
runtime/batch variation.

An optional `verified` prompt adds a reasoning checklist for requested columns
and their order, row granularity, evidence-defined metrics, joins, NULLs, and result
shape. It asks the model to avoid repeating failed calls, reserve a tool round to
verify the final query, and preserve the verified SQL in its final answer. It uses
the same full M-Schema/statistics and the same three BM25 examples; there is no
gold-answer feedback, schema pruning, or prompt/history trimming. Baseline is
still the default profile. Prepare the separate experiment with:

```bash
.venv/bin/python -m bird_inference.prepare --prompt-profile verified
.venv/bin/python scripts/render_prompt.py --input data/prepared/california_schools.verified.jsonl --output data/prepared/california_schools.verified.prompt.txt
```

The verified-prompt experiment used these commands (choose a fresh output directory
for a rerun):

```bash
.venv/bin/python -m bird_inference.run --input data/prepared/california_schools.verified.jsonl --output outputs/california_schools_verified_temp0
.venv/bin/python -m bird_inference.evaluate --output outputs/california_schools_verified_temp0
```

Both experiments were completed sequentially on the same two replicas, using
temperature 0, eight tool rounds, TP4 with two shards, context 43,000, and the
unchanged evaluator:

| Run | Correct / 89 | EX | Missing final SQL | Tool errors | Inference seconds |
| --- | --- | --- | --- | --- | --- |
| Original baseline | 47 | 52.81% | 0 | 54 | 126 |
| Fixed tools + baseline prompt | 48 | 53.93% | 0 | 44 | 122 |
| Fixed tools + verified prompt | 36 | 40.45% | 28 | 39 | 256 |

Fixed tools gained six questions and lost five versus the original. All 89 final
queries executed successfully. This is a one-question net gain from one run per
variant, not proof of a stable accuracy improvement. The verified prompt gained
three and lost fifteen versus fixed tools with the baseline prompt. It reached
the ninth assistant turn on 41 questions; 28 ended without a parseable final SQL,
and one additional final SQL failed execution. All failures count in the score.
No samples were filtered or trimmed. Maximum actual input lengths were 17,829
and 28,431 respectively, with every preflight count matching server usage.

**Keep the baseline prompt as the default.** The verified profile is retained as
a failed experiment, not recommended for inference. Its additional instructions
were associated with more rounds, approximately twice the output tokens, and
missing final answers. A subsequent experiment should isolate a short output
format instruction or constrained final response before adding more verification
instructions. Repeated runs are needed to assess batch-dependent variation;
latencies also reflect warm-up/prefix-cache differences.

Full paired predictions, gained/lost question IDs, token counts, and failure
statistics are in `outputs/california_accuracy_experiments/comparison.json`.
Input hashes, inference logs/timings, and server logs are saved alongside it.
Both replicas were shut down after the experiments. Compare question-level
predictions with the other workspace before attributing its higher score to a
specific setting. These runs have not reproduced 58%+ accuracy.

Relevant primary research and possible later experiments:

- [CHESS](https://arxiv.org/abs/2405.16755) uses retrieval, candidate refinement,
  and natural-language unit tests. Our checklist is a lightweight adaptation,
  not an implementation of its full system. A future verifier should check
  semantics and expected output shape, not merely successful SQL execution.
- [CHASE-SQL](https://arxiv.org/abs/2410.01943) generates candidates with different
  reasoning strategies and ranks them using a trained pairwise selector. An
  adaptation could generate three temperature-0 candidates with different prompts
  (direct, decomposed, execution-plan reasoning), execute them, and compare their
  compliance with the question. This costs more inference; simple voting or an
  untrained selector is not equivalent to the paper's method.
- [DAIL-SQL](https://arxiv.org/abs/2308.15363) studies example selection and prompt
  organization. A compatible experiment would retrieve a BM25 candidate pool,
  rerank for question/SQL-pattern relevance and diversity, and retain exactly
  three training examples. It must use a generated draft, never dev gold SQL,
  if query structure is used for retrieval.
- NVIDIA's [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
  recommends temperature 1.0/top-p 0.95 generally. That is not evidence of a gain
  on these questions, and this project retains the requested temperature 0.

The research techniques above have not been evaluated here beyond the prompt
experiment just reported. Changes developed from California dev errors should also be validated on untouched
dev databases to distinguish general improvements from adaptation to this subset.

## Reproduce

Run from this repository root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
python3 scripts/download_data.py
.venv/bin/python -m bird_inference.prepare
.venv/bin/python scripts/render_prompt.py
.venv/bin/python -m unittest discover -s tests -v
```

Start the server in a separate terminal:

```bash
bash scripts/serve_shards.sh
```

The server uses the existing clean vLLM 0.30.0 environment at
`/home/ec2-user/watermarking-test1008/.venv-v030`, whose vLLM source checkout has no
local modifications. Override `SERVER_PYTHON`, `MODEL_PATH`, `CUDA_VISIBLE_DEVICES`
or `PORT` as needed. The shared environment has a CUDA 13.4 compiler with 13.2
runtime headers. The launcher selects supported Triton MoE/Mamba kernels,
FlashAttention, and native sampling to avoid incompatible FlashInfer JIT builds;
it does not modify that shared environment. BF16 precision is unchanged.
Each replica uses TP4, BF16 weights, native float32 Mamba state, and a 43,000-token
context. Shard 0 uses GPUs 0–3 / localhost port 8000; shard 1 uses GPUs 4–7 / port
8001. Questions are partitioned by input row index modulo 2 (45 and 44 California
questions). Logs are `logs/server-shard0.log` and `logs/server-shard1.log`.
The single-server launcher remains available as `scripts/serve.sh`.

After both localhost `/health` endpoints respond:

```bash
.venv/bin/python -m bird_inference.run --limit 2 --output outputs/smoke_tp4x2
.venv/bin/python -m bird_inference.run
.venv/bin/python -m bird_inference.evaluate
```

The default run uses eight concurrent questions **per shard**, at most eight
tool-enabled rounds followed by one final-answer turn, and 8,192 new tokens per
turn. The final turn disables tool choice; multiple tool calls in one round are
rejected without executing them. The prompt explicitly requests at most one
tool call per round and exactly one SQL query inside the final-answer tags.

Every turn is counted using the serving model's `/tokenize` endpoint, including
tool definitions and history. Input plus the full output allowance must fit
43,000 tokens. Otherwise the sample records `filtered_context_length`; neither input nor
output allowance is silently shortened. History reasoning trimming is explicitly
disabled, and `truncate_prompt_tokens` is explicitly unset. Filtered samples retain
their traces and token counts in per-question files and `filtered.json`; they are
excluded from `predict.json` and the EX denominator. Evaluation reports requested,
scored, and filtered counts separately. Other inference failures count as incorrect.
Each completed sample is saved atomically
with its full trace, tool results, finish reasons, and token usage. `--resume`
skips already saved samples and checks the run configuration. It does not silently
retry saved failed samples. Use a new output directory for a changed configuration.
New runs also fingerprint the runner, tools, and database implementation and record
chat-template settings, preventing resume across code changes. Older run manifests
without these fields cannot be resumed into a new-code run.

Outputs are `outputs/california_schools_temp0/{question_id}.json`, `predict.json`,
`summary.json`, `config.json` and `evaluation.json`. Execution accuracy compares
sets of full result rows, with positional columns, as in BIRD's execution metric;
it does not assess ordering or duplicate multiplicity. Gold execution failures
are reported separately. No evaluation feedback is given to the model.

All 1,534 prompts have also been prepared in `data/prepared/all.jsonl`. To rebuild, run
`.venv/bin/python -m bird_inference.prepare --db all`. This profiles each of the
11 databases once. Inference still defaults to California schools; running the
full set requires explicitly passing `--input data/prepared/all.jsonl` and a new
output directory. Larger schemas may need a larger context budget.

To reproduce the extended-statistics full run with both servers running, use a
fresh output directory:

```bash
.venv/bin/python -m bird_inference.prepare --db all --stats-profile extended
.venv/bin/python -m bird_inference.run --input data/prepared/all.stats-extended.jsonl --output outputs/bird_dev1534_extended_stats_rerun
.venv/bin/python -m bird_inference.evaluate --output outputs/bird_dev1534_extended_stats_rerun
```

Downloaded data, database-stat caches, rendered prompts, logs, and traces stay
local and are ignored by Git. Compact aggregate results under `results/` are
committed; they do not contain benchmark questions, gold SQL, or database rows.
