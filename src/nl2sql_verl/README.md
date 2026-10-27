# Gemma 4 NL2SQL tool-calling RL in verl

This ports the task behavior from `origin/consensus` commit
`588e1378777c64fa1a48d2548efccd51edd29270` to verl, with a repository-local
NeMo AutoModel Gemma ring-context-parallel engine. Existing parquet data and
SQLite databases are inputs. No question/schema/teacher-data generation runs.

The implementation is experimental. The BF16 TP2/DP4/CP1 actor has passed full
31B microbatch gradient checks, a 13,312-token optimizer update, and a complete
vLLM rollout → SQL tools/rewards → GRPO/DAPO update → full checkpoint → 640-row
evaluation cycle. See `configs/cp1.yaml` and the CP1 section below. **CP1 is a
validated control, not completion of the CP2 objective.** Full-model BF16 CP2
gradient correctness and training-memory scaling still require validation.

## What the consensus recipe does

The active launcher is `scripts/launch_train.sh`, calling
`src/nl2sql_gspo/train_gspo_nl2sql.py`. It uses
`DynamicSamplingGRPOTrainer`, a TRL GRPO subclass, normally with DeepSpeed
ZeRO-3 and BF16. There is also an optional experimental async/FSDP path.
`configs/train_gspo_gemma4.yaml` is empty in the inspected branch; the shell
launcher and Python trainer define the actual recipe.

The active code uses GRPO group-relative advantages, `scale_rewards=batch`,
and TRL's DAPO token-mean loss reduction. Despite its GSPO names/comments,
`GRPOConfig` never sets `importance_sampling_level="sequence"`; the pinned
[TRL 1.3.0 default is token-level IS](https://github.com/huggingface/trl/blob/v1.3.0/trl/trainer/grpo_config.py).
The custom trainer delegates its final policy loss to TRL. Sequence-level vLLM
sampling correction is a separate setting and does not make that policy loss GSPO.

Active launcher defaults include 16 generations per prompt, LR `1e-6`,
temperature 1.2, top-p 0.95, a 15,500-token prompt filter, 8,000 completion tokens,
clipping `0.2` below / `0.28` above, one policy update pass, beta=0, dynamic group
selection, and a soft overlong penalty with a 512-token buffer. The old branch
README's 16k/4k description is stale. TRL's server-mode vLLM rollout loop executes SQLite tools between
assistant turns. Data-generation scripts supply schema prompts and tool schemas;
their teacher traces are not required for online RL with prepared prompts.

The active trainer returns exactly these three callable tools:

| Tool | Purpose |
|---|---|
| `bm25_search_sqlite` | Find exact stored text values using BM25 and substring/regex matching |
| `sqlite_peek` | Inspect example values, types, nulls and column profiles |
| `sqlite_query` | Check a proposed read-only SQL query and return bounded rows |

The branch also defines `consensus_at_1`, but the active TRL training tool list
does not expose it. This recipe follows that list. Input tool definitions that
include another tool fail validation rather than silently changing the policy's
available actions. The original query tool's quote/name-repair behavior is
retained; terminal BIRD reward executes the extracted prediction with the
original branch scoring helpers.

| Reward | Definition retained from consensus | Weight |
|---|---|---:|
| format | Required scratch pad, relevant-table/column tags, final-answer SQL tags | 0.2 |
| execution | Predicted SQL executes successfully | 0.5 |
| result | BIRD raw-row set equality with gold SQL results | 2.0 |
| table linking | Exact equality of extracted table sets | 0.5 |
| column linking | Jaccard similarity of extracted column sets | 0.5 |
| nonnull | Successful execution with a nonempty, non-null cell | 0.1 |
| length penalty | Zero below the buffer; linear ramp to −1 at the generated-token limit | 0.1 |

The maximum positive weighted reward is 3.8. BIRD result matching ignores row
order and duplicate rows and retains raw SQLite cell values; it does not add
numeric tolerances. Format scoring also requires `<relevant_tables>` and
`<relevant_columns>`; the source README's shorter format description omits those
checks. Copied task code and source hashes are in `consensus/provenance.json`.

## Data-generation pipeline on consensus

Inspected branch: `origin/consensus`, commit
[`588e1378777c64fa1a48d2548efccd51edd29270`](https://github.com/kpratik41/nl2sql-gspo-gemma4/tree/588e1378777c64fa1a48d2548efccd51edd29270).
The main RL preparation pipeline constructs prompts deterministically from
existing questions and SQL answers:

```text
Existing BIRD questions + evidence + gold SQL
  → optional BM25 few-shot retrieval from the training split
  → SQLite schema introspection and prompt formatting
  → tool-aware system prompt + JSON tool definitions
  → training/evaluation JSONL
  → TRL normalizes prompts and generates tool trajectories during RL
```

1. **Source records.** `data/bird_train_data/raw/train-6601.jsonl` is already
   present on that branch, alongside the full `train.json`. The 6,601-row file
   is the filtered training input; these scripts do not implement the original
   dataset filtering. The default dev input is `dev_20251106.json`.
2. **Optional few-shot retrieval.**
   [`few_shot_bm25.py`](https://github.com/kpratik41/nl2sql-gspo-gemma4/blob/588e1378777c64fa1a48d2548efccd51edd29270/scripts/data_generation/few_shot_bm25.py)
   indexes training questions plus evidence and attaches the top five matching
   question/evidence/SQL examples. Training retrieval excludes examples from the
   target database. Dev retrieval draws only from training records, not dev gold.
3. **Schema prompts.**
   [`schema_build.py`](https://github.com/kpratik41/nl2sql-gspo-gemma4/blob/588e1378777c64fa1a48d2548efccd51edd29270/scripts/data_generation/schema_build.py)
   opens each SQLite database, collects table/column names, primary and foreign
   keys, inferred types, optional column descriptions, and optional statistics
   and example values. It combines those with the database ID, question,
   evidence/hint, and optional few-shots. Schemas are cached per database.
   `--no-fewshots`, `--no-comments`, `--no-stats` and `--no-nullability` control
   prompt variants. `--no-stats` still samples values for type inference.
   **Use `--n-examples -1` for a complete split; the script defaults to five.**
   Its intermediate JSONL includes a gold assistant answer.
4. **Tool prompts.**
   [`build_tool_dataset.py`](https://github.com/kpratik41/nl2sql-gspo-gemma4/blob/588e1378777c64fa1a48d2548efccd51edd29270/scripts/data_generation/build_tool_dataset.py)
   retains the schema/question user message, replaces the system prompt, adds
   tool definitions, and puts `gold_sql` in metadata. It removes the target gold
   assistant answer from `prompt` and `messages`. Its default exposes
   `bm25_search_sqlite`, `sqlite_peek`, and `sqlite_query`. The optional
   `--prompt-template consensus` additionally declares `consensus_at_1`, which
   is not in the active trainer's callable tool list or this verl recipe.
5. **RL.** The active launcher uses `outputs/train-6601-schema-tool.jsonl` and
   `outputs/dev-20251106-schema-tool.jsonl`; its README contains older defaults.
   The trainer generates assistant/tool turns online and scores final SQL
   against the metadata and the matching SQLite database. Pre-generated teacher
   trajectories are unnecessary for this path.

For example, a source question asking how many people meet a condition becomes
one user message containing the relevant database schema, the question and its
hint. During RL, the model can inspect a column with `sqlite_peek`, test a query
with `sqlite_query`, and write its final SQL. The reward code executes the final
SQL and the gold SQL on the same database and compares their result sets.

There is a separate **teacher/SFT/RFT pipeline** under `scripts/teacher/`.
`gen_teacher_traces.py` generates tool trajectories, optionally with privileged
SQL hints during teacher generation, and marks execution-correct traces without
hard leaks as eligible. `build_rft_from_traces.py` selects eligible traces and
assembles prompt-plus-transcript SFT rows; copy/tool-round filters are optional.
`build_run2_sft_dataset.py` mixes selected correct student pass@16 trajectories
with teacher traces and applies additional phrase filtering. These files train
on assistant trajectories; online RL instead starts from the prompt and samples
new trajectories. Do not carry privileged teacher hints into RL prompts.

For verl, the input boundary is already-built **parquet**, not raw BIRD JSON or
those JSONL files. If starting with consensus tool JSONL, serialize its rows to
`data/nl2sql/train.parquet` and `validation.parquet`, then run
`src.nl2sql_verl.prepare_data` (the launcher does this normalization). It strips
assistant answers, binds tools to `db_id`, and stores gold SQL in reward metadata.
It does not build missing schemas or remove arbitrary hints embedded in user
text. The upstream generation scripts remain on `consensus`; they were not
copied into this branch as part of the directory move.

## Which framework we used, and what changes here

Earlier four-GPU measurements ran **NeMo AutoModel directly**, with PyTorch TP
and Gemma's own ring CP attention. They were not NeMo RL or Megatron-LM training.
The `megatron_fsdp` import/fallback warning in environment logs does not mean we
trained a Megatron model. NeMo RL documentation was an integration reference.

Here, **verl** owns the RL trainer, group advantages, policy loss, agent loop and
rollout scheduling; **vLLM** generates trajectories; **NeMo AutoModel + PyTorch
FSDP2/TP** execute the policy. The default four-GPU topology is TP2 × CP2 × DP1.
Each CP partition has two TP GPUs. CP2 splits one example; it is not two
independent examples. vLLM uses TP2 replicas on the same four GPUs in alternating
rollout/training phases. Generation itself does not use the AutoModel CP ring.

The stock inspected verl AutoModel engine still uses retired mesh/loader APIs
and does not invoke the current model-owned sharder. `engine.py` supplies a
vendor-specific `EngineRegistry` registration through `model.external_lib`.
Installed libraries are unchanged. It handles:

- `DistributedSetup` and AutoModel's Gemma-owned TP/tied-weight plan, followed
  by FSDP2 sharding across DP×CP.
- The tested cuBLASLt policy disabling BF16 reduced-precision and split-K
  reductions. Weights used for computation and activations are BF16.
- SM120 Flex tiles: 32×32 forward, 16×16 backward, forced regular Flex path;
  compiled mask construction; actual causal/window/document masks retained.
- Model-owned auxiliary sharding: full input IDs enter Gemma once, next-token
  targets shift globally before CP partitioning, and padding/document IDs survive.
- Chunked TP-global vocabulary normalization and analytical gradients without
  a full-vocabulary gather. Only per-token logprobs/entropy are gathered by CP's
  differentiable token-layout API.
- Full-sequence token order for the default loss and optional GSPO ratios after gathering. CP gather's backward multiplicity
  is cancelled by FSDP's DP×CP gradient average. True DP, CP and output-source
  rank roles stay separate.
- TP replica-gradient synchronization, CPU master-weight updates, full
  model/optimizer/scheduler checkpoints, per-rank RNG restore, and one-tensor-at-a-time
  BF16 HF weight reconstruction for rollout synchronization.

For full-parameter updates on four cards, master weights and Adam state reside
on CPU in FP32 via FSDP2's CPUOffloadPolicy; compute uses BF16. This avoids losing
small updates directly into BF16-stored master weights. The host has roughly
1 TiB RAM. It is a memory-oriented starting recipe; CPU transfers/Adam can be
slow. This state placement differs from our no-optimizer BF16 capacity benchmark.

## Multi-turn token and reward handling

`agent_loop.py` extends verl's standard tool loop, with Gemma-native call parsing
and array-argument support for `columns`. It uses the loop's token-preserving
context merge, retains its generated-token mask, and stops on `<tool_call|>`.
No model-generated text is decoded and re-encoded by this adapter.

Tool observations remain in context but have response mask 0. Rewards receive
only assistant-generated text and a generated-token count through
`tool_extra_fields`; a missing field fails explicitly. An observation containing
a correct SQL answer cannot earn final-answer credit. Tool callbacks contribute
zero additional reward. Each invocation is bound to the example's `db_id`, with
bounded concurrency, SQLite time/VM limits and capped returned rows.

Verl's response-length budget includes inserted observations/template tokens.
The length penalty counts generated tokens only. Therefore the response budget
does not have exactly the same meaning as TRL's completion-token budget.

## Inputs

Default locations (the parquet files still need to be supplied):

```text
data/nl2sql/train.parquet
data/nl2sql/validation.parquet
databases/<db_id>/<db_id>.sqlite
```

Flat `databases/<db_id>.sqlite` and nested `train_databases` / `dev_databases`
layouts also work. Override with `TRAIN_PARQUET`, `VAL_PARQUET` and
`NL2SQL_DATABASE_DIR`.

The official [BIRD train and dev releases](https://bird-bench.github.io/)
were downloaded and extracted on this host on 2026-10-23:

```text
databases/train_databases/<db_id>/<db_id>.sqlite
databases/dev_databases/<db_id>/<db_id>.sqlite
data/bird_train_data/raw/    # official train questions, tables and gold SQL
data/bird_dev_data/raw/      # official dev questions, tables and gold SQL
cache/bird/                 # original ZIPs and nested database ZIPs
```

The train release contains **69 databases / 9,428 questions** and the dev release
contains **11 databases / 1,534 questions**. All 80 database files were extracted
with ZIP CRC validation and opened read-only; every bundled question has a
resolvable database. Total SQLite file size is approximately 33.90 GB.

These folders remain gitignored. The official dev ZIP contains
`dev_20240627/dev.json`; the consensus default is the corrected
`dev_20251106.json`. Downloaded databases do not select the question/gold version
for evaluation. Use the intended annotation file explicitly when preparing
validation parquet, and record that choice when comparing with consensus runs.
Download provenance, archive hashes, and the per-database size/schema inventory
are recorded in [`benchmark_results/bird_data_download/manifest.json`](../../benchmark_results/bird_data_download/manifest.json).
The download does not include the branch's 6,601-row filtered training file or
its additional column-meaning annotations; those are available on `consensus`.

Each source parquet row needs a chat-message list in `prompt` or `messages`,
plus `db_id` and `gold_sql`. `SQL`/`sql` aliases and an existing verl
`reward_model.ground_truth` JSON object are accepted. Example:

```python
{
    "prompt": [
        {"role": "system", "content": "Your schema/tool NL2SQL instructions..."},
        {"role": "user", "content": "<db_id>my_db</db_id>...schema and question..."},
    ],
    "db_id": "my_db",
    "gold_sql": "SELECT COUNT(*) FROM people;",
}
```

The adapter removes assistant answers from prompts, stores gold SQL only in
reward metadata, adds per-example tool bindings and `agent_name`, and writes
normalized parquet under `outputs/verl_data/`. It preserves supplied system/user
content; if no system prompt exists it inserts the branch's tool-aware template.
The existing system prompt should request the required response tags. Prompt
length filtering uses the actual tokenizer/chat template **with tool schemas**;
overlength rows are dropped, never silently truncated. No schema generation runs.

## Environment and launch

`setup_env.sh` builds a separate Python 3.12 `.venv-verl`; it does not modify
`.venv-bench`. It downloads packages/source, not training data or model weights.
Use a local cached checkpoint path to avoid a model download at launch.

```bash
VERL_BASE_PYTHON=/path/to/python3.12 bash src/nl2sql_verl/setup_env.sh

# Configuration validation without data or GPU allocation:
CONFIG_ONLY=1 bash src/nl2sql_verl/launch.sh

# After the parquet files and databases exist: one short RL step.
MODEL_PATH=/path/to/cached/gemma-4-31B-it bash src/nl2sql_verl/launch.sh

# The longer recipe preserves n=16 but still starts at a modest context.
RECIPE_CONFIG=nl2sql MODEL_PATH=/path/to/model bash src/nl2sql_verl/launch.sh

# Explicit GSPO sequence-ratio variant, with the same task/reward scaling:
RECIPE_CONFIG=gspo MODEL_PATH=/path/to/model bash src/nl2sql_verl/launch.sh
```

The source pins are AutoModel `65866fbce209938cd153f2bc26f01fa0d34cc61e` and
verl `d430157d15516d89e0cc853fd10fcd21d9234c44`; the rollout runtime proposal
uses vLLM 0.19.1+cu130 and PyTorch 2.10/cu130. The CUDA-specific vLLM wheel
is required: the plain PyPI wheel links CUDA 12 and fails to import in this
environment. Setup now checks the native extensions as well as package metadata.
The launcher prioritizes the environment's CUDA libraries to avoid mixing the
B200 host's CUDA 13.2 cuBLASLt with the wheel's CUDA 13.0 cuBLAS.
The recipe entrypoint `train.py` resolves Ray runtime environment interpolations
before delegating to verl; the pinned upstream launcher otherwise passes literal
`${oc.env:...}` strings to reward/tool workers.
The inspected verl package metadata caps
Transformers below 5.13, whereas the tested AutoModel revision requires 5.18.
The script intentionally loads the pinned verl source through a `.pth` file,
using the local engine adapter, instead of installing a wheel that downgrades
Transformers. **This is not an upstream-supported combined dependency matrix.**
The full Python 3.12 runtime has now been installed and checked on B200.
`vllm_compat.py` registers a recipe-local serving adapter: Transformers 5.18's
per-layer Gemma4 config is checked for representability before restoring the
legacy local/global head fields used by vLLM 0.19.1. This preserves the 31B
model's 256/512 head dimensions and 16/4 KV-head counts; arbitrary per-layer
heterogeneity is rejected. The training model config is not flattened.
`pip check` checks installed package metadata but cannot certify this source
compatibility exception or rollout correctness.

The smoke profile uses TP2/CP2, two prompts, two generations each, 1,024 prompt
tokens and a 1,024-token multi-turn response budget, with one training step.
The main profile uses four prompts, 16 generations each, 2,048 prompt tokens,
4,096 response-budget tokens and up to 100 steps. Both use one example per
policy microbatch, activation checkpointing and beta=0; no reference model or
critic is allocated. Do not enable KL without adding the corresponding CP
reference configuration.

The default `smoke` and `nl2sql` profiles match the active TRL objective:
token-level clipped ratios, token-mean reduction, prompt-group mean subtraction,
and a batch-wide reward standard deviation plus `1e-4`. `advantages.py` registers
this small task-specific estimator. The optional `gspo` profile changes only
the policy loss to explicit sequence-level ratios. This avoids silently changing
the old algorithm merely because the branch is named GSPO.

To extend context later, set `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH`; the
launcher keeps trainer and rollout length budgets aligned. Our measured
199,680-token **forward-only** limit is not the training/generation limit.
Start with the smoke profile. Full 31B CP1/CP2 probability sensitivity from the
benchmark investigation remains unresolved; the recipe carries the mitigation
but does not mark that numerical gate as passed.
The prepared B200 pilot's real BIRD tool prompts are 4,525–5,124 tokens, so
neither default prompt budget fits them. Use `MAX_PROMPT_LENGTH=8192` for that
pilot. `prepare_data.py` explicitly requests token IDs from Transformers;
counting the default `BatchEncoding` would incorrectly report two tokens.

Resume with `trainer.resume_mode=resume_path
trainer.resume_from_path=/path/to/global_step_N` as launcher overrides.
Checkpoints contain training state; model-only continuation is separate.
Checkpoint retention is not enabled by default. Keep the original HF base path
for the model/tokenizer while loading sharded training state through resume.

## Deliberate differences from TRL

| Consensus behavior | This verl recipe |
|---|---|
| Actual TRL defaults: token IS + DAPO loss reduction | Default `vanilla` token ratios + token-mean aggregation; optional `gspo` profile |
| Group-centered rewards scaled by batch-wide std | Same formula and `1e-4` denominator epsilon, registered in verl |
| Separate vLLM server on spare GPUs | Colocated verl vLLM replicas, alternating phases |
| ZeRO-3, usually six trainer + two serving GPUs | TP2/CP2/FSDP2 on four GPUs, CPU master/optimizer |
| Oversample-and-replace, K=12 in the generic launcher; filter by result_reward; zero-mask leftover groups | Native V1 result-reward filtering; the pinned V1 replay buffer ignores `max_num_gen_batches`, so it does not provide a total retry bound |
| Optional beta schedule | beta=0 only in this initial recipe |
| Optional truncation masking, default off | Existing verl response mask; no extra truncation penalty/masking hidden in the adapter |
| Model-only rotating checkpoints plus latest full checkpoint | Full sharded training checkpoints with RNG; no automatic model-only export |

Temperature 1.2 and top-p 0.95 follow the active consensus launcher. The
`gspo` profile is a deliberate optional policy-loss change. Old-policy logprobs are
recomputed through the training engine; rollout logprobs are retained and bypass
mode is disabled. Monitor serving/training probability mismatch before long
runs. The native group filter is a comparable control, not a byte-for-byte
reimplementation of TRL's tail queue, minimum-std threshold or padding behavior.
TRL also enables its own vLLM sampling correction by default; this initial
recipe retains mismatch diagnostics but leaves verl's IS/rejection correction
disabled rather than assuming the similarly named controls have identical math.

The latest production launcher on `origin/consensus` (`588e137`),
`scripts/teacher/run_rl_sft_ga16_ckpt25.sh`, adds requirements beyond that
generic configuration:

| Production consensus setting | Current verl gap |
|---|---|
| SFT ga16 checkpoint-25 warm start | Uses the original instruction model; the SFT checkpoint has not been restored |
| Type-corrected 6,574-example corpus, 27 defective gold queries removed | Diagnostic subset comes from restored train-6601; full typefix corpus not recreated |
| `old-dev-schema-bare-tool` evaluation | Pilot uses eight questions from the corrected dev annotation version; scores are not comparable |
| 24 selected prompt groups/update, 16 samples/group, K=14 candidates | Base uses four groups; native filtering has different queue/selection behavior; smoke filtering is off |
| Learning rate `3e-6`, beta `0:0.005,20:0.001,35:0` | Base lr `1e-6`, no reference model or beta schedule |
| Batch reward std computed over all candidates before selection | Current estimator sees selected samples after filtering |
| Prompt cap 15,500; completion cap 8,000; eight tool rounds | Base caps 2,048/4,096; pilot overrides 8,192/2,048 and five assistant turns |
| Save every five steps, rolling full checkpoint on NVMe | Full-runtime save/resume and retention have not been validated on the 31B pilot |

The existing seven terminal rewards already cover format, SQL execution,
result matching, table linking, column linking, nonnull results and length
penalty, weighted `0.2,0.5,2,0.5,0.5,0.1,0.1`. Tool observations are masked
from the policy loss and excluded from reward text. The pilot checks real
SQLite rewards; adding a second reward implementation is unnecessary.

## Validation and boundaries

Checks performed in `.venv-verl-check`, a Python 3.13 source-check environment
that reads the established benchmark stack and adds only test/runtime Python
packages in its own directory:

```bash
.venv-verl-check/bin/python -m pytest -q src/nl2sql_verl/tests
.venv-verl-check/bin/python -m src.nl2sql_verl.preflight --config smoke --config-only
.venv-verl-check/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m src.nl2sql_verl.tests.distributed_logprobs
.venv-verl-check/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m src.nl2sql_verl.tests.tiny_engine
CUDA_VISIBLE_DEVICES=0,1 .venv-verl-check/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m src.nl2sql_verl.tests.tiny_engine --cp 1
```

The tiny-engine test creates a random tiny HF checkpoint at
`outputs/verl_recipe_checks/tiny_gemma` from its tracked configuration if absent;
this is not the 31B model and requires no model download. Logs and
machine-readable test results are under `outputs/verl_recipe_checks/`.
The original CPU suite had 13 passing tests; `validation.json` in that directory
records package/source versions and the distributed test results.
Compact validation reports and the unit-test summary are also versioned under
`benchmark_results/verl_recipe/`; local checkpoints and verbose logs stay in `outputs/`.
Distributed GSPO gradient error against a full-vocabulary/full-sequence
reference was **2.9802322387695312e-8**, with identical scalar losses. Tiny model
tests establish execution/save/restore/export behavior, not CP numerical gradient
parity, long-context training capacity, rollout weight-sync completion or NL2SQL
accuracy. A real 31B cycle now reaches generation, reward scoring, old logprobs,
backward/optimizer execution and rollout weight synchronization. The first
cycle was **not a learning success**: all eight outputs were corrupted and
truncated, producing identical rewards and zero gradients. A standalone
reproducer isolated corruption to vLLM 0.19.1 Gemma4 level-2 sleep/refit.
All 1,189 actor-exported tensors matched the original HF weights exactly;
direct refit without sleep worked. Level-1 sleep/refit reproduced the original
answer and sampled-token logprobs exactly, so the recipe-local serving adapter
now requests level 1. This retains approximately 59 GiB of CPU backing per
TP2 rollout replica and adds sleep-copy time. The lost internal state is the fixed `v_norm.weight` tensor in all 60
attention layers: vLLM RMSNorm with `has_weight=False` stores it as an ordinary
tensor, outside both parameters and registered buffers. Registering these
constants as nonpersistent buffers also passes the isolated level-2 refit
reproducer, but the full pilot uses the validated level-1 workaround.

B200 diagnostics are stored separately under `outputs/b200_mini/`. The current
CPU suite has 22 passing tests. Eight-GPU NCCL, synthetic distributed loss,
tiny TP2/CP2 train/update/checkpoint restore/export, and Flex ring attention
output/gradient checks pass. Full 31B TP2 CP1/CP2 probability screening still
fails at 128 tokens (maximum target-logprob error 0.356054, threshold 0.1),
while the 1,024-token case passes (0.069377). Finite standalone FFPA gradients
do not establish correctness: FFPA ring dQ/dK checks fail at 128 tokens, and
dK fails at 1,024, against the existing FP32 reference tolerances. Reproduce
without changing the training backend:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0,1 .venv-verl/bin/torchrun \
  --standalone --nproc-per-node=2 benchmarks/cp_attention_gradients.py \
  --backend ffpa --full-length 128 --output outputs/ffpa_ring_check
```

The pilot also exposed a prompt/parser mismatch: consensus asks for bare
`call:...` lines, while the upstream Gemma parser required `<|tool_call>`.
The recipe parser now accepts either opener, requires a complete native closing
token and a line boundary for bare calls, and preserves the sampled token IDs.
Regression tests cover the observed SQL/quoted-brace form and reject incomplete
calls and inline mentions.

The actual recipe explicitly selects **Flex** for both full and sliding ring
attention. FFPA remains an optional backend requiring further numerical work.

Follow-up on B200 identified a specific FFPA 0.2.4 SM100 ring-backward defect:
the D512 dQ/dK kernels explicitly zero the score gradient for a causal row
with one **local** key. That shortcut assumes local softmax normalization.
The first query on CP rank 1 also sees rank 0's keys, so it is not globally
a one-key row. NeMo supplies globally merged output/LSE to each chunk's
backward; the shortcut incorrectly removes a real boundary contribution.
The installed `_dq_d512_sm100.py` around line 2381 and
`_dk_d512_sm100.py` around line 2696 contain the assumption; upstream main
also contained it when inspected on 2026-10-27.

`outputs/ffpa_backward_investigation/kernel_probe.py` separates standalone
attention from chunk backward against FP32 math. Native standalone backward
passes at 128/256/512/1024/2048 tokens for BF16 and FP16. Supplying oracle
output/LSE does not repair chunk backward. Restoring the omitted boundary
term passes all those cases. The diagnostic-only `ring_boundary_probe.py`
also passes the original two-GPU CP2 checks at 128 and 1024 tokens on both
ranks. At 128 tokens, rank 1 maximum dQ/dK errors fall from 0.2600/0.3771
to 0.0070/0.0092 without relaxing tolerances. This correction covers only
the tested dense, unpacked geometry; it is not installed in the recipe and
does not validate packed/padded FFPA or full-model gradients. A durable
kernel/API fix must distinguish global normalization from local one-key
softmax, preserve zero-key masking, and cover varlen/LSE-gradient cases.

FFPA does support B200 forward and backward; this is a specific integration
contract failure. The full-model discrepancy below was measured with Flex,
so it is a separate issue. Sources:
[FFPA CuTeDSL backend](https://ffpa-attn.readthedocs.io/en/latest/user_guide/backends/cutedsl/),
[NeMo Gemma FFPA CP integration](https://github.com/NVIDIA-NeMo/Automodel/discussions/2928).

Full 31B backward probes now compare TP2/CP1 and TP2/CP2 on the same SQL
fixture, before clipping or optimizer updates. All selected gradients are
finite, but 13/16 sampled parameter tensors fail the declared 3% relative-L2 /
0.999 cosine gate at 128 tokens (worst relative L2 22.9%). At 1,024 tokens,
15/16 fail; early/middle-layer relative errors reach 287%, while the loss
difference is only 4.9e-5. The latter fixture repeats text and has very low
loss (~8e-4), making relative gradient errors particularly sensitive. These
are selected-gradient diagnostics, not an all-parameter comparison or a proven
root-cause diagnosis. Forward closeness and finite gradients alone are not
sufficient acceptance criteria. See `outputs/b200_mini/gradient_comparison*.json`.

A subsequent 128-token full-model control promoted only Flex attention to
FP32 while retaining BF16 model compute. CP1/CP2 losses became
1.241046667/1.241170526 (absolute difference 0.000123858 versus the earlier
0.006451964), but 0/16 sampled parameter gradients passed the unchanged
relative-L2/cosine gates; worst relative L2 was 33.16%. This does not establish
which full-model path is correct, and promoting attention alone does not
resolve gradient parity. Results and diagnostic wrapper are under
`outputs/ffpa_backward_investigation/fp32_attention_*`.

The next control promoted **all actor compute** to FP32. TP2/CP1 versus
TP2/CP2 passed all 16 selected-gradient gates at 128 tokens (worst relative
L2 0.00876%; loss difference 1.31e-6). The requested TP4/CP2 layout with
GPU-resident FP32 master weights/optimizer also passed against the TP2/CP1
FP32 control: worst sampled relative L2 0.0171% at 128 tokens and 0.8736%
at 1,025 tokens, including padding and the sliding-window boundary. This
supports the FP32 path; it does not certify BF16 training or every parameter.
Raw results are in `outputs/b200_two_hour/fp32*gradient_comparison*.json`. The FP32
engine selects standard cuBLAS; forcing the experimental cuBLASLt backend
failed at the first RoPE product in a colocated training worker. Inspection
found mixed wheel CUDA 13.0 cuBLAS and system CUDA 13.2 cuBLASLt: the host
`.bashrc` prepends system CUDA during Ray worker startup. `worker_python.sh`
restores wheel library precedence after that shell and before Python starts;
a real Ray actor test verified matching loaded libraries and threaded matmul.
Full-model gradient probes also cover the standard cuBLAS path with the launcher CUDA
library environment.

The opt-in `b200` recipe uses all eight GPUs as TP4/CP2 (logical DP1), full
FP32 actor compute and GPU-resident FP32 state, Flex full/sliding attention,
and BF16 vLLM TP4 rollout replicas. `model_dtype` now controls compute as
well as the FSDP precision policy; the older recipes retain BF16 compute and
CPU master state. A real eight-rank tiny-model test passed optimizer update,
exact model/optimizer restore, and weight export with the GPU-state policy.

For this bounded run, `bounded_runtime` starts a 7,200-second training clock
after initial validation and stops after completing the active update,
checkpoint, weight sync, validation and metric logging. The external job
watchdog allows three hours including initialization/cleanup. A full sharded
checkpoint is written every step to `/opt/dlami/nvme/nl2sql/`; it includes model,
optimizer/scheduler, per-rank RNG and trainer data-loader state. Only after the
whole save succeeds does the controller publish a completion marker and prune
older marked checkpoints, retaining the newest two. The GPU-resident state is
not itself a durable checkpoint. Local NVMe survives process restarts but is
ephemeral instance storage. Resume repairs the latest pointer to the newest
completed checkpoint; unmarked directories are never pruned automatically.

The timed run uses 6,218 restored BIRD training examples after the 8,192-token
prompt filter (383 overlength rows removed), 2,048 response tokens, four prompts
and four samples per update, lr 1e-6, no KL, and disabled group filtering to
keep runtime bounded. All 6,601 original training gold queries passed a SQLite
compile-only check; this is not the consensus branch's unavailable 6,574-row
typefix corpus or a result-correctness audit. Sixteen seeded eligible dev rows
from disjoint databases are evaluated before training and after every step.
The file logger records each reward component, total reward, execution-result
accuracy, gradient norm, actor/rollout logprob discrepancy and timings. Run
artifacts, configuration, data provenance, live plots and final completion
evidence are under `outputs/b200_two_hour/`.

The requested relaunch expands evaluation to 32 fixed eligible dev examples
across 10 databases, containing the original 16. It resumes completed checkpoint
1 and evaluates the expanded set before the next update and after every step.
Its metrics live separately in `outputs/b200_two_hour_eval32/` so the 16- and
32-example evaluation curves are not combined. The training clock preserves
completed runtime across checkpoint resumes and excludes restart downtime.

Measured B200 Flex global-attention forward/backward memory is approximately
linear over the tested range, without replacing dense causal attention.
For real CP2 with TP2's per-rank head geometry (16 query heads, 2 KV heads,
D512, BF16), warm peak allocated GPU memory including ring buffers, gradients,
inputs, output and compiled block masks was:

| Total sequence tokens | Tokens per CP rank | Peak MiB per GPU |
|---:|---:|---:|
| 4,096 | 2,048 | 465.18 |
| 8,192 | 4,096 | 930.60 |
| 16,384 | 8,192 | 1,862.20 |
| 32,768 | 16,384 | 3,728.41 |

Both ranks produced finite gradients. These are attention-only memory
measurements, not full-model memory or numerical parity at these lengths.
Compilation is outside the warm measurement; cold peaks are saved separately.
Dense global compute remains quadratic. Compact block-mask metadata also has
a quadratic component, so the measurements support practical near-linear
memory over this range, not a strict asymptotic claim for the entire wrapper.
The CP2 Flex backward currently collects KV chunks and builds their autograd
graphs; its attention working-memory constant is larger than the single-GPU
kernel's. CP does not guarantee a twofold reduction in that specific workspace.
See `outputs/ffpa_backward_investigation/flex_cp_memory_rank*.json` and the
matching probe script. The separate single-GPU D512 measurement was
144.32/288.75/578.00/1,158.01 MiB at 2k/4k/8k/16k tokens.

The repaired B200 pilot completed two actual optimizer updates and final dev
evaluation in **1,666.5 seconds (27.8 minutes)**, including initialization and
cleanup. It used the original 31B instruction model, TP2/CP2 on four GPUs,
Flex attention, lr `1e-6`, two prompt groups per step and four samples per group.
Four distinct train prompts produced 16 training trajectories. Filtering was
disabled; prompt/response caps were 8,192/2,048; no full checkpoint was saved.

| Metric | Step 1 | Step 2 |
|---|---:|---:|
| Gradient norm before clipping | 6.8970 | 36.4747 |
| Mean reward | 3.3250 | 2.3304 |
| Mean actor/rollout absolute token-logprob difference | 0.003709 | 0.003457 |
| Maximum actor/rollout absolute token-logprob difference | 0.394307 | 0.221462 |
| Training-step wall time, seconds | 599.7 | 628.5 |

On the **same eight held-out diagnostic dev prompts**, greedy result matching
went from 2/8 before training to 7/8 afterward; valid final SQL went from 3/8
to 8/8. This small, selected subset and two updates do not establish a reliable
accuracy improvement. The observed change includes much better completion of
tool/final-answer formatting. Numerical CP parity remains unresolved. The
controller exited with code 0; the watchdog subsequently terminated remaining
Ray/vLLM descendants (late EngineCore shutdown messages belong to that cleanup).
Metrics, paired dev outputs, exact commands and failure diagnostics are in
`outputs/b200_mini/assessment.json`, `pilot31b_repaired_metrics.json`,
`paired_dev.json`, and `pilot31b_repaired.status.json`.

## Next: exact attention, near-linear memory, and CP2

The target remains exact Gemma global/sliding attention with near-linear
attention working-memory growth and usable CP2. Dense attention compute remains
quadratic. FSDP shards model state; CP shards sequence/attention work; neither
alone prevents materializing a quadratic attention-score matrix.

1. Establish the simplest supported full-model BF16 control: native verl FSDP2,
   TP1/CP1, microbatch one, short real BIRD sequences and a plain token loss.
   Compare against the independent HF SDPA-math reference before adding RL.
   Native Gemma support must be verified; do not assume a backend/config switch
   preserves the heterogeneous D256/D512 heads or sliding masks.
   Native verl FSDP sequence parallelism uses Ulysses; this is distinct from
   the current adapter's ring CP. Verify the pinned implementation's model and
   attention support before choosing the sequence-sharding path. See the
   [upstream FSDP extension checklist](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html).
2. Use exact memory-efficient attention in that CP1 control, then change only
   CP to two. Proposed eight-GPU topology: TP1/CP2/DP4, with FSDP sharding model
   state across DP×CP. Compare identical global examples, loss normalization,
   log-probabilities and sampled gradients against CP1. Cover causal boundaries,
   sliding windows, padding and unequal sequence lengths. The current adapter
   still restricts CP>1 to one sequence per microbatch; TP1 support is not yet
   validated. Use the custom Gemma CP adapter only if the native path lacks the
   required model/attention support, introducing it as a separate tested change.
3. Measure warm forward **and backward** peak allocated/reserved memory at
   4k/8k/16k/32k global tokens, fixed batch size and precision, both CP1 and CP2.
   Separate model/optimizer baseline memory from attention working memory and
   inspect the executed backend to exclude a dense-score fallback. Earlier
   attention-only and forward-only capacity results do not certify full training.
4. Add checkpointing, accumulation and optimizer state one at a time, followed
   by rewards/tools, GRPO, native DAPO filtering, rollout weight sync and a full
   checkpoint/evaluation cycle. Rollout TP remains independent of actor TP/CP.

Keep FFPA disabled until its CP backward contract passes. The CP1 recipe remains
the known working control; it is not evidence that BF16 CP2 is fixed. The runtime
uses a pinned local Transformers/vLLM bridge rather than an upstream-supported
version combination. Full optimizer resume from a 31B checkpoint, durable export
off instance-local NVMe, and serving/training probability mismatch remain checks
to complete beyond the existing small-model optimizer round trip and full-model
model-only checkpoint loads.

The implementation was introduced in commit `e95b4cf`; subsequent history moves
it from `recipes/` to `src/` and updates import/config/launch references. Git
history is the rollback source. Local reverse patches under
`outputs/verl_recipe_baseline/` predate the move and refer to the old paths.

Primary upstream references: [verl tool/agent extension interfaces](https://verl.readthedocs.io/en/latest/extend_guide.html),
[pinned verl requirements](https://github.com/verl-project/verl/blob/d430157d15516d89e0cc853fd10fcd21d9234c44/requirements.txt),
[Gemma AutoModel TP plan](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/models/gemma4_moe/parallelization.py),
and [vLLM 0.19.1 model registry](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/registry.py).


## CP1 production recipe

`configs/cp1.yaml` implements the agreed 6,601-train / 640-dev recipe. The
640 fixed dev rows include the previous 32 and cover all eleven dev databases.
The 9,428-row raw training dataset is not used. All 6,601 filtered-corpus rows,
including one duplicate prompt, are retained. Measured prompts range from 4,402
to 10,786 tokens including the actual tool schema and chat template. The 11,264
prompt allowance plus the unchanged 2,048-token response budget gives 13,312
context tokens. Preparation fails rather than dropping a row above that allowance.

The actor topology is TP2 / DP4 / CP1; serving remains TP4 / DP2. Thirty-two accepted prompt groups with sixteen
trajectories each produce 512 training trajectories. The PPO mini-batch is 32
prompt-equivalents, with one PPO epoch and one optimizer update per rollout
batch. Each actor TP group processes two independent trajectories at a time and 128
per update: 64 accumulation microbatches. Inputs are right-padded into 256-token
buckets; padding is excluded from attention keys and the loss. Each row keeps
its own labels, temperature, position IDs and jagged output offsets. Native
FSDP optimizations defer gradient synchronization until the final microbatch.

GRPO retains consensus's group centering and batch-wide reward standard
deviation. Native V1 DAPO filtering uses `result`, individual prompt refills,
and at most two train batches of prompts in flight. That concurrency setting
is not a total retry budget. Evaluation is greedy with n=1. Full model,
optimizer/scheduler, per-rank RNG and dataloader checkpoints are saved every
five updates; the latest two completed saves are retained. The 640-row dev
set is evaluated before training and every ten updates. Default run length
is 100 optimizer updates, not the earlier two-hour pilot timer.

The actor uses BF16 compute with FP32 master weights, Adam state and FSDP gradient
reductions. Serving also uses BF16. TP4 actor compute failed BF16 microbatch
gradient checks; removing DP alone did not fix it. TP2/DP4 passed the full-model
BIRD comparison (worst sampled relative L2 0.267%, cosine > 0.999996) and a real
Adam update with four sequences per DP replica up to 13,312 tokens. That test
peaked at 117.3 GiB allocated and 164.7 GiB reserved per GPU. TP8 is unsupported
by the pinned adapter because global-attention layers have only four KV heads.
The full-model HF BF16/SDPA-math single-GPU control also passed (0.288%). These
are sampled-gradient checks, not a proof for every parameter or input. End-to-end
readiness is still governed by `outputs/cp1_recipe/readiness.json`.

Prepare/verify data and inspect the validation report:

```bash
.venv-verl/bin/python -m src.nl2sql_verl.prepare_cp1_data
python3 -m src.nl2sql_verl.manage_cp1 check
```

Start a new run from the base model after the readiness checks pass:

```bash
python3 -m src.nl2sql_verl.manage_cp1 start
```

To initialize from the pilot's last completed model instead, use
`--initial-model /opt/dlami/nvme/nl2sql/b200-tp4-cp2-20261027T015717Z/global_step_6/actor/model`.
This is model-only initialization with fresh optimizer and dataloader state:
the new corpus and recipe differ from the pilot. Subsequent starts recover
complete CP1 checkpoints from `/opt/dlami/nvme/nl2sql/cp1-grpo` automatically.

The persistent user service is `nl2sql-grpo-cp1`. Logs, metrics, heartbeat,
progress JSON, reward plot and final process status live in
`outputs/cp1_training/`. Inspect with
`python3 -m src.nl2sql_verl.manage_cp1 status`. Do not run a second GPU job
concurrently; the manager checks for existing GPU processes before starting.

Validation commands (the GPU commands each require all eight GPUs):

```bash
mkdir -p outputs/cp1_recipe
# Generate the shared tiny fixture and check FP32 save/restore first:
OMP_NUM_THREADS=2 src/nl2sql_verl/worker_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 -m src.nl2sql_verl.tests.tiny_engine --tp 4 --cp 2 --dtype fp32 --gpu-state
.venv-verl/bin/python -m pytest src/nl2sql_verl/tests -q > outputs/cp1_recipe/cpu_tests.log 2>&1
CUDA_VISIBLE_DEVICES='' .venv-verl/bin/python -m pytest outputs/verl_runtime/verl/tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py -q -k 'dapo or filter' > outputs/cp1_recipe/native_dapo_tests.log 2>&1
OMP_NUM_THREADS=2 src/nl2sql_verl/worker_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 -m src.nl2sql_verl.tests.cp1_microbatch --dtype bf16 --tp 2
OMP_NUM_THREADS=2 src/nl2sql_verl/worker_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 -m src.nl2sql_verl.tests.full_cp1_probe --dtype bf16 --tp 2 --parity-only --output-tag _tp2dp4_synthetic
OMP_NUM_THREADS=2 src/nl2sql_verl/worker_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 -m src.nl2sql_verl.tests.full_cp1_probe --dtype bf16 --tp 2 --bird --output-tag _bird_tp2dp4
python3 -m src.nl2sql_verl.smoke_cp1
# After nl2sql-cp1-smoke.service has finished successfully:
python3 -m src.nl2sql_verl.validate_cp1
```

The smoke run uses the production dataset, 16 generations, two-trajectory
microbatches, native result filtering, all 640 dev examples, and full checkpoint
saving. Only its accepted training batch is reduced from 32 prompt groups to two.
It is a validation job, not the 100-update production run.

The completed base-model smoke run processed 32 accepted trajectories, recorded
mean training reward 3.11745 and finite gradient norm 3.00863, saved a 346-GiB full
training checkpoint, and scored 422/640 (65.9375%) on the fixed dev set. It exited
successfully in 847.5 seconds. One update does not establish a learning trend.
The small-model BF16 checkpoint round trip restored both model and Adam state
exactly. CPU recipe tests: 31 passed; native verl DAPO tests: 12 passed.
The full-model BIRD and synthetic microbatch checks had worst sampled relative
gradient differences of 0.267% and 0.381%, respectively, below the existing 3%
gate. A compact record is tracked in
`benchmark_results/verl_cp1/validation_summary.json`; detailed local logs and
checkpoints are not committed. The main 100-update run has not been started.
