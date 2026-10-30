# Gemma 4 NL2SQL tool-calling RL in verl

The active **16-H200 screened DAPO run** uses native verl FSDP2, FlexAttention,
Ulysses CP2 / DP8, IEEE FP32 actor updates and four BF16 async vLLM TP4 rollout
replicas. It trains on 1,039 screened prompts with 16 generations per prompt and
an 8K cumulative response budget. The current path does not use NeMo AutoModel,
Megatron, DeepSpeed or Ring Attention. Native tools execute during rollouts and
their actual executions are recorded.

See the [run results through update 7](reports/screened1039-8k-2026-10-30.md)
for the timestamped reward progression, evaluation results, filtering costs,
numerical observations and checkpoint details. At that snapshot, step-5
validation accuracy was **73.28%**, versus **72.50%** before training, on the
same 640 questions at temperature 0.

**W&B:** [screened1039/8K training dashboard](https://forge.coreweave.com/wandb/kpratik41/nl2sql-gemma4-verl/runs/75et1rl9).

The earlier incorrect run, using old data and 2K limits, was stopped at nine
completed updates on 2026-10-29 at the user's request. Its complete step-5
model-only export is preserved separately on node 0. Cleanup manifests are in
`outputs/h200_checks/cleanup_20261029/`.

The active run uses the versioned `gemma4_native_v1` profile with native Gemma
call delimiters, resolved implicit foreign keys and actual tool-execution
telemetry. Its run-specific data changes the system budget instruction to 8K;
the separate 4K config remains available. See
[versioned data preparation](DATA_PREPARATION.md) for provenance, rebuilding,
and the fixed-sample validation workflow.

The environment instructions below describe that native FSDP2 environment.
The later consensus/AutoModel sections retain historical adapter notes and are
not the setup instructions for the H200 run. Task behavior originated from
`origin/consensus` commit `588e1378777c64fa1a48d2548efccd51edd29270`.

## Exact environment of the retired H200 run

Captured on 2026-10-29 from both selected nodes:

- [`requirements.txt`](requirements.txt): all **240 installed Python package
  versions**, including transitive dependencies and build tools. The repository
  root [`requirements.txt`](../../requirements.txt) includes this file.
- [`environment-h200.json`](environment-h200.json): OS, driver, compiler,
  CUDA-library, interconnect and package-version snapshot, plus wheel provenance
  and the pinned source revision. It contains no authentication credentials.
- [`setup_fsdp_env.sh`](setup_fsdp_env.sh): installs the exact native environment
  into a separate virtualenv, installs the pinned verl source path, and verifies
  package versions and CUDA-extension imports.

| Component | Recorded version |
|---|---|
| Hardware / architecture | NVIDIA H200, SM90; Linux x86_64 |
| OS / libc | Ubuntu 24.04.5 LTS / glibc 2.39 |
| Python / GCC | CPython 3.12.3 / GCC 13.3.0 |
| SQLite / PyTorch C++ ABI | SQLite 3.45.1 / CXX11 ABI enabled |
| NVIDIA driver | 595.112.07 |
| System CUDA toolkit (`nvcc`) | CUDA 13.2, compiler 13.2.51 |
| PyTorch / CUDA build | 2.10.0+cu130 / CUDA 13.0 |
| torchvision / torchaudio | 0.25.0+cu130 / 2.10.0+cu130 |
| torchao / Triton | 0.16.0+cu130 / 3.6.0 |
| CUDA runtime / NVRTC Python wheels | 13.0.96 / 13.0.88 |
| cuBLAS / cuDNN / NCCL wheels | 13.1.0.3 / 9.15.1.9 / 2.28.9 |
| vLLM | **0.19.1+cu130**, exact wheel URL and SHA256 in the lock file |
| Transformers | 5.18.0 |
| FlashInfer Python / cubin | 0.6.6 / 0.6.6 |
| CUTLASS DSL / quack-kernels | 4.6.2 / 0.6.4 |
| Ray / TensorDict / TransferQueue | 2.54.1 / 0.10.0 / 0.1.11 |
| Accelerate / PEFT | 1.15.0 / 0.21.2 |
| W&B | 0.31.0 |
| AWS EFA installer / aws-ofi-nccl / libfabric | 1.51.0 / 1.22.0 / 2.7.0amzn2.0 |
| verl source | `d430157d15516d89e0cc853fd10fcd21d9234c44`, clean upstream checkout |

Both nodes import the same package versions. Node 1 retains stale
`pip-24.0.dist-info` alongside `pip-26.2.2.dist-info`, so its `pip freeze --all`
reports an installer-version discrepancy. Its imported pip is 26.2.2. The
canonical requirements use node 0's unambiguous snapshot; the active environment
was not altered to remove stale metadata.

### CUDA versions and library loading

Three versions describe different layers: the **driver** runs on the host,
the system **toolkit** supplies `nvcc`, and the **PyTorch/vLLM wheels** target
CUDA 13.0 and supply their CUDA runtime libraries. `torch.version.cuda == '13.0'`
is expected even though `nvcc --version` says 13.2. The CUDA version displayed
by `nvidia-smi` describes driver support, not the wheel's build version.
The tested driver is 595.112.07; NVIDIA documents CUDA 13.x minor compatibility
on the R580+ driver family with restrictions, which is not a claim that this
whole JIT stack was tested on every R580 driver.
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Retain the explicit **vLLM +cu130 wheel** in the requirements. The default
vLLM 0.19.1 PyPI wheel linked CUDA 12 and failed here with missing
`libcudart.so.12`. The pinned CUDA 13 wheel imports correctly with our torch
build. [Official vLLM CUDA build instructions](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/)
and [the exact 0.19.1 CUDA 13 wheel index](https://wheels.vllm.ai/0.19.1/cu130/vllm/).

Use [`worker_python.sh`](worker_python.sh) for GPU commands and Ray workers.
It prepends the virtualenv's `site-packages/nvidia/cu13/lib` to
`LD_LIBRARY_PATH` after shell startup. System CUDA libraries preceding the wheel
libraries previously caused cuBLASLt initialization failures. Python requirements
do not install the host driver, CUDA toolkit, compiler or EFA software; match the
recorded host stack when reproducing the full workload, including JIT extensions.

### Replicate in another workspace

Copy this repository **including its local adapter changes**. The recorded git
base revision alone does not contain the uncommitted implementation. Use Linux
x86_64, Python 3.12 (tested 3.12.3), a compatible NVIDIA driver and the recorded
toolkit/compiler for JIT compilation. Install OS tools such as `git`, `rsync`,
`python3.12-venv` and `build-essential` separately.

From the copied repository root, create a fresh environment:

```bash
export VERL_BASE_PYTHON=python3.12
export VERL_VENV="$PWD/.venv-h200-replica"
export VERL_SOURCE="$PWD/outputs/verl_runtime/verl-d430157"
bash src/nl2sql_verl/setup_fsdp_env.sh
```

The script installs the version-pinned requirements using the PyTorch CUDA 13.0
index, checks out the exact verl commit, and adds both verl and this repository
to `nl2sql-verl-source.pth`. **verl is intentionally a source checkout rather
than a PyPI requirement.** Our local FSDP, Gemma config and vLLM adapters are
required alongside it. This is a tested local integration, not a claim that the
stock upstream dependency matrix supports the combined versions.

The generic `setup_env.sh` and `runtime-requirements.txt` cover the legacy NeMo
adapter; use the new native setup above for the ongoing H200 environment.
For package installation alone, `python -m pip install -r requirements.txt`
works, but does not perform the source-checkout or `.pth` setup.

Verify an existing environment without changing it:

```bash
VERL_VENV="$PWD/.venv-h200-replica" \
VERL_SOURCE="$PWD/outputs/verl_runtime/verl-d430157" \
  bash src/nl2sql_verl/setup_fsdp_env.sh --check-only
```

For two-node training, replicate the environment/source on both nodes and stage
the same model/tokenizer, parquet files and SQLite databases. The current model
snapshot is `842da3794eaa0b77d5f08bae87a17459d91ff475`. Networking and checkpoint
sharing remain separate host setup: this cluster uses EFA, `enp71s0` for the
NCCL/Gloo interface, and a private NFS checkpoint staging path. Select interfaces
and an interconnect plugin appropriate to the destination workspace. Regenerate
Triton/Inductor caches for its hardware. Authenticate W&B separately; do not copy
API keys into requirements or config files.

This reproduces the software environment. Distributed correctness, capacity and
end-to-end checks still apply on a new cluster. The active run retains a 2,048-token
response limit; the checked-in **future-run** profile has 4,096. Those are training
configuration choices, not Python environment differences. See
[`DAPO_16GPU.md`](DAPO_16GPU.md) for the recipe and monitoring commands.

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
uses vLLM 0.19.1 and PyTorch 2.10/cu130. The inspected verl package metadata caps
Transformers below 5.13, whereas the tested AutoModel revision requires 5.18.
The script intentionally loads the pinned verl source through a `.pth` file,
using the local engine adapter, instead of installing a wheel that downgrades
Transformers. **This is not an upstream-supported combined dependency matrix.**
The full Python 3.12/vLLM runtime installation has not been exercised here.
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
| Oversample-and-replace, K=12; filter by result_reward; zero-mask leftover groups | Native bounded result-reward group filtering, up to four generation batches |
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
The final CPU suite has 13 passing tests; `validation.json` in that directory
records package/source versions and the distributed test results.
Compact validation reports and the unit-test summary are also versioned under
`benchmark_results/verl_recipe/`; local checkpoints and verbose logs stay in `outputs/`.
Distributed GSPO gradient error against a full-vocabulary/full-sequence
reference was **2.9802322387695312e-8**, with identical scalar losses. Tiny model
tests establish execution/save/restore/export behavior, not CP numerical gradient
parity, long-context training capacity, rollout weight-sync completion or NL2SQL
accuracy. The next integration test is the complete one-step 31B smoke cycle
once the real parquet/databases and a resolved rollout runtime are available.

## Remaining integration work

1. Build real schema/tool prompt parquet files and verify database coverage and
   the intended dev annotation version. Downloading databases alone does not
   create the prepared parquet inputs.
2. Build and resolve the Python 3.12/vLLM rollout runtime. The existing
   `.venv-verl-check` is a source-check environment, not a validated full rollout
   installation; the Transformers compatibility exception above remains open.
3. Run the complete 31B one-step cycle: generation, tool execution, rewards,
   old-policy logprobs, backward/update, weight synchronization, and a second
   rollout. Verify checkpoint resume in that complete runtime.
4. Compare full-model TP2/CP1 versus TP2/CP2 response logprobs and gradients on
   identical short examples. BF16 mitigations are included, but tiny-model
   execution and synthetic gradient tests do not resolve full-model CP parity
   or serving/training probability mismatch.
5. Measure training and rollout capacity, then increase context gradually and
   run dev execution-accuracy evaluation. The 199,680-token result is a
   forward-only measurement, not an RL context-length guarantee.

The implementation was introduced in commit `e95b4cf`; subsequent history moves
it from `recipes/` to `src/` and updates import/config/launch references. Git
history is the rollback source. Local reverse patches under
`outputs/verl_recipe_baseline/` predate the move and refer to the old paths.

Primary upstream references: [verl tool/agent extension interfaces](https://verl.readthedocs.io/en/latest/extend_guide.html),
[pinned verl requirements](https://github.com/verl-project/verl/blob/d430157d15516d89e0cc853fd10fcd21d9234c44/requirements.txt),
[Gemma AutoModel TP plan](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/models/gemma4_moe/parallelization.py),
and [vLLM 0.19.1 model registry](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/registry.py).
# Full-dev greedy baseline (2026-10-29)

The inference-only full-dev run uses the original base
`google/gemma-4-31B-it` revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, the corrected native-tool
`outputs/gemma4_native_v1/data/dev_full.parquet` (all 1,534 rows), temperature
0, one generation per question, and an 8,192-token cumulative response budget.
Each question retains three retrieved training examples and the enriched schema
with descriptions, statistics and sample values. Only the system instruction's
budget number is adjusted for this evaluation; the canonical parquet is unchanged.
Native tool execution and the strict final-answer scorer are used, with at most
nine assistant generations. The response budget includes tool observations and
formatting; it is not renewed after each tool call.

The 24 GPUs host twelve independent async vLLM replicas, each TP2, four per
node. Each replica has concurrency 16, a 38,912-token context, chunked prefill,
prefix caching and 80% GPU memory allocation. The pinned compatibility path uses
eager execution. Each node reads its single model snapshot on NVMe; four disk
copies are unnecessary. There is no actor, optimizer or training Ray cluster.

Artifacts live in `outputs/full_dev_base_t0_8k_20261029/`. `status.json` reports
all shards; each `shard-XX.log` contains raw logs and each shard directory contains
per-example results, tool events, raw generated tokens and a heartbeat. Detached
user services `nl2sql-dev-eval-s00` through `s11` run the engines. A separate
`nl2sql-dev-eval-monitor` service collects completed shards onto node 0 and merges
them only after verifying all reference indices 0–1533 occur exactly once.

```bash
tmux attach -t nl2sql-eval
# Or query all nodes without attaching:
.venv-verl/bin/python -m src.nl2sql_verl.manage_full_dev_eval status \
  --output-dir outputs/full_dev_base_t0_8k_20261029
# Final aggregate and sorted per-example outputs:
# outputs/full_dev_base_t0_8k_20261029/{summary.json,results.jsonl}
```

Completed result: **BIRD EX 70.93% (1,088/1,534)** using the saved official
evaluator and its 30-second timeout. Format validity99.15%, SQL execution
success99.09%, zero8K response caps. Native result reward with its longer60s
timeout is1,090/1,534; the two differences were confirmed30s timeouts.
See `outputs/full_dev_base_t0_8k_20261029/REPORT.md` and
`bird_official/summary.json` for the official metric and provenance.

## Offline 16-rollout training-prompt screening (2026-10-29)

The detached inference-only run in `outputs/train_pass16_base_t12_8k_20261029/`
screens all6,601 native training prompts with16 stochastic responses each
(105,616 responses). Original base Gemma weights, temperature1.2/top-p0.95,
8K cumulative response budget, nine assistant turns, actual native tools and
binary SQL-result correctness are used. The correctness scorer uses the same
60-second SQL timeout as DAPO training rewards. Training prompts have no
retrieved few-shots. The twelve async vLLM TP2 engines run four per node with
32 concurrent requests per replica, prefix caching, chunked prefill8192 and
80% memory allocation. They reuse one local model snapshot per node.

After all16 responses exist for every prompt, the monitor automatically writes
`group_scores.jsonl`, `selection.json`, `selected_train.parquet` and
`training_overrides.json`. It retains every group with1–15 correct responses,
plus a seeded uniform sample of floor(10% of the16/16-correct groups).
All0/16 groups are excluded. Seed20261029 and source positional IDs fix the
sample reproducibly. Final parquet rows retain the original source order,
prompts, schemas, tool configuration, gold SQL and source reference IDs.
Screening completions are evidence; they are not inserted into the RL parquet.
The final size is measured, not forced below1,500.

Pass@16 is the fraction of prompts with at least one correct response; the
selection uses the entire0–16 solve-count distribution. Online DAPO filtering
remains enabled on `result`: future training generates fresh groups and
continues dropping newly homogeneous groups. The raw6601-row input remains
available; the completed `training_overrides.json` specifies the selected
parquet and its exact expected row count for the next authorized launch.
This pipeline never starts training automatically. The separate screening uses
8K; the actor training recipe retains its previously validated4K response budget.

```bash
tmux attach -t nl2sql-screen
.venv-verl/bin/python -m src.nl2sql_verl.manage_prompt_screening status \
  --output-dir outputs/train_pass16_base_t12_8k_20261029
```

User services `nl2sql-screen-s00` through`s11` own the engines. The separate
`nl2sql-screen-monitor` service polls all nodes, backs up worker results every
five minutes, collects complete shards, and validates all105,616 unique
(data_index, rollout_index) pairs before exporting the subset. Per-shard raw
text, token IDs, tool events, seeds and provenance remain in the run directory.

The future16-GPU DAPO recipe explicitly enables `trainer.val_before_train=true`
and `test_freq=5`. All validation uses the same `val_kwargs` profile: temperature0,
top-p1, top-k-1, do_samplefalse and n1. Thus the fixed640-row evaluations at
steps0,5,10,... are all greedy. Training sampling remains temperature1.2/n16.
This was checked both through Hydra composition and through the actual pinned
V1 dispatch method at steps0,5,10 and a training step; evidence is in
`evaluation_sampling_validation.json`. Async vLLM and concurrent thread-backed
SQLite tool execution were already enabled for training. Optimizer updates
continue to consume complete rollout batches.

## Active screened1039/8K training run (2026-10-29)

The user authorized a fresh base-model training run on the1,039 screened
prompts,16 generations, node0+node1 (16 H200s), CP2/DP8/FSDP2. Node2 is outside
this Ray cluster. Entry point is `run_screened_pipeline`; actual configuration
is `dapo_screened_8k_16gpu.yaml`. The shared DAPO body is in
`dapo_fsdp_cp2_recipe.yaml`; public4K and8K primary configs each own Hydra's
search path. Old retired legacy readiness is not reused for this run.

All limits now agree:8,192 cumulative response tokens,30,720 prompt tokens,
38,912 context,8K reward-length setting, and8K prompt instruction. Versioned
run inputs in `outputs/screened_dapo_8k/data/` differ from their originals only
in the system budget number; selected membership, schema, few-shots, gold,
source IDs and tools are preserved. Eval640 is greedy at step0 and every5;
training remainsT1.2/top-p.95/n16 with online binary-result DAPO filtering.

Actor arithmetic/state is IEEE FP32; rollout weights areBF16. Per update:
32 accepted groups×16=512 trajectories, DP8, microbatch1,64 accumulation
rounds/GPU. AdamW lr1e-6, weight_decay0, no warmup or LR decay, gradient clip1.0;
DAPO ratio bounds0.2/0.28, token-mean, one PPO epoch, no KL or entropy bonus.
The run targets100 optimizer updates.

Capacity passed on all16 ranks: longest selected prompt10,897 +8,192 loss
response tokens =19,089 (padded19,200), two actual Adam updates/rank, max
allocated71.82GiB/reserved110.29GiB. First probe attempt failed only when
writing worker reports to an absent directory; the corrected clean rerun is
preserved as passing evidence. The remote launcher now uses `setsid --wait`
so nonzero worker exits propagate. One actual16-GPU native-tool integration
update passed with16 trajectories,32 tools and finite grad norm19.07.

Checkpoint policy changed at the user's request: **retain every model-only
export**, every5 updates. `max_actor_ckpt_to_keep=0` disables the custom
publisher's pruning; no optimizer/RNG checkpoint is requested. Twenty portable
exports require about1.25TB, exceeding root disk free space, so exports live in
`/opt/dlami/nvme/nl2sql/model-exports/screened1039-8k`. This NVMe is ephemeral;
copy exports elsewhere before terminating the instance. The source HF export
folders are also retained on NVMe. Keep-all publication and config tests pass.

The first integration startup exposed Ray's inherited1,024-file soft limit.
All affected job processes were raised to at least65,536 and narrow systemd
user-service drop-ins persist the limits for this job's head/worker/controller.
No model/data/optimizer change was needed to recover generation.

W&B is online: https://forge.coreweave.com/wandb/kpratik41/nl2sql-gemma4-verl/runs/75et1rl9

```bash
tmux attach -t nl2sql-train-1039
.venv-verl/bin/python -m src.nl2sql_verl.monitor_screened_run
```

The tmux status/training/metrics/gpus windows monitor a systemd-owned run;
closing tmux does not stop training. Root artifacts are
`outputs/h200_training/screened1039-8k/`: `train/training.log`,
`train/metrics.jsonl`, `train/validation/`, `train/rollouts/`, `run_manifest.json`,
command/source/data records, W&B URL/verification and pipeline state. Check
current phase from monitor/metrics; initial evaluation precedes optimizer step1.

### DAPO rollout accounting

Each candidate prompt group has 16 trajectories. The sampler keeps 32 mixed
groups (512 trajectories) for each update, discards groups whose binary result
is identical across all 16 trajectories, and drains requests already dispatched
before selecting the final batch. Mixed groups beyond the required 32 are
discarded as surplus; they are not carried into the next policy version.

The pinned replay buffer reports prompt-group counts despite naming the metrics
`evicted_samples` and `discarded_surplus_samples`. For a completed update without
rollout failures, generated groups equal 32 plus those two counters. The first
three updates generated 132, 165, and 152 groups (2,112, 2,640, and 2,432
trajectories) respectively. Only 512 trajectories per update entered the actor
loss. These totals include 28, 45, and 37 surplus mixed groups respectively.

Run CPU tests in the H200 environment without touching the training GPUs:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
VERL_CONFIG_DIR=/opt/dlami/nvme/nl2sql/runtime/verl/verl/trainer/config \
.venv-verl/bin/python -m pytest src/nl2sql_verl/tests -q
```

The three legacy AutoModel config cases require optional `nemo_automodel` and
skip in the FSDP-only environment. Native dataset/tokenizer integration cases
also skip when their local reference artifacts are unavailable. Recipe tests
require `VERL_CONFIG_DIR` to point to the pinned upstream config directory.
