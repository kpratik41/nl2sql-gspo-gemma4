# Gemma 4 NL2SQL tool-calling RL in verl

This ports the task behavior from `origin/consensus` commit
`588e1378777c64fa1a48d2548efccd51edd29270` to verl, with a repository-local
NeMo AutoModel Gemma ring-context-parallel engine. Existing parquet data and
SQLite databases are inputs. No question/schema/teacher-data generation runs.

The implementation is experimental. CPU task/config tests and distributed
synthetic GSPO checks pass. A tiny two-layer Gemma executes TP2/CP1 and TP2/CP2
training, optimizer updates, checkpoint restore and weight export. **A complete
31B vLLM rollout → tool execution → policy update → weight-sync cycle has not been
run.** Full-model CP numerical parity remains an open validation item.

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

Expected locations, already created locally:

```text
data/nl2sql/train.parquet
data/nl2sql/validation.parquet
databases/<db_id>/<db_id>.sqlite
```

Flat `databases/<db_id>.sqlite` and nested `train_databases` / `dev_databases`
layouts also work. Override with `TRAIN_PARQUET`, `VAL_PARQUET` and
`NL2SQL_DATABASE_DIR`.

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
VERL_BASE_PYTHON=/path/to/python3.12 bash recipes/nl2sql_verl/setup_env.sh

# Configuration validation without data or GPU allocation:
CONFIG_ONLY=1 bash recipes/nl2sql_verl/launch.sh

# After the parquet files and databases exist: one short RL step.
MODEL_PATH=/path/to/cached/gemma-4-31B-it bash recipes/nl2sql_verl/launch.sh

# The longer recipe preserves n=16 but still starts at a modest context.
RECIPE_CONFIG=nl2sql MODEL_PATH=/path/to/model bash recipes/nl2sql_verl/launch.sh

# Explicit GSPO sequence-ratio variant, with the same task/reward scaling:
RECIPE_CONFIG=gspo MODEL_PATH=/path/to/model bash recipes/nl2sql_verl/launch.sh
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
.venv-verl-check/bin/python -m pytest -q recipes/nl2sql_verl/tests
.venv-verl-check/bin/python -m recipes.nl2sql_verl.preflight --config smoke --config-only
.venv-verl-check/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m recipes.nl2sql_verl.tests.distributed_logprobs
.venv-verl-check/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m recipes.nl2sql_verl.tests.tiny_engine
CUDA_VISIBLE_DEVICES=0,1 .venv-verl-check/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m recipes.nl2sql_verl.tests.tiny_engine --cp 1
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

All new recipe code is under this directory. Existing benchmark/library code
was not edited. Delete this added directory to remove the recipe; the only
shared-file edits are the environment/database ignore entries and the roadmap
link. Existing uncommitted benchmark/debugging work is preserved.
For this workspace, the checked reverse patch and instructions are saved at
`outputs/verl_recipe_baseline/recipe-changes.patch` and `ROLLBACK.txt`.

Primary upstream references: [verl tool/agent extension interfaces](https://verl.readthedocs.io/en/latest/extend_guide.html),
[pinned verl requirements](https://github.com/verl-project/verl/blob/d430157d15516d89e0cc853fd10fcd21d9234c44/requirements.txt),
[Gemma AutoModel TP plan](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/models/gemma4_moe/parallelization.py),
and [vLLM 0.19.1 model registry](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/registry.py).
