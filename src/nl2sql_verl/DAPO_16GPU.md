# DAPO candidate on sixteen H200s

Config: `configs/dapo_fsdp_cp2_16gpu.yaml`. Launcher:
`bash src/nl2sql_verl/launch_dapo_16gpu.sh` on the Ray head, after staging and
validation. No training is launched merely by creating the config.

Future-run defaults allow 4,096 retained response tokens. The incorrect legacy
2K/32K run was stopped at nine updates on 2026-10-29. No new training run is
active. An inference-only comparison of 4K versus 8K budgets is being evaluated;
8K would require 38,912 context capacity and its own training launch checks.

The future profile now selects `outputs/gemma4_native_v1/data/`, preserving all
6,601 train rows and the fixed 640 validation rows while correcting implicit
foreign keys and aligning Gemma examples with the actual native tool parser.
It uses the separate observed native agent/tools/reward entry, logging real
tool executions, errors, exact final-SQL verification and budget exhaustion.
The task reward weights and DAPO recipe are unchanged. See
[data preparation](DATA_PREPARATION.md) for rebuilding and validation commands.
The legacy service is stopped. Its base evaluation cannot be reused
as the baseline for the new prompt version; a future run needs a fresh baseline.

Use two eight-GPU nodes in a dedicated Ray cluster. This is a sixteen-GPU
total job: actor training and vLLM rollout alternate on the same GPUs. The
third node's eight GPUs are not a separate rollout pool in this recipe.

| Component | Initial layout/settings |
|---|---|
| Actor | Native FSDP2, TP1 / Ulysses CP2 / DP8 |
| Actor state | FSDP shards parameters, gradients and optimizer state across 16 ranks |
| Actor precision | Existing static IEEE FP32 policy |
| Actor microbatch | One trajectory per CP pair |
| Global update | 32 accepted prompt groups × 16 generations = 512 trajectories |
| Accumulation | 512 / 8 DP replicas = 64 microbatches per replica |
| Rollout | BF16 vLLM TP4 / DP4, initially at most 16 sequences per replica |
| Future prompt / response / context limits | 30,720 / 4,096 / 34,816 tokens |

Keep CP pairs and TP4 rollout replicas within a node. FSDP state synchronization
still crosses nodes. Verify actual rank placement and inter-node NCCL before
measuring throughput. TP1/CP2 retains the native actor implementation that passed
the previous machine's tests; adding actor TP would require a different backend.
The chosen rollout concurrency is a conservative starting point for H200 memory,
not a measured throughput optimum.

The new DAPO profile explicitly uses:

- Upstream `grpo` advantages with group standard-deviation normalization.
  This changes the earlier `nl2sql_batch_grpo` batch-wide scaling.
- Native V1 dynamic filtering on binary BIRD `result`, rejecting groups whose
  sixteen generations all have the same result. Refills dispatch one prompt
  at a time, with at most 64 prompt groups pending/running.
- Token-level `vanilla` clipped policy loss, token-mean reduction, lower/upper
  clipping 0.2/0.28 and dual-clip parameter 10. There is no `loss_mode: dapo`.
- No KL reward/loss or entropy bonus.
- The existing NL2SQL composite rewards, including its soft overlong penalty:
  zero through 3,584 generated assistant tokens, ramping to -1 at 4,096,
  with the existing composite weight 0.1. This is a task-adapted DAPO recipe,
  not a claim to reproduce a math benchmark's rewards.

The pinned V1 sampler does not enforce a total refill/retry limit through
`max_num_gen_batches`; the in-flight cap is not a total sampling budget.

All measured train/full-dev prompts fit the new prompt limit. The full-dev
maximum is 29,695 tokens; adding the 4,096-token response budget gives
33,791, below the 34,816-token context. The actor and log-probability token
limits and rollout prefill budget also cover the full context. Preparation
manifests retain their original 11,264-token audit; this config supersedes that
limit without modifying the datasets or their provenance.

Identical runtimes, model/tokenizer, SQLite databases and parquet data are staged
on both selected nodes. The dedicated Ray cluster reports sixteen GPUs, and
sixteen-rank NCCL collectives and the tiny FSDP2/model-only export checks passed.
`ray_kwargs.ray_init.address: auto` attaches to that existing cluster; the
launcher supplies its explicit address. Checkpoint paths are separate from prior
eight-GPU runs.

Training remains gated on the full-model numerical and memory checks, followed
by the end-to-end smoke. Initial CP2 long-prompt gradient checks failed despite
close log probabilities. The cause was narrowed to cuBLAS changing its FP32
reduction for the narrow global-key projection when the token count is halved.
The H200 profile gathers full-sequence inputs for that projection only; all
other linears remain partitioned. The first eight-GPU real-prompt comparison
then passed, with worst sampled relative gradient error 0.0000215 versus the
independent HF reference (previously 0.827). TF32 failed and remains disabled.
Sixteen-GPU numerical/memory validation is still required; the first diagnostic
does not establish production readiness. Prior eight-B200 evidence does not
validate this topology or the new evaluation format.

Configuration checks can run without installing the GPU stack. Set
`VERL_CONFIG_DIR` to the pinned commit's `verl/trainer/config` directory and run:

```bash
.venv-data/bin/pip install hydra-core pytest
VERL_CONFIG_DIR=/path/to/pinned-verl/verl/trainer/config \
  .venv-data/bin/python -m pytest src/nl2sql_verl/tests/test_dapo_16gpu_config.py -q
```

The upstream pin is `d430157d15516d89e0cc853fd10fcd21d9234c44`.
Official references: [verl DAPO settings](https://verl.readthedocs.io/en/latest/ascend_tutorial/en/model_support/examples/dapo_multi_model_optimization_practice.html)
and [the DAPO paper](https://arxiv.org/abs/2503.14476).

## Current checkpoint and evaluation policy

Saving and evaluation both run every five optimizer updates. Actor saves use
`checkpoint.save_contents: [hf_model]`; the native engine streams BF16 HF
weights while leaving training parameters/optimizer arithmetic unchanged. No
full model/Adam/RNG shard set is stored. The completed HF export is copied from
NVMe to `/home/ubuntu/nl2sql-checkpoints/` on node 0's EBS disk, retaining the
latest two completed model exports per run. Each export includes the tokenizer
and configs and can initialize a new run with fresh optimizer/dataloader state.
Automatic full-state resume is disabled. The lifecycle hooks do not require
NeMo; legacy AutoModel recipes use their own `automodel_runtime` entry point.

The Ray controller is pinned to node 0 so EBS publication and W&B authentication
happen on the intended host. Ray head/worker services run under systemd user
units `nl2sql-ray-head` and `nl2sql-ray-worker`; the cluster contains only the two
selected nodes. Checkpoints are staged on node 0 NVMe and shared with node 1
through a private NFS export restricted to node 1's address. Model weights,
SQLite databases, compilation caches and environments use node-local NVMe.

`run_h200.py` supervises the gated five-update smoke and bounded full-batch pilot.
The pilot keeps the save/eval cadence of five; after its two-hour training target
is reached, it finishes through the next checkpoint/eval boundary, with a
six-hour watchdog. W&B keys are kept outside the repository. If authentication
is unavailable, the supervisor records W&B offline files plus JSONL metrics.
Non-secret W&B mode/directory controls are forwarded to the Ray controller;
API credentials are read from the controller user's SDK credential store.

`prepare_h200_run.py` executes the remaining stages under the user service
`nl2sql-h200-prepare`: full-model sixteen-GPU comparisons against independent
HF references, warm Adam/memory checks at 13K and the maximum context, optional microbatch-2
comparison, readiness validation, five-update smoke, integration validation,
and launch of the authorized bounded full-batch pilot. A failed required stage
stops the pipeline. A failed optional microbatch-2 check retains microbatch 1.
Progress is in `outputs/h200_checks/pipeline_status.json` and `pipeline.log`;
per-node probe logs are in `stable_ieee16_baseline/` and `stable_ieee16_micro2/`.
No H200 readiness marker is issued until the required gates pass.

## Live terminal monitoring

On node 0, attach with `tmux attach -t nl2sql`. Detach with Ctrl-b, then d;
the training job is supervised by systemd and continues independently. Select
windows with Ctrl-b, then 0–4: status, live logs, both nodes' GPUs, Ray status,
and the pipeline supervisor log. The live-log view switches automatically from
smoke to pilot when the pilot heartbeat appears. The monitor itself is read-only.

Console output remains in `outputs/h200_training/{smoke,pilot}/training.log`,
with metrics in `metrics.jsonl` and supervisor progress in
`outputs/h200_checks/pipeline.log`. `tail -F` works without tmux. The Ray web
dashboard is disabled in this cluster; the Ray CLI window provides resource
status. Ray resource reservations/fractions describe scheduling, while the GPU
window reports physical compute utilization. W&B provides reward/loss/evaluation
curves; retain the local logs for initialization errors and worker diagnostics.

## Full training run

The user requested starting full training before the smoke finished. The smoke
was stopped after three completed updates, and the queued bounded pilot was
cancelled. `nl2sql-h200-dapo-train` starts fresh from the base snapshot for 100
updates, using 32 accepted prompt groups × 16 generations, CP2/DP8, microbatch 1,
save/eval every five updates and model-only EBS exports. It has no pilot time
limit. The completed base evaluation is referenced from the smoke instead of
being repeated; the full run's reference metadata is in
`outputs/h200_training/train/initial_evaluation_reference.json`.

The explicit launch command was
`.venv-verl/bin/python -m src.nl2sql_verl.run_h200 train --allow-incomplete-smoke`.
This flag retains numerical/memory/source/data gates but allows this user-selected
start before end-to-end smoke integration completes. It does not mark integration
checks as passed; `launch_policy.json` records that distinction. Ordinary launches
continue to require integration readiness. The live log is
`outputs/h200_training/train/training.log`; tmux window 1 follows it automatically.
