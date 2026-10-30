# Gemma 4 31B: incremental validation of CP=2 and FFPA for verl RL

> Historical planning/benchmark record. The current native FSDP2/Flex/Ulysses
> run and its dated results are described in [the runtime guide](src/nl2sql_verl/README.md)
> and [WORK_LOG.md](WORK_LOG.md). Older AutoModel proposals and pending
> statements below describe their original experiment stage.

## Tool-calling verl recipe — 2026-10-22 (America/New_York)

The task code from `consensus` has been adapted into
[`src/nl2sql_verl/`](src/nl2sql_verl/README.md): prepared parquet inputs,
the three active SQLite tools, seven weighted rewards, native Gemma tool
parsing and a local AutoModel TP2/CP2 engine for verl RL. Defaults preserve
the actual TRL token-ratio/batch-reward-scaling objective; an explicit GSPO
profile is also provided. It uses BF16 compute,
FSDP2 CPU master weights/optimizer state, the measured accumulation policy and
SM120 tiles. Installed libraries and prior benchmark files are unchanged.

Task/config tests, a four-rank synthetic GSPO gradient comparison, and tiny
Gemma engine training/checkpoint/weight-export checks pass. A complete 31B
vLLM tool rollout and RL update is not yet validated. The recipe documents
the verl/AutoModel Transformers version conflict and the source-adapter
exception; its separate rollout runtime still needs installation/validation.
Full-model CP probability sensitivity remains open. Earlier experiments used
NeMo AutoModel directly, not NeMo RL or Megatron-LM training.

## TP=2 measurement update — 2026-10-23

The requested TP2/CP1 (two GPUs) and TP2/CP2 (four GPUs, two GPUs per CP
partition) now execute with the standard Gemma TP plan and BF16 ring Flex
attention. With batch one, no gradients/cache, and all-token TP-vocabulary-sharded
logits, the largest confirmed forwards are **104,448** and **199,680** total
tokens respectively. The next 1,024-token step fails with CUDA OOM in each
layout under the default allocator; these are measured forward limits.

The TP2 short probability comparison passes the existing screening gates at
1,024 tokens but fails the max-token logprob gate at 128 tokens (0.356054 nats).
The large original BF16 discrepancy is mitigated, but probability sensitivity
remains. Long forwards measure capacity; they do not validate long-context
probability parity, whole-model gradients, optimizer steps or RL integration.
No data download was needed. See the newest section of
[GEMMA4_BENCHMARK_RESULTS.md](GEMMA4_BENCHMARK_RESULTS.md) and
`benchmark_results/tp2_capacity/report.json` for exact results and reproduction.
The earlier dated updates and original allocation assumptions below are historical.

## Current execution update — 2026-10-22

The host now has four RTX PRO 6000 Blackwell GPUs; references below to one GPU
and waiting for a second node describe the earlier allocation. A clean AutoModel
environment is installed. Real two-rank ring-Flex CP2 forward runs on GPUs 0–1
at 128 and 1,024 total tokens now execute, but full-model probability screening
fails against both native Flex CP1 and a one-rank ring control. Separate two-rank
global/sliding attention output and Q/K/V gradient checks pass their FP32-reference
tolerances with explicit SM120 tile overrides. This does not validate whole-model
gradients or RL. See the measured CP2 update in
[GEMMA4_BENCHMARK_RESULTS.md](GEMMA4_BENCHMARK_RESULTS.md) and run
`bash benchmarks/run_cp_checks.sh` to reproduce. Resolve the model probability
discrepancy before increasing context or integrating training.

Follow-up traced the large difference to shape-dependent BF16 GEMM reductions:
one GPU on 64 versus 128 tokens reproduces the original worst probability error,
and the first difference is layer 0 Q projection. Opt-in cuBLASLt with full
reduction/no split-K mitigates it; remaining ring-chunk precision effects shrink
further with optional FP32 attention. The target training precision remains
BF16; FP32 intermediate accumulation is compatible with BF16 weights and
activations. The efficient profiles still fail the unchanged max-token screening gate.
Installed AutoModel source is unchanged; the benchmark report records flags,
raw evidence, reproduction and rollback. Continue validation with BF16 and
appropriate accumulation precision before treating the path as ready for RL.

## Main objective and why we are doing this

Enable **CP=2 context parallelism** and **FFPA, or another validated memory-efficient attention backend**, for **`google/gemma-4-31B-it`**, with **verl** as the target framework for eventual **full-parameter NL2SQL RL**. CP=2 distributes one long sequence across two GPU ranks to reduce per-rank sequence/activation pressure. Efficient attention avoids materializing quadratic attention-score tensors; FFPA is a candidate for faster eligible global-attention operations. We want measured capacity and speed improvements while preserving model semantics, token probabilities, and gradients. Global-attention compute/latency is not expected to become linear merely because working memory is approximately linear.

We currently have one GPU, so we validate local behavior first and design the distributed recipe for a future two-node allocation with an explicit GPUs-per-node setting. CP=2 counts GPU ranks, not nodes. Standalone kernels or successful forward execution do not establish distributed training correctness. Existing TRL/ZeRO-3 code on `consensus` is prior task work; NL2SQL integration comes after the systems checks.

**Incremental validation sequence:**

1. Establish a reproducible environment and model snapshot; benchmark single-GPU attention and full-model forward latency/memory across context lengths, with and without FFPA. Measure mask construction and cached generation separately.
2. Resolve numerical discrepancies with identical inputs/weights, teacher-forced token probabilities, FP32 references, fixed-input layer replay, and causal/window/padding/document-boundary checks. This is why we are investigating precision and masks before scaling further.
3. Validate real CP=2 forward correctness, per-rank memory, communication, and speed on separate GPUs; compare efficient-attention backends under the same allocation and effective batch.
4. Validate attention backward, model gradients, activation checkpointing, optimizer steps/state memory, and save/resume; increase context incrementally toward the provisional 40,960-total-token target.
5. Integrate the validated AutoModel/Gemma path into verl, including CP token mapping, loss normalization, actor/reference probabilities, rollout weight synchronization, and one complete resumable RL update.
6. Add NL2SQL data, executable rewards, and held-out database evaluation; measure complete RL iteration performance before choosing a production recipe.

**Success means correct long-context training with CP=2 and an efficient attention path. FFPA is optional:** retain a correct FlexAttention or other efficient baseline if FFPA is unsupported or offers an unfavorable speed/memory tradeoff. Record performance observations separately from correctness acceptance. The sections below expand this validation sequence.

Updated: 2026-10-22 (America/New_York).

## Goal and current status

Use **`google/gemma-4-31B-it`**, context parallelism **CP=2**, and optionally **FFPA**, eventually for full-parameter NL2SQL RL in **verl**. First establish correct execution, memory scaling, and speed; implement the NL2SQL training workflow afterward. TRL/DeepSpeed ZeRO-3 work on `consensus` is prior work, not evidence that this verl configuration works.

This document is a research-backed implementation plan. No benchmark, dependency installation, distributed run, or training run was performed for this update. Configuration fragments describe intended AutoModel behavior; they are not runnable verl recipes.

The recommended order is:

1. Single-GPU attention and model forward tests, with and without FFPA.
2. Real two-GPU CP=2 forward correctness and performance.
3. Attention backward, model backward, and optimizer validation.
4. Integrate the validated AutoModel path into verl.
5. Validate one complete RL update, then add NL2SQL data/rewards.

**CP=2 is the primary capability goal. FFPA is an optional optimization.** A correct FlexAttention CP=2 path is a useful result even if FFPA is unavailable or slower on the selected hardware.

## Corrections to the earlier note

| Earlier assumption | Current interpretation |
|---|---|
| The current host has eight H100s | This workspace currently exposes one NVIDIA RTX PRO 6000 Blackwell Server Edition, reporting 97,887 MiB total memory. Earlier eight-H100 observations belong to a previous environment. |
| Gemma dense cannot use CP | That was a restriction of the examined Megatron Bridge provider. AutoModel has a Gemma-specific ring CP implementation; investigate that path first. |
| Two nodes means 16 GPUs | GPUs per future node are unspecified. Parameterize the plan by GPUs per node; show both two total GPUs and sixteen total GPUs below. |
| Linear attention memory means linear inference latency | Tiled attention can avoid quadratic score storage while global attention computation remains quadratic. Measure memory and latency independently. |
| Setting `attn_implementation=ffpa` enables CP+FFPA | Direct HF FFPA is rejected with CP>1. AutoModel CP uses `sdpa` as the entry point and a separate full-attention backend setting. |
| A generic verl CP or Ulysses flag enables Gemma ring CP | These are different integration paths. Correct model construction, batch sharding, masks, token mapping, and gradient normalization must all be established. |
| A forward test establishes training support | Training also needs correct backward kernels, CP communication in backward, activation checkpointing, optimizer memory, and loss scaling. |
| Nemotron is the preferred next step | Gemma 4 31B remains the chosen model. The earlier alternative-model investigation is deferred. |

The old checkpoint paths under `/home/ubuntu`, screen-session identifiers, branch/worktree setup, and historical PR-status tables are not current environment facts. Git history retains the earlier investigation. This roadmap supersedes its hardware and implementation recommendations.

## What CP and FFPA each do

**CP=2 splits the sequence computation across two GPU ranks.** For an approximately 40k-token sequence, each rank processes roughly half the token positions and exchanges attention information with its partner. Both halves still participate in the same example. The implementation determines the actual layout; do not manually assume that arbitrary token slicing is sufficient.

**FSDP2 shards model training state.** It addresses parameter, gradient, and optimizer storage. CP addresses sequence-dependent work and activations. They complement each other, but their process groups and memory effects must be inspected rather than multiplied blindly.

**FFPA accelerates eligible attention operations.** AutoModel uses its CuTeDSL kernels for eligible 512-dimensional full-attention heads. Local sliding layers use a separate backend. The model's attention semantics remain the same.

The published [Gemma checkpoint configuration](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) specifies 60 text layers: 50 sliding layers with head dimension 256 and a 1,024-token window, plus 10 full-attention layers with head dimension 512. It has 32 query heads, 16 local KV heads, four global KV heads, a 262,144-token configured maximum, and final-logit softcapping of 30. Preserve those settings, attention scaling, normalization, and positional embeddings.

### What “linear memory” should mean in our tests

For fixed batch size, heads, and dtype, an efficient full-attention kernel should avoid storing an `S × S` attention-score matrix. Its sequence-dependent working memory can scale approximately linearly with sequence length `S`. Full causal attention still performs approximately quadratic arithmetic. Fixed-window sliding attention can have approximately linear arithmetic when its implementation exploits the window.

For perspective, one hypothetical BF16 score tensor at 40,960 tokens and 32 heads would occupy:

```text
1 × 32 × 40,960 × 40,960 × 2 bytes = 100 GiB
```

Avoiding that allocation is a key success criterion. CP=2 does not remove the need for a memory-efficient kernel.

Whole-model memory includes fixed weights, logits, masks, temporary copies, communication buffers, and cached allocations. Training adds activations and optimizer state. Therefore, total memory need not double when context doubles, and CP=2 need not halve the total peak.

At this model's vocabulary size, BF16 logits for all 40,960 positions alone occupy 20 GiB. Use last-position logits for generation-style forward tests where supported, and separately test response log probabilities for training. Record whether logits are materialized; otherwise they can hide attention-memory improvements.

## What the upstream evidence establishes

The [requested discussion #2928](https://github.com/NVIDIA-NeMo/Automodel/discussions/2928) explains two distinct routes:

- Direct HF FFPA: CP=1, no packed sequences; eligible global layers use FFPA and sliding layers use FlexAttention.
- Ring CP: initialize with SDPA and select FFPA through `text_config.cp_full_attn_backend`; eligible full-attention chunks use FFPA while other chunks retain the ring fallback.

Its published H100 results report FFPA gains over Flex of about 17–24% for CP1 at 4k/8k and about 11% for packed CP8 at 16k. These are upstream results, not forecasts for our GPU or CP2. The discussion also identifies SDPA fallback to math and allocator pressure as reasons a comparison can be misleading.

Current source confirms the routing in [FFPA integration](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/attention/ffpa_attention.py), [Gemma model setup](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/models/gemma4_moe/model.py), and [ring attention](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/models/gemma4_moe/cp_attention.py). The shared implementation lives under `gemma4_moe` even though our target is dense.

The [FFPA project](https://github.com/xlite-dev/ffpa-attn/blob/b11d103e4adeed630f77d2cc84038100cd5ebc2c/README.md) lists CuTeDSL forward/backward support for large heads including SM120-class devices. That makes our GPU a candidate. Runtime tests must still establish support for the exact BF16 D512 dense and varlen operations that AutoModel invokes. B200/SM100 and RTX PRO/SM120 have different kernel paths; published B200 numbers do not transfer directly. FP8/FP4 forward-only benchmarks are outside this BF16 training comparison.

[NeMo RL's Gemma guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/gemma/gemma4.html) lists 31B CP1/CP2 as functionally ready, with short-run coverage. Its current [base recipe](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/dapo-gemma4-31b-it-4n8g-automodel.yaml) uses four eight-GPU nodes and a 4,096-token total-sequence limit; the [CP2 override](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/dapo-gemma4-31b-it-4n8g-automodel-cp2.yaml) changes CP. This supports the integration approach, not a claim that 40k full-parameter RL fits our future cluster. NeMo RL is a reference implementation; verl remains the requested framework.

## Backend combinations to test

| Case | CP | HF attention entry point | Global attention | Sliding attention | Purpose |
|---|---:|---|---|---|---|
| A | 1 | `flex_attention` | Flex | Flex | Main efficient baseline |
| B | 1 | `ffpa` | Eligible calls use FFPA | Flex | Single-GPU optimization candidate |
| C | 2 | `sdpa` | Ring Flex | Ring Flex | Primary CP capability baseline |
| D | 2 | `sdpa` | Ring FFPA when eligible | Ring Flex | CP optimization candidate |
| E | 1 | `sdpa` | Profile actual dispatch | Profile actual dispatch | Short-context diagnostic |

For C/D, hold the sliding backend at `flex` to isolate the global-kernel change. Current source has additional options, but they can be tested later. Never label a run “FFPA” solely from its configuration: record executed kernels and fallback reasons.

Conceptual AutoModel settings for **C**:

```yaml
model:
  pretrained_model_name_or_path: google/gemma-4-31B-it
  torch_dtype: torch.bfloat16
  attn_implementation: sdpa
  text_config:
    use_cache: false
    cp_full_attn_backend: flex
    cp_sliding_attn_backend: flex
distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 2
  activation_checkpointing: true
```

For **D**, change only `text_config.cp_full_attn_backend` to `ffpa`. Exact loader arguments and schema depend on the pinned AutoModel revision and eventual verl adapter. For forward-only tests, disable activation checkpointing and omit optimizer construction.

For **B**, use direct `attn_implementation: ffpa`, CP1, and unpacked examples. Direct FFPA's eligibility includes BF16/FP16, D512, zero attention dropout, supported masks/scaling, and available kernels. Unsupported calls may fall back. The CP FFPA gate also needs document metadata supplied by the model's CP batch preparation, including synthesized IDs for an unpacked single document.

## Stage 0 — establish the reproducible environment

Before benchmarking:

1. Record GPU model, compute capability, driver, usable memory, topology, CPU RAM, CUDA runtime, and available disk. Do not install over the existing working environment; create an isolated benchmark environment.
2. Pin model revision and tokenizer/chat template. Use the exact case-sensitive model ID above or a verified local snapshot. Keep BF16 weights and computation for the primary comparison.
3. Resolve a compatible PyTorch/Transformers/AutoModel/FFPA stack and save a lock/container manifest. Verify dense and varlen FFPA forward/backward imports and execution on the actual GPU.
4. Log the resolved attention configuration, model class, trainable parameter policy, and kernel fallbacks. For text-only full training, train all language-model weights including embeddings/tied output weights; unused vision/audio components can be frozen.
5. Create deterministic token inputs and a small realistic text set. Define length after tokenization/chat templating. Start with one unpacked example, batch size one, no padding.

Source snapshots inspected on this date, **not a tested compatible installation**:

| Project | Inspected revision |
|---|---|
| AutoModel | `65866fbce209938cd153f2bc26f01fa0d34cc61e` |
| verl | `dc709afb7a74fd2645640a4e5f2fdf40166b4fbb` |
| FFPA | `b11d103e4adeed630f77d2cc84038100cd5ebc2c` |

AutoModel's inspected [dependency manifest](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/pyproject.toml) pins Transformers 5.18.0 and its FFPA extra specifies `ffpa-attn>=0.2.4` plus `nvidia-cutlass-dsl==4.6.2`. These are source constraints, not instructions to combine independently latest packages. Resolve the full dependency set before choosing the runtime pin.

**Exit gate:** short BF16 Flex execution works; FFPA either executes the required operations or has a documented blocker. FFPA failure does not prevent the Flex CP track.

## Stage 1 — single-GPU forward latency and memory

This is the first implementation milestone. It establishes local kernel behavior before distributed complexity.

### 1A. Attention-only tests

Use model-shaped Q/K/V for both layer types: D512 with 32/4 query/KV heads for full attention, and D256 with 32/16 heads plus the 1,024-token window for sliding attention. Include actual model scaling and causal masks.

At short lengths, compare outputs against a trusted reference before measuring speed. Add boundary cases around 1,024 tokens, padding, unequal lengths, and eventually packed documents. Keep eager/dense reference calculations at small lengths.

Sweep **1,024 → 2,048 → 4,096 → 8,192 → 16,384 → 32,768 → 40,960** tokens, stopping on OOM or unsupported execution. Compare Flex and FFPA global attention. Profile SDPA only at safe lengths first; investigate any math fallback before increasing length.

### 1B. Full-model forward without a KV cache

Run A/B with identical tokens, batch size, dtype, output selection, and mask semantics, using eval mode and no gradients, `use_cache=false`. Measure the complete model forward and separately identify attention cost. This is the closest initial probe of the forward computation used by training/logprob evaluation.

Load BF16 weights once per test process. A 31B checkpoint consumes roughly 58 GiB for weights alone, so full-model 40k execution on the current card is a measurement target, not an assumed fit. If it fails, retain the attention-only results and report the first failing model length and allocation category.

### 1C. Cached generation as a separate experiment

After 1B, measure prompt prefill plus a fixed short decode, for example 128 tokens, with `use_cache=true`. Report prefill/TTFT, decode milliseconds per token, and KV-cache memory separately. Ensure the selected backend supports cached unequal query/key lengths. Use fixed-length decoding or report actual generated lengths if EOS ends early.

Later repeat generation in the chosen rollout engine. Its serving kernels and TP layout can differ from AutoModel; a vLLM result does not validate the trainer's FFPA or CP implementation.

### Measurement protocol

- Warm up each backend/shape until compilation and allocation settle; start with five warmups and twenty measured repetitions, increasing only if needed. Record cold compilation time separately.
- Synchronize CUDA at timing boundaries. Record model-only device time and end-to-end wall time, including mask construction where applicable. Keep profiling separate from timed repetitions.
- After warmup, release previous outputs, record live allocated/reserved memory, reset peak counters, run the measured operation, and record peak allocated/reserved memory plus device-level usage. Use fresh processes for clean backend/length comparisons; also retain cold-start peak memory.
- Report incremental peak allocated memory above the live pre-forward baseline. Keep allocator configuration fixed and log allocation retries/OOMs. Reserved memory is not live-tensor memory.
- Save median and p95 latency, tokens/s, numerical error, dispatch evidence, and failures. Use actual non-padding tokens in throughput calculations.

Fit the incremental memory trend to `a + b*S`, examine doubling-length increments and residuals, and compare with a quadratic trend. A fixed weight offset can conceal growth in total memory. Several lengths plus a profiler trace are needed; one ratio does not establish asymptotic complexity. Compilation, mask construction, and logits must be accounted for separately.

**Exit gate:** output parity passes at short lengths, dispatched kernels are known, and the memory curve shows no unexplained quadratic score/mask allocation. Report speedup or slowdown relative to Flex at each length; no speedup is guaranteed.

## Stage 2 — CP=2 forward on two nodes

Design the future launcher around `NNODES=2`, `GPUS_PER_NODE=G`, `NODE_RANK`, rendezvous address/port, CP size, backend, and sequence length. No launcher is implemented yet.

| Future allocation | World size | With TP=PP=1 and CP=2 | Implication |
|---|---:|---|---|
| Two nodes × one GPU | 2 | One context-parallel pair | Tests cross-node CP; full training likely needs substantial offload/state-memory changes. |
| Two nodes × eight GPUs | 16 | Eight logical data-parallel groups, each with a CP pair | Candidate full-training allocation; verify actual FSDP sharding groups in the pinned mesh. |

CP degree counts ranks, not nodes. With multiple GPUs per node, keep CP partners on fast intra-node links initially, and use both nodes for the wider training mesh. Test cross-node CP placement separately. With one GPU per node the pair necessarily communicates over the network. Print group membership and GPU placement; never infer them from world size alone.

First validate rendezvous/NCCL collectives and measure inter-node bandwidth. One physical GPU cannot provide a meaningful real CP2 correctness or speed result; two processes on that card are not a substitute.

Then:

1. Run a small two-rank attention forward parity test using global reference outputs and each rank's actual token mapping. Include tokens whose causal/window dependencies cross the rank boundary.
2. Run C versus D at the same length and placement, starting at 2k/4k. Confirm FFPA actually executes on eligible global ring chunks.
3. Compare CP1 and CP2 at identical full sequence length, effective examples, dtype, outputs, and total GPU allocation where feasible. Record changes in logical DP and adjust accumulation to preserve the effective batch.
4. Measure both fixed-length latency (strong scaling) and maximum feasible sequence length. Report per-rank peak memory, worst-rank elapsed time, useful global tokens/s, and communication time. Do not count the same sequence twice because two CP ranks cooperate on it.
5. Grow to 8k, 16k, 32k, and 40,960 total tokens. Add padding/packing only after single-document parity passes.

Where CP1 cannot fit the full model at a shared length, report that capacity result; use smaller lengths or attention-only tests for a direct parity/speed comparison. CP communication can make short sequences slower. CP's first benefit may be capacity rather than latency.

**Exit gate:** CP2 agrees with CP1/reference, the intended kernels execute, and memory/latency/communication are measured on real separate GPUs.

## Stage 3 — backward and optimizer checks

Use the validated forward configuration and introduce one change at a time:

1. **Attention backward:** compare Q/K/V gradients for Flex versus FFPA and CP1 versus CP2 at small lengths. Cover local-window boundaries, causal offsets, padding, and later packed-document isolation. Check gradients contributed by remote ranks.
2. **Model backward:** compute a deterministic token loss without an optimizer. Compare token log probabilities, scalar loss, selected parameter gradients, and gradient norms. Verify label shifting and final-logit softcapping.
3. **Activation checkpointing:** repeat with the actual production checkpoint policy. FFPA's integration has selective-save behavior; do not assume identical recomputation or memory savings across backends.
4. **Optimizer step:** enable full language-model updates with an explicitly selected optimizer/state dtype and sharding/offload policy. Run multiple steps so lazily allocated optimizer state is included. Compare one deterministic update across CP layouts.
5. **Length growth:** repeat at 8k → 16k → 32k → 40k only after short-length correctness passes. Record forward, backward, optimizer, communication, and peak memory separately.
6. **Recovery:** save and restore model, optimizer, scheduler, and RNG state; compare the next update with uninterrupted execution.

Use the upstream numerical-test tolerances as a starting reference, calibrate against the same BF16 baseline, and fix tolerances before performance evaluation. Record absolute/relative errors and nonfinite values; a finite loss alone is insufficient.

For a rough capacity check, a conventional mixed-precision Adam layout at 16 bytes per trainable parameter is about 496 decimal GB for 31B parameters before activations and temporary buffers. Two 96-GiB cards cannot hold that whole layout on-device. CP alone does not fix it. Actual optimizer precision, FSDP groups, CPU offload, and additional GPUs determine feasibility. An offloaded run must include transfer/CPU time in its performance report.

**Exit gate:** correct distributed gradients and updates, stable multi-step memory including optimizer state, and successful resume. Keep CP2+Flex as the fallback if FFPA backward is blocked.

## Stage 4 — integrate into verl

This is a concrete engineering task, not a configuration-only promise. The inspected verl [AutoModel documentation](https://github.com/verl-project/verl/blob/dc709afb7a74fd2645640a4e5f2fdf40166b4fbb/docs/workers/automodel_workers.rst) describes SFT examples and older dependency requirements. Its [utilities](https://github.com/verl-project/verl/blob/dc709afb7a74fd2645640a4e5f2fdf40166b4fbb/verl/workers/engine/automodel/utils.py) use older mesh/model-setup interfaces. Its [engine batch path](https://github.com/verl-project/verl/blob/dc709afb7a74fd2645640a4e5f2fdf40166b4fbb/verl/workers/engine/automodel/transformer_impl.py) does not show the newer model-owned sharder invocation. These observations are integration gaps to resolve, not proof that all verl RL use is impossible.

The newer [NeMo RL integration design](https://docs.nvidia.com/nemo/rl/nightly/design-docs/automodel-context-parallel.html) provides a useful reference: construct the distributed setup, delegate model-specific CP layout to AutoModel, and use that layout to map token tensors. Its documented compatible stack uses AutoModel v0.6.0 and Transformers 5.12.1, which differs from the inspected AutoModel main. Choose a compatible pin rather than mixing those examples unchanged.

Required adapter work to investigate:

- Load the correct Gemma model class and preserve checkpoint weight names/tied weights. Pass text-config overrides through the verl configuration to AutoModel.
- Update distributed/model-setup APIs for the selected pin, with explicit CP/FSDP groups.
- Use AutoModel's [ContextParallelSharder](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/distributed/context_parallel/sharder.py) and Gemma's [batch preparation](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/nemo_automodel/components/models/gemma4_moe/cp_batch.py), including document IDs and model-owned input handling. Current Gemma handling leaves primary input preparation partly inside the model; do not add a second manual token split.
- Map response masks, shifted targets, advantages, old/reference log probabilities, and returned log probabilities through the exact layout. Avoid gathering full-vocabulary prompt logits just to compute response probabilities.
- Verify CP/DP loss normalization and gradient reduction against CP1 with the same global batch. Do not copy a scaling factor without checking the adapter's reduction semantics.
- Validate actor/reference forward-only paths, offload lifecycle, checkpointing, and full-policy weight export to the chosen rollout engine.
- Start with unpacked single-document batches; generic remove-padding/packing paths need separate document-isolation tests.

First reproduce the earlier deterministic forward/backward test through verl's engine. Next perform one tiny RL update with a simple deterministic reward, followed by rollout weight synchronization and fresh generation. Compare actor/rollout probabilities on identical tokens with matching temperature and distribution settings. Introduce GRPO/GSPO settings, reference/KL choice, and response-group size explicitly.

**Exit gate:** one complete, correct, resumable verl RL cycle. A standalone AutoModel SFT result does not satisfy this gate.

## Stage 5 — NL2SQL after the systems gates

Only now adapt the `consensus` data and reward logic to verl. Preserve a separate implementation boundary between the validated backend and SQL task code.

Use real schema/documentation contexts, executable SQL rewards, held-out databases, and bounded response lengths. Begin at short context, then extend to examples whose required evidence spans the long input. Many short documents packed into 40k tokens test execution capacity but do not establish reasoning over one 40k schema/context.

The earlier 40k goal is retained as a provisional benchmark target: **40,960 total forward tokens**. Also record prompt and response lengths separately. If the eventual requirement is 40,960 input tokens plus a response, test that larger total explicitly before claiming it fits.

Measure the entire RL iteration: rollout, actor/reference logprob passes, update, weight transfer, and checkpointing. Kernel speedups may be small in end-to-end RL if another phase dominates.

## Planned benchmark artifacts and acceptance record

The next coding pass should create these in order, sharing configuration and result formats:

| Artifact | Responsibility |
|---|---|
| Environment manifest | Exact dependencies/model revision and hardware inventory |
| Attention benchmark | Local/global forward parity, memory, and latency; backward added in Stage 3 |
| Model forward benchmark | No-cache forward first, cached generation separately |
| Two-node launcher | Parameterized allocation, topology/group logging, CP1/CP2 and Flex/FFPA selection |
| Training smoke benchmark | Backward, optimizer state, checkpointing, and resume |
| verl adapter/test | Same validated behavior inside the requested framework |

For every run, save resolved configuration, seed/input identity, batch and token counts, model/kernel revisions, actual kernel dispatch, cold/warm timing, median/p95, peak allocated/reserved/device memory, allocator retries, communication timing, numerical errors, and OOM/fallback status. Distributed metrics include every rank and the worst-rank summary.

Produce three first-stage plots: sequence length versus incremental memory, versus forward latency, and versus FFPA/Flex speedup. Add CP1/CP2 plots when the second GPU allocation exists. Keep measured results empty until execution; upstream numbers belong in source context only.

## Reference recipes and tests to reuse carefully

- [AutoModel FFPA CP1 8k recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/examples/vlm_finetune/gemma4/gemma4_31b_ffpa_mock_8k.yaml) and [packed FFPA CP8 16k recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/examples/vlm_finetune/gemma4/gemma4_31b_ffpa_mock_packing_cp8_16k.yaml): backend and distributed configuration references. Their synthetic VLM data, packing, and disabled checkpoint defaults need adaptation for our text tests.
- [FFPA parity tests](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/tests/functional_tests/models/gemma4/test_ffpa_cp_ring_parity.py): short reference checks, including CP1 ring forward/backward. CP1 coverage does not establish two-rank correctness.
- [Distributed FFPA/Flex throughput comparison](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/tests/functional_tests/models/gemma4/run_ffpa_vs_flex_throughput_cp.py): multi-rank attention comparison and worst-rank timing patterns. Validate its shapes, scenarios, and assertions for our CP2 tests before reuse.
- [Gemma long-context validation](https://github.com/NVIDIA-NeMo/Automodel/blob/65866fbce209938cd153f2bc26f01fa0d34cc61e/examples/long_context_validation/gemma4_31B/README.md): later long-context reference; its large-cluster SFT results do not establish our cluster's RL capacity.

The immediate next deliverable is the Stage 0/1 benchmark implementation and measured single-GPU results. The two-node interface should be designed alongside it; distributed claims wait for real multi-GPU execution.
