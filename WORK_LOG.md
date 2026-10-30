# Chronological work log: Gemma 4 NL2SQL training

This log records completed work, failed checks, decisions and pending work in
chronological order. New results are appended; a later passing result does not
retroactively validate an earlier experiment. Dates/times below are UTC unless
an older source explicitly uses America/New_York. Early entries are reconstructed
from Git history and the linked reports; current experiment times are artifact
completion times. Large logs, datasets, model weights and checkpoints remain local.

The objective is full-parameter Gemma 4 31B NL2SQL RL in verl, preserving Gemma's
exact dense global attention and sliding attention, with practical near-linear
attention memory and CP2. Dense global attention compute remains quadratic.
FFPA is optional; correctness takes priority over backend choice.

## 2026-09-28 — branch initialized

- Initialized the `verl` branch (`5ee6c6c`). The prior NL2SQL task recipe remains
  available on `consensus`; its existence is separate from a working verl port.

## 2026-10-08 — architecture and implementation planning

- Documented the full-parameter long-context RL objective, framework choices,
  Gemma context-parallel restrictions, and resource/recipe references.
- Investigated NeMo AutoModel rather than assuming a generic model could use
  arbitrary TP/CP layouts. CP counts GPU ranks, not nodes.
- Established incremental validation: kernels, model probabilities, distributed
  gradients, optimizer/checkpoints, rollout synchronization, then task training.
- Commits: `6b7ac51`, `732092c`, `1bbc72e`, `6101465`.
- Details: [long-context plan](GEMMA4_LONG_CONTEXT_RL.md).

## 2026-10-22 — attention benchmarks and numerical isolation

- Benchmarked the cached Gemma snapshot
  `842da3794eaa0b77d5f08bae87a17459d91ff475`. Early tests used the SM120
  environment, not the later eight-B200 allocation.
- Flex and FFPA attention working memory grew approximately linearly through
  40,960 tokens. FFPA was slower at short lengths and faster at long lengths
  in that environment; these are not B200 training-throughput claims.
- Default D512 Flex tiles exceeded shared-memory capacity. Explicit 32×32
  forward tiles and smaller backward tiles provided an executable baseline.
- Eager mask construction produced large quadratic temporaries: at 40,960
  tokens it peaked near 15.63 GiB, versus about 131 MiB for compiled construction.
  Kernel-only memory measurements therefore cannot establish end-to-end scaling.
- Full-model eager/SDPA/Flex outputs differed substantially even with identical
  weights and tokens. BF16 eager was not treated as a definitive oracle.
- Fixed-input FP32 decoder replay matched FP32 SDPA math exactly. Optimized
  BF16 layer outputs were comparable to BF16 eager. Forty-two causality checks
  and six sliding-window isolation checks passed; this did not certify CP or
  whole-model backward.
- Changing sequence/GEMM shape on one GPU reproduced important discrepancies;
  the first observed divergence was in layer-0 Q projection. cuBLASLt and
  disabling reduced-precision/split-K BF16 reductions mitigated some errors.
- Commits: `fbca939`, `c4f2a9f`.
- Evidence: [benchmarks](GEMMA4_BENCHMARK_RESULTS.md),
  [model parity](GEMMA4_MODEL_PARITY.md),
  [layer replay](GEMMA4_LAYER_REPLAY_RESULTS.md), `benchmark_results/`.

## 2026-10-23 — four-GPU CP experiments, consensus port and BIRD downloads

- Compared TP2/CP1 against TP2/CP2 with model-owned NeMo Gemma sharding.
  The 128-token probability check failed its maximum-token gate (0.356054);
  the 1,024-token case passed (0.069377). Top-1 agreement alone concealed
  differences relevant to RL probability ratios.
- Forward-only capacity with all-token vocabulary-sharded logits reached
  104,448 tokens for TP2/CP1 and 199,680 for TP2/CP2; the next 1,024-token
  increment OOMed. These are not training limits.
- Ported the consensus task into verl and moved the implementation under
  `src/nl2sql_verl`. Source provenance is retained under its `consensus/` directory.
- Established that the active consensus recipe uses GRPO group centering,
  batch-wide reward standard deviation and token-level policy ratios despite
  GSPO-related names. Sequence-level sampling correction is a different option.
- Retained three active tools: `bm25_search_sqlite`, `sqlite_peek`, `sqlite_query`.
  The optional `consensus_at_1` tool was not exposed by the active source trainer.
- Ported seven rewards with weights: format 0.2, execution 0.5, result 2.0,
  table linking 0.5, column linking 0.5, nonnull 0.1, length penalty 0.1.
  BIRD result matching retains the original raw-row set comparison semantics.
- Downloaded/extracted all 69 train and 11 dev SQLite databases, approximately
  33.90 GB, with archive CRC validation and read-only opening checks. Downloaded
  official train questions contain 9,428 rows; that is not the selected RL corpus.
  The official dev ZIP annotation date differs from consensus's corrected dev
  annotations, so annotation provenance must be explicit.
- Commits: `e95b4cf`, `43770a6`.
- Evidence: [recipe/task documentation](src/nl2sql_verl/README.md),
  [download manifest](benchmark_results/bird_data_download/manifest.json),
  `benchmark_results/tp2_capacity/`, `benchmark_results/verl_recipe/`.

## 2026-10-26 — B200 environment and integration investigation

- Continued in the eight-B200 environment. The later pinned stack is PyTorch
  2.10.0+cu130, Transformers 5.18.0, vLLM 0.19.1+cu130, verl
  `d430157d15516d89e0cc853fd10fcd21d9234c44`, NeMo AutoModel
  `65866fbce209938cd153f2bc26f01fa0d34cc61e`.
- Developed the B200 task/runtime checks and isolated rollout/refit failures.
  The first complete rollout/update cycle produced corrupted outputs and zero
  learning signal; successful process execution was not counted as learning.
- Kept workspace handoff archives ignored (`2590742`). Detailed B200 outcomes
  below were finalized during the following UTC day; older notes may date these
  sessions using America/New_York.

## 2026-10-27 — B200 pilots and the validated CP1 control

### Before 05:13 — integration fixes, FFPA diagnosis and pilot results

- vLLM Gemma4 level-2 sleep/refit corrupted output despite all 1,189 exported
  tensors matching the source weights. Fixed `v_norm.weight` constants were
  ordinary tensors outside parameters/buffers. Level-1 sleep/refit reproduced
  baseline outputs/logprobs; the recipe uses this validated workaround.
- The parser now accepts complete native tool calls and consensus-style bare
  `call:` lines, while preserving sampled IDs and rejecting incomplete calls.
- FFPA standalone forward/backward works on B200. A specific D512 SM100 ring
  backward shortcut zeroed a one-local-key row even when other CP ranks
  contributed keys to the global softmax. A diagnostic correction restored
  the missing boundary term and passed tested short CP2 checks. It was not
  installed as the production backend and does not validate packed FFPA.
- Flex full-model BF16 CP comparisons also failed, independently of FFPA.
  Promoting attention alone did not fix those full-model gradient differences.
- Full FP32 actor controls passed the selected-gradient gates, including a
  TP4/CP2 versus TP2/CP1 check at 128 and 1,025 tokens. This did not validate BF16.
- Mixed wheel CUDA 13.0 and system CUDA 13.2 BLAS libraries caused worker failures.
  `worker_python.sh` restores wheel-library precedence after Ray shell startup.
- A repaired four-GPU pilot completed two updates in 1,666.5 seconds. An eight-row
  diagnostic dev subset changed from 2/8 to 7/8 correct. This small subset and
  two updates do not establish general learning improvement.
- An eight-GPU FP32 TP4/CP2 pilot initially used 16 dev examples, then resumed
  with 32. It was stopped at the user's request after completed step 6, retaining
  checkpoints 5 and 6. Recorded training time was 3,753.8 seconds, not the full
  two-hour target. This pilot still used the old 8,192-token filter and small
  rollout batches; it is distinct from the proposed production recipe.

| Completed step | Correct / 32 | Dev mean reward | Train mean reward |
|---:|---:|---:|---:|
| 1, resumed baseline | 17 | 2.25720 | — |
| 2 | 19 | 2.57573 | 3.51488 |
| 3 | 22 | 2.94086 | 3.13984 |
| 4 | 21 | 2.74262 | 3.64028 |
| 5 | 20 | 2.70843 | 3.14673 |
| 6 | 21 | 2.82248 | 3.14375 |

The series fluctuates; training batches differ between steps. Evidence:
`outputs/b200_mini/`, `outputs/ffpa_backward_investigation/`,
`outputs/b200_two_hour/`, `outputs/b200_two_hour_eval32/stopped_summary.json`.

### 05:00–05:13 — CP1 recipe validation completed

- Selected all 6,601 filtered-corpus examples; excluded the 9,428-row corpus.
  Removed dataset length filtering, including the previous 383 exclusions.
- Measured prompts are 4,402–10,786 tokens with the actual tool template.
  Set allowances to 11,264 prompt + 2,048 response = 13,312 total tokens.
- Selected a fixed 640-row dev subset across all eleven dev databases, containing
  the prior 32. Eval is greedy, one generation per prompt.
- Agreed production batch: 32 accepted prompt groups × 16 generations = 512
  trajectories, GRPO, native verl V1 DAPO result filtering, LR 1e-6,
  clipping 0.2/0.28, one PPO pass, no KL, save every five updates, eval every ten,
  retain the latest two completed full checkpoints. Default length: 100 updates.
- BF16 TP4 actor microbatch checks failed. TP8 is unsupported by the pinned
  adapter because global layers have four KV heads. TP2/DP4/CP1 passed;
  removing DP alone had not resolved the numerical issue.
- TP2/DP4/CP1 full-model microbatch checks: worst selected-gradient relative L2
  0.267% on BIRD and 0.381% on synthetic data. The independent HF BF16
  microbatch-consistency control was 0.288%; it was not a cross-implementation
  equality claim.
- A real 13,312-token Adam update peaked at 117.3 GiB allocated / 164.7 GiB
  reserved per GPU. Compute is BF16; master weights/Adam/reductions are FP32.
- The base-model smoke completed 32 accepted trajectories, native DAPO,
  tool rewards, one optimizer update, rollout refresh, a ~346-GiB full checkpoint
  and 640-row evaluation. Reward 3.11745, grad norm 3.00863, dev 422/640
  (65.9375%), exit 0, 847.5 seconds. One update is not a learning trend.
- CPU tests: 31; native DAPO tests: 12. Committed as `d31b350`.
- This is a validated CP1 control. The 100-update production job was not launched.
- Evidence: [CP1 validation summary](benchmark_results/verl_cp1/validation_summary.json),
  `outputs/cp1_recipe/`, [CP1 recipe](src/nl2sql_verl/configs/cp1.yaml).

### 05:23 onward — native verl FSDP2 and Ulysses CP investigation

The requested four stages are: native FSDP2 CP1 control; CP2 gradient checks;
warm attention/full-model memory scaling; real optimizer/save/restore and
GRPO/DAPO integration. The candidate actor topology is TP1 × CP2 × DP4 on eight
GPUs. Rollouts independently use TP4 × DP2 on the same allocation in alternating
phases. There are four CP pairs; each pair shares one sequence partition.

- Stock Gemma initialization failed because generic verl patching queried an
  ambiguous global KV-head attribute. Gemma has heterogeneous global/sliding
  head counts. A repository-local compatibility guard bypasses that unsuitable
  patch only for supported Gemma paths and validates each layer's divisibility.
- The attention adapter uses upstream HF Flex attention/mask construction plus
  verl's differentiable Ulysses collectives. It does not use NeMo TP projections
  or ring-softmax backward. Installed libraries are not edited.
- Masks preserve causal, sliding-window, document and suffix-padding semantics.
  CP slices inputs before embedding; next-token labels are shifted globally.
- Gradient probes execute the full 31B model, comparing 20 selected parameter
  tensors across layers 0, 1, 5 and 59. They are not all-parameter gradient audits.
  Short fixtures are SQL-context tails; full BIRD fixtures contain 5,088 and
  5,252 tokens. Gates remain relative L2 <3%, cosine >0.999, loss delta <0.01,
  maximum response-token logprob delta <0.1.

| UTC completion | Experiment | Result / limitation |
|---|---|---|
| 05:38 | Native FSDP2 CP1, BF16 SDPA math versus independent HF | Exact loss/logprobs; worst selected gradient 0.181%; pass |
| 05:41 | Initial custom Flex CP1 versus HF SDPA math | 89.35% gradient difference, max logprob 3.96; fail |
| 05:47 | FP32-attention-only variant versus HF SDPA math | 295.17% gradient difference; fail |
| 05:49–05:52 | Captured pretrained Q/K/V and upstream HF Flex control | Standalone attention is close to FP32 math; upstream HF Flex also differs from HF SDPA math in full-model BF16. This is not proof that either full-model gradient is correct |
| 05:57 | HF wrapper and HF masks, native FSDP2 CP1 versus HF Flex | Exact loss/logprobs; worst gradient 0.205%; pass |
| 06:00 | Same path, CP2 | Exact loss/logprobs; worst gradient 3.63%; fail |
| 06:08 | Matched 256-token padding buckets, CP2 | Exact loss/logprobs; worst gradient 4.49%; fail |
| 06:12 | Matched FP32 attention on both sides, otherwise BF16 | Exact loss/logprobs; worst gradient 3.23%; fail |
| 06:16 | FP32-weight/residual + BF16-autocast reference | Required Q/K/V dtype alignment: RoPE made Q/K FP32 while V remained BF16 |
| 06:18 | That AMP path, CP2, short fixtures | Exact loss/logprobs; worst gradient 1.35%; pass only on these short fixtures |
| 06:23 | AMP CP1, complete BIRD examples | Exact loss/logprobs; worst gradient ~0.000007%; pass |
| 06:26 | AMP CP2, two packed complete examples | Worst gradient 29.03%, max logprob 0.375; fail |
| 06:30 | AMP CP2, complete examples processed separately | Exact loss/logprobs but worst gradient 8.60%; fail; packing is not the only issue |
| 06:33 | FP32 matmul outputs with lower-precision Tensor Core arithmetic, BF16 Flex | Independent HF reference completed; CP2 comparison pending at this entry |

Additional findings and implementation corrections:

- Early harness attempts accidentally cleared accumulated gradients on context
  exit and mismatched checkpoint-wrapper parameter names. Those attempts are
  invalid evidence; corrected comparisons above preserve accumulation and
  normalize names. Gates were not loosened.
- Identical Q/K/V with the same HF mask give identical wrapper output. Equivalent
  mask representations can still change a BF16 output by one ULP. Full-model
  amplification remains a numerical investigation, not a proven complete cause.
- Captured real-Q/K/V CP2 backward checks passed the 3% operator gate (worst BF16
  1.247%, FP32-attention 0.167%) on tested odd-length padded fixtures. Operator
  correctness alone did not predict full-model parity.
- A single-GPU FP32-weight reference initially OOMed because autocast retained
  BF16 weight casts. Disabling autocast caching and releasing parameter gradients
  after accumulation allowed the independent full-model reference to fit.
- The FP32-attention-only 32K memory sweep was stopped after short cases because
  that candidate failed the numerical gate and was substantially slower.
- PyTorch `float32_matmul_precision='medium'` was profiled on this B200 stack:
  the observed forward/backward GEMM kernels were **TF32 Tensor Core kernels**
  with FP32 outputs. Do not label this diagnostic BF16 linear compute.
- CPU suite reached 33 passing tests. The native checkpoint-completion adapter
  requires every rank's model, optimizer and extra-state files plus trainer data
  before publishing/pruning; committed as `eddd66f` at 06:22.

The BF16 Flex/Ulysses attention sweep, including mask-construction peaks, passed
with finite forward/backward gradients. Geometry is 32 Q heads, 4 KV heads,
D512; these are attention-only peaks, not full-model/Adam residency:

| Global tokens | CP1 peak MiB/GPU | CP2 peak MiB/GPU |
|---:|---:|---:|
| 4,096 | 1,696.57 | 928.32 |
| 8,192 | 3,393.19 | 1,856.69 |
| 16,384 | 6,786.63 | 3,713.63 |
| 32,768 | 13,574.26 | 7,428.26 |

This supports practical near-linear memory over 4K–32K. Compact mask metadata
still has a quadratic component; dense attention compute remains quadratic.
Evidence: `outputs/fsdp_cp_recipe/`; compact tracked observations are under
`benchmark_results/verl_fsdp_cp/` as results are recorded.

### 06:35 onward — SSH publishing restored

- HTTPS pushes failed because this environment lacked Git HTTPS credentials;
  the connected GitHub integration had also rejected repository writes.
- Found an existing SSH key already authorized for `kpratik41`. Verified the
  server key against GitHub's published Ed25519 host key, configured this
  repository's SSH command, and changed `origin` to the SSH URL.
- Successfully pushed `2590742..eddd66f` to `origin/verl`. Local commits were no
  longer stranded. New reviewed work continues to be committed and pushed.
- Generated a separate Ed25519 pair as requested. Public key:
  `/home/ubuntu/.ssh/id_ed25519_nl2sql_github.pub`; private counterpart stays
  outside Git. The new key is not yet registered on GitHub; current pushes use
  the already-authorized `/home/ubuntu/.ssh/id_ed25519_github`.

### Pending at this entry

1. Finish the TF32/FP32-output CP2 comparison; repeat CP1 and packing/accumulation
   controls for any precision policy selected for production.
2. Measure warm full-model forward/backward/Adam memory at 4K, 8K, 16K, 32K for
   CP1/CP2. Test the production 13,312-token boundary and proposed microbatch.
3. Execute a full native FSDP checkpoint round trip with exact checks of all
   model/Adam tensors, scheduler and CPU/CUDA RNG, after deliberate perturbation.
4. Run two actual GRPO/DAPO updates with tools, reward components, rollout weight
   refresh, complete checkpoints and 640-row baseline/final validation.
5. Finalize the native CP2 precision/microbatch recipe only after these gates.
   The new launcher is fail-closed; no native-FSDP sustained RL job has started.
6. Keep recording results and pending items here, committing reviewed code and
   compact evidence. Leave unrelated local TP-backward diagnostics and archive
   files out of these commits.

### 06:37 — FP32-output/TF32 packed comparison

- The CP2 two-example packed test completed. Maximum response logprob delta was
  0.0851 and loss delta 0.00328, but the worst selected gradient difference was
  100.17% with poor early-layer cosine similarity. It fails the gradient gate.
- The same precision policy is being tested with examples processed separately
  to isolate packing. No production readiness has been declared.
- A repeated SSH push succeeded (`Everything up-to-date`); the chronological log
  and compact experiment evidence are now being committed alongside the gated
  experimental FSDP/Ulysses code. Optimizer/memory/restore/RL gates remain pending.

### 06:43–06:46 — single-trajectory TF32 gate passed; capacity testing started

- The TF32/FP32-output CP2 test passed on the complete 5,088/5,252-token BIRD
  examples when processed separately: exact scalar loss, maximum response-token
  logprob difference 1.43e-6, worst selected gradient relative L2 2.5876%.
- The candidate config now explicitly selects this TF32 policy, BF16 Flex
  attention, and microbatch one. This is not BF16 linear compute. Two packed
  trajectories remain disabled because they failed the gradient check.
- With CP2/DP4 and 512 accepted trajectories, microbatch one implies 128
  accumulation rounds per optimizer update. Each CP pair shares one trajectory
  per microbatch; these are not 128 distinct prompt groups per GPU.
- Started full-model CP2 memory/Adam sweeps at 4K/8K/16K/32K. Additional
  longest-prompt parity, CP1 control, restore and RL integration remain pending.
- The longest measured train prompts are rows 4837 and 2737, at 10,786 and
  10,774 prompt tokens. They are retained in the training set.

### 06:48–06:50 — harness correction and tiny restore gate

- The first full-model memory sweep failed before its first optimizer update:
  the diagnostic loss callback omitted a required `default` argument in native
  verl's metadata accessor. This was a harness/API error, not a capacity result.
  The failed attempt is retained as `full_memory_tf32_sp2_attempt0.log`.
- Corrected the callback and added a CPU regression for its native metadata and
  global-token normalization. The CPU suite now has 34 passing tests.
- An eight-rank tiny Gemma test passed two accumulation rounds, two Adam updates,
  full checkpoint save/load, exact comparison of every model/Adam tensor,
  scheduler state and CPU/CUDA RNG on all eight ranks. This verifies the harness;
  the full 31B round trip is still pending.
- Restarted the full 31B CP2 4K/8K/16K/32K sweep. The production candidate uses
  zero weight decay, LR 1e-6, constant scheduling, and FP32 gradient reductions.
- Candidate generation batches now contain two prompt groups, with two native
  DAPO batches allowed in flight. That permits up to four prompt groups / 64
  sampled trajectories in flight, subject to agent and serving scheduling.
  It does not change the 32 accepted groups / 512 trajectories per update.

### 06:53 — CP2 full-model memory sweep completed

Eight actual synthetic Adam updates completed on the full 31B model (cold/warm
updates at each length), with finite gradients on all data-parallel replicas.
Warm worst-rank measurements for microbatch one:

| Global tokens | Peak allocated GiB/GPU | Peak reserved GiB/GPU | Update seconds |
|---:|---:|---:|---:|
| 4,096 | 79.76 | 107.12 | 2.88 |
| 8,192 | 79.76 | 117.37 | 5.59 |
| 16,384 | 86.08 | 127.67 | 12.63 |
| 32,768 | 128.84 | 174.71 | 36.99 |

These include model, gradients and Adam state. They are not RL-step timings
(rollout, SQL tools, logprob recomputation, checkpoint I/O and evaluation add
work). The 32K reserved peak is close to the GPU limit; production remains at
13,312 total tokens. The candidate's longest-real-prompt parity, CP1 sweep,
full-model checkpoint round trip and real RL integration remain pending.

### 06:55–06:57 — production attention I/O and longest-reference capacity

- The attention sweep with FP32 Q/K/V inputs and FP32 outputs around the BF16
  Flex kernel passed, including mask construction. Peak allocated MiB at
  4K/8K/16K/32K: CP1 = 1888.57/3777.19/7554.63/15110.26;
  CP2 = 944.32/1888.69/3777.63/7556.26. All gradients were finite.
- The longest-prompt independent HF reference OOMed during backward on one
  B200. Restarted that reference with saved-tensor CPU offload, preserving its
  arithmetic. This is a reference-only capacity workaround; the distributed
  production actor does not offload its parameters or optimizer.
- The new dedicated SSH key now authenticates successfully as `kpratik41`.
  Updated this repository's SSH command to use
  `/home/ubuntu/.ssh/id_ed25519_nl2sql_github`; pushes no longer depend on the
  older key. Neither private key is committed.

### 07:00 — longest-prompt CP2 backward still outside the gate

- CP2 on the two longest-prompt fixtures (10,884 and 10,844 total tokens)
  matched the reference loss within 8.94e-8 and response logprobs within 1.08e-6.
  Worst selected-gradient relative L2 was 4.7774%, above the unchanged 3% gate.
- Therefore the TF32 candidate is not numerically ready at the required length,
  despite passing the roughly 5K-token examples and the capacity/optimizer sweep.
- Testing a specific layout hypothesis: Ulysses makes Q/K/V and the Flex output
  gradient contiguous in BHSD layout, while unsharded HF can pass strided
  tensors. The diagnostic normalizes layouts on the HF reference, preserving
  tensor values and attention equations, and compares with saved CP2 gradients.
  This is a hypothesis under test, not a confirmed root cause.

### 07:04–07:12 — layout hypothesis rejected; attention precision isolated

- Canonical contiguous attention inputs/output-gradient layouts did not fix the
  longest-prompt discrepancy (about 5.10% relative L2 against saved CP2 gradients).
  The layout normalization remains an opt-in diagnostic, disabled in the recipe.
- Started testing FP32 attention inputs, outputs and intermediate state together
  with FP32-output linear layers. Linear arithmetic still uses TF32 Tensor Cores.
- A 4K standalone attention forward/backward benchmark (16 Q / 2 KV heads,
  D512) found large precision-setting differences: warm `tf32x3` ~1.18 s at
  tile 16, `tf32` ~0.054 s at tile 16 / ~0.052 s at tile 32, and IEEE ~0.267 s
  at tile 16. All tested variants produced finite gradients. These are kernel
  timings, not full-model throughput or accuracy acceptance.
- The faster all-TF32/FP32-output HF reference completed on both longest-prompt
  fixtures in 61.4 s, peaking at 160.1 GiB on one B200 with CPU saved-tensor
  offload. A CP2/DP1 two-GPU diagnostic is comparing with it while the slower
  `tf32x3` reference finishes independently on GPU 0. The final eight-GPU
  CP2/DP4 topology must still pass before training.

### 07:13–07:20 — longest-prompt precision improvement and publishing confirmed

- FP32 attention tensors/intermediate state with TF32 arithmetic passed the
  full 31B CP2/DP1 comparison on 10,884/10,844-token real BIRD fixtures. Worst
  of 20 selected gradient tensors: 0.3967% relative L2; loss difference 2.98e-8;
  largest response logprob difference 1.095e-6. This retains the existing 3%
  gradient gate. It does not validate BF16 attention or every parameter gradient.
- The independent HF reference using higher-accuracy `tf32x3` attention also
  finished (397.4 s versus 61.4 s for `tf32`). Both use FP32-output linear layers.
- Started the eight-GPU CP2/DP4 longest-fixture comparison followed by real Adam
  updates at 4K/8K/16K/32K. Added an explicit attention precision setting to the
  gated recipe and source/argument metadata to subsequent probe launches.
- CPU regression suite: 34 passed. Fixed attention-memory artifact names to
  distinguish FP32/TF32 from earlier BF16-kernel measurements. Full-model restore
  and the real two-update RL integration remain pending.
- Retried `git push origin verl` with the dedicated new SSH key: successful,
  remote already current through `7493892`.

### 07:21 — cross-precision reference check blocks the fast attention choice

- Comparing the saved independent HF `tf32` attention run against the completed
  higher-accuracy `tf32x3` attention reference revealed substantial disagreement:
  maximum response logprob difference 0.3534; worst selected-gradient relative
  L2 15.83 (1,583%). Loss difference alone was only 0.000414 and hid the issue.
- Matching CP and non-CP arithmetic is therefore insufficient evidence for
  selecting fast TF32 attention. The current TF32 attention config remains
  gated/experimental. Next: test IEEE FP32 attention against the higher-accuracy
  reference; retain the completed capacity measurements as diagnostics.

### 07:25 — fast attention capacity complete; accurate attention tests started

- Eight-GPU CP2/DP4 longest-fixture parity with matching TF32 arithmetic passed
  (worst selected-gradient relative L2 0.3967%, response logprobs 1.095e-6).
- Full-model Adam updates completed twice each at 4K, 8K, 16K and 32K using
  FP32 attention/TF32 arithmetic. This confirms capacity for that policy only;
  its cross-precision discrepancy still blocks training.
- Started IEEE FP32 attention HF references on short, full ~5K and longest ~11K
  BIRD fixtures; a two-GPU CP2 comparison against the higher-accuracy `tf32x3`
  reference; and standalone CP1/CP2 attention-memory sweeps through 32K.
- Committed and pushed `a386bf5` with diagnostic harnesses, explicit candidate
  precision configuration and the failed cross-precision evidence.

### 07:29–07:31 — IEEE attention comparison and near-linear attention memory

- The FP32-output/IEEE-attention CP2 run agrees with an independent HF run using
  IEEE attention: loss difference 2.98e-8, largest response logprob difference
  1.431e-6, worst selected-gradient relative L2 0.4246%. This was CP2/DP1; the
  final eight-GPU layout still needs validation for this arithmetic.
- IEEE versus `tf32x3` attention also showed substantial full-model gradient
  disagreement (4.230 relative L2), despite response logprobs within 0.0422.
  Therefore `tf32x3` is not treated as an exact FP32 oracle. IEEE attention is
  the current reference; attention Tensor Core approximations remain excluded.
  Linear layers still use the separately recorded TF32 or BF16 policy.
- Warm IEEE attention peaks including mask construction at 4K/8K/16K/32K:
  CP1 1856.57/3713.19/7426.63/14854.26 MiB;
  CP2 1088.32/2176.69/4353.63/8708.26 MiB. All gradients finite, approximately
  linear memory over the tested range. Timings remain quadratic: CP2
  0.270/0.981/3.862/15.367 seconds for one global attention forward/backward.
- Testing BF16 linear compute with FP32 residuals and IEEE attention as a speed
  option, plus packed microbatch 2 with FP32-output linears. Neither is approved
  until its full-model comparisons pass.

### 07:32 — BF16 linear option fails the unchanged gradient gate

- BF16 linears with FP32 residuals and IEEE attention matched longest-fixture
  loss and response logprobs exactly across HF and CP2, but worst selected
  gradient relative L2 was 3.26696%, above 3%. Rejected without relaxing the gate.
- Cancelled its remaining ~5K reference/microbatch-2 follow-ups after this
  disqualifying result. Those cancelled jobs are not numerical failures.
- Continue with FP32-output linears and IEEE attention; packed microbatch 2
  is still being checked for that policy.

### 07:33 — packed microbatch 2 rejected; final topology validation starts

- FP32-output linears/IEEE attention with two packed ~5K BIRD trajectories
  disagreed with the independent per-trajectory HF reference: largest response
  logprob difference 0.05446 and worst selected-gradient relative L2 28.1748.
  Forward loss difference alone was only 0.000506. Microbatch remains 1.
- Started the eight-GPU CP2/DP4 longest-fixture comparison using IEEE attention,
  followed by full-model Adam updates at 4K/8K/16K/32K. The candidate uses
  FP32 parameters/residuals/linear outputs, TF32 linear arithmetic, IEEE attention,
  FP32 gradient reduction and microbatch 1. It retains 512 trajectories per
  production update: 128 accumulation rounds per CP pair, four DP replicas.

### 07:36–07:37 — final eight-GPU CP2 numerical check passes

- IEEE-attention, FP32-output linears on TP1/CP2/DP4 passed the full 31B
  longest-real-prompt comparison: selected-gradient worst relative L2 0.42461%,
  response logprobs within 1.431e-6, loss within 2.98e-8. All 20 selected
  gradient tensors meet the unchanged relative-L2/cosine/finite gates.
- The same initialized model continues with the final-policy Adam/memory sweep.
  CP1 control, 13,312-token accumulation/full-model restore and real two-update
  GRPO/DAPO/tool integration with 640-row evaluation remain pending.
- CPU regression suite rerun for the IEEE recipe setting: 34 passed. Updated
  readiness checks to require IEEE-policy, eight-GPU evidence and matching
  engine/attention/runtime source hashes. No sustained training launched yet.

### 07:49 — final-policy CP2 full-model capacity complete

- IEEE attention / FP32-output linears completed two actual Adam updates at
  each of 4K, 8K, 16K and 32K on all eight B200s (CP2/DP4, microbatch 1).
  All reported source-rank gradient norms were finite and positive.
- Started short and full ~5K SQL comparisons using the same initialized model,
  followed by two 13,312-token updates with two accumulation rounds, then an
  exact full-model/Adam/scheduler/CPU+CUDA RNG checkpoint round trip on NVMe.
- The remaining separate control is CP1 numerical/capacity validation. Real
  GRPO/DAPO/tool integration remains gated on these checks.

Final-policy CP2 warm update maxima across all ranks:

| Tokens | Allocated GiB | Reserved GiB | Seconds |
|---|---:|---:|---:|
| 4,096 | 79.76 | 105.42 | 8.89 |
| 8,192 | 79.76 | 117.28 | 22.35 |
| 16,384 | 86.08 | 128.01 | 67.62 |
| 32,768 | 128.84 | 174.14 | 231.86 |

These are synthetic full-model optimizer updates, not complete RL steps; they
exclude rollout, SQL tools, evaluation and checkpoint I/O. IEEE attention trades
throughput for the tested numerical behavior. Production throughput must be
measured in the integration run.

### 07:52–08:00 — shape-history failure isolated; true FP32 baseline added

- The combined short→~5K CP2 probe failed on the ~5K fixture: worst gradient
  relative L2 5.341, maximum response logprob difference 0.17014. It stopped
  before accumulated optimizer updates or checkpoint writing.
- A fresh CP2 process passed that same fixture (0.2608% gradient relative L2),
  and a fresh HF rerun was bit-identical. The discrepancy affected the first
  5,088-token example; the second 5,252-token example matched within 1.91e-6.
  This implicates shape history, not a universally failing length.
- Testing automatic dynamic-shape compilation explicitly, including HF
  short→full and static-shape variants. A related upstream report exists at
  https://github.com/pytorch/pytorch/issues/186775, but it reports NaNs under
  different mask conditions; it is not proof that our failure is the same bug.
- Added a true IEEE FP32 linear-arithmetic control (TF32 disabled, standard
  cuBLAS), retaining IEEE attention. The fresh full-model CP2 comparison on
  ~5K fixtures passed at 3.296e-6 relative L2 (0.000330%), much tighter than
  the TF32-linear result; response logprobs within 1.91e-6. Runtime was not
  worse in these diagnostic runs. Static true-FP32 short/full/long HF
  references are now running. All earlier TF32-policy capacity results remain
  diagnostic evidence; they do not override this new shape-history failure.

### 08:01–08:07 — HF reproduces shape-history drift; static policy established

- Unsharded HF also changes on short→full shape history: the same ~5K example
  shifts response logprobs by 0.170145, matching the native CP2 drift. This
  failure does not require context parallelism or FSDP.
- Disabling automatic dynamic shapes makes fresh and warmed HF forward results
  identical. Their selected-gradient difference is 0.2044% with TF32 linears.
  CP2 against the matching static HF reference passes at 0.2210% gradient
  relative L2 and 1.073e-6 logprob difference. Comparisons of static runs against
  older dynamic references are retained as failures, not silently overwritten.
- Current candidate selects IEEE FP32 linears (standard cuBLAS), IEEE attention
  and static compilation explicitly in config. True-FP32 HF references are
  complete for short/~5K/~11K contexts; longest reference took 224.8 seconds.
- Added shape-history sequences and sequential memory/restore exercises to the
  probe harness, so reuse is tested before any optimizer step. CPU suite remains
  34 passed. Eight-GPU checks and final-policy capacity/restore/integration are
  still required; earlier TF32 passes do not certify the new candidate.

### 08:09 — static FP32 attention memory complete; eight-GPU suite launched

- Static IEEE attention passed CP1 and CP2 4K–32K memory sweeps, including mask
  construction and finite backward gradients. Peaks remain approximately linear
  and match the earlier IEEE attention measurements.
- Launched full31B TP1/CP2/DP4 static IEEE FP32 validation on all eight B200s:
  short→full→long→full comparisons against independent static HF references;
  then 13,312-token accumulated updates/full-state restore; then the final-policy
  4K–32K Adam/memory sweep. Each stage stops on a failed numerical assertion.
- GitHub is current through `c6ccce3`. No full RL training has been launched.

### 08:18 — eight-GPU static true-FP32 numerical and shape-history gates pass

- Full 31B, TP1/CP2/DP4, IEEE linear and attention arithmetic, static compilation:
  worst selected-gradient relative L2 was 0.002158% (short), 0.000413% (~5K),
  0.003361% (~11K), and 0.000413% (repeated ~5K after ~11K).
- The two ~5K runs have bit-identical response logprobs and every one of the
  20 selected gradient tensors. All four comparisons pass the unchanged
  independent-HF gates. This resolves the observed shape-history drift for
  these tested fixtures and policy; it does not certify every possible input.
- Accumulated 13,312-token updates are now running, before full-state save/load
  and the final-policy capacity sweep. CP1 control and real RL integration
  remain pending. The recipe remains fail-closed until those checks complete.

### 08:24 — full 31B checkpoint round trip passes on all eight ranks

- Two 13,312-token Adam updates with two accumulation rounds completed. Warm
  update: 141.0 seconds, maximum 93.12 GiB allocated / 118.89 GiB reserved.
- Saved full model/optimizer/scheduler/RNG state to
  `/opt/dlami/nvme/nl2sql/fsdp-static-fp32-full-restore`, deliberately changed
  every parameter and optimizer tensor and advanced scheduler/RNG, then loaded.
- All eight ranks passed exact comparisons of 1,126 model parameter tensors,
  2,316 Adam-state tensors, scheduler state and CPU/CUDA RNG. This is a full
  model/state restoration check, unlike the selected-gradient numerical checks.
- The same process continues with the final-policy 4K–32K capacity sweep.
  CP1 control and the real two-update RL/640-row evaluation test remain pending.

### 08:39 — bounded pilot launch option prepared

- Added an optional `run_fsdp pilot` mode without changing the regular `train`
  profile (100 updates, checkpoint every 5, eval every 10). Pilot keeps the
  full 512-trajectory batch and 16 generations per prompt.
- Pilot targets two hours after initial validation, checking at completed-update
  boundaries. It saves/evaluates every update to satisfy the already-tested
  bounded-runtime contract, retains the latest two complete checkpoints, and
  has a three-hour total-job watchdog. The watchdog can interrupt an unfinished
  update; status records that reason and the last complete checkpoint remains.
- Both pilot and train require completed numerical/memory/restore and real RL
  integration evidence. No pilot or sustained training is launched yet.
- CPU suite: 35 passed, including preservation of the pilot's full batch and
  checkpoint/evaluation contract.

### 08:43 — CP2 validation suite complete; CP1 control starts

- The static true-FP32 CP2 suite completed successfully: four numerical
  comparisons including shape-history reuse, two accumulated 13,312-token
  optimizer updates, full31B state restoration, and two Adam updates each at
  4K/8K/16K/32K. All eight ranks completed the final 32K test.
- Started the matching eight-GPU CP1 full-context numerical comparison and
  capacity sweep. This control will establish the actual non-CP capacity bound
  rather than assuming it can fit 32K. Real RL integration follows these checks.

Final static true-FP32 CP2 warm update maxima across ranks:

| Tokens | Allocated GiB | Reserved GiB | Seconds |
|---|---:|---:|---:|
| 4,096 | 79.76 | 101.71 | 16.47 |
| 8,192 | 79.76 | 112.37 | 37.21 |
| 16,384 | 86.07 | 127.62 | 96.60 |
| 32,768 | 128.83 | 174.03 | 297.69 |

The pilot supervisor now grants 60 seconds after its watchdog sends SIGTERM,
then kills the owned process group if needed. This is separate from the normal
cooperative stop after a completed update.

### 09:01 — CP1 capacity boundary recorded; real RL integration launched

- CP1 full-model numerical check passed. Two Adam updates each passed at 4K,
  8K and 16K. At 32K all eight ranks OOMed requesting another 32 GiB with
  about 18 GiB free (already about 150.35 GiB allocated by PyTorch).
- Readiness records CP1's measured maximum successful length as 16K and its
  failed 32K attempt separately; it does not report a 32K CP1 pass. CP2 passed
  32K and is the deployment topology. No numerical thresholds were relaxed.
- The fail-closed validator passed numerical, attention-memory, full-model
  capacity, accumulated-update, full-state restore, CPU and data-hash checks.
- Launched the native FSDP2/CP2 real integration service: 2 updates, 2 accepted
  groups × 16 generations = 32 trajectories/update, native DAPO filtering,
  SQL tools/rewards, checkpoint after each update, 640-row evaluation before
  training and after update 2. Production remains 512 trajectories/update.

### 09:03 — native DAPO dispatch semantics clarified during integration

- The running V1 trainer explicitly overrides `data.gen_batch_size=2` to 1 for
  exact DAPO refills. Set the checked-in value to 1 to express the actual behavior;
  the running integration already uses 1, so no behavioral restart is needed.
- Correction to earlier concurrency descriptions: `max_inflight_gen_batches`
  multiplies **train_batch_size**, not gen_batch_size. With value 2 the
  production pending/running cap is 64 prompt groups (32 × 2), versus 4 for
  the two-group integration test. This includes queued requests; vLLM's
  32-sequence limit per TP4 replica bounds active GPU sequences.
- Each rejected result-constant group grants credit for two replacement prompts.
  Dispatch respects the in-flight cap, then drains outstanding work and discards
  surplus once enough accepted groups are available. There is no total retry
  limit enforced by `max_num_gen_batches` in this pinned V1 implementation.

### 09:14 — 640-example baseline complete; first actor update in progress

- Native FSDP2/CP2 integration baseline: 337/640 correct (52.65625%),
  mean composite reward 2.22944, format success 72.34375%, execution
  success 72.5%. All 640 trajectories completed without scheduler failures.
- The historical CP1 result 422/640 was measured **after one optimizer
  update**, with no pre-training evaluation. The input sets are identical,
  but this is not a paired baseline comparison or evidence that CP caused
  the difference. Final evaluation in this run will provide its own paired
  before/after result.
- First actor processing is active on all eight GPUs. No completed real RL
  optimizer update or integration checkpoint has been claimed yet.

### 09:17 — first real GRPO update and complete checkpoint pass

- Step 1 used 32 accepted trajectories, mean composite reward 3.00859,
  result success 81.25%, finite pre-clipping gradient norm 8.71929.
  Training reward is measured on DAPO-selected groups, not the dev set.
- Timings: generation 31.9 s, old log probabilities 106.5 s, actor update
  216.9 s, checkpoint 48.9 s, rollout refresh 4.35 s; total 408.8 s.
  This small-batch timing is not a measured 512-trajectory throughput result.
- The first 345.52-GiB checkpoint has eight model, eight optimizer and eight
  extra-state shards plus trainer data, its completion marker and the latest
  pointer. The full state is on local NVMe.
- Actor/rollout mean absolute logprob difference 0.006705, maximum 0.77814;
  reported rollout-correction KL 0.004394. Actor remains IEEE FP32 and
  serving BF16. No importance-sampling correction was enabled.
- Second update and final 640-example validation remain pending.

### 09:36 — all four validation stages pass; final paired dev result

- Native two-update integration exited 0 after 2,040 seconds. Both updates
  processed 32 trajectories and published complete full checkpoints. Step 2
  gradient norm was 10.04689; selected training reward was 1.69178 versus
  3.00859 at step 1. These batches contain different DAPO-selected prompts.
- On the same 640 dev inputs, execution-result accuracy rose from **337/640
  (52.65625%) to 439/640 (68.59375%)**. Composite dev reward rose from
  **2.22944 to 2.91306**. Paired outcomes: 121 newly correct, 19 newly wrong,
  318 correct both times and 182 wrong both times. Two small updates are
  encouraging integration evidence, not a sustained learning trend.
- Step 2 native DAPO metrics: six result-constant groups rejected, two surplus
  accepted groups discarded. Generation took 368.3 s, old log probabilities
  106.1 s, actor update 254.4 s, checkpoint 50.3 s, refresh 5.38 s. The step
  took 784.6 s excluding its 296.7-s final evaluation.
- Step 2 actor/rollout mean absolute logprob difference was 0.02378, maximum
  19.022, reported KL 0.01271. The average remained modest but the tail is
  a pending monitoring concern; no sampling correction was silently added.
- A DataLoader worker emitted an ignored `atexit` cleanup exception after
  final validation and metrics had been written. Supervisor exit was 0,
  checkpoints were complete and GPU workers exited. This is recorded as
  cleanup noise, not a failed optimizer/evaluation step.
- Re-ran the fail-closed validator: numerical/memory and integration readiness
  are both true. The CPU suite has 35 passing checks. All four requested
  validation stages are complete for static IEEE FP32, microbatch 1, CP2.
- Remaining: measure actual 512-trajectory throughput and reward progression
  in the bounded pilot; BF16/TF32 and microbatch-2 performance optimization
  remain unapproved because their earlier numerical tests failed.

### 09:37 — bounded full-batch pilot launched

- Started `nl2sql-fsdp-cp2-pilot.service` after all readiness gates passed.
  Supervisor PID 877614; training launcher PID 877615. Initial state is
  initialization, followed by 640-example baseline evaluation.
- Uses all eight B200s, actor TP1/CP2/DP4 with native FSDP2, static IEEE
  FP32, microbatch 1 and 128 accumulation rounds per update. Production
  batch is 32 accepted prompts × 16 generations = 512 trajectories.
  Rollout remains BF16 TP4/DP2, with native DAPO result filtering.
- This is a fresh base-model run, separate from the two-update smoke.
  Targets two hours after baseline evaluation, stopping at an update
  boundary, with a three-hour total-job watchdog. Full checkpoint and
  fixed-640 evaluation every completed update; retain two checkpoints.
- Logs/metrics/status: `outputs/fsdp_cp_recipe/pilot/`. Full checkpoints:
  `/opt/dlami/nvme/nl2sql/fsdp-cp2-pilot`. Actual full-batch throughput and
  reward progression are pending. The regular train profile remains
  100 updates with save every 5 and evaluation every 10.


### 2026-10-29 UTC — two-node H200 setup and model-only cadence

- Staged identical Python/CUDA runtimes, pinned verl, Gemma snapshot, BIRD SQLite
  databases and prepared parquet data on node0/node1 NVMe. Dedicated Ray cluster
  has 16 H200 GPUs. Sixteen-rank NCCL all-reduce/all-to-all and node-local CP-pair
  placement passed. Node2 is outside this job.
- DAPO profile uses 32 accepted prompts × 16 generations = 512 trajectories per
  optimizer update, CP2/DP8, 32K total context, microbatch 1 pending measurement.
- User-selected save/eval cadence is 5 optimizer updates. Save contents are HF
  model-only BF16 weights/config/tokenizer, published to node0 EBS under
  /home/ubuntu/nl2sql-checkpoints/, retaining two completed exports per run.
  No optimizer/RNG checkpoint set is written; full-state resume is disabled.
  Tiny real FSDP2 model export/reload and publication/retention checks passed.
- W&B credentials supplied in chat were rejected by the SDK. They are kept
  outside the repository; no secret appears in commands/configs/job metadata.
  Supervisor supports offline W&B plus JSONL metrics until authentication works.
- Training has NOT started. Full-model CP1 matches independent HF real-prompt
  gradients (worst sampled relative L2 ~0.000104), whereas native CP2 fails
  (~0.827); log probabilities remain close. TF32 failed the short comparison
  (~0.306) and is rejected. Fresh-process, canonical-layout and SDPA-math
  diagnostics did not resolve the CP2 failure. Isolated random and real-operand
  attention forward/backward replay checks passed. Layer traces show tiny
  forward differences growing across the model and large early-layer adjoint
  differences. Replicated full-shape attention diagnostic is in progress.
- Numerical gates have not been relaxed. Full-model microbatch/memory testing,
  end-to-end five-update smoke and the production-batch pilot remain gated.
  No H200 readiness marker has been issued. Prior eight-B200 pilot entries in
  this log describe a different machine/run.


### H200 global-key projection correction and gated continuation

- Repeating full attention on both CP ranks did not change the failed gradient
  result, ruling out head partitioning as the cause. An isolated matrix test
  showed full-token versus half-token IEEE SGEMM produced exactly equal Q/O/MLP
  results for the tested real shape, but the narrow global K projection differed
  by relative L2 ~1.26e-6. That small change is amplified through this model.
- Gathering inputs only for global-attention K projections made the eight-GPU
  real-prompt comparison pass: worst sampled gradient relative L2 2.142e-5,
  max log-probability difference 9.97e-7, loss difference 2.98e-8. Added the
  modular fsdp_projection.py implementation behind an explicit H200 config flag.
- Ten focused configuration/checkpoint/cadence tests passed. Verified W&B mode
  and directory reach the Ray controller without putting an API key in Ray env.
- Started nl2sql-h200-prepare.service. It performs sixteen-GPU production-code
  numerical comparisons, 13K/32K warm Adam/memory checks, optional microbatch-2
  checks, then gated five-update smoke and the authorized full-batch pilot.
  Every required stage must pass; failures stop progression. Pipeline status/log:
  outputs/h200_checks/pipeline_status.json and pipeline.log. No production
  readiness or RL-training success is claimed by launching the validation job.


### H200 validation complete; W&B authenticated; integration retry

- All required 16-GPU microbatch-1 numerical checks passed, including independent
  HF long references and repeatability. 13K warm-update peak allocated/reserved:
  64.36/89.62 GiB; 32K: 114.37/135.19 GiB. Microbatch 2 failed the numerical
  comparison (worst sampled gradient relative L2 ~0.516); retain microbatch 1.
- First smoke stopped before model setup: Ray reported only 8 available GPUs.
  Node1 Ray worker had been killed by the host OOM killer during the earlier
  no-checkpoint CPU-offload diagnostic. Replaced its transient service with an
  enabled persistent user unit, Restart=on-failure. Verified Ray reports 16 GPUs.
  Preserved failed logs at outputs/h200_training/smoke_failed_ray_worker/.
- Complete W&B credentials verified against the service and saved in node0's
  mode-600 SDK credential store. Removed the superseded incomplete private key
  file. Credentials are excluded from source, commands and Ray runtime env.
  Enabled W&B for the smoke as well as the pilot; kept offline fallback.
- Restarted the gated continuation with --from-smoke. It revalidates measured
  evidence/source hashes, retries smoke, and launches the pilot only after
  integration checks pass. Save/eval cadence remains 5; model-only BF16 exports
  remain on node0 EBS, latest two retained per run.


### vLLM CUDA build mismatch fixed

- The retried smoke exited during rollout initialization: the default vLLM
  0.19.1 wheel linked libcudart.so.12, while this environment uses CUDA 13.
  The gated pipeline stopped without launching the pilot; GPUs were idle.
- Installed the official 0.19.1+cu130 wheel without changing torch/dependencies
  and synchronized it to node1. Actual vllm._C, LLM and verl ServerAdapter imports
  now pass; pip check passes on both nodes. Requirements pin the CUDA-13 wheel,
  and setup_env.sh explicitly checks the CUDA extension before declaring setup.
- Preserved the failed attempt at outputs/h200_training/smoke_failed_vllm_cuda12/.
  Restarted the supervised smoke/pilot continuation. Model initialization and
  first generation are still being checked; no optimizer-update success is
  implied by process startup.


### Distributed paths corrected; real rollout generation confirmed

- After the CUDA fix, node1 reward workers exposed a relative-path failure.
  train.py now resolves local train/eval parquet, reward module, tool config and
  agent config paths against the repository before serializing Ray config.
  Node1 Ray service also has an explicit repository WorkingDirectory.
- Verified reward-module import and agent/tool/data files from /tmp on node1.
  Four focused config tests passed. Retained failed attempt separately under
  outputs/h200_training/smoke_failed_relative_reward_path/.
- Restarted the gated smoke. Actor initialization and all four TP4 rollout
  servers succeeded. Real initial validation generation is confirmed: last
  observed 130/640 finished, 510 running, zero failures. Optimizer updates have
  not yet begun. W&B is syncing run half8sml; live link:
  https://forge.coreweave.com/wandb/kpratik41/nl2sql-gemma4-verl/runs/half8sml


### Attachable monitoring installed

- Created tmux session nl2sql on node0: status, automatically following active
  run logs, GPU queries on both nodes, Ray CLI and supervisor logs. Training
  remains under systemd; attaching/detaching the terminal does not restart it.
- Added read-only monitor_h200.py. Status reports heartbeat age, last logged
  optimizer update, baseline evaluation completion, checkpoint/eval boundaries,
  and active W&B/log paths. Live logs follow smoke → pilot automatically.
- The initial 640-example evaluation completed; baseline result accuracy is
  0.1671875. Smoke is in its first training cycle, last logged update zero.
  Last physical-GPU sample: all 16 selected H200s at 100% utilization, roughly
  61–63 GiB used each. Node2's eight GPUs remain outside the job and idle.
- Verified tmux windows, status output, GPU queries, Ray CLI and Python syntax.
  The web dashboard is disabled; no cluster restart was needed for monitoring.

- Subsequent check confirmed the first smoke optimizer update completed; metrics are logged at step 1.


### Full training started at the user's explicit request

- User instructed starting the full training run immediately. Stopped the smoke
  after 3 completed updates and cancelled its queued pilot. No checkpoint export
  was in progress. Preserved smoke logs/metrics and recorded the requested stop.
- Revalidated existing numerical/memory evidence and source/data hashes. Added an
  explicit train-only --allow-incomplete-smoke switch; full integration remains
  false, and the default gate remains enforced. No readiness evidence was forged.
- Started nl2sql-h200-dapo-train.service: fresh base Gemma model, 16 H200s, native
  DAPO CP2/DP8, 32 accepted prompts × 16 generations = 512 trajectories, microbatch
  1 / 64 accumulation rounds, target 100 updates, no bounded-pilot time limit.
- Save/eval every 5, BF16 model-only exports to node0 EBS, latest two retained.
  Skip repeating the completed base evaluation; a provenance reference and its
  metrics are stored in outputs/h200_training/train/initial_evaluation_reference.json.
- Online W&B enabled. Existing tmux nl2sql now follows the train log automatically;
  attach with tmux attach -t nl2sql. The full run is initializing, not yet updated.

- Full-run actor/rollout initialization succeeded and the 0/100 training loop started. Online W&B run: https://forge.coreweave.com/wandb/kpratik41/nl2sql-gemma4-verl/runs/cwxyqeok. No full-batch update is completed yet.


### Live utilization diagnosis during full update 7

- Six full optimizer updates completed; step-5 BF16 HF export is complete on EBS
  (62,546,340,072 weight bytes). Current worker process names indicate actor update.
- Read-only three-second NVML GPM samples across all eight GPUs on each node:
  node0 mean GPU-busy 100%, SM-active 93.16%, SM occupancy 12.87%, Tensor activity
  0.020%, DRAM bandwidth activity 11.41%, VRAM used 81.68 GiB; node1 100%,
  92.89%, 13.20%, 0.023%, 10.80%, 80.94 GiB respectively.
- The actor's IEEE FP32 arithmetic largely bypasses Tensor Cores; GPU-busy time
  alone overstates effective hardware use. Microbatch 1/CP2 and repeated FSDP
  collectives/recomputation also need kernel profiling to apportion their costs.
- Step 6 took 98.0 minutes: rollout generation 27.3, old log probabilities 16.8,
  actor update 53.8. DAPO filtered 263 prompt groups and discarded 9 surplus groups
  while collecting 32 accepted groups. This increases sampling cost.
- Pinned verl explicitly warns Gemma4 is unsupported by its FLOP estimator, so
  perf/mfu/actor=0 is not a valid measured utilization. GPU samples and timing
  diagnosis saved in outputs/h200_checks/gpu_utilization_*.json. No runtime
  settings were changed by this diagnosis.


### Multi-turn budget and retrospective tool-call audit

- Verified vLLM each-call max_tokens defaults to min(response_length=2048,
  remaining overall context). ToolAgentLoop also stops/slices the accumulated
  retained response sequence at 2048, including tool-message context. This is
  not an independent retained allowance of 2048 for each of nine turns.
- Saved eval JSONL has decoded mixed response text but no executed tool counters
  or traces. Regex counts: base 701 call expressions /640=1.0953; step5 905/640
  =1.4141 (849 sqlite_query, 40 bm25_search_sqlite, 16 sqlite_peek), median 1,
  max 14. These are emitted call-shaped expressions, not verified executions.
- Both eval turn counters report min=max=mean=2, so genuine multi-assistant-turn
  tool round trips are not demonstrated. Actual executed-call averages cannot
  be reconstructed reliably from these dumps; tool execution/termination needs
  further tracing. No live configuration was changed. Audit JSON is in
  outputs/h200_checks/eval_tool_call_audit.json.


### Future runs: response limit raised to 4096; active run untouched

- User explicitly limited this change to future runs. Updated only future-launch
  H200 config/launcher: response length 4096 for training and eval, prompt limit
  remains 30720, total context and actor/log-prob/prefill budgets 34816.
- Updated the soft overlong reward maximum to 4096, retaining its 512-token
  buffer (penalty starts at 3584). Reaching the cap does not itself remove a
  rollout; DAPO filtering remains based on homogeneous result groups.
- Four config tests pass; launcher shell syntax and Python compilation pass.
  Future validation now requires capacity evidence at the configured context;
  a direct check confirmed 32768-token results cannot validate 34816 tokens.
  No new GPU capacity workload was launched on the busy training cluster.
- Staged future-launch files on both nodes. Verified active driver PID 345962
  is unchanged, its environment still has LENGTH_MAX=2048/BUFFER=512, heartbeat
  remains live and no exit status exists. Current training was not restarted;
  its existing readiness evidence and saved model/config were not overwritten.


### Ongoing environment replication artifacts

- Captured the installed 240-package environment in src/nl2sql_verl/requirements.txt
  with exact version pins, PyTorch cu130 index, and the exact vLLM CUDA-13 wheel
  URL/SHA256. Root requirements.txt includes this file. Broader unpinned runtime
  requirement files remain as development inputs, not the exact snapshot.
- Added environment-h200.json with both hosts' Ubuntu/Python/glibc/compiler,
  driver/toolkit/runtime-library, EFA/libfabric/NCCL and source revision details.
  Both nodes import the same versions. Node1 pip freeze's 24.0 report is stale
  duplicate metadata; imported pip is 26.2.2. Active packages were not modified.
- Added setup_fsdp_env.sh for a separate fresh virtualenv, exact clean verl
  d430157 source checkout and .pth wiring. --check-only verifies an existing
  environment read-only. It does not install the unused NeMo adapter.
- README now leads with the current native FSDP2 H200 environment and explains
  NVIDIA driver 595.112.07 vs system nvcc 13.2.51 vs torch/vLLM CUDA-13.0 builds,
  CUDA library precedence, replication commands, source/workspace copying,
  host/network prerequisites and distinction from future 4K token settings.
- Validation: read-only verification succeeded on both nodes (240 package pins,
  dependency check, CUDA extension + native FSDP2/vLLM adapter imports), pip
  requirements dry run was already satisfied, shell syntax and git diff checks
  passed. Snapshot hash verified; artifacts staged on both nodes. A fresh full
  environment installation was not performed. Training PID 345962 stayed active.

### 2026-10-29 — Rollout format/execution investigation

- Replayed all 4,096 accepted rollout responses from updates 1–8 with the actual
  format predicates and SQLite executor. All eight batches exactly matched their
  logged format/execution aggregates. Replayed the 640 step-5 dev responses with
  zero per-row execution-label disagreements.
- Update 8: 474 executable, format-valid answers; 7 format-valid SQL errors;
  31 incomplete final answers. Thirty incomplete answers have the fully saturated
  length penalty, matching the 30 responses at the active 2,048-token cap. The
  remaining incomplete response ends after a bare sqlite_peek call. Execution
  among format-valid responses is 474/481 = 98.54%.
- The seven actual SQL errors are five nonexistent torrent_tags tables, one
  ambiguous ProductID, and one unquoted reserved index identifier. No missing DB
  or timeout issue explains these failures.
- Found implicit foreign-key targets rendered as `.None` in the reproduced
  prompts (2,068/6,601 train; 82/640 fixed dev). For music_tracker, SQLite's
  omitted target references torrents.id; several outputs invent a junction table
  while reasoning about tags.id=torrents.None. Existing parquets were preserved.
- Confirmed a mismatch between the prompt's bare call syntax and the native
  Gemma parser's required special-token wrappers/string delimiters. A controlled
  actual-agent check executes zero tools for a bare call, but executes SQLite and
  feeds back its real row for a correctly delimited call.
- Staged a separate runtime, completed step-5 model export, source and read-only
  databases on idle node 2. It never joined the training Ray cluster. Ran TP4
  vLLM at temperature 1.2/top-p .95 on four selected prompts x two seeds:
  existing syntax/2K: format 4/8, execution 2/8, zero actual tools, four capped;
  existing syntax/4K: format 6/8, execution 5/8, zero actual tools, two capped;
  explicit native syntax/4K: format 8/8, execution 8/8, 13 actual tools, zero capped.
  Raw tokens confirm absent call wrappers in all 16 current-syntax responses.
  The native condition includes actual ambiguous-column feedback and a repaired
  executable query. These selected cases establish the mechanism and candidate
  repair, not a population pass-rate estimate or SQL correctness improvement.
- User confirmed the other environment uses a 4,096-token limit and temperature
  1.2. The running job remains at 2,048; future profile remains at 4,096.
- Added independent audit_rollout_failures.py and tests/h200_tool_trace.py.
  Full report, per-row failures, raw-token GPU traces, actual tool events and
  machine-readable diagnosis are under outputs/h200_checks/rollout_failure_audit.
  Compile/diff checks and assertions over all 24 probe trajectories/13 tool
  events passed. Node-2 GPU memory returned to zero after the probe. No active
  training package, prompt data, config, tool/parser or model was modified.

### 2026-10-29 — Versioned native Gemma corrections and revalidation

- User authorized correcting the future Gemma profile, versioning FK-corrected
  data, retaining the future 4K budget, and larger-sample actual-tool validation
  on the spare node. The ongoing legacy training service remains unchanged.
- Added gemma4_native_v1 with a model-specific native serializer and corrected
  instructions/demonstrations; semantic tool schemas and legacy gemma4 prompts
  are preserved. All five native demonstrations round-trip through the parser.
- Created outputs/gemma4_native_v1/data/{train,validation,dev_full}.parquet with
  6601/640/1534 rows, preserving all ordering, gold SQL and evaluation membership.
  Only the system wire contract and implicit FK targets in the schema change.
  Corrected 2068 train, 82 fixed-dev and 193 full-dev prompts; 179 unique edges
  were checked independently against SQLite PK order, including composite keys.
  Maximum prompt lengths: 10911/29618/29775, all below 30720. No row was dropped
  or truncated; user content outside the schema, including few-shots, is intact.
- Added a dedicated native agent/tools/reward/runtime route. Task reward weights
  remain unchanged. Actual executions/errors, successful SQL checks, exact final
  SQL verification, cumulative-budget exhaustion and generation traces are
  recorded. The V1 runtime writes companion <step>.tools.jsonl and numeric
  metrics suitable for console/file/W&B. Each generation receives only the
  remainder of the cumulative 4096-token response/context budget.
- Updated future 16-GPU config to the new data and route; it retains DAPO,
  TP1/CP2/DP8 actor, TP4 rollout replicas and 34816 context. Future full launches
  now request a fresh baseline instead of reusing the legacy prompt evaluation.
  Old full-run readiness is not overwritten or relabeled.
- Fixed sample: 128 train + 128 validation prompts, every 69/11 DB covered, eight
  longest each split plus seeded fill. Ran both base and saved step-5 weights
  on independent TP4 engines on node 2, temperature 1.2/top-p .95, 34816 context.
  Base: format 255/256 (99.61%), execution 253/256 (98.83%), result 191/256.
  Step5: format 255/256 (99.61%), execution 254/256 (99.22%), result 190/256.
  Both passed all 128 train format/execution checks. All 512 trajectories used
  real tools, with 672 executions, four observed errors followed by recovery,
  no 4K caps, and maximum retained response 3690. Remaining dev failures include
  turn-limit exhaustion and final SQL damaged after successful tool checks.
  Raw traces and summaries are under outputs/h200_checks/native_v1/{base,step5}.
- Eight-GPU CP2/DP4 actor capacity: longest corrected prompt 10911 plus 4096
  response loss tokens, sequence 15007 padded 15104, microbatch 1, IEEE FP32/Flex
  and canonical global-key projection. Two actual forward/backward/Adam updates
  passed every rank; max allocated 82.78 GiB/reserved 124.40 GiB, finite gradient
  norms 5.7523/3.5831. Prior unchanged 16-GPU selected-gradient gates still pass.
- Fresh original-parent validation verified all 6,601 raw/gold rows and all 4,602
  retrieved eval demonstrations. Targeted CPU tests: 17 passed; compile/diff
  checks passed. Separate node-2 Ray6381 one-update DAPO integration is running;
  results will be appended once actual optimizer/telemetry persistence completes.

- The initial Ray DAPO smoke completed one real 32-trajectory update, exit 0,
  grad norm 7.9142, 175 actual tools, and saved companion traces/metrics. Its two
  accepted groups were harder than the broad fixed sample: format23/32,
  original execution24/32, six 4K caps and three additional turn-limit failures.
  Thirteen homogeneous groups were discarded. The movielens question requests
  movie names although movies has only movieid/year/isEnglish/country/runningtime;
  responses inspect repeatedly. No question/gold correction was made.
- This exposed a native reward boundary bug: one unfinished trajectory could
  earn draft-SQL execution/result credit through the shared legacy fenced-SQL
  fallback. Native reward_entry now requires a completed final answer before
  semantic rewards; legacy/current-run scorer remains unchanged. Regression
  coverage added; 18 tests pass. Strict offline replay of the 32 original smoke
  trajectories gives format23/execution23/result13. No completion was fabricated.
- Re-scored all 512 immutable fixed-sample generations with the strict scorer:
  every reward was unchanged. Original raw outputs/provenance remain preserved;
  strict_results/strict_summary and strict_rescoring.json record input/output
  hashes and the terminal-only source change, without rerunning generation.
- Fresh strict-scorer Ray smoke: 8 GPUs, CP2/DP4, reduced 1 group × 4 generations,
  one optimizer update, exit 0, all 4 format/executable, 3 correct, 21 actual tools,
  zero caps/errors, finite grad norm 4.5886. Metrics and all 4 tool traces persisted.
  Full runtime 590s, training step 313s. This is integration evidence, not another
  pass-rate estimate. Shutdown DataLoader weakref tracebacks occurred after the
  successful step/exit 0 and were retained in logs; they did not invalidate results.
- validate_native_readiness passed source/data/sample/trace/count reconciliation,
  capacity and unchanged 16-rank numerical controls. New evidence is ready for a
  full 16-GPU native smoke; full two-node training is not marked revalidated. The
  original readiness artifact was not overwritten. Both private node2 Ray runs
  were stopped and all 8 spare GPUs returned to 0 MiB. No active training restart,
  runtime-package install, optimizer checkpoint or full model export was done.


### 2026-10-29 — User-requested stop, cleanup and inference-only 8K test

- User explicitly authorized stopping the incorrect legacy run and removing
  obsolete smoke/debug artifacts; explicitly prohibited starting a new run.
  Stopped nl2sql-h200-dapo-train after update 9, stopped its Ray head/worker,
  disabled the persistent node-1 worker, and verified all 24 GPUs initially idle.
  No new training/optimizer/actor smoke was started in this task.
- Preserved the complete EBS step-5 model-only checkpoint, verified all 12 shards
  against its index/completion marker. Kept original/versioned data, DB/model
  caches, environments, compiled kernels, source and useful numerical/native
  validation evidence. Archived compact old metrics/config/eval scores and
  compressed stdout; removed the bulk retired run folders and raw debug dumps.
- Cleanup manifests: outputs/h200_checks/cleanup_20261029/. Apparent bytes removed
  total 237.08 GiB (node0 170.56, node1 8.19, node2 58.33), across root and NVMe.
  Node0 root free space about 485 GiB. No current datasets/models were deleted;
  removed full-size checkpoint copies were redundant NVMe stages/old test copies.
  Old default readiness was archived and replaced with not-ready/stopped flags.
- Added test_output_budget.py, an inference-only paired-seed comparison. Same
  hard prompts (food_inspection_2 row2552, movielens row5358), 16 generations each
  per budget; 64 responses total across 4K and 8K. Base weights, native tools,
  strict scorer, T1.2/top-p.95, TP4, max9 assistant turns, shared38912 context.
  Matched seeds; matching budget instruction updated per arm. No actor loaded.
- Fresh4K: format27/32, execution27/32, correct14/32, three token caps,163 tools.
  8K: format28/32, execution28/32, correct14/32, zero token caps,172 tools.
  All64 responses used real tools. Four remaining8K format failures hit nine
  assistant turns. Largest8K retained response5404 tokens, assistant-only4745.
  Format pairing:3 recoveries/2 regressions; correctness6 recoveries/6 regressions.
  No observed net correctness gain; sample is small and deliberately difficult.
- Validated response bounds, per-call budgets, matching sample IDs/seeds,
  actual execution counters and strict no-final-answer reward gating. Compilation
  and diff checks passed. Existing default4K training config/data were not changed
  to8K. Full8K actor/context launch validation remains a separate scope.
- Results and report: outputs/h200_checks/output_budget_8k/. Both inference engines
  completed and released their GPUs. No training is running; all24 GPUs idle.
### 2026-10-29 — Full 1,534-question greedy base-model evaluation

- User selected original base Gemma-4-31B-it. Launched inference only, without
  restarting training: twelve independent async vLLM TP2 engines, four per node,
  all 24 H200s. One local model snapshot per node is shared by its four replicas.
- Verified matching model index/config/tokenizer, weight-file sizes, corrected
  full-dev parquet and native tool/scorer source hashes across all three nodes.
  All 1,534 prompts retain exactly three retrieved examples, enriched schema and
  original reference indices. No prompt filtering or dataset rewrite.
- Added evaluate_full_dev.py and manage_full_dev_eval.py: explicit greedy
  temperature0/n1, 8K cumulative response budget, context38912, max9 turns,
  async concurrency16 per replica, 80% memory allocation, prefix caching,
  chunked prefill8192 and the validated eager Gemma compatibility path.
- TP2 preflight produced five valid/executable results with actual tools.
  Fixed startup-only version reporting for source-installed verl before the
  full run generated answers. Preflight outputs/log preserved separately.
- Detached services nl2sql-dev-eval-s00..s11 and a collecting monitor run outside
  tmux; tmux session nl2sql-eval exposes progress plus all three nodes' logs.
  Results/provenance/heartbeats are persisted per sample, and the monitor merges
  only once all reference indices 0–1533 are present exactly once.
- Artifacts: outputs/full_dev_base_t0_8k_20261029/. Initial progress111/1534;
  all24 GPUs active, approximately114GiB perGPU during inference.
- Completed all1534 exactly once; full prompt/source/token/tool accounting passed.
  Native format1521/1534 (99.15%), executable1520 (99.09%), result1090 (71.06%).
  All1534 used actual tools,1971 calls,5 errors; zero8K caps, max5464 totaltokens,
  seven over4K. Thirteen unfinished answers;15 trajectories reached nine turns.
- Independently ran saved official BIRD evaluator, default30s timeout, with
  read-only SQLite connections and original BIRD gold/metadata. EX1088/1534
  (70.925684485%, reported70.93%). Simple701/925, moderate297/464, hard90/145.
  Native-vs-official differences518/1127 both independently confirmed30s
  timeouts; retained as incorrect in official EX. Dependency isolated in
  output-local target; existing GPU runtime unchanged. All24 GPUs idle and
  all inference/monitor/scoring units exited. Training remains stopped.
- Final report, official labels, predictions, provenance, diagnosis and validation
  saved under outputs/full_dev_base_t0_8k_20261029/.

### 2026-10-29 — Full training-prompt screening and greedy periodic evaluation

- User requested all6601 train prompts ×16 responses, keep heterogeneous groups
  plus random10% all-correct groups, using all24 GPUs; explicitly confirmed
  retaining DAPO's online result filter. No optimizer training is authorized here.
- Added screen_training_prompts.py and manage_prompt_screening.py. Detached
  twelve TP2 async vLLM engines, four/node, base weights, T1.2/top-p.95,
 8K response/context38912/max9 turns, concurrency32, GPUmem.8, actual native
  tools and same binary SQL-result/60s reward. All69 training DBs, data, model
  files and pipeline source hashes matched across nodes before launch.
- Train parquet is shuffled: data_index denotes position; reference_index retains
  original source ID. Preserved both rather than assuming source IDs are sorted.
  Full16 samples and independently seeded requests are required for classification.
- Automatic completion stage validates105616 distinct pairs, exports every1..15
  group plus floor(N_16/10) uniformly sampled16/16 groups with seed20261029.
  It excludes0/16, keeps exact original RL rows/schema/order, produces per-group
  labels/selection hashes and next-launch overrides, and never starts training.
  Raw source data and training4K actor settings are retained; screening is8K.
- systemd units nl2sql-screen-s00..s11 plus nl2sql-screen-monitor are detached;
  tmux nl2sql-screen shows progress and all nodes' logs. Monitor retries transient
  collection errors and backs up in-progress worker results every five minutes.
- Updated dapo_fsdp_cp2_16gpu.yaml: explicit step0 validation, evalevery5, common
  greedy eval settings T0/top-p1/top-k-1/n1/do_samplefalse; training remainsT1.2,
  n16, DAPO result filterenabled. Config synchronized to all three nodes.
- Nine targeted tests passed, including complete synthetic6601×16 export,
  shuffled reference IDs, correct subset sampling and rejection of incomplete
  groups. Actual pinned V1 generation dispatch verified greedy temperatures at
  steps0,5,10 and stochasticT1.2 training; saved evidence alongside run outputs.
- First real group validation confirmed16 distinct seeds and genuine output
  variation with consistent token/tool accounting. Screening remains in progress;
  all12 engines healthy and all24 GPUs active. No training restart occurred.

### 2026-10-29 — Screening completed and selected dataset verified

- All6601 prompts ×16 =105616 rollouts completed; all12 shards finished and
  collector/selector exited successfully. No training/optimizer was started.
- Mixed1..15/16:561 (all retained). All16/16:4783 (478 retained, seeded10%
  rounded down). All0/16:1257 (excluded). Selected train parquet has1039 rows.
- Pass@16:5344/6601=80.9574%; individual sample correctness81413/105616=77.0840%.
  Native training reward/60s metric; not the separate official30s full-dev EX.
  Format104238, executable104017;153508 tool calls,1714 errors,six8K caps.
- Independently reproduced seed20261029 selection, checked all category counts,
  selected/input hashes, exact source-row equality/order and all12 completed
  shard counts. Saved completion_validation.json and REPORT.md in the run dir.
- Selected parquet plus compact selection/provenance/launch overrides staged to
  both worker nodes for future use. All24 GPUs verified0MiB/0% and idle.
  Online DAPO filtering and step0/every5-step greedy evaluation remain configured.

### 2026-10-29 — Authorized screened1039/8K two-node training launch

- User authorized training1039 prompts/n16/two nodes/W&B/detached tmux; chose
  8192 cumulative output and actor-memory validation, then requested retaining
  every checkpoint. No further permission was required to launch after checks.
- Prepared run-specific8K train1039/validation640/full1534 parquets, changing
  only the system budget number. Input membership/gold/user/tool data unchanged;
  source/output hashes in outputs/screened_dapo_8k/data/manifest.json.
- Added separate8K config, shared DAPO config body (Hydra primary search-path
  restriction), and owned capacity->integration->full-run supervisor pipeline.
  No retired legacy readiness flags were promoted. Ten targeted config/model
  retention tests pass. Keep0 now means retain all completed model exports.
- Actual16-rank CP2/DP8 capacity:10897 prompt+8192 response=19089 tokens,
  padded19200, two real Adam updates, finite norms2.69866/4.42799, max allocated
 71.82/reserved110.29GiB. Initial computation passed but worker report writing
  failed on missing parent directory; preserved that failed attempt, fixed
  directory creation and reran cleanly with [0,0] exit statuses/all16 reports.
  Fixed setsid --wait in the remote launcher and verified nonzero exit7 propagation.
- Started owned Ray head/node1 worker (16 GPUs only); node2 excluded. Integration
  initially hit inherited softNOFILE1024 on Ray daemons. Raised all affected job
  processes to at least65536 and persisted narrow per-job service limits.
  Generation recovered without changing the model or training recipe.
- Full two-node native integration passed: one real optimizer update,16 accepted
  trajectories,32 actual tools, grad norm19.06982, exit0. Shutdown EngineCore
  errors were emitted after successful work during actor cleanup; full run
  started in a fresh job from original base weights, not updated smoke weights.
- Full run is active, initially performing freshT0/n1/640 step0 evaluation.
  Batch32 accepted prompts×16=512 trajectories; micro1/accum64/DP8/CP2,
  async rollout4×TP4, trainT1.2/top-p.95, constantAdamW1e-6/no warmup/decay,
  clipgrad1.0, DAPO clipping.2/.28, result filtering,100 target updates.
- Every5 updates save model-only and evaluateT0. All checkpoints retained on
  localNVMe (20 exports≈1.25TB; source copies also retained), since root had
  only484GiB free. Ephemeral storage implication was explained to the user.
- systemd:nl2sql-screened-training/head/worker; tmux:nl2sql-train-1039 with
  status,training,metrics,gpus windows. W&B API verified state=running, rows1039
  andresponse8192. URL:https://forge.coreweave.com/wandb/kpratik41/nl2sql-gemma4-verl/runs/75et1rl9
- Current evidence/artifacts:outputs/h200_checks/screened_8k_launch/ and
  outputs/h200_training/screened1039-8k/. Custom status monitor accurately
  separates initialization/step0 from completed optimizer updates.

### 2026-10-29 — Source snapshot and DAPO rollout accounting

- Prepared the native-tool, data preparation, full-dev evaluation, pass@16
  selection, H200 environment, 8K DAPO launch, checkpoint and monitoring source
  changes for Git. Local environments, handoff manifests and temporary logs
  remain excluded; runtime outputs, downloaded data and model weights remain
  local. No change to the ongoing training recipe or process.
- Step0 held-out native SQL-result accuracy was464/640=72.50%. First three
  updates completed with mean rewards2.61234/2.62191/2.58674 and accepted-batch
  result accuracy57.4219%/50.7813%/53.1250%. Step5 evaluation is pending.
- Verified pinned upstream replay-buffer counters count prompt groups, not
  individual trajectories. Updates1/2/3 generated132/165/152 groups, including
  homogeneous72/88/83, surplus mixed28/45/37 and accepted32 each. With16
  trajectories/group this is2112/2640/2432 generated,512 accepted each.
  Mixed surplus is drained then discarded, not reused across updates.
- CPU-only test invocation documents the pinned upstream config path. Legacy
  AutoModel tests explicitly skip if the optional NeMo backend is absent,
  instead of failing imports in the active FSDP-only environment.
- Preserved explicit return_dict=False tokenization and its regression test in
  the legacy converter: Transformers mappings count fields rather than tokens.
- Validation: full CPU suite51 passed/3 optional-NeMo skips; after restoring the
  token-count regression test, all8 task tests passed. Shell syntax and Git
  whitespace checks passed; source scan found no credential patterns.
