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

### 09:16 — 640-example baseline complete; first actor update in progress

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
