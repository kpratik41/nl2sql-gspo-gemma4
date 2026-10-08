# Long-context full-parameter RL: Gemma 4 and Nemotron Lightning

Last updated: 2026-10-07 (America/New_York).

This is the running record for this conversation on the `verl` branch. It records requirements, inspected facts, recommendations, and unresolved questions. Proposed configurations have not yet been benchmarked. No verl RL training run has been launched as part of this discussion.

## Current requirements and documentation workflow

- Model: the previously downloaded Gemma 4 31B instruction-tuned checkpoint.
- Alternative under evaluation: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. Comparing it does not constitute a decision to replace Gemma.
- Research update: Gemma 4 31B CP exists in NVIDIA AutoModel and has NeMo RL functional coverage. The `CP=1` restriction discussed here is specific to Megatron Bridge's current dense provider. Evaluate that existing AutoModel path before deciding that Gemma requires a new CP implementation or a model switch; integration into our required verl workflow remains to be verified.
- Framework: verl.
- Training: **full-parameter RL of the language model; no LoRA or QLoRA**. The earlier LoRA recommendation is superseded by this explicit requirement. For text-only tasks, vision/audio components are outside the proposed training scope.
- Context: approximately 40,000 tokens. The working assumption is about 40,960 input tokens plus a short response; the user has not yet confirmed whether the target instead means total sequence length or long generated responses.
- Data: synthetic data or an appropriate existing dataset. General long-context reasoning versus continued NL2SQL specialization remains open.
- Hardware: one existing H100 node; additional H100 nodes can be procured. Clarify whether “2 nodes later” means two total or two additional when planning procurement.
- Keep the VS Code checkout on `verl`; use separate worktrees for other branches.
- The user explicitly authorized creating this note, committing it, pushing it to `origin/verl`, and continuing to update, commit, and push this same file with important discussion findings and decisions. During continued work, maintain this file rather than creating separate discussion notes. This is an ongoing conversation workflow, not a scheduled background job.

## How the components fit together

| Component | Responsibility |
|---|---|
| verl | Runs the RL workflow: prompts, rollouts, rewards, advantages/loss, training updates, weight synchronization, evaluation, and checkpoints. |
| GRPO / GSPO | Learning algorithms/objectives used within that workflow. They are independent of the choice of distributed training backend. |
| FSDP2 | Shards model parameters, gradients, and optimizer state across training GPUs. Layers gather the parameters needed for computation. |
| Megatron Core / Megatron-LM | Another training backend with distributed model computation and several parallelism strategies; actual availability depends on the specific model implementation. |
| Megatron Bridge | Adapts supported Hugging Face architectures/checkpoints to Megatron and handles conversion. A model “provider” is the code that constructs that architecture for Megatron. |
| vLLM / SGLang | Generation engines for producing candidate answers during rollouts. Generation kernels need not be the same as training kernels. |

Typical cycle: question and context → candidate answers → reward function → RL loss → forward/backward and optimizer step → updated weights sent to the generator → repeat.

For NL2SQL, rewards can check whether generated SQL returns the expected results. For synthetic reasoning, rewards can compare the answer with a programmatically computed target.

References: [verl](https://github.com/verl-project/verl), [engine workers](https://verl.readthedocs.io/en/latest/workers/engine_workers.html), [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

## Inspected hardware and model

The current host has eight NVIDIA H100 80GB HBM3 GPUs. `nvidia-smi topo -m` reported NV18 connectivity between each pair. The host reported approximately 2 TiB of system RAM. GPU memory is distributed across devices; it is not one automatically pooled allocation.

The downloaded checkpoint was found at:

```text
/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31b-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
```

Both safetensors shards resolve to existing cache blobs. The index reports 62,546,177,752 weight bytes, approximately 62.5 decimal GB. The cached configuration has no quantization configuration. The repository's `gemma-4-31b-it-local` directory in the consensus worktree contains configuration/tokenizer files; the actual weight files were verified in the cache above.

The inspected model configuration specifies:

| Property | Value |
|---|---|
| Architecture | `Gemma4ForConditionalGeneration`, dense text backbone |
| Text layers | 60 |
| Sliding-attention layers | 50, head dimension 256 |
| Global-attention layers | 10, head dimension 512 |
| Attention pattern | Five sliding layers followed by one global layer |
| Sliding window | 1,024 tokens |
| Query heads | 32 |
| Local / global KV heads | 16 / 4 |
| Configured maximum positions | 262,144 |
| Vocabulary | 262,144 |
| Final logit softcapping | 30.0 |

40k tokens is within the configured context window. That alone establishes neither training feasibility nor task accuracy at that length.

The existing consensus code uses TRL/DeepSpeed and SDPA paths. It is not evidence of an already working verl 40k configuration. The `verl` branch was initially empty before this note.

## Attention at 40k tokens

The usual FlashAttention-2 training path cannot handle this model's 512-dimensional global heads. The local layers have 256-dimensional heads, so the limitation does not apply identically to every layer. Kernel capabilities vary by version, GPU, and forward/backward mode; inference support does not establish training support.

The first training candidate is **PyTorch FlexAttention**, preserving the model's global causal attention and local sliding windows. A second candidate is hybrid attention: FlashAttention-2 for local layers and a verified memory-efficient training kernel for global layers. Axolotl documents such a hybrid implementation and a 32k Gemma 4 31B FSDP2 LoRA example. That example is implementation evidence, not a full-parameter verl RL benchmark.

Selecting `attn_implementation="sdpa"` does not by itself prove memory efficiency. SDPA dispatches to a backend, and the actual forward and backward kernels must be inspected. Avoid a dense attention-score allocation:

```text
1 sample × 32 heads × 40,960² positions × 2 BF16 bytes
= 107,374,182,400 bytes = 100 GiB
```

That is one hypothetical dense score tensor for one layer, before gradients and other intermediates. Tiled memory-efficient attention avoids materializing it, although global attention still has quadratic compute cost.

FlexAttention needs a real H100 forward/backward test with the 512-dimensional global heads. Compilation settings, block sizes, masks, and grouped-query attention can affect feasibility and performance. Preserve Gemma's attention scaling, normalization, positional embeddings, and sliding-window semantics.

References: [Axolotl Gemma 4 guidance](https://docs.axolotl.ai/docs/models/gemma4.html), [32k example](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/gemma4/31b-lora-fsdp.yaml), [FlexAttention API](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html).

## CP=1 and PP=1 in simple terms

The numbers specify how many cooperating partitions a parallelism method uses. A value of one means that method is not splitting the work.

- **Context parallelism (CP)** divides the tokens of one sequence among GPUs. For example, CP=4 could assign about 10k of a 40k sequence to each of four ranks. They exchange the information needed for correct attention across the complete sequence. CP=1 means there is no division along the sequence dimension.
- **Pipeline parallelism (PP)** divides model layers into stages. For example, PP=2 could put layers 1–30 on one GPU group and layers 31–60 on another. PP=1 means there is one pipeline stage.
- **Tensor parallelism (TP)** divides computations within a layer across GPUs. It is distinct from both methods above.
- **Data parallelism (DP)** processes different examples on different replicas and combines their gradients. FSDP additionally shards model states; it does not automatically divide each example's sequence computation.

The examined NVIDIA Megatron Bridge `Gemma4DenseProvider.provide()` explicitly raises errors for PP other than one and CP other than one. Its CP explanation points to a local attention implementation without context-parallel support. These guards concern that Gemma 4 dense integration; they are not a fundamental limitation of Gemma's architecture or of Megatron as a whole. Other implementations or future versions may differ.

In particular, **CP=1 and PP=1 do not mean only one GPU can be used**. Other parallelism may still distribute work, subject to model support. They mean that this provider cannot currently use two specific ways of spreading one large training example across devices. Adding GPUs does not remove those implementation checks.

Consequence: do not assume generic Megatron CP/PP configuration recipes will work for this checkpoint. FSDP2 is the first integration candidate, with AutoModel's existing model-specific CP path now a priority to investigate. Reconsider Megatron if model support improves or a custom integration is justified by measurements.

Source checked during this discussion: [Gemma 4 provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/gemma/gemma4_provider.py). This link tracks upstream main; recheck and pin a commit before implementation.

## Investigation: why Megatron Bridge restricts dense Gemma 4 to CP=1

This follow-up examined upstream source, the original dense-model contribution, the explicit CP guard change, developer issue comments, two pending CP implementations, their Transformer Engine dependency, and working AutoModel/NeMo RL alternatives. Repository/API states below were checked during this discussion on 2026-10-07 America/New_York. Upstream main and open-PR states can change.

### Direct cause: the dense provider selects an attention implementation without CP

The construction path is:

```text
Gemma4DenseProvider
  → get_gemma4_layer_spec()
  → LocalSpecProvider
  → Megatron Core's non-TE DotProductAttention
  → requires context_parallel_size == 1
```

The dense layer spec explicitly selects the local backend. The resulting attention implementation checks CP size and does not implement the distributed exchange of attention information. Its forward code constructs attention scores explicitly. This confirms an implementation constraint, not a limit on how many tokens the pretrained architecture can understand. Sources: [dense layer spec](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/gemma/modeling_gemma4.py), [Megatron Core local attention](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/dot_product_attention.py).

Bridge PR #5895 added the early error, merged September 2, 2026, as commit `d69c7c4292c062c83648a92349a6bf1d87b3f783`. Before that change, CP already failed deeper in layer construction. The new guard made the failure understandable rather than introducing a new capability restriction. Its author encountered this while attempting dense Gemma 4 GRPO with CP2. [Guard PR and rationale](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5895)

Plain-language example: splitting 40k tokens into two 20k pieces is only the allocation step. Tokens on one GPU may still need information from the other GPU, and backward must return the appropriate gradients. The local attention implementation lacks that communication protocol. Removing its error checks would not add the missing operations.

### Second obstacle: a compatible distributed attention backend for 512-dimensional heads

Simply choosing Transformer Engine does not complete the repair. Draft Bridge PR #6120 explains that the released TE path it targets has no suitable backend for symmetric 512-dimensional heads together with CP. It depends on TE PR #3527; without that dependency, model construction can succeed but attention subsequently fails. This corroborates the head-dimension concern, while keeping the immediate cause distinct: the current dense provider chooses the local path in the first place. [Bridge TE integration proposal](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6120)

TE PR #3527 describes the backend gap: FA2/FA3 kernels stop at 256; the applicable C++ cuDNN fused path also caps at 256; the unfused implementation accepts 512 but cannot run CP. Its proposed FROST backend uses cuDNN's Python/CuTe-DSL path. **That proposal targets SM100/SM103 Blackwell, not our SM90 H100s**, and was still unmerged when checked. It also states that its previous numerical tests precede a dispatch refactor and require rerunning on the current head. Treat this as evidence about the missing integration, not an available H100 solution. [TE proposal and hardware scope](https://github.com/NVIDIA/TransformerEngine/pull/3527)

The official [FlashAttention README](https://github.com/Dao-AILab/flash-attention) independently documents the FA2 head-size limit. These are version/backend constraints, not a claim that no possible attention algorithm can process 512-dimensional heads. FlexAttention and FFPA offer alternative implementations.

### Why it is still unfinished: prioritization and correctness work

In Bridge issue #4663, NVIDIA contributors explained in July that limited engineering bandwidth and different model priorities left Gemma 4 outside their current performance-optimization work; they welcomed a contribution with numerical, distributed, and performance validation. This is explicit project-priority evidence, not speculation about the model's theoretical suitability. [Developer response](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/4663#issuecomment-4898117463)

There are also real correctness problems to solve:

- Local and global layers use different dimensions, head counts, and attention windows. Their behavior must survive backend replacement.
- CP requires correct global token positions, cross-GPU K/V exchange, softmax normalization, and backward gradient accumulation.
- Packed examples require document boundaries to remain isolated after distributed rearrangement; successful single-document tests are insufficient.
- Smaller kernel tests must be followed by full-model loss/logit/gradient comparisons and checkpoint checks.

Draft PR #6213 implements a hybrid approach: TE for supported sliding-layer layouts and differentiable K/V gathering with compiled FlexAttention for global layers. It explicitly handles document IDs, padding, and rank-major packed layouts. The author reports CP2/CP4 attention-level gradient checks, but full 31B parity, full-model performance, and the pinned H100 validation are still listed as unfinished. [Hybrid CP proposal and validation limits](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6213)

Do not attribute this 31B restriction to cross-layer KV sharing in the smaller E-series: our inspected 31B config has `num_kv_shared_layers=0` and no per-layer input embeddings. Those features complicate other Gemma variants but are not the demonstrated cause here. CP and PP are separate implementations; this investigation does not establish the reason for the separate PP=1 guard.

### Development status and working alternatives

| Evidence | Status at inspection | What it establishes |
|---|---|---|
| [Bridge #3885](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3885) | Merged May 23 | Initially added limited dense Gemma support; support for loading a model was not a promise of every parallelism feature. |
| [Bridge #5895](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5895) | Merged September 2 | Clear error for an already unsupported CP path. |
| [Bridge #6120](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6120) | Open draft, not merged | TE wiring proposal dependent on an additional backend change; reported reduced-model B200 tests. |
| [TE #3527](https://github.com/NVIDIA/TransformerEngine/pull/3527) | Open, not merged | Proposed Blackwell-specific D512 backend, not an H100 fix. |
| [Bridge #6213](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6213) | Open draft, not merged | Alternative hybrid CP implementation with attention-level tests and remaining full-model validation. |
| [AutoModel #2592](https://github.com/NVIDIA-NeMo/Automodel/pull/2592) | Merged June 17 | Dense Gemma 4 31B CP through model-specific ring FlexAttention; reports CP parity and a single-node 16k CP8 SFT run. |
| [AutoModel #2436](https://github.com/NVIDIA-NeMo/Automodel/pull/2436) | Merged June 30 | Optional D512 FFPA kernels with forward/backward and a CP ring integration. |

**Gemma 4 31B can use context parallelism in another implementation today.** AutoModel's ring path exchanges attention data and applies model-specific masks around compiled FlexAttention; it can optionally dispatch eligible global chunks to FFPA. This is a concrete alternative to writing a new Bridge backend. [AutoModel ring implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/gemma4_moe/cp_attention.py)

For the documented FFPA route, direct HF `attn_implementation: ffpa` is restricted to non-CP/non-packed execution. Its CP configuration instead uses the model's ring interface with `attn_implementation: sdpa` and `text_config.cp_full_attn_backend: ffpa`. The same word `sdpa` therefore does not imply the same executed kernel across these integrations. Inspect dispatch and profile it. [AutoModel FFPA explanation](https://github.com/NVIDIA-NeMo/Automodel/discussions/2928)

### What this changes for our 40k full-parameter RL plan

NeMo RL now documents Gemma 4 31B CP1/CP2 functional training comparisons with AutoModel. This is evidence that CP also works in an RL workflow, not only SFT. Its guide describes 100-step comparisons and labels support functionally ready rather than claiming broad long-run convergence. [NeMo RL Gemma guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/gemma/gemma4.html)

However, the associated [base recipe](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/dapo-gemma4-31b-it-4n8g-fsdp2-automodel.yaml) uses four eight-GPU nodes and a 4,096-token total-sequence limit. The [CP2 override](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/dapo-gemma4-31b-it-4n8g-fsdp2cp2-automodel.yaml) retains that limit. Those runs do not validate 40k RL on our eight or sixteen H100s.

AutoModel also documents 64k CP8 SFT, but that [long-context recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/long_context_validation/gemma4_31B/README.md) uses the base checkpoint on 16 nodes / 128 GPUs. It supports architectural feasibility, not a small-cluster memory claim.

Revised next investigation: evaluate **AutoModel FSDP2 plus its Gemma-specific CP kernels** for the requested verl workflow before attempting a new Megatron patch or deciding to switch models. verl has an [AutoModel engine](https://github.com/verl-project/verl/blob/main/docs/workers/automodel_workers.rst), but its documentation/examples inspected here emphasize SFT and do not establish Gemma-specific CP integration or our end-to-end RL path. Check the exact versions, CP batch/mask hooks, response log probabilities, full-policy weight transfer to vLLM, and checkpoint restore. Porting may be required. NeMo RL is the demonstrated framework alternative if the user later relaxes the verl requirement.

This qualifies the earlier preference for Lightning: Lightning remains attractive for its architecture and Megatron support, but **Gemma's lack of CP is not universal**, and AutoModel provides existing work we should assess. We have not changed frameworks, models, runtime configuration, or procured hardware during this research.

## Revised approach: full-parameter training

The original suggestion to begin with LoRA has been rejected by the user. The following replaces it.

### One node: eight H100s

Use verl with FSDP2, BF16 computation, full language-model parameter updates, and a validated memory-efficient attention backend. Start with all eight GPUs available for training and alternate generation/training phases using supported sleep/offload behavior. Begin with one sequence per GPU per training microbatch and accumulate gradients for the desired effective batch.

Required profiling candidates:

1. Layer-level FSDP sharding and gradient checkpointing/recomputation.
2. Supported optimizer/model-state CPU offload if model states leave inadequate GPU headroom. Verify the exact FSDP2/verl offload semantics in the pinned stack rather than assuming every offload flag is independent.
3. Activation offload or compatible MLP tiling if activations remain the bottleneck. System RAM is abundant, but transfers and CPU optimization may be slow.
4. Limited rollout concurrency and explicit memory release between phases. Test vLLM TP=4 as a starting generation layout, not a settled optimal configuration.
5. Efficient response log-probability computation without keeping full-vocabulary logits for every prompt position. At 40,960 positions, a single BF16 logits tensor with this vocabulary is 20 GiB; FP32 is 40 GiB. Preserve final logit softcapping and correct response-position indexing.

A conventional mixed-precision Adam budget can be about 16 bytes per trainable parameter: BF16 weights and gradients, FP32 master weights, and two FP32 moments. For 31 billion parameters that is roughly 496 decimal GB, or 462 GiB, before activations, temporary layer gathers, communication buffers, rollout state, and any reference model. Evenly sharded eight ways, that illustrative budget is about 58 GiB per GPU. Actual optimizer/dtype choices change this estimate.

This makes one-node full training a plausible but unproven engineering target, likely requiring offload and careful memory management at 40k. Aggregate GPU capacity alone is not a feasibility test. No throughput estimate is justified yet.

Start with GRPO or GSPO and four responses per prompt. These approaches avoid a separate learned critic. A reference policy is still needed if the chosen KL objective requires one; full tuning cannot recover a frozen base reference by disabling an adapter. Budget reference weights/offload and reference log-probability passes explicitly, or make a deliberate decision about a no-reference objective. Keep generated answers short initially, around 512–1,024 tokens.

Sequence parallelism may ultimately be needed. The examined verl generic Ulysses path wraps FlashAttention calls. Do not assume that setting Ulysses size above one will automatically parallelize FlexAttention or correctly support Gemma's heterogeneous heads. That combination requires integration and correctness testing.

References: [verl configuration](https://verl.readthedocs.io/en/latest/examples/config.html), [verl attention/sequence-parallel implementation](https://github.com/verl-project/verl/blob/main/verl/models/transformers/monkey_patch.py).

### Additional nodes

| Hardware, assuming eight H100s per node | Full-training planning option |
|---|---|
| One node / 8 GPUs | All eight train; generation and training alternate. Profile offload, attention, and activations. |
| Two nodes total / 16 GPUs | If model-state memory is the problem, shard training across all 16 and alternate rollout phases. If training already fits comfortably on eight and generation is limiting, dedicate the second node to rollout. |
| Three nodes total / 24 GPUs | Consider 16 training GPUs plus eight rollout GPUs, subject to measured bottlenecks and interconnect performance. |

The earlier suggestion to dedicate a second node to rollout was made for LoRA. It is conditional for full training: using all 16 for training may be more valuable. With the illustrative 496 GB state budget, ideal 16-way sharding reduces the state share to about 29 GiB per GPU. It does not automatically reduce the activation/attention footprint of one sequence.

Check cross-node InfiniBand/RoCE bandwidth and latency before procurement. Cross-node FSDP communication and full-policy weight synchronization can be substantial. Start with synchronous policy updates; asynchronous rollout overlap introduces policy staleness and should be a separate optimization. Buying more nodes cannot fix an unsupported kernel or provider feature.

## Proposed data and rewards

The first general-purpose candidate is freshly generated **RULER-style synthetic data**, using configurable long-context tasks rather than training on a fixed benchmark test split.

| Task | Training behavior | Verifiable reward |
|---|---|---|
| Multi-key retrieval | Retrieve several relevant records spread across the input | Exact set match |
| Multi-hop tracing | Follow references between distant records | Exact final answer |
| Aggregation | Count/combine information across many records | Exact numeric result |

Generate a few thousand pilot examples; measure baseline performance before choosing difficulty. Adjust difficulty so groups contain a useful mix of successful and unsuccessful answers. Identical rewards within a group give little or no relative-advantage learning signal.

Measure length with Gemma's tokenizer after applying the chat template. Put necessary evidence at varied positions across roughly 40k tokens. Padding an otherwise short task with irrelevant text is not sufficient evidence of useful long-context reasoning. Keep evaluation seeds, entities, and templates separate. Evaluate at 8k, 16k, 32k, and 40k to track both long-context gains and shorter-context regressions.

Use a separate realistic evaluation such as LongBench to test transfer. Synthetic task improvements alone do not establish broader ability.

For NL2SQL specialization, an alternative is large schema catalogs and documentation with relevant information distributed across the context. Reward executable SQL using multiple database instances where possible, reducing accidental equivalence on one instance. The target task choice remains open.

Sources: [RULER generators and tasks](https://github.com/NVIDIA/RULER), [LongBench](https://github.com/THUDM/LongBench).

## Implementation milestones and open questions

1. Pin compatible verl, Transformers, PyTorch, vLLM, and attention-kernel versions; identify any required model patches. General framework support is not proof of this exact combination.
2. Validate attention outputs and gradients at short lengths against a trusted baseline, then test global/local kernels at 40k on H100.
3. Validate full-model loading and sensible generation before training; check trainer/generator log-probability agreement and weight synchronization.
4. Run full-parameter forward/backward/optimizer smoke tests at increasing lengths: 8k → 16k → 32k → 40k. These are engineering checks; the intended training target remains 40k.
5. Complete one end-to-end RL update at 40k: generation → reward → policy/reference log probabilities as required → backward → optimizer step → rollout weight synchronization → fresh generation.
6. Record peak memory and time separately for each phase, numerical stability, reward distribution, and validation accuracy. Test checkpoint save/resume before a long run.
7. Use those measurements to choose offloading, node count, and GPU placement; do not procure another node based only on parameter-count arithmetic.

Unresolved: meaning of the 40k requirement, general reasoning versus SQL, final algorithm and KL/reference choice, supported FlexAttention/sequence-parallel integration, and additional-node interconnect. No claim of a working 40k full-parameter training configuration has been made.

## Alternative: Nemotron 3.5 Lightning

The concrete comparison target is **`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`**. NVIDIA identifies the BF16 release as the customization/post-training checkpoint. The requirement remains full-parameter RL, without adapters. NVFP4 serving recipes and single-GPU inference claims are not training-memory estimates.

### What changes architecturally

| Property | Gemma 4 31B | Nemotron 3.5 Lightning |
|---|---|---|
| Main architecture | Dense transformer with local/global attention | Hybrid Mamba-2, mixture-of-experts (MoE), and attention |
| Advertised parameter scale | About 31B dense | About 30B total, 3B active per token |
| Attention head dimension | 256 local / 512 global | 128 for attention; Mamba has separate dimensions |
| Attention structure | 50 sliding and 10 global layers | Six attention blocks in the 52-block main stack; other blocks are Mamba or MoE |
| Vocabulary | 262,144 | 131,072 |
| First training backend to investigate | FSDP2 with a compatible attention path | Megatron via a pinned Lightning-compatible verl integration |

The inspected Lightning config has 128 routed experts, six selected per token, and one shared expert. A router chooses a subset of expert networks for each token. Mamba blocks maintain and transform sequence state rather than building a full token-to-token attention matrix. These differences give a reason to expect lower computation for many workloads, but do not establish a speedup factor on our machine. Some global attention remains, as do activation and communication costs.

Gemma's specific 512-dimensional-head obstacle is absent. A supported Transformer Engine fused/FlashAttention path is the first candidate for Lightning's attention blocks, alongside optimized Mamba and MoE kernels. Every part of the hybrid model still needs a compatible backward pass; a generic transformer attention patch is insufficient.

The model card advertises up to one million tokens; the inspected HF config defaults `max_position_embeddings` to 262,144. Neither number proves training feasibility. Our 40k target is below both. Regenerate/tokenize datasets using Lightning's own tokenizer and native chat template, including an explicit thinking-mode choice.

Sources: [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), [configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json), [native template guidance](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/nemotron/nemotron_3/lightning/README.md).

### Active parameters reduce computation, not total training-state storage

“3B active” does not mean a 3B model's optimizer-memory requirement. Full tuning retains all expert weights and their training state; different tokens select different experts. Full-parameter training makes the policy parameters trainable, although an individual token only exercises its routed subset.

At the same illustrative 16 bytes per parameter used above, 30B parameters imply about 480 decimal GB of state, not 48 GB. NVIDIA's conversion verification reports approximately 32.9B checkpoint parameters including auxiliary components; budgeting all of those at 16 bytes would be about 527 GB. Actual trainable components, precision, optimizer state, and sharding determine the real allocation. This remains similar in order of magnitude to Gemma.

The smaller vocabulary halves the hypothetical full-sequence logits allocation: at 40,960 positions, BF16 logits would be 10 GiB instead of 20 GiB. Efficient response-logprob computation is still worthwhile.

**Expert parallelism (EP)** becomes useful: different GPUs own different experts, and tokens are routed to the appropriate devices. This distributes expert storage and computation but creates all-to-all communication. Plan placement around NVLink and the cross-node network. EP groups overlap with other parallelism dimensions in Megatron; do not multiply TP, CP, and EP sizes together as if they were independent GPU counts. See [Megatron parallelism guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html).

### What is actually supported and verified

There are three distinct pieces of evidence:

1. **Megatron Bridge training:** NVIDIA records full-parameter BF16 SFT at 32,768 tokens on two eight-H100 nodes, using TP=2, CP=2, EP=8, and expert TP=1, with full recomputation. This establishes a working context-parallel training path for Lightning, unlike the examined Gemma dense provider. It is SFT at 32k, not our 40k verl RL workload. [Verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/nemotron-3.5-lightning/card.yaml)
2. **NVIDIA's NeMo RL reference:** its Lightning RLVR recipe uses a 73,728-token maximum and TP=4, CP=4, EP=16, PP=1. It runs on a much larger GB200 deployment, including 128 training GPUs. This demonstrates long-context RL support in that stack, not feasibility on eight or sixteen H100s. NeMo RL is an alternative framework; switching to it has not been requested. [Reference recipe](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/nemotron-3.5-lightning.md)
3. **verl integration:** PR #7192 was open and draft when checked. Its author reports full-parameter H100 GRPO validation on a v0.7 backport, while the rebased main-branch implementation still needs GPU validation. Pin the tested revision rather than assuming this work is available in a released verl version. [PR and validation scope](https://github.com/verl-project/verl/pull/7192)

The PR records H100-tested backport `dc3421cf63fbfbf7aecb191bbca1e0fb0707e90d`; rebased PR head `1ba0bcbef9697acc20ff5863f592a096c5d75474`; and model revision `d468880b6ad3c6e0d21377ce7242adaea4cc884d`. These are reproduction references, not versions installed in this workspace.

The examined backport launcher defaults to two eight-H100 nodes, actor TP=2 / CP=1 / PP=1 / EP=8 / expert TP=1, with offloading. Its prompt and response defaults are 2,048 tokens each. That supplied configuration is not a 40k validation. The same launcher requires **CP=1 when R3 router replay is enabled**. [Pinned launcher](https://github.com/Phlip79/verl/blob/dc3421cf63fbfbf7aecb191bbca1e0fb0707e90d/examples/grpo_trainer/run_nemotron_3_5_lightning_30b_a3b_megatron.sh)

Router replay records and reuses expert-routing decisions to reduce discrepancies between rollout and training computations. At 40k we must determine whether CP=1 fits, or validate a CP-compatible routing strategy. Simply disabling replay to allow CP changes numerical behavior and requires checking actor/rollout probability agreement and training stability.

Consequently, Lightning has stronger evidence for useful model parallelism, but **CP support in Bridge does not imply every verl RL feature combination supports CP**. The cited recipes use PP=1; they are not proof that arbitrary pipeline layouts or PP combined with MTP work. Keep PP=1 initially.

### Full-training approach on our hardware

| Available hardware | Proposed Lightning plan and evidence limit |
|---|---|
| One node / 8 H100s | Attempt a reduced Megatron configuration with expert parallelism, BF16, recomputation, and offload as needed. TP=1 / EP=8 / CP=1 / PP=1 is a candidate to test, not a verified 40k configuration. Alternate generation and training. |
| Two nodes total / 16 H100s | Prefer all 16 for initial full training. Reproduce the reported short-context verl setup first, then grow sequence length. The NVIDIA TP2/CP2/EP8 SFT recipe is a separate long-context starting point; reconcile CP with the selected RL routing path before combining them. |
| Three nodes total / 24 H100s | Consider 16 training GPUs plus eight rollout GPUs after full-policy synchronization and throughput are measured. |

The initial alternative-model recommendation was to prioritize a **Lightning + Megatron feasibility study**, particularly with two H100 nodes. It removes Gemma's unusual attention-head constraint and brings published hybrid/MoE training evidence. The subsequent CP investigation above qualifies that recommendation: assess Gemma's already implemented AutoModel CP path as well before choosing a model. Neither route has been benchmarked here, and architecture/support evidence does not establish task accuracy or a training-speed winner.

Full-policy model/optimizer storage remains large. Eight H100s are a pilot target, not a guaranteed 40k training configuration. With two nodes, CP can potentially reduce per-sequence activation pressure in addition to distributing model states, provided the exact RL path supports it. Interconnect quality matters for both expert dispatch and CP communication.

### Extra correctness checks and data choices

- Preserve Mamba state resets between packed examples. Incorrect packing can leak information across training samples.
- Track expert-load balance and router precision, plus actor/rollout log-probability differences. Start with the pinned reproduction's settings before tuning.
- Treat multi-token prediction (MTP) training and speculative generation as separate choices. For a minimal baseline, ordinary generation avoids speculative plumbing; reproducing the cited validated recipe instead requires its exact MTP settings. Explicitly decide how auxiliary MTP weights are trained/preserved and exported. Do not silently freeze the main policy or substitute adapters.
- Revalidate save/resume and model export, including experts, Mamba state-space parameters, and any MTP components. Support for training does not by itself prove checkpoint restoration.
- Keep the proposed RULER-style retrieval, tracing, and aggregation pilot. Baseline and difficulty calibration must be rerun because the model and tokenizer changed. Keep necessary evidence distributed through the full 40k input and use independent held-out examples.
- Lightning's open RL training blend and math recipes can inform reward formats, but do not assume they contain 40k input examples. Short math data with padding does not replace the long-context requirement.

No Lightning weights have been downloaded and no Lightning training run has been launched for this comparison. Gemma remains the original target until the user chooses otherwise.

## Earlier environment setup

At the user's request, `consensus` was checked out in a separate worktree at `/home/ubuntu/verl-fun/nl2sql-gspo-gemma4-consensus`; the VS Code worktree remained on `verl`. A Python 3.12.3 `.venv` was created there.

`python -u temp.py` was launched in a detached GNU screen named `consensus-temp`. It was verified at setup to run four CPU workers targeting approximately 2% of the host's 192 logical CPUs. This script is unrelated to GPU model training.

The session identifier at setup was `13223.consensus-temp`. While it remains running:

```bash
screen -r 13223.consensus-temp
```

Detach with Ctrl+A, then D; stop the script with Ctrl+C. Use `screen -ls` to check current session state. This note records the earlier verification, not a guarantee that the process remains running indefinitely.

## Decision history

- Explained verl as the RL coordinator, Megatron/FSDP as training backends, and vLLM/SGLang as generation engines.
- Inspected the eight-H100 node, cached model weights, and heterogeneous attention configuration.
- Initially suggested LoRA for memory headroom. **Superseded:** the user requires full-parameter RL.
- Identified CP/PP limitations in the examined Megatron Bridge Gemma 4 dense provider; these are implementation-specific.
- Revised the starting plan to full-parameter FSDP2 with memory-efficient attention, checkpointing, and measured offloading requirements.
- Authorized this single running Markdown document and ongoing commits/pushes to `origin/verl` for important discussion updates.
- Added Nemotron 3.5 Lightning as an alternative: its hybrid/MoE architecture and 128-dimensional attention heads make Megatron more attractive, while full training-state memory remains substantial. Documented the separate evidence for Bridge CP training, large-scale NeMo RL, and draft verl integration, including the tested launcher’s R3/CP restriction and short-context defaults. No model switch has been decided.
- Investigated the Gemma CP restriction through source, history, developer responses, and pending fixes. Confirmed local-attention wiring, D512 distributed-kernel gaps, and explicit project-priority constraints. Found merged AutoModel CP support and NeMo RL CP2 functional evidence; revised the plan to evaluate that existing path while preserving the full-parameter, 40k, and verl requirements. Recorded why neither the 4k RL recipe nor the large-cluster 64k SFT recipe validates our target hardware/workload.
