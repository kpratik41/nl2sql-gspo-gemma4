"""Recipe-local verl engine for the pinned AutoModel Gemma4 ring-CP contract.

This uses AutoModel FSDP2 across DP×CP, with TP inside each CP partition.
Only per-token scalars are gathered over CP; logits remain vocabulary-sharded.
"""
import gc
import importlib
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.workers.config import AutomodelEngineConfig
from verl.workers.engine.base import BaseEngine, EngineRegistry
from verl.workers.engine.automodel.transformer_impl import AutomodelEngineWithLMHead
from verl.workers.engine.utils import detach_tree

from ...parallel_logprobs import vocab_logprobs
from ... import advantages  # noqa: F401; register on controller and worker imports


@dataclass
class Gemma4CPEngineConfig(AutomodelEngineConfig):
    cpu_master_weights: bool = True
    logprob_chunk_size: int = 128


@EngineRegistry.register(model_type='language_model', backend='automodel', device='cuda', vendor='nvidia')
class Gemma4CPEngine(AutomodelEngineWithLMHead):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **kwargs):
        # The upstream initializer still calls the retired create_device_mesh API.
        BaseEngine.__init__(self)
        from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
        from nemo_automodel.components.distributed.mesh import ParallelismSizes
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
        self.model_config, self.engine_config = model_config, engine_config
        self.optimizer_config, self.checkpoint_config = optimizer_config, checkpoint_config
        self.rank, self.mode = dist.get_rank(), None
        if engine_config.distributed_strategy != 'fsdp2' or engine_config.pp_size != 1 or engine_config.ep_size != 1:
            raise ValueError('This recipe supports FSDP2 with TP/CP only')
        if engine_config.sequence_parallel:
            raise ValueError('TP sequence parallelism is separate from the Gemma ring CP path')
        if not engine_config.cpu_master_weights:
            raise ValueError('This four-GPU recipe requires CPU FP32 master weights/optimizer and BF16 compute')
        policy = FSDP2Config(sequence_parallel=False, activation_checkpointing=engine_config.activation_checkpointing,
            defer_fsdp_grad_sync=False, reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16), offload_policy=CPUOffloadPolicy(pin_memory=True))
        self.setup = DistributedSetup.build(strategy=policy,
            parallelism_sizes=ParallelismSizes(tp_size=engine_config.tp_size, cp_size=engine_config.cp_size,
                                               dp_replicate_size=1),
            activation_checkpointing=engine_config.activation_checkpointing, world_size=dist.get_world_size())
        self.device_mesh, self.moe_mesh = self.setup.mesh_context.device_mesh, None
        self.distributed_config = policy
        self._is_offload_param = self._is_offload_optimizer = True
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.preferred_blas_library('cublaslt')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        torch._dynamo.config.recompile_limit = 128
        torch._dynamo.config.accumulated_recompile_limit = 1024
        torch._dynamo.config.fail_on_recompile_limit_hit = True

    def initialize(self):
        from nemo_automodel import NeMoAutoModelForImageTextToText
        from nemo_automodel.components.models.common import BackendConfig
        self.module = NeMoAutoModelForImageTextToText.from_pretrained(self.model_config.path,
            dtype=torch.float32, distributed_setup=self.setup,
            attn_implementation='sdpa', use_liger_kernel=False, use_sdpa_patching=False,
            backend=BackendConfig(linear='torch', attn='sdpa', rms_norm='torch', experts='torch',
                                  dispatcher='torch', enable_hf_state_dict_adapter=False),
            text_config={'use_cache': False, 'cp_full_attn_backend': 'flex', 'cp_sliding_attn_backend': 'flex'})
        if not hasattr(self.module, '_cp_shard_batch_aux_only'):
            raise TypeError('Expected the custom AutoModel Gemma4 model-owned CP implementation')
        # Text-only recipe: freeze unused vision/audio towers and their adapters.
        for name, p in self.module.named_parameters():
            if any(part in name for part in ('vision_tower', 'multi_modal_projector', 'audio_tower', 'embed_audio')):
                p.requires_grad_(False)
        from benchmarks.cp_flex_options import configure_flex_tiles
        for module in self.module.modules():
            if hasattr(module, '_cp_manual_metadata_keys'):
                configure_flex_tiles(module, backward_block=16, attention_precision='bf16')
        flex = importlib.import_module('torch.nn.attention.flex_attention')
        original = flex.create_block_mask
        compiled = torch.compile(original, dynamic=False)
        def create_mask(*args, **kwargs):
            kwargs.pop('_compile', None)
            return compiled(*args, **kwargs)
        flex.create_block_mask = create_mask
        self.optimizer = None if self.engine_config.forward_only else torch.optim.AdamW(
            [p for p in self.module.parameters() if p.requires_grad], lr=self.optimizer_config.lr,
            betas=tuple(self.optimizer_config.betas), eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay, foreach=False)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer) if self.optimizer is not None else None
        self._build_checkpointer()

    def get_data_parallel_rank(self):
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
        return get_flat_mesh(self.device_mesh, 'dp').get_local_rank()

    def get_data_parallel_size(self):
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
        return get_flat_mesh(self.device_mesh, 'dp').size()

    def get_data_parallel_group(self):
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
        return get_flat_mesh(self.device_mesh, 'dp').get_group()

    def is_mp_src_rank_with_outputs(self):
        return self.device_mesh['tp'].get_local_rank() == 0 and self.device_mesh['cp'].get_local_rank() == 0

    def to(self, device, model=True, optimizer=True, grad=True):
        # FSDP CPUOffloadPolicy owns parameter movement. Moving the complete model
        # or optimizer to CUDA here defeats it and exhausts the four-card budget.
        if device not in {'cpu', 'cuda'}:
            raise ValueError(device)
        if device == 'cpu' and hasattr(self, 'module'):
            self.module.reshard()
            self._clear_output_holder()
            gc.collect()
            torch.cuda.empty_cache()

    def _clear_output_holder(self):
        for module in self.module.modules():
            if hasattr(module, '_gemma4_tp_output_holder'):
                module._gemma4_tp_output_holder[0] = None

    def optimizer_step(self):
        from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
        synchronize_tp_replica_gradients([self.module], self.device_mesh)
        return super().optimizer_step()

    def _build_checkpointer(self):
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
        self.checkpointer = Checkpointer(config=CheckpointingConfig(enabled=True,
            checkpoint_dir='checkpoints/', model_save_format='safetensors',
            model_repo_id=self.model_config.path, save_consolidated=False,
            cpu_offload=True, load_metadata_with_gloo=True),
            dp_rank=get_flat_mesh(self.device_mesh, 'dp_cp').get_local_rank(),
            tp_rank=self.device_mesh['tp'].get_local_rank(), pp_rank=0)

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        import numpy as np
        if hdfs_path is not None:
            raise ValueError('This recipe checkpoints to a shared local filesystem')
        self.checkpointer.save_model(self.module, local_path)
        if self.optimizer is not None:
            self.checkpointer.save_optimizer(self.optimizer, self.module, local_path, scheduler=[self.lr_scheduler])
        Path(local_path).mkdir(parents=True, exist_ok=True)
        torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(), 'global_step': global_step},
            Path(local_path)/f'rng_rank{self.rank}.pt')
        dist.barrier()

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        import numpy as np
        if hdfs_path is not None:
            raise ValueError('This recipe restores from a shared local filesystem')
        self.checkpointer.load_model(self.module, os.path.join(local_path, 'model'))
        if self.optimizer is not None:
            self.checkpointer.load_optimizer(self.optimizer, self.module, local_path, scheduler=[self.lr_scheduler])
        rng = torch.load(Path(local_path)/f'rng_rank{self.rank}.pt', map_location='cpu', weights_only=False)
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        dist.barrier()

    def forward_step(self, micro_batch, loss_function, forward_only):
        from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
        from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices
        micro_batch = micro_batch.to(get_device_id())
        if micro_batch.batch_size[0] != 1:
            raise ValueError('Use micro_batch_size_per_gpu=1; packed/multi-example CP is not validated here')
        if tu.get_non_tensor_data(micro_batch, 'use_fused_kernels', False):
            raise ValueError('The Gemma CP adapter owns the vocabulary-sharded logprob calculation')
        ids = micro_batch['input_ids'][0].reshape(1, -1)
        positions = micro_batch['position_ids'][0].reshape(1, -1)
        length = ids.shape[1]
        unit = 2*self.engine_config.cp_size
        padded = (length+unit-1)//unit*unit
        pad_id = self.model_config.hf_config.text_config.pad_token_id or 0
        ids = torch.nn.functional.pad(ids, (0, padded-length), value=pad_id)
        positions = torch.nn.functional.pad(positions, (0, padded-length))
        padding_mask = torch.arange(padded, device=ids.device)[None] >= length
        # Global shift BEFORE sharding is essential at the CP partition boundary.
        targets = ids.roll(-1, dims=1)
        batch = {'input_ids': ids, 'labels': targets, 'position_ids': positions, 'padding_mask': padding_mask}
        if self.engine_config.cp_size == 1:
            sharder = ContextParallelSharder(shard_batch=self.module._cp_shard_batch_aux_only,
                local_token_global_indices=contiguous_local_indices, device_mesh=self.device_mesh)
        else:
            sharder = ContextParallelSharder(self.module, self.device_mesh, batch)
        context, batch = sharder.shard(batch)
        local_targets = batch.pop('labels')
        batch['attention_mask'] = {'full_attention': None, 'sliding_attention': None}
        calculate_entropy = tu.get_non_tensor_data(micro_batch, 'calculate_entropy', False)
        with context(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = self.module(**batch, use_cache=False, logits_to_keep=0).logits
            if not isinstance(logits, DTensor):
                raise TypeError('Expected TP vocabulary-sharded DTensor logits')
            local = logits.to_local()
            temperature = micro_batch['temperature']
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.reshape(-1)[0]
            lp, entropy = vocab_logprobs(local, local_targets, temperature,
                group=self.device_mesh['tp'].get_group(),
                vocab_start=self.device_mesh['tp'].get_local_rank()*local.shape[-1],
                chunk_size=self.engine_config.logprob_chunk_size, entropy=calculate_entropy)
            # Autograd's CP gather sums gradients across replicated losses. FSDP's
            # DP×CP average cancels the CP multiplicity, retaining the global loss.
            full_lp = sharder.gather_token_tensor(lp)[:, :length]
            model_output = {'log_probs': torch.nested.nested_tensor_from_jagged(
                full_lp.reshape(-1), micro_batch['input_ids'].offsets())}
            if calculate_entropy:
                full_entropy = sharder.gather_token_tensor(entropy)[:, :length]
                model_output['entropy'] = torch.nested.nested_tensor_from_jagged(
                    full_entropy.reshape(-1), micro_batch['input_ids'].offsets())
            if loss_function is not None:
                loss, metrics = loss_function(model_output=model_output, data=micro_batch,
                                               dp_group=self.get_data_parallel_group())
            elif forward_only:
                loss, metrics = full_lp.new_zeros(()), {}
            else:
                raise ValueError('Training needs a verl loss function')
        self._clear_output_holder()
        return loss, {'model_output': detach_tree(model_output), 'loss': loss.detach().item(), 'metrics': metrics}

    def get_per_tensor_param(self, **kwargs):
        # Materialize one HF tensor at a time for colocated rollout weight sync.
        # All ranks participate in DTensor collectives, even output-non-source ranks.
        from verl.utils.model import convert_weight_keys
        state = convert_weight_keys(self.module.state_dict(), self.module)
        def parameters():
            for name, value in state.items():
                # Offloaded DTensors must move their local shard to CUDA before
                # NCCL reconstruction; never all-gather a CPU shard with NCCL.
                on_device = value.to(device=get_device_id(), dtype=torch.bfloat16)
                full = on_device.full_tensor() if isinstance(on_device, DTensor) else on_device
                yield name, full
        return parameters(), None
