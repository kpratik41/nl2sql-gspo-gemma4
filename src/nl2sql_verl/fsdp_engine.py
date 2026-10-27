"""Small Gemma extension of native verl FSDP2 for exact Flex/Ulysses attention."""
import torch
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.utils.ulysses import get_ulysses_sequence_parallel_rank
from .fsdp_attention import BACKEND,make_hf_masks
from . import advantages, vllm_compat
vllm_compat.register()


@EngineRegistry.register(model_type='language_model',backend='fsdp2',device='cuda',vendor='nvidia')
class GemmaFSDPEngine(FSDPEngineWithLMHead):
    def _build_fsdp_module(self, module):
        module=super()._build_fsdp_module(module)
        # Keep FP32 embeddings/residuals when param_dtype=fp32, while linear
        # projections still use BF16 autocast. Avoid retaining a second full
        # model's worth of cached casts across a checkpointed forward.
        self._autocast_dtype=torch.bfloat16
        torch.set_autocast_cache_enabled(False)
        return module

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        # Controller publishes a completion marker before pruning old saves.
        # Disable native pre-save rotation so a failed new save preserves them.
        return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep=None, **kwargs)

    def _build_module(self):
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.preferred_blas_library('cublaslt')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
        torch._dynamo.config.recompile_limit=512
        torch._dynamo.config.accumulated_recompile_limit=4096
        model=super()._build_module()
        if model.config.model_type!='gemma4':raise ValueError('Expected HF Gemma4')
        if model.config.text_config._attn_implementation!=BACKEND:raise ValueError('Select the recipe Flex attention backend')
        if model.config.text_config.num_kv_shared_layers:raise ValueError('Shared-KV variants are unvalidated')
        for name,param in model.named_parameters():
            if any(x in name for x in ('vision_tower','embed_vision','multi_modal_projector','audio_tower','embed_audio')):
                param.requires_grad_(False)
        return model

    def prepare_model_inputs(self,micro_batch):
        inputs,output_args=super().prepare_model_inputs(micro_batch)
        pos=inputs['position_ids']
        # Native verl leaves multimodal input IDs unsliced, expecting the model
        # to slice after embedding. Text-only Gemma can slice before embedding.
        # This avoids modifying HF forwards or duplicating embedding work.
        if not self.use_remove_padding:
            raise ValueError('The recipe uses packed text with suffix padding')
        inputs['attention_mask']=make_hf_masks(pos,self.module.config.text_config.sliding_window)
        if self.use_ulysses_sp:
            if not hasattr(self.module.config,'vision_config'):raise ValueError('Expected unsliced Gemma VLM inputs')
            rank=get_ulysses_sequence_parallel_rank();sp=self.ulysses_sequence_parallel_size
            for key in ('input_ids','position_ids'):
                inputs[key]=inputs[key].chunk(sp,dim=-1)[rank].contiguous()
        return inputs,output_args


def install_hf_compat():
    """Skip verl's homogeneous-head/FlashAttention patch for supported HF paths.

    The pinned generic patch reads an ambiguous global KV-head count. Native
    padded SDPA needs no model patch; our explicit Flex backend handles each
    layer and its Ulysses collectives itself. Fail on other packed Gemma paths.
    """
    import verl.workers.engine.fsdp.transformer_impl as impl
    original=impl.apply_monkey_patch
    if getattr(original,'_nl2sql_gemma',False):return
    def apply(model,use_remove_padding=False,ulysses_sp_size=1,use_fused_kernels=False,**kwargs):
        if model.config.model_type!='gemma4':
            return original(model,use_remove_padding=use_remove_padding,ulysses_sp_size=ulysses_sp_size,use_fused_kernels=use_fused_kernels,**kwargs)
        if use_fused_kernels:raise ValueError('Fused Gemma forward is unvalidated')
        if (use_remove_padding or ulysses_sp_size>1) and model.config.text_config._attn_implementation!=BACKEND:
            raise ValueError('Packed/SP Gemma requires the explicit Flex/Ulysses backend')
        for cfg in model.config.text_config.per_layer_config:
            if cfg.num_key_value_heads%ulysses_sp_size:raise ValueError('SP must divide each layer KV-head count')
    apply._nl2sql_gemma=True
    impl.apply_monkey_patch=apply


install_hf_compat()
