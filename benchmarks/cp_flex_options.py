"""Benchmark-local SM120 tiles and opt-in precision diagnostics; masks stay intact."""
import torch
from torch.nn.attention.flex_attention import flex_attention

OPTIONS = {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_M1': 32, 'BLOCK_N1': 32,
           'BLOCK_M2': 32, 'BLOCK_N2': 32, 'num_stages': 1, 'num_warps': 4,
           'FORCE_USE_FLEX_ATTENTION': True}


def configure_flex_tiles(module, backward_block=32, attention_precision='bf16'):
    if attention_precision not in {'bf16', 'fp32'}:
        raise ValueError(f'Unsupported attention precision: {attention_precision}')
    compiled = torch.compile(flex_attention, dynamic=True)
    def call(*args, **kwargs):
        kwargs['kernel_options'] = dict(OPTIONS)
        if attention_precision == 'fp32':
            args = tuple(x.float() if i < 3 else x for i, x in enumerate(args))
            kwargs['kernel_options']['BLOCK_M'] = 16
            kwargs['kernel_options']['BLOCK_N'] = 16
        for name in ('BLOCK_M1', 'BLOCK_N1', 'BLOCK_M2', 'BLOCK_N2'):
            kwargs['kernel_options'][name] = backward_block
        return compiled(*args, **kwargs)
    module._gemma4_cp_compiled_flex_attn = call
