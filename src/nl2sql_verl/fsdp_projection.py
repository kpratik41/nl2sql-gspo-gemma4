"""Keep Gemma global-key FP32 projection arithmetic stable under CP.

On H200, cuBLAS selects a different SGEMM reduction for the narrow global-key
projection when CP halves the token count. Gemma amplifies that small forward
difference into large prefix-gradient differences. Only this small projection
uses full-sequence inputs; the other linears retain token partitioning.
"""
from types import MethodType

import torch
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather
from verl.utils.ulysses import (get_ulysses_sequence_parallel_group,
                                get_ulysses_sequence_parallel_rank,
                                get_ulysses_sequence_parallel_world_size)


def full_sequence_linear(module, inputs):
    sp = get_ulysses_sequence_parallel_world_size()
    if sp == 1:
        return F.linear(inputs, module.weight, module.bias)
    full = torch.cat(all_gather(inputs.contiguous(),
                     group=get_ulysses_sequence_parallel_group()), dim=1)
    output = F.linear(full, module.weight, module.bias)
    return output.chunk(sp, dim=1)[get_ulysses_sequence_parallel_rank()].contiguous()


def install_global_key_projection(model):
    for layer in model.model.language_model.layers:
        attention = layer.self_attn
        if not attention.is_sliding:
            if attention.is_kv_shared_layer:
                raise ValueError('Shared global keys are unsupported')
            attention.k_proj.forward = MethodType(full_sequence_linear, attention.k_proj)
