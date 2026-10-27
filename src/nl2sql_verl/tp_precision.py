"""Use FP32 for Gemma's TP activation sums while retaining BF16 computation."""
import torch


def enable_fp32_tp_reductions():
    enable_tp_reductions(torch.float32)


def enable_tp_reductions(accum_dtype):
    """Patch the pinned Gemma-owned sum, retaining its stream and backward contract.

    FSDP's reduce_dtype controls gradient reductions, not this forward sum.
    Summing BF16 contributions before rounding back to BF16 made full-model
    results depend strongly on microbatch shape. The existing custom backward
    passes the replicated output gradient to each local contribution unchanged;
    changing the forward accumulator dtype does not change that derivative.
    """
    from nemo_automodel.components.models.gemma4_moe.parallelization import (
        _ReduceFromTensorParallelRegion,
    )

    if accum_dtype not in (torch.float32, torch.float64):
        raise ValueError('TP accumulator must be float32 or float64')
    current = _ReduceFromTensorParallelRegion.forward
    if getattr(current, '_nl2sql_reduce_dtype', None) == accum_dtype:
        return
    original = getattr(current, '_nl2sql_original', current)

    def forward(ctx, local_output, process_group):
        if local_output.dtype not in (torch.bfloat16, torch.float16):
            return original(ctx, local_output, process_group)
        output = original(ctx, local_output.to(accum_dtype), process_group)
        return output.to(local_output.dtype)

    forward._nl2sql_reduce_dtype = accum_dtype
    forward._nl2sql_original = original
    _ReduceFromTensorParallelRegion.forward = staticmethod(forward)
