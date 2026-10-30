"""Diagnostic: keep HF's full attention shape while partitioning token states."""
import torch
from torch.distributed.nn.functional import all_gather
from transformers import AttentionInterface
from transformers.integrations.flex_attention import flex_attention_forward
from verl.utils.ulysses import (get_ulysses_sequence_parallel_group,
                                get_ulysses_sequence_parallel_rank,
                                get_ulysses_sequence_parallel_world_size)
from src.nl2sql_verl.fsdp_attention import BACKEND, OPTIONS


def attention(module, query, key, value, attention_mask, scaling=None, dropout=0., **kwargs):
    assert not dropout
    group = get_ulysses_sequence_parallel_group()
    sp = get_ulysses_sequence_parallel_world_size()
    rank = get_ulysses_sequence_parallel_rank()
    if sp > 1:
        query, key, value = [torch.cat(all_gather(x.contiguous(), group=group), dim=2)
                             for x in (query, key, value)]
    options = dict(OPTIONS)
    options.update(BLOCK_M=16, BLOCK_N=16, FLOAT32_PRECISION=repr('ieee'))
    output, weights = flex_attention_forward(module, query, key, value, attention_mask,
                                             scaling=scaling, kernel_options=options)
    return output.chunk(sp, dim=1)[rank].contiguous(), weights


if __name__ == '__main__':
    AttentionInterface.register(BACKEND, attention)
    from src.nl2sql_verl.tests.fsdp_cp_probe import main
    main()
