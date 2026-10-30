"""Diagnostic: preserve the global key projection's full-sequence GEMM shape."""
from types import MethodType
import torch
from torch.distributed.nn.functional import all_gather
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
from verl.utils.ulysses import (get_ulysses_sequence_parallel_group,
                                get_ulysses_sequence_parallel_rank,
                                get_ulysses_sequence_parallel_world_size)

original_init = Gemma4TextAttention.__init__


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if not self.is_sliding:
        original = self.k_proj.forward

        def forward(module, inputs):
            sp = get_ulysses_sequence_parallel_world_size()
            if sp == 1:
                return original(inputs)
            gathered = torch.cat(all_gather(inputs.contiguous(),
                                 group=get_ulysses_sequence_parallel_group()), dim=1)
            return original(gathered).chunk(sp, dim=1)[get_ulysses_sequence_parallel_rank()].contiguous()
        self.k_proj.forward = MethodType(forward, self.k_proj)


if __name__ == '__main__':
    Gemma4TextAttention.__init__ = initialize
    from src.nl2sql_verl.tests.fsdp_cp_probe import main
    main()
