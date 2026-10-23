"""Two-rank Flex ring attention output/gradient checks, not model training.

Unpacked/unpadded Gemma-shaped full and sliding QKV; independent FP32 SDPA
math reference over the full sequence, compared to each local output/gradient.
"""
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel, SDPBackend
from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    assert dist.get_world_size() == 2
    root = Path('benchmark_results/cp_forward')
    rows = []
    for name, length, dim, kvheads, window in [('global', 128, 512, 4, None), ('sliding', 1280, 256, 16, 1024)]:
        torch.manual_seed(42)
        q, k, v = [torch.randn(1, h, length, dim, device='cuda', dtype=torch.bfloat16) for h in (32, kvheads, kvheads)]
        q, k = q * 0.1, k * 0.1
        upstream_grad = torch.randn_like(q)
        reference_inputs = [x.float().requires_grad_() for x in (q, k, v)]
        positions = torch.arange(length, device='cuda')
        allowed = positions[:, None] >= positions[None, :]
        if window:
            allowed &= positions[:, None] - positions[None, :] < window
        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(
                *reference_inputs, attn_mask=allowed[None, None], scale=1.0, enable_gqa=True)
        reference.backward(upstream_grad.float())
        start, end = rank * (length//2), (rank+1) * (length//2)
        local = [x[:, :, start:end].clone().requires_grad_() for x in (q, k, v)]
        module = SimpleNamespace(sliding_window=window, config=SimpleNamespace(use_bidirectional_attention=None))
        from cp_flex_options import configure_flex_tiles
        configure_flex_tiles(module, backward_block=16)
        ctx = cpa.CPRingAttentionContext(module=module, query=local[0], key=local[1], value=local[2],
            cp_mesh=None, cp_group=dist.group.WORLD, cp_size=2, cp_rank=rank,
            seq_local=length//2, seq_full=length, seq_global_start=start,
            attn_mask=None, dropout_p=0.0, is_causal=True, scale=1.0, enable_gqa=True,
            kwargs={}, metadata={'_packed_seq_ids': torch.ones(1, length//2, device='cuda', dtype=torch.long)},
            metadata_seq_dims={'_packed_seq_ids': 1})
        print(f'[rank {rank}] {name}: ring forward/backward', flush=True)
        actual = cpa._Gemma4FlexRingAttention.apply(*local, ctx)
        actual.backward(upstream_grad[:, :, start:end].contiguous())
        metrics = {}
        for stage, got, expected in [('output', actual, reference[:, :, start:end])] + [
                (label, tensor.grad, ref.grad[:, :, start:end])
                for label, tensor, ref in zip(('dq', 'dk', 'dv'), local, reference_inputs)]:
            delta = got.float() - expected
            ok = bool(torch.allclose(got.float(), expected, atol=.03, rtol=.03))
            metrics[stage] = {'max_abs': delta.abs().max().item(),
                              'relative_l2': (delta.norm()/expected.norm().clamp_min(1e-12)).item(),
                              'pass': ok}
        row = {'layer': name, 'rank': rank, 'tokens': length, 'head_dim': dim,
               'query_heads': 32, 'kv_heads': kvheads, 'window': window,
               'reference': 'full-sequence FP32 SDPA math with BF16 QKV values',
               'atol': .03, 'rtol': .03, 'metrics': metrics}
        row['tile_policy'] = '32x32 forward, 16x16 backward, one stage, four warps, forced Flex'
        rows.append(row)
        print(json.dumps(row), flush=True)
        (root/f'attention_gradients_rank{rank}.json').write_text(json.dumps(rows, indent=2)+'\n')
        dist.barrier()
    status = torch.tensor([int(all(m['pass'] for row in rows for m in row['metrics'].values()))], device='cuda')
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    if not status.item():
        raise SystemExit('Attention parity gate failed; inspect saved JSON.')


if __name__ == '__main__':
    main()
