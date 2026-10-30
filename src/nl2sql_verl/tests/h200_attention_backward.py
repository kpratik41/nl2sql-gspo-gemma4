"""Long-input H200 attention backward check against independent SDPA math."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from verl.utils.ulysses import set_ulysses_sequence_parallel_group
from src.nl2sql_verl.fsdp_attention import attention, make_hf_masks


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    rank, sp = dist.get_rank(), dist.get_world_size()
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    results = []
    for n in (5120, 11008):
        for dimension, kv_heads, window in ((256, 16, 1024), (512, 4, None)):
            torch.manual_seed(71)
            q = torch.randn(1, 32, n, dimension, device='cuda')
            k = torch.randn(1, kv_heads, n, dimension, device='cuda')
            v = torch.randn_like(k)
            # Earlier model layers receive gradients at prompt positions too.
            dy = torch.randn_like(q)
            dy[:, :, -3:] = 0
            pos = torch.arange(n, device='cuda')[None]
            pos[:, -3:] = torch.arange(3, device='cuda')
            masks = make_hf_masks(pos, 1024)
            docs = (pos == 0).cumsum(-1)
            allowed = (docs[:, :, None] == docs[:, None, :]) & (pos[:, :, None] >= pos[:, None, :])
            if window:
                allowed &= (pos[:, :, None] - pos[:, None, :]) < window
            # Gemma4 normalizes Q/K and uses scale=1 rather than 1/sqrt(D).
            scale = 1.0
            reference = [x.detach().requires_grad_() for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.MATH):
                y = F.scaled_dot_product_attention(*reference, attn_mask=allowed[:,None],
                                                  scale=scale, enable_gqa=True)
                (y * dy).sum().backward()
            for canonical in ('0', '1'):
                os.environ['NL2SQL_FLEX_CANONICAL_LAYOUT'] = canonical
                local = [x.chunk(sp, dim=2)[rank].contiguous().detach().requires_grad_() for x in (q,k,v)]
                output, _ = attention(SimpleNamespace(training=True), *local,
                                      masks['sliding_attention' if window else 'full_attention'], scaling=scale)
                # Match a real o_proj's contiguous BSHD output gradient.
                (output * dy.chunk(sp, dim=2)[rank].transpose(1,2).contiguous()).sum().backward()
                checks = []
                for name, actual, expected in zip(('output','dq','dk','dv'),
                        [output.transpose(1,2), *[x.grad for x in local]],
                        [y, *[x.grad for x in reference]]):
                    target = expected.chunk(sp, dim=2)[rank]
                    checks.append({'name': name, 'relative_l2': float((actual-target).norm()/target.norm().clamp_min(1e-12)),
                                   'finite': bool(actual.isfinite().all())})
                row = {'length': n, 'dimension': dimension, 'window': window,
                       'canonical': canonical == '1', 'checks': checks,
                       'pass': all(c['finite'] and c['relative_l2'] < .03 for c in checks)}
                results.append(row)
                print(json.dumps({'rank': rank, **row}), flush=True)
            del q,k,v,dy,reference,y,local,output
            torch.cuda.empty_cache()
            Path(f'outputs/h200_checks/attention_backward_rank{rank}.json').write_text(json.dumps(results, indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
