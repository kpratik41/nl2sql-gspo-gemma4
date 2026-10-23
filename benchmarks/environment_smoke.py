"""Environment checks only: these do not validate model parity or context parallelism.

Run normally for GPU/attention checks, or with torchrun and --distributed-only
for an NCCL all-reduce across the visible devices.
"""
import argparse
import json
import os
from datetime import timedelta

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distributed-only', action='store_true')
    args = parser.parse_args()
    if args.distributed_only:
        import torch.distributed as dist
        rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl', timeout=timedelta(seconds=90))
        try:
            value = torch.tensor([dist.get_rank() + 1.0], device='cuda')
            dist.all_reduce(value)
            size = dist.get_world_size()
            assert value.item() == size * (size + 1) / 2
            print(json.dumps({'rank': rank, 'nccl_all_reduce': 'pass', 'sum': value.item()}), flush=True)
        finally:
            dist.destroy_process_group()
        return

    import importlib.metadata as metadata
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.attention import ffpa_attention as binding
    from ffpa_attn import ffpa_attn_func, CuTeDSLBackend
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    print(json.dumps({'versions': {name: metadata.version(name) for name in
        ['torch', 'transformers', 'nemo-automodel', 'ffpa-attn', 'nvidia-cutlass-dsl']}}), flush=True)
    assert NeMoAutoModelForImageTextToText is not None
    assert binding.register_ffpa_attention()
    assert binding._ffpa_low_level_ready() and binding._ffpa_varlen_ready()
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
            assert (x @ x.T).isfinite().all()
            print(json.dumps({'gpu': device, 'name': torch.cuda.get_device_name(device),
                              'bf16_matmul': 'pass'}), flush=True)

    torch.cuda.set_device(0)
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, h, 128, 512, device='cuda', dtype=torch.bfloat16)
               for h in (32, 4, 4)]
    # Model scaling is 1.0. Scale the random Q/K inputs to avoid saturated softmax.
    q, k = q * 0.1, k * 0.1
    kr, vr = [x.repeat_interleave(8, dim=1) for x in (k, v)]
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), kr.float(), vr.float(), is_causal=True, scale=1.0)
    out = ffpa_attn_func(q, k, v, is_causal=True, scale=1.0,
                         enable_gqa=True, backend=CuTeDSLBackend())
    torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)
    print('FFPA dense BF16 D512 GQA forward: pass', flush=True)
    train_inputs = [x.detach().requires_grad_() for x in (q, k, v)]
    ffpa_attn_func(*train_inputs, is_causal=True, scale=1.0,
                   enable_gqa=True, backend=CuTeDSLBackend()).float().square().mean().backward()
    assert all(x.grad is not None and x.grad.isfinite().all() for x in train_inputs)
    print('FFPA dense backward finite gradients: pass (not gradient parity)', flush=True)

    dense, dense_lse = binding._ffpa_dense_fwd(q, kr, vr, scale=1.0, causal=True)
    torch.testing.assert_close(dense.float(), ref, atol=0.02, rtol=0.02)
    grads = binding._ffpa_dense_bwd(torch.ones_like(dense), q, kr, vr,
                                   dense, dense_lse, scale=1.0, causal=True)
    assert all(g.isfinite().all() for g in grads)
    packed = [x.squeeze(0).transpose(0, 1).contiguous() for x in (q, kr, vr)]
    cu = torch.tensor([0, 128], device='cuda', dtype=torch.int32)
    varlen, varlen_lse = binding._ffpa_varlen_fwd(*packed, cu, cu, 128, 128, scale=1.0, causal=True)
    torch.testing.assert_close(varlen.transpose(0, 1).unsqueeze(0).float(), ref, atol=0.02, rtol=0.02)
    grads = binding._ffpa_varlen_bwd(torch.ones_like(varlen), *packed, varlen,
                                    varlen_lse, cu, cu, 128, 128, scale=1.0, causal=True)
    assert all(g.isfinite().all() for g in grads)
    print('AutoModel dense/varlen FFPA forward and finite backward: pass', flush=True)

    mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, 1, 1, 128, 128, device='cuda')
    out = torch.compile(flex_attention)(q, k, v, block_mask=mask, scale=1.0,
        enable_gqa=True, kernel_options={'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 1})
    torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)
    torch.cuda.synchronize()
    print('FlexAttention compiled BF16 D512 GQA forward: pass', flush=True)


if __name__ == '__main__':
    main()
