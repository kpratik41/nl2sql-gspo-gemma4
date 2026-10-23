"""Four-rank TP2/CP2 GSPO forward/backward and CP loss-scaling check.

Run with python -m torch.distributed.run --standalone --nproc-per-node=4 -m
src.nl2sql_verl.tests.distributed_logprobs
"""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.context_parallel.sharder import (
    contiguous_local_indices, gather_token_tensor_by_indices,
)
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config import ActorConfig

from src.nl2sql_verl.parallel_logprobs import vocab_logprobs


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl')
    mesh = init_device_mesh('cuda', (2,2), mesh_dim_names=('cp','tp'))
    cp, tp = mesh['cp'].get_local_rank(), mesh['tp'].get_local_rank()
    torch.manual_seed(17)
    x = torch.randn(1, 32, 16, device='cuda')
    initial_weight = torch.randn(16, 64, device='cuda')*.2
    labels = torch.randint(64, (1, 32), device='cuda')
    mask = torch.ones(1,32, device='cuda')
    mask[:, 8:16] = 0  # Tool observations must not receive policy loss.
    reference_weight = initial_weight.clone().requires_grad_()
    reference_logdist = (x@reference_weight/.8).log_softmax(-1)
    reference_lp = reference_logdist.gather(-1, labels[...,None]).squeeze(-1)
    old_lp = reference_lp.detach()+torch.linspace(-.1,.1,32,device='cuda')[None]
    cfg = ActorConfig(strategy='automodel', rollout_n=2, ppo_micro_batch_size_per_gpu=1,
                      clip_ratio_low=.2, clip_ratio_high=.28, loss_agg_mode='token-mean')
    advantages = torch.full_like(reference_lp, .7)
    expected_loss, _ = compute_policy_loss_gspo(old_lp, reference_lp, advantages, mask, 'token-mean', cfg)
    expected_loss.backward()
    index = contiguous_local_indices(mesh['cp'],32,device=x.device)
    weight = initial_weight[:, tp*32:(tp+1)*32].clone().requires_grad_()
    logits = x.index_select(1,index)@weight
    lp, entropy = vocab_logprobs(logits, labels.index_select(1,index), .8,
        group=mesh['tp'].get_group(), vocab_start=tp*32, chunk_size=3, entropy=True)
    full_lp = gather_token_tensor_by_indices(mesh['cp'],lp,index)
    full_entropy = gather_token_tensor_by_indices(mesh['cp'],entropy,index)
    expected_entropy = -(reference_logdist.exp()*reference_logdist).sum(-1)
    loss, _ = compute_policy_loss_gspo(old_lp, full_lp, advantages, mask, 'token-mean', cfg)
    loss.backward()
    # FSDP across CP averages these per-partition contributions in the engine.
    dist.all_reduce(weight.grad,group=mesh['cp'].get_group())
    weight.grad /= mesh['cp'].size()
    expected_gradient = reference_weight.grad[:,tp*32:(tp+1)*32]
    torch.testing.assert_close(full_lp,reference_lp,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(full_entropy,expected_entropy,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(weight.grad,expected_gradient,atol=2e-6,rtol=2e-6)
    error = (weight.grad-expected_gradient).abs().max()
    dist.all_reduce(error,op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        result={'status':'pass','tp':2,'cp':2,'max_gradient_absolute_error':error.item(),
                'loss':loss.item(),'reference_loss':expected_loss.item(),
                'scope':'synthetic TP-vocabulary logprobs, CP gather, actual verl GSPO and loss masking; not full Gemma'}
        destination=Path('outputs/verl_recipe_checks/distributed_logprobs.json')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
