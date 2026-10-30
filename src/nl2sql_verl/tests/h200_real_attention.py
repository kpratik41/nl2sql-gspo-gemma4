"""Capture real Gemma attention operands/adjoints and replay CP2 independently."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MODEL = '/opt/dlami/nvme/nl2sql/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'
ROOT = Path('/opt/dlami/nvme/nl2sql/checks/real_attention')


def capture():
    from transformers import AttentionInterface, Gemma4ForConditionalGeneration
    from transformers.integrations.flex_attention import flex_attention_forward
    from src.nl2sql_verl.tests.fsdp_cp_probe import fixture
    from src.nl2sql_verl.fsdp_attention import OPTIONS
    ROOT.mkdir(parents=True, exist_ok=True)
    tensors, response = fixture(MODEL, [13312], [0])[0]
    ids = tensors.cuda()[None]
    pos = torch.arange(ids.shape[1], device='cuda')[None]
    pad = (-ids.shape[1]) % 256
    mask = response.cuda()[None, 1:]
    if pad:
        ids = F.pad(ids, (0,pad))
        mask = F.pad(mask, (0,pad))
        pos = torch.cat([pos, torch.arange(pad, device='cuda')[None]], -1)
    captured = {}

    def attention(module, q, k, v, block, **kwargs):
        options = dict(OPTIONS)
        options.update(BLOCK_M=16, BLOCK_N=16, FLOAT32_PRECISION=repr('ieee'))
        output, weights = flex_attention_forward(module, q.float(), k.float(), v.float(), block,
                                                 kernel_options=options, **kwargs)
        layer = module.layer_idx
        if layer in (0, 5, 59) and layer not in captured:
            entry = dict(q=q.detach().cpu(), k=k.detach().cpu(), v=v.detach().cpu(),
                         output=output.detach().cpu(), pos=pos.cpu(), scaling=kwargs['scaling'],
                         sliding=module.is_sliding)
            captured[layer] = entry

            def gradient(dy):
                entry['dy'] = dy.detach().cpu()
                return dy
            output.register_hook(gradient)
        return output, weights

    AttentionInterface.register('flex_attention', attention)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, device_map={'':'cuda'}, attn_implementation='flex_attention')
    for name, parameter in model.named_parameters():
        if any(x in name for x in ('vision_tower','embed_vision','multi_modal_projector')):
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            parameter.register_post_accumulate_grad_hook(lambda p: setattr(p, 'grad', None))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        logits = model(input_ids=ids, position_ids=pos, use_cache=False,
                       logits_to_keep=mask[0].nonzero(as_tuple=True)[0]).logits[0]
        loss = F.cross_entropy(logits, ids[:,1:][mask])
        loss.backward()
    for layer, item in captured.items():
        assert 'dy' in item
        torch.save(item, ROOT/f'layer{layer}.pt')
    print(json.dumps({'status':'captured','layers':sorted(captured),'loss':float(loss.detach())}), flush=True)


def replay():
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group
    from src.nl2sql_verl.fsdp_attention import attention, make_hf_masks
    dist.init_process_group('nccl', device_id=torch.device('cuda',torch.cuda.current_device()))
    rank, sp = dist.get_rank(), dist.get_world_size()
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    rows = []
    for layer in (0,5,59):
        item = torch.load(ROOT/f'layer{layer}.pt', map_location='cpu', weights_only=False)
        q,k,v = [item[name].cuda() for name in ('q','k','v')]
        pos,dy = item['pos'].cuda(),item['dy'].cuda()
        masks = make_hf_masks(pos,1024)
        docs = (pos==0).cumsum(-1)
        allowed = (docs[:,:,None] == docs[:,None,:]) & (pos[:,:,None] >= pos[:,None,:])
        if item['sliding']:
            allowed &= pos[:,:,None]-pos[:,None,:] < 1024
        ref = [x.detach().requires_grad_() for x in (q,k,v)]
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(*ref, attn_mask=allowed[:,None],
                                                      scale=item['scaling'], enable_gqa=True)
            (expected.transpose(1,2)*dy).sum().backward()
        for canonical in ('0','1'):
            os.environ['NL2SQL_FLEX_CANONICAL_LAYOUT'] = canonical
            local = [x.chunk(sp,dim=2)[rank].contiguous().detach().requires_grad_() for x in (q,k,v)]
            output,_ = attention(SimpleNamespace(training=True), *local,
                                 masks['sliding_attention' if item['sliding'] else 'full_attention'],
                                 scaling=item['scaling'])
            (output*dy.chunk(sp,dim=1)[rank].contiguous()).sum().backward()
            checks = []
            for name,actual,target in zip(('output','dq','dk','dv'),
                    [output.transpose(1,2),*[x.grad for x in local]], [expected,*[x.grad for x in ref]]):
                target = target.chunk(sp,dim=2)[rank]
                checks.append({'name':name,'relative_l2':float((actual-target).norm()/target.norm().clamp_min(1e-12))})
            row = {'layer':layer,'canonical':canonical=='1','checks':checks,
                   'q_std':float(q.std()),'k_std':float(k.std()),'scale':item['scaling']}
            rows.append(row)
            print(json.dumps({'rank':rank,**row}), flush=True)
        del q,k,v,dy,ref,expected,local,output
        torch.cuda.empty_cache()
    Path(f'outputs/h200_checks/real_attention_rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','replay']);args=p.parse_args()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.preferred_blas_library('cublas')
    capture() if args.mode=='capture' else replay()
