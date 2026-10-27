"""Independent single-GPU HF Gemma reference, without TP, DP, or recipe attention."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Gemma4ForConditionalGeneration
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.nl2sql_verl.tests.full_cp1_probe import MODEL, make_bird_batch
from src.nl2sql_verl.tests.cp1_microbatch import make_batch
from src.nl2sql_verl.batching import pad_sequences


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial-model-checkpoint',required=True)
    p.add_argument('--fixture',choices=['bird','synthetic'],default='bird')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.preferred_blas_library('cublaslt')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
    model=Gemma4ForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa')
    state=model.state_dict()
    covered={}
    with torch.no_grad():
        for path in sorted(Path(a.initial_model_checkpoint).glob('*.safetensors')):
            with safe_open(path,framework='pt',device='cpu') as f:
                info=json.loads(f.metadata()['DCP_SHARDING_INFO'])
                for name in f.keys():
                    if name not in state:
                        raise KeyError(f'Checkpoint tensor missing from HF model: {name}')
                    shape=f.get_slice(name).get_shape()
                    offsets=info[name]['saved_offsets']
                    key=(tuple(offsets),tuple(shape))
                    if key in covered.setdefault(name,set()):continue
                    covered[name].add(key)
                    region=tuple(slice(o,o+s) for o,s in zip(offsets,shape,strict=True))
                    state[name][region].copy_(f.get_tensor(name).to(device='cuda',dtype=state[name].dtype))
            print('loaded',path.name,flush=True)
    for name,regions in covered.items():
        count=sum(__import__('math').prod(shape) for offsets,shape in regions)
        assert count==state[name].numel(),(name,count,state[name].numel())
    del state
    for name,param in model.named_parameters():
        if any(x in name for x in ('vision_tower','embed_vision','multi_modal_projector','audio_tower','embed_audio')):
            param.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    device=torch.device('cuda:0')
    batch=make_bird_batch(0,device) if a.fixture=='bird' else make_batch([1025,769],device,seed=27)
    ids=list(batch['input_ids'].unbind());positions=list(batch['position_ids'].unbind())
    masks=list(batch['loss_mask'].unbind());temperatures=batch['temperature']
    count=float(sum(x.sum() for x in masks))
    selected=lambda n: 'language_model' in n and any(f'.layers.{i}.' in n for i in (0,5,59)) and any(x in n for x in ('q_proj','k_proj','o_proj','down_proj','input_layernorm'))
    samples=[];passes=[]
    for micro in (1,2):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        loss_sum=0.;start=time.monotonic()
        for begin in range(0,len(ids),micro):
            rows=ids[begin:begin+micro]
            nested=torch.nested.as_nested_tensor(rows,layout=torch.jagged)
            pos=torch.nested.as_nested_tensor(positions[begin:begin+micro],layout=torch.jagged)
            data,lengths=pad_sequences(nested,pos,0,bucket_size=256)
            valid=~data['padding_mask']
            loss_mask=torch.stack([F.pad(x,(0,valid.shape[1]-len(x))) for x in masks[begin:begin+micro]])[:,1:].bool()
            with sdpa_kernel(SDPBackend.MATH):
                logits=model(input_ids=data['input_ids'],position_ids=data['position_ids'],
                    attention_mask=valid,use_cache=False).logits
                selected_logits=logits[:,:-1][loss_mask].float()
                temp=temperatures[begin:begin+micro,None].expand_as(valid)[:,1:][loss_mask]
                loss=F.cross_entropy(selected_logits/temp[:,None],data['input_ids'][:,1:][loss_mask],reduction='sum')/count
                loss.backward()
            loss_sum+=float(loss.detach())
            del logits,selected_logits,loss
        samples.append({n:v.grad.detach().float().cpu().clone() for n,v in model.named_parameters() if selected(n) and v.grad is not None})
        row={'microbatch':micro,'loss':loss_sum,'seconds':time.monotonic()-start,
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        passes.append(row);print(json.dumps(row),flush=True)
    rows=[]
    for name,x in samples[0].items():
        y=samples[1][name]
        rows.append({'name':name,'relative_l2':float((x-y).norm()/x.norm().clamp_min(1e-12)),
            'reference_norm':float(x.norm()),'difference_norm':float((x-y).norm()),
            'cosine':float(F.cosine_similarity(x.flatten().double(),y.flatten().double(),dim=0))})
    result={'implementation':'HF Gemma4, BF16 weights, SDPA math, single GPU, no optimizer',
        'checkpoint':a.initial_model_checkpoint,'fixture':a.fixture,'passes':passes,'rows':rows,
        'worst_relative_l2':max(r['relative_l2'] for r in rows),
        'min_cosine':min(r['cosine'] for r in rows)}
    Path(f'outputs/cp1_recipe/hf_bf16_{a.fixture}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
