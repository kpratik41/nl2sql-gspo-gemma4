#!/usr/bin/env python3
# Main objective: Gemma 4 31B CP=2 + FFPA (or validated efficient attention)
# for long-context full-parameter NL2SQL RL in verl. CP reduces per-rank
# sequence pressure; efficient attention avoids quadratic score storage.
# Gates: forward memory/speed -> numerical/semantic parity -> real CP2
# -> backward/optimizer/resume -> verl RL -> NL2SQL. FFPA is optional.
# This file: Isolate local precision and mask behavior with fixed decoder inputs and FP32 references before scaling training.
# See GEMMA4_LONG_CONTEXT_RL.md for the objective and complete roadmap.
"""Replay real Gemma decoder inputs against FP32, with causal/window invariants."""
import copy,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.nn.attention.flex_attention import create_block_mask
from ffpa_attn import ffpa_attn_func,CuTeDSLBackend

root=Path('benchmark_results/layer_replay');root.mkdir(parents=True,exist_ok=True)
path=Path.home()/'.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'
torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
text=Path('benchmark_results/model_parity/fixture.txt').read_text();unit=tok.encode(text,add_special_tokens=False)
layers={m.layer_idx:m for m in model.modules() if m.__class__.__name__=='Gemma4TextDecoderLayer' and m.layer_idx in [0,5,6,59]}
inputs={};handles=[];rows=[]
def save(i):
 def hook(module,args,kwargs):
  kw=dict(kwargs)
  if args: kw['hidden_states']=args[0]
  kw['hidden_states']=kw['hidden_states'].detach().clone()
  kw['shared_kv_states']={}
  inputs[i]=kw
 return hook
for i,m in layers.items():handles.append(m.register_forward_pre_hook(save(i),with_kwargs=True))
def ffpa(module,q,k,v,attention_mask,**kw):
 assert module.sliding_window is None and q.shape[-1]==512
 return ffpa_attn_func(q,k,v,is_causal=True,enable_gqa=True,scale=kw['scaling'],backend=CuTeDSLBackend()).transpose(1,2),None
ALL_ATTENTION_FUNCTIONS.register('replay_ffpa',ffpa)

def cast(x):
 if isinstance(x,torch.Tensor) and x.is_floating_point():return x.float()
 if isinstance(x,tuple):return tuple(cast(a) for a in x)
 if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
 return x

def run(layer,kw,backend):
 layer.config._attn_implementation=backend
 layer.self_attn.config._attn_implementation=backend
 kw=dict(kw);kw['shared_kv_states']={};kw.pop('kernel_options',None)
 if backend=='flex_attention':
  n=kw['hidden_states'].shape[1];window=layer.self_attn.sliding_window
  def allowed(b,h,q,k):return (q>=k)&((q-k<window) if window else True)
  kw['attention_mask']=create_block_mask(allowed,1,1,n,n,device='cuda',_compile=True)
  kw['kernel_options']={'BLOCK_M':32,'BLOCK_N':32,'num_stages':1}
 values={}
 def pre(module,args):values['attention']=args[0].detach().float().cpu()
 def post(module,args,out):values['projected']=out[0].detach().float().cpu()
 h1=layer.self_attn.o_proj.register_forward_pre_hook(pre)
 h2=layer.self_attn.register_forward_hook(post)
 try:
  with torch.inference_mode():values['decoder']=layer(**kw).detach().float().cpu()
 finally:h1.remove();h2.remove()
 return values

def error(a,b):return {'max_abs':(a-b).abs().max().item(),'relative_l2':((a-b).norm()/b.norm().clamp_min(1e-12)).item(),'rms_error':(a-b).square().mean().sqrt().item()}

for length in [128,1024,1025]:
 model.set_attn_implementation('eager');inputs.clear()
 ids=torch.tensor(([tok.bos_token_id]+unit*(length//len(unit)+1))[:length],device='cuda').unsqueeze(0)
 with torch.inference_mode():model(input_ids=ids,use_cache=False,logits_to_keep=1)
 # Remove capture hooks during replay, otherwise replay would overwrite fixtures.
 for h in handles:h.remove()
 handles=[]
 for i,layer in layers.items():
  kw=inputs[i]
  fp=copy.deepcopy(layer).float().eval()
  reference=run(fp,cast(kw),'eager')
  # An independent FP32 SDPA math reference controls eager implementation accuracy.
  from torch.nn.attention import sdpa_kernel,SDPBackend
  with sdpa_kernel(SDPBackend.MATH):mathref=run(fp,cast(kw),'sdpa')
  rows.append({'length':length,'layer':i,'backend':'fp32_sdpa_math','errors':{k:error(mathref[k],reference[k]) for k in reference}})
  del fp,mathref
  for backend in ['eager','sdpa','flex_attention']+(['replay_ffpa'] if layer.self_attn.sliding_window is None else []):
   actual=run(layer,kw,backend)
   r={'length':length,'layer':i,'backend':backend,'input_sha256':hashlib.sha256(kw['hidden_states'].cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),'scale':layer.self_attn.scaling,'window':layer.self_attn.sliding_window,'errors':{k:error(actual[k],reference[k]) for k in reference}}
   suffix=dict(kw);suffix['hidden_states']=kw['hidden_states'].clone();suffix['hidden_states'][:,length//2:]+=1
   changed=run(layer,suffix,backend)
   r['causal_prefix_max_abs']=(actual['decoder'][:,:length//2]-changed['decoder'][:,:length//2]).abs().max().item()
   if layer.self_attn.sliding_window and length>layer.self_attn.sliding_window:
    older=dict(kw);older['hidden_states']=kw['hidden_states'].clone();older['hidden_states'][:,0]+=1
    changed=run(layer,older,backend)
    r['outside_window_last_query_max_abs']=(actual['attention'][:,-1]-changed['attention'][:,-1]).abs().max().item()
   rows.append(r)
   (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(length,i,backend,r['errors']['decoder'],r['causal_prefix_max_abs'],r.get('outside_window_last_query_max_abs'),flush=True)
  del reference
 for i,m in layers.items():handles.append(m.register_forward_pre_hook(save(i),with_kwargs=True))
for h in handles:h.remove()
