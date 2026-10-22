#!/usr/bin/env python3
# Main objective: Gemma 4 31B CP=2 + FFPA (or validated efficient attention)
# for long-context full-parameter NL2SQL RL in verl. CP reduces per-rank
# sequence pressure; efficient attention avoids quadratic score storage.
# Gates: forward memory/speed -> numerical/semantic parity -> real CP2
# -> backward/optimizer/resume -> verl RL -> NL2SQL. FFPA is optional.
# This file: Check same-weight teacher-forced probabilities and layer outputs before accepting full-model backend equivalence.
# See GEMMA4_LONG_CONTEXT_RL.md for the objective and complete roadmap.
"""Same-weight teacher-forced Gemma parity diagnostic; no generation sampling."""
import hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer

root=Path('benchmark_results/model_parity');root.mkdir(parents=True,exist_ok=True)
path=Path.home()/'.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'
torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
text='''Database schema: customers(customer_id, name, city); orders(order_id, customer_id, amount, order_date).
Question: Which customers placed at least three orders in 2025? Return their names and total order amounts.
SQL: SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' GROUP BY c.customer_id, c.name HAVING COUNT(*) >= 3;
Explanation: Join on the customer key, filter dates before aggregation, and apply the count condition with HAVING.
'''
(root/'fixture.txt').write_text(text)
unit=tok.encode(text,add_special_tokens=False)
rows=[];states={};attention_checks=[]
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward
originals={b:ALL_ATTENTION_FUNCTIONS.get_interface(b,eager_attention_forward) for b in ['eager','sdpa','flex_attention']}
def wrap(original):
    def dispatch(module,q,k,v,attention_mask,**kw):
        output=original(module,q,k,v,attention_mask,**kw)
        if module.layer_idx in [0,5,6,59]:
            positions=sorted(set([0,q.shape[2]//2,q.shape[2]-1]))
            idx=torch.tensor(positions,device=q.device);keys=torch.arange(k.shape[2],device=q.device)
            references=[]
            for h in range(q.shape[1]):
                kh=h//(q.shape[1]//k.shape[1])
                scores=(q[0,h,idx].float() @ k[0,kh].float().T)*kw['scaling']
                allowed=idx[:,None]>=keys[None,:]
                if module.sliding_window is not None: allowed &= idx[:,None]-keys[None,:]<module.sliding_window
                scores.masked_fill_(~allowed,float('-inf'))
                references.append(scores.softmax(-1) @ v[0,kh].float())
            ref=torch.stack(references)
            actual=output[0][0,idx].transpose(0,1).float()
            attention_checks.append({'layer':module.layer_idx,'scale':kw['scaling'],'window':module.sliding_window,'max_abs':(actual-ref).abs().max().item(),'relative_l2':((actual-ref).norm()/ref.norm().clamp_min(1e-12)).item()})
        return output
    return dispatch
for name,fn in originals.items(): ALL_ATTENTION_FUNCTIONS.register(name,wrap(fn))
modules=[(n,m) for n,m in model.named_modules() if m.__class__.__name__=='Gemma4TextDecoderLayer']
assert len(modules)==60
handles=[]
def hook(name):
    def capture(module,args,out):
        x=out[0] if isinstance(out,tuple) else out
        idx=sorted(set([0,x.shape[1]//2,x.shape[1]-1]))
        states[name]=x[0,idx].detach().float().cpu()
    return capture
for n,m in modules: handles.append(m.register_forward_hook(hook(n)))

def measure(ids,backend):
    model.set_attn_implementation(backend)
    states.clear();attention_checks.clear()
    with torch.inference_mode():
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=0,**({'kernel_options':{'BLOCK_M':32,'BLOCK_N':32,'num_stages':1}} if backend=='flex_attention' else {})).logits
        # Chunk vocabulary normalization so FP32 temporaries stay bounded.
        lp=[]
        for start in range(0,ids.shape[1]-1,32):
            chunk=logits[0,start:min(start+32,ids.shape[1]-1)].float()
            labels=ids[0,start+1:min(start+33,ids.shape[1])]
            lp.append(chunk.gather(1,labels[:,None]).squeeze(1)-chunk.logsumexp(-1))
        lp=torch.cat(lp).cpu()
        pos=torch.linspace(0,ids.shape[1]-1,32,device='cuda').long().unique()
        sampled=logits[0,pos].float().log_softmax(-1).cpu()
        del logits
    return lp,sampled,dict(states)

for length in [128,1023,1024,1025,2048]:
    ids=torch.tensor(([tok.bos_token_id]+unit*((length//len(unit))+1))[:length],device='cuda').unsqueeze(0)
    digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    ref_lp,ref_dist,ref_states=measure(ids,'eager')
    for backend in ['eager','sdpa','flex_attention']:
        lp,dist,hidden=measure(ids,backend)
        kl=(ref_dist.exp()*(ref_dist-dist)).sum(-1)
        layers=[]
        for name,_ in modules:
            a,b=hidden[name],ref_states[name]
            layers.append({'name':name,'relative_l2':((a-b).norm()/b.norm().clamp_min(1e-12)).item(),'max_abs':(a-b).abs().max().item()})
        r={'length':length,'backend':backend,'input_sha256':digest,'reference':'eager BF16, same loaded weights','nll':-lp.mean().item(),'reference_nll':-ref_lp.mean().item(),'mean_logprob_abs_error':(lp-ref_lp).abs().mean().item(),'max_logprob_abs_error':(lp-ref_lp).abs().max().item(),'nll_delta':(ref_lp.mean()-lp.mean()).item(),'sampled_mean_kl':kl.mean().item(),'sampled_max_kl':kl.max().item(),'sampled_top1_agreement':(dist.argmax(-1)==ref_dist.argmax(-1)).float().mean().item(),'layers':layers,'same_qkv_fp32_attention_checks':list(attention_checks)}
        # Predeclared investigation gates, not universal BF16 guarantees.
        r['investigation_gate_pass']=abs(r['nll_delta'])<=.01 and r['max_logprob_abs_error']<=.1 and r['sampled_max_kl']<=.001
        rows.append(r)
        (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print({k:v for k,v in r.items() if k!='layers'},flush=True)
for h in handles:h.remove()
