"""Ulysses/Flex output and backward checks against independent FP32 SDPA math."""
import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from src.nl2sql_verl.fsdp_attention import attention,make_masks
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    dist.init_process_group('nccl',device_id=torch.device('cuda',torch.cuda.current_device()))
    rank=dist.get_rank();sp=dist.get_world_size();set_ulysses_sequence_parallel_group(dist.group.WORLD)
    results=[]
    for n,split in [(128,0),(1028,0),(1152,517)]:
        for d,hk,window in [(256,16,1024),(512,4,None)]:
            torch.manual_seed(71);h=32
            q=torch.randn(1,h,n,d,device='cuda',dtype=torch.bfloat16)*.1
            k=torch.randn(1,hk,n,d,device='cuda',dtype=torch.bfloat16)*.1
            v=torch.randn_like(k);grad=torch.randn_like(q)
            pos=torch.arange(n,device='cuda')[None]
            if split:pos[:,split:]-=split
            valid=torch.arange(n,device='cuda')[None]<n-3
            masks=make_masks(pos,valid,1024)
            docs=(pos==0).cumsum(-1)
            allowed=(valid[:,:,None]&valid[:,None,:]&(docs[:,:,None]==docs[:,None,:])&(torch.arange(n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None,:]))
            if window:allowed &= torch.arange(n,device='cuda')[:,None]-torch.arange(n,device='cuda')[None,:]<window
            ref=[x.float().detach().requires_grad_() for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.MATH):
                y=F.scaled_dot_product_attention(*ref,attn_mask=allowed[:,None],scale=1.,enable_gqa=True)
                (y*grad.float()).sum().backward()
            local=[x.chunk(sp,dim=2)[rank].contiguous().detach().requires_grad_() for x in (q,k,v)]
            out,_=attention(SimpleNamespace(training=True),*local,masks['sliding_attention' if window else 'full_attention'],scaling=1.)
            (out.transpose(1,2)*grad.chunk(sp,dim=2)[rank]).sum().backward()
            targets=[y.detach(),*[x.grad for x in ref]]
            actual=[out.transpose(1,2),*[x.grad for x in local]]
            checks=[]
            for name,a,b in zip(['output','dq','dk','dv'],actual,targets):
                b=b.chunk(sp,dim=2)[rank];delta=(a.float()-b).norm()/b.norm().clamp_min(1e-12)
                checks.append({'name':name,'relative_l2':float(delta),'finite':bool(a.isfinite().all())})
            row={'length':n,'split':split,'head_dim':d,'window':window,'checks':checks,'pass':all(x['finite'] and x['relative_l2']<.03 for x in checks)}
            results.append(row);print(json.dumps({'rank':rank,**row}),flush=True)
    path=Path(f'outputs/fsdp_cp_recipe/attention_rank{rank}.json');path.write_text(json.dumps(results,indent=2)+'\n')
    assert all(x['pass'] for x in results)
    dist.destroy_process_group()

if __name__=='__main__':main()
