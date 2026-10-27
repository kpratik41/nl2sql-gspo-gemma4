"""CP2 backward on real pretrained Q/K/V, including an odd-length padded boundary."""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from src.nl2sql_verl.fsdp_attention import attention,make_masks
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',device_id=torch.device('cuda',torch.cuda.current_device()))
    sp=dist.get_world_size();rank=dist.get_rank();set_ulysses_sequence_parallel_group(dist.group.WORLD)
    rows=[]
    for layer in (0,5,59):
        data=torch.load(f'outputs/fsdp_cp_recipe/attention_capture_layer{layer}.pt',map_location='cpu',weights_only=False)
        n=data['q'].shape[2];width=(n+sp-1)//sp*sp
        tensors=[F.pad(data[name].cuda(),(0,0,0,width-n)) for name in ('q','k','v')]
        pos=torch.arange(width,device='cuda')[None];valid=pos<n;masks=make_masks(pos,valid,1024)
        allowed=valid[:,:,None]&valid[:,None,:]&(pos[:,:,None]>=pos[:,None,:])
        if layer==0:allowed&=pos[:,:,None]-pos[:,None,:]<1024
        torch.manual_seed(311);dy=torch.randn_like(tensors[0]);dy[:,:,n:]=0
        reference=[t.float().detach().requires_grad_() for t in tensors]
        with sdpa_kernel(SDPBackend.MATH):
            out=F.scaled_dot_product_attention(*reference,attn_mask=allowed[:,None],scale=data['scaling'],enable_gqa=True)
            (out*dy.float()).sum().backward()
        for precision in ('0','1'):
            os.environ['NL2SQL_ATTENTION_FP32']=precision
            local=[t.chunk(sp,dim=2)[rank].contiguous().detach().requires_grad_() for t in tensors]
            y,_=attention(None,*local,masks['sliding_attention' if layer==0 else 'full_attention'],scaling=data['scaling'])
            (y.transpose(1,2)*dy.chunk(sp,dim=2)[rank]).sum().backward()
            checks=[]
            for name,a,b in zip(('output','dq','dk','dv'),[y.transpose(1,2),*[t.grad for t in local]],[out,*[t.grad for t in reference]]):
                a=a.detach().float();b=b.detach().chunk(sp,dim=2)[rank]
                checks.append({'name':name,'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-12)),'cosine':float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)),'finite':bool(a.isfinite().all())})
            row={'layer':layer,'attention_fp32':precision=='1','checks':checks,'pass':all(t['finite'] and t['relative_l2']<.03 for t in checks)}
            rows.append(row);print(json.dumps({'rank':rank,**row}),flush=True)
    Path(f'outputs/fsdp_cp_recipe/captured_backward_rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    dist.destroy_process_group()

if __name__=='__main__':main()
