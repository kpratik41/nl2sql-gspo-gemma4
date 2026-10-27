"""Warm exact global-attention forward/backward memory for CP1 and Ulysses CP2."""
import json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from src.nl2sql_verl.fsdp_attention import attention,make_hf_masks
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    dist.init_process_group('nccl',device_id=torch.device('cuda',torch.cuda.current_device()))
    sp=dist.get_world_size();rank=dist.get_rank()
    set_ulysses_sequence_parallel_group(dist.group.WORLD if sp>1 else None)
    rows=[]
    for length in (4096,8192,16384,32768):
        torch.manual_seed(19+rank)
        dtype=torch.float32 if any(os.environ.get(k)=='1' for k in ('NL2SQL_ATTENTION_BF16_IO','NL2SQL_ATTENTION_FP32')) else torch.bfloat16
        tensors=[torch.randn(1,h,length//sp,512,device='cuda',dtype=dtype,requires_grad=True) for h in (32,4,4)]
        pos=torch.arange(length,device='cuda')[None]
        torch.cuda.reset_peak_memory_stats()
        masks=make_hf_masks(pos,1024)
        torch.cuda.synchronize()
        mask_peak=torch.cuda.max_memory_allocated()/2**20
        for repeat in range(2):
            for t in tensors:t.grad=None
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic();baseline=torch.cuda.memory_allocated()
            out,_=attention(None,*tensors,masks['full_attention'],scaling=1.)
            out.float().square().mean().backward();torch.cuda.synchronize()
            row={'global_tokens':length,'local_tokens':length//sp,'repeat':repeat,'baseline_mib':baseline/2**20,'mask_construction_peak_mib':mask_peak,'peak_allocated_mib':max(mask_peak,torch.cuda.max_memory_allocated()/2**20),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'seconds':time.monotonic()-start,'finite':all(bool(t.grad.isfinite().all()) for t in tensors)}
            assert row['finite'];print(json.dumps({'rank':rank,**row}),flush=True)
            del out
            if repeat==1:rows.append(row)
        del masks,tensors,pos
    precision=('fp32_'+os.environ.get('NL2SQL_FLEX_FP32_PRECISION','tf32x3')) if os.environ.get('NL2SQL_ATTENTION_FP32')=='1' else ('bf16_fp32io' if os.environ.get('NL2SQL_ATTENTION_BF16_IO')=='1' else 'bf16')
    Path(f'outputs/fsdp_cp_recipe/attention_memory_{precision}_sp{sp}_rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    dist.destroy_process_group()

if __name__=='__main__':main()
