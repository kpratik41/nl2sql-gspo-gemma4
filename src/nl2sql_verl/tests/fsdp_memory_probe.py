"""Warm full-model forward/backward memory and actual Adam updates with FSDP2/SP."""
import argparse,json,os,time,math
from pathlib import Path
import torch
import torch.distributed as dist
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.config import HFModelConfig,FSDPEngineConfig,FSDPOptimizerConfig
from verl.trainer.config import CheckpointConfig
from src.nl2sql_verl.fsdp_engine import GemmaFSDPEngine
from src.nl2sql_verl.fsdp_attention import BACKEND
from src.nl2sql_verl.tests.fsdp_cp_probe import MODEL


def make_batch(lengths,seed):
    g=torch.Generator(device='cuda').manual_seed(seed)
    ids=torch.nested.as_nested_tensor([torch.randint(3,128,(n,),device='cuda',generator=g) for n in lengths],layout=torch.jagged)
    pos=torch.nested.as_nested_tensor([torch.arange(n,device='cuda') for n in lengths],layout=torch.jagged)
    masks=torch.nested.as_nested_tensor([(torch.arange(n,device='cuda')>=n-64).float() for n in lengths],layout=torch.jagged)
    batch=TensorDict({'input_ids':ids,'position_ids':pos,'loss_mask':masks,'temperature':torch.ones(len(lengths),device='cuda')},batch_size=[len(lengths)])
    tu.assign_non_tensor(batch,use_remove_padding=True,use_fused_kernels=False,calculate_entropy=False,use_dynamic_bsz=False,micro_batch_size_per_gpu=len(lengths))
    return batch


def token_loss(model_output,data,dp_group):
    mask=data['loss_mask'].values().roll(-1)
    return -(model_output['log_probs'].values()*mask).sum()*tu.get_non_tensor_data(data,'dp_size')/tu.get_non_tensor_data(data,'batch_num_tokens'),{}


def main():
    p=argparse.ArgumentParser();p.add_argument('--sp',type=int,default=2);p.add_argument('--lengths',type=int,nargs='+',default=[4096,8192,16384,32768]);p.add_argument('--micro',type=int,default=1);p.add_argument('--accum',type=int,default=1);p.add_argument('--fp32-residual',action='store_true');p.add_argument('--output',required=True);p.add_argument('--model',default=MODEL);p.add_argument('--checkpoint');a=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2);dist.init_process_group('nccl',device_id=torch.device('cuda',torch.cuda.current_device()))
    precision={'param_dtype':'fp32' if a.fp32_residual else 'bf16','reduce_dtype':'fp32','buffer_dtype':'fp32'}
    if a.fp32_residual:torch.set_autocast_cache_enabled(False)
    e=GemmaFSDPEngine(HFModelConfig(path=a.model,load_tokenizer=False,use_remove_padding=True,override_config={'attn_implementation':BACKEND}),FSDPEngineConfig(strategy='fsdp2',mixed_precision=precision,use_torch_compile=False,ulysses_sequence_parallel_size=a.sp,use_no_sync_for_gradient_accumulation=False,pad_to_length=True,pad_to_length_bucket=256),FSDPOptimizerConfig(lr=1e-6,total_training_steps=100),CheckpointConfig());e.initialize()
    if a.fp32_residual:e._autocast_dtype=torch.bfloat16
    reports=[]
    for length in a.lengths:
        batch=make_batch([length-i*113 for i in range(a.micro*a.accum)],51+e.get_data_parallel_rank())
        tu.assign_non_tensor(batch,micro_batch_size_per_gpu=a.micro)
        passes=[]
        for repeat in range(2):
            e.optimizer_zero_grad();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();start=time.monotonic()
            with e.train_mode(disable_auto_offload=True):
                out=e.train_batch(batch,token_loss)
            torch.cuda.synchronize()
            norm=float(out['metrics']['grad_norm']) if e.is_mp_src_rank_with_outputs() else None
            row={'repeat':repeat,'baseline_gib':base/2**30,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-start,'grad_norm':norm}
            assert norm is None or math.isfinite(norm)
            passes.append(row)
            print(json.dumps({'rank':dist.get_rank(),'length':length,'micro':a.micro,**row}),flush=True)
        reports.append({'length':length,'microbatch':a.micro,'passes':passes})
        Path(f'{a.output}_rank{dist.get_rank()}.json').write_text(json.dumps({'sp':a.sp,'fp32_residual':a.fp32_residual,'accumulation_rounds':a.accum,'reports':reports},indent=2)+'\n')
    if a.checkpoint:
        rng_before={'cpu':torch.get_rng_state().clone(),'cuda':torch.cuda.get_rng_state().clone()}
        scheduler_before=e.lr_scheduler.state_dict().copy()
        e.save_checkpoint(a.checkpoint,global_step=1)
        # Host RAM holds the sharded snapshots: check every model/Adam tensor,
        # rather than assuming that a successful load restored every layer.
        chosen=dict(e.module.named_parameters())
        before={n:(p.to_local() if hasattr(p,'to_local') else p).detach().cpu().clone() for n,p in chosen.items()}
        optim_before={(n,k):(v.to_local() if hasattr(v,'to_local') else v).detach().cpu().clone() for n,p in chosen.items() for k,v in e.optimizer.state[p].items() if isinstance(v,torch.Tensor)}
        with torch.no_grad():
            for p in chosen.values():p.add_(.1)
            for p in chosen.values():
                for v in e.optimizer.state[p].values():
                    if isinstance(v,torch.Tensor):v.add_(1)
        torch.rand(7);torch.rand(7,device='cuda');e.lr_scheduler.step()
        e.load_checkpoint(a.checkpoint,del_local_after_load=False)
        torch.testing.assert_close(torch.get_rng_state(),rng_before['cpu'],rtol=0,atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(),rng_before['cuda'],rtol=0,atol=0)
        assert e.lr_scheduler.state_dict()==scheduler_before
        for n,p in chosen.items():
            assert torch.equal((p.to_local() if hasattr(p,'to_local') else p).cpu(),before[n]),n
            for k,v in e.optimizer.state[p].items():
                if isinstance(v,torch.Tensor):assert torch.equal((v.to_local() if hasattr(v,'to_local') else v).cpu(),optim_before[n,k]),(n,k)
        Path(f'{a.output}_restore_rank{dist.get_rank()}.json').write_text(json.dumps({'pass':True,'checkpoint':a.checkpoint,'scope':'Exact comparison of every model parameter/Adam tensor, scheduler and CPU/CUDA RNG','parameters':len(chosen),'optimizer_tensors':len(optim_before)})+'\n')
    dist.destroy_process_group()

if __name__=='__main__':main()
