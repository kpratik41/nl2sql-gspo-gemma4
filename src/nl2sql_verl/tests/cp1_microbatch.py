"""Eight-rank TP4/DP2 parity for unequal-length independent CP1 sequences."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from tensordict import TensorDict
from transformers import AutoConfig
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import AutomodelOptimizerConfig
from src.nl2sql_verl.engine import Gemma4CPEngine,Gemma4CPEngineConfig


def make_batch(lengths, device, seed=17):
    gen=torch.Generator(device=device).manual_seed(seed)
    ids=torch.nested.as_nested_tensor([torch.randint(3,128,(n,),device=device,generator=gen) for n in lengths],layout=torch.jagged)
    positions=torch.nested.nested_tensor_from_jagged(torch.cat([torch.arange(n,device=device) for n in lengths]),ids.offsets())
    mask=torch.nested.nested_tensor_from_jagged(torch.cat([(torch.arange(n,device=device)>=n//2).float() for n in lengths]),ids.offsets())
    batch=TensorDict({'input_ids':ids,'position_ids':positions,'loss_mask':mask,'temperature':torch.tensor([.8 if i%2==0 else 1.2 for i in range(len(lengths))],device=device)},batch_size=[len(lengths)])
    tu.assign_non_tensor(batch,calculate_entropy=False,use_fused_kernels=False,use_dynamic_bsz=False,micro_batch_size_per_gpu=1)
    return batch


def loss_fn(model_output,data,dp_group):
    lp=model_output['log_probs'].values();mask=data['loss_mask'].values().roll(-1)
    return -(lp*mask).sum()*tu.get_non_tensor_data(data,'dp_size',None)/tu.get_non_tensor_data(data,'batch_num_tokens',None),{}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dtype',default='fp32');p.add_argument('--tp',type=int,default=4);p.add_argument('--tp-reduce-dtype',choices=['fp32','fp64']);args=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2);dist.init_process_group('nccl')
    path=str(Path('outputs/verl_recipe_checks/tiny_gemma_tp4').resolve())
    engine=Gemma4CPEngine(SimpleNamespace(path=path,hf_config=AutoConfig.from_pretrained(path)),
        Gemma4CPEngineConfig(tp_size=args.tp,tp_reduce_dtype=args.tp_reduce_dtype,cp_size=1,model_dtype=args.dtype,cpu_master_weights=False,param_offload=False,optimizer_offload=False,activation_checkpointing=True,defer_fsdp_grad_sync=True,sequence_bucket_size=32),
        AutomodelOptimizerConfig(lr=1e-6,total_training_steps=1),CheckpointConfig());engine.initialize()
    dp=engine.get_data_parallel_rank();batch=make_batch([33+dp*16,67+dp*16],torch.device('cuda',torch.cuda.current_device()),seed=17+dp)
    samples=[]; losses=[]
    for micro in (1,2):
        tu.assign_non_tensor(batch,micro_batch_size_per_gpu=micro)
        engine.optimizer_zero_grad()
        with engine.train_mode(disable_auto_offload=True):
            engine.forward_backward_batch(batch,loss_fn,forward_only=False)
            from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
            synchronize_tp_replica_gradients([engine.module],engine.device_mesh)
            samples.append({n:p.grad.to_local().detach().float().clone() for n,p in engine.module.named_parameters() if p.grad is not None})
    rows=[]
    for name,a in samples[0].items():
        b=samples[1][name];rel=float((a-b).norm()/a.norm().clamp_min(1e-12));rows.append({'name':name,'relative_l2':rel,'reference_norm':float(a.norm()),'difference_norm':float((a-b).norm()),'cosine':float(torch.nn.functional.cosine_similarity(a.flatten().double(),b.flatten().double(),dim=0)), 'finite':bool(torch.isfinite(b).all())})
    worst=max(x['relative_l2'] for x in rows);ok=all(x['finite'] for x in rows) and worst < (.03 if args.dtype=='bf16' else .001)
    gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,{'rank':dist.get_rank(),'dp':dp,'worst_relative_l2':worst,'pass':ok})
    Path(f'outputs/cp1_recipe/tiny_diagnostics_{args.dtype}_rank{dist.get_rank()}.json').write_text(json.dumps({'rows':rows,'aggregate_relative_l2':(sum(r['difference_norm']**2 for r in rows)/sum(r['reference_norm']**2 for r in rows))**.5},indent=2)+'\n')
    assert all(x['pass'] for x in gathered),gathered
    # Exercise real optimizer update + distributed model/optimizer save and restore.
    tu.assign_non_tensor(batch,micro_batch_size_per_gpu=2)
    engine.train_batch(batch,loss_fn)
    ckpt=str(Path(f'outputs/cp1_recipe/tiny_checkpoint_{args.dtype}').resolve())
    before={n:p.to_local().detach().clone() for n,p in engine.module.named_parameters()}
    before_optim={(id(p),k):v.to_local().detach().clone() for p,state in engine.optimizer.state.items() for k,v in state.items() if hasattr(v,'to_local')}
    engine.save_checkpoint(ckpt,global_step=1)
    with torch.no_grad():
        for p in engine.module.parameters():p.add_(.1)
        for state in engine.optimizer.state.values():
            for v in state.values():
                if hasattr(v,'to_local'):v.add_(1.)
    engine.load_checkpoint(ckpt)
    for n,p in engine.module.named_parameters():torch.testing.assert_close(p.to_local(),before[n],rtol=0,atol=0)
    for p,state in engine.optimizer.state.items():
        for k,v in state.items():
            if hasattr(v,'to_local'):torch.testing.assert_close(v.to_local(),before_optim[id(p),k],rtol=0,atol=0)
    if dist.get_rank()==0:
        result={'tp':args.tp,'tp_reduce_dtype':args.tp_reduce_dtype,'dp':engine.get_data_parallel_size(),'cp':1,'dtype':args.dtype,'microbatch_parity':gathered,'checkpoint_restored':True,'pass':True}
        Path(f'outputs/cp1_recipe/tiny_microbatch_{args.dtype}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    dist.destroy_process_group()
if __name__=='__main__':main()
