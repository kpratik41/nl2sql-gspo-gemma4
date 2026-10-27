"""31B CP1/TP4/DP2 microbatch parity and longest-context memory gate."""
import argparse,json,math,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from transformers import AutoConfig
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import AutomodelOptimizerConfig
from src.nl2sql_verl.engine import Gemma4CPEngine,Gemma4CPEngineConfig
from src.nl2sql_verl.tests.cp1_microbatch import make_batch,loss_fn
MODEL='/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'


def make_bird_batch(dp,device):
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer
 from tensordict import TensorDict
 from src.nl2sql_verl.prepare_data import TOOLS
 tok=AutoTokenizer.from_pretrained(MODEL)
 records=pq.read_table('outputs/cp1_recipe/data/train.parquet').slice(dp*2,2).to_pylist()
 tokens=[];masks=[]
 for record in records:
  prompt=tok.apply_chat_template(record['prompt'],tools=TOOLS,tokenize=True,add_generation_prompt=True,return_dict=False)
  sql=json.loads(record['reward_model']['ground_truth'])['gold_sql']
  response=tok.encode('```sql\n'+sql+'\n```',add_special_tokens=False)
  tokens.append(torch.tensor(prompt+response,device=device));masks.append(torch.tensor([0.]*len(prompt)+[1.]*len(response),device=device))
 ids=torch.nested.as_nested_tensor(tokens,layout=torch.jagged)
 pos=torch.nested.nested_tensor_from_jagged(torch.cat([torch.arange(t.numel(),device=device) for t in tokens]),ids.offsets())
 mask=torch.nested.nested_tensor_from_jagged(torch.cat(masks),ids.offsets())
 batch=TensorDict({'input_ids':ids,'position_ids':pos,'loss_mask':mask,'temperature':torch.ones(2,device=device)},batch_size=[2])
 tu.assign_non_tensor(batch,calculate_entropy=False,use_fused_kernels=False,use_dynamic_bsz=False,micro_batch_size_per_gpu=1)
 return batch


def main():
 p=argparse.ArgumentParser();p.add_argument('--dtype',default='fp32');p.add_argument('--long-length',type=int,default=13312);p.add_argument('--initial-model-checkpoint');p.add_argument('--sequence-bucket-size',type=int,default=256);p.add_argument('--output-tag',default='');p.add_argument('--parity-only',action='store_true');p.add_argument('--tp',type=int,default=4);p.add_argument('--blas',choices=['cublas','cublaslt']);p.add_argument('--trace',action='store_true');p.add_argument('--tp-reduce-fp32',action='store_true');p.add_argument('--bird',action='store_true');p.add_argument('--attention-fp32',action='store_true');p.add_argument('--tp-reduce-fp64',action='store_true');a=p.parse_args()
 suffix=a.dtype+a.output_tag
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2);dist.init_process_group('nccl')
 engine=Gemma4CPEngine(SimpleNamespace(path=MODEL,hf_config=AutoConfig.from_pretrained(MODEL)),Gemma4CPEngineConfig(tp_size=a.tp,cp_size=1,model_dtype=a.dtype,cpu_master_weights=False,param_offload=False,optimizer_offload=False,activation_checkpointing=True,defer_fsdp_grad_sync=True,sequence_bucket_size=a.sequence_bucket_size,initial_model_checkpoint=a.initial_model_checkpoint,tp_reduce_dtype='fp64' if a.tp_reduce_fp64 else 'fp32' if a.tp_reduce_fp32 else None),AutomodelOptimizerConfig(lr=1e-6,total_training_steps=1),CheckpointConfig());engine.initialize()
 if a.blas:torch.backends.cuda.preferred_blas_library(a.blas)
 if a.attention_fp32:
  from benchmarks.cp_flex_options import configure_flex_tiles
  for module in engine.module.modules():
   if hasattr(module,'_cp_manual_metadata_keys'):configure_flex_tiles(module,backward_block=16,attention_precision='fp32')
 dp=engine.get_data_parallel_rank();device=torch.device('cuda',torch.cuda.current_device())
 batch=make_bird_batch(dp,device) if a.bird else make_batch([1025+dp*16,769+dp*16],device,seed=27+dp)
 selected=lambda n: 'language_model' in n and any(f'.layers.{i}.' in n for i in (0,5,59)) and any(x in n for x in ('q_proj','k_proj','o_proj','down_proj','input_layernorm'))
 results=[];gradients=[];traces=[];current_trace={}
 if a.trace:
  def capture(name):
   def hook(module,inputs,output):
    if name in current_trace:return
    x=output[0] if isinstance(output,tuple) else output
    if hasattr(x,'to_local'):x=x.to_local()
    if isinstance(x,torch.Tensor) and x.ndim>=3:current_trace[name]=x[0,:32].detach().float().cpu().clone()
   return hook
  for name,module in engine.module.named_modules():
   if ('language_model.layers.' in name and (name.rsplit('.',1)[-1].isdigit() or '.layers.0.' in name)) or name.endswith('embed_tokens') or name.endswith('language_model.norm'):
    module.register_forward_hook(capture(name))
 for micro,precision in ((1,'highest'),(2,'highest')):
  torch.set_float32_matmul_precision(precision)
  current_trace={}
  tu.assign_non_tensor(batch,micro_batch_size_per_gpu=micro);engine.optimizer_zero_grad();losses=[]
  def tracked_loss(**kwargs):
   loss,info=loss_fn(**kwargs);losses.append(float(loss.detach()));return loss,info
  start=time.monotonic()
  with engine.train_mode(disable_auto_offload=True):
   engine.forward_backward_batch(batch,tracked_loss,forward_only=False)
   from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
   synchronize_tp_replica_gradients([engine.module],engine.device_mesh)
   gradients.append({n:p.grad.to_local().detach().float().cpu().clone() for n,p in engine.module.named_parameters() if selected(n) and p.grad is not None})
  results.append({'microbatch':micro,'matmul_precision':precision,'local_loss':sum(losses),'seconds':time.monotonic()-start})
  print(json.dumps({'rank':dist.get_rank(),'completed':results[-1]}),flush=True)
  traces.append(current_trace)
 if a.trace:
  trace_rows=[]
  for name,x in traces[0].items():
   y=traces[1][name];trace_rows.append({'name':name,'shape':list(x.shape),'relative_l2':float((x-y).norm()/x.norm().clamp_min(1e-12)),'max_abs':float((x-y).abs().max())})
  Path(f'outputs/cp1_recipe/full_trace_{suffix}_rank{dist.get_rank()}.json').write_text(json.dumps(trace_rows,indent=2)+'\n')
 rows=[]
 for name,x in gradients[0].items():
  y=gradients[1][name];rel=float((x-y).norm()/x.norm().clamp_min(1e-12));cos=float(torch.nn.functional.cosine_similarity(x.flatten().double(),y.flatten().double(),dim=0));rows.append({'name':name,'relative_l2':rel,'reference_norm':float(x.norm()),'difference_norm':float((x-y).norm()),'cosine':cos,'finite':bool(torch.isfinite(y).all())})
 assert rows, 'No parameter gradients selected'
 Path(f'outputs/cp1_recipe/full_diagnostics_{suffix}_rank{dist.get_rank()}.json').write_text(json.dumps({'rows':rows,'aggregate_relative_l2':(sum(r['difference_norm']**2 for r in rows)/sum(r['reference_norm']**2 for r in rows))**.5,'passes':results},indent=2)+'\n')
 ok=all(r['finite'] and r['relative_l2']<.03 and r['cosine']>.999 for r in rows)
 rank_results=[None]*dist.get_world_size();dist.all_gather_object(rank_results,{'rank':dist.get_rank(),'pass':ok,'worst_relative_l2':max(r['relative_l2'] for r in rows),'min_cosine':min(r['cosine'] for r in rows),'passes':results})
 if dist.get_rank()==0:Path(f'outputs/cp1_recipe/full_microbatch_{suffix}.json').write_text(json.dumps({'pass':all(r['pass'] for r in rank_results),'dtype':a.dtype,'tp':a.tp,'dp':engine.get_data_parallel_size(),'cp':1,'sequence_bucket_size':a.sequence_bucket_size,'blas':a.blas,'tp_reduce_fp32':a.tp_reduce_fp32,'tp_reduce_fp64':a.tp_reduce_fp64,'attention_fp32':a.attention_fp32,'bird':a.bird,'ranks':rank_results,'initial_model_checkpoint':a.initial_model_checkpoint},indent=2)+'\n')
 assert all(r['pass'] for r in rank_results),rank_results
 if a.parity_only:
  dist.destroy_process_group();return
 del gradients
 # Worst configured context, two independent sequences; allocate actual Adam states.
 long=make_batch([a.long_length,a.long_length-113,a.long_length-17,a.long_length-65],device,seed=41+dp);tu.assign_non_tensor(long,micro_batch_size_per_gpu=2)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 out=engine.train_batch(long,loss_fn);torch.cuda.synchronize()
 memory={'rank':dist.get_rank(),'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-start,'grad_norm':float(out['metrics']['grad_norm']) if engine.is_mp_src_rank_with_outputs() else None}
 all_memory=[None]*dist.get_world_size();dist.all_gather_object(all_memory,memory)
 assert all(r['grad_norm'] is None or (math.isfinite(r['grad_norm']) and r['grad_norm']>0) for r in all_memory),all_memory
 if dist.get_rank()==0:
  result={'pass':True,'dtype':a.dtype,'tp':a.tp,'cp':1,'dp':engine.get_data_parallel_size(),'microbatch':2,'lengths':[a.long_length,a.long_length-113,a.long_length-17,a.long_length-65],'deferred_gradient_sync':True,'initial_model_checkpoint':a.initial_model_checkpoint,'ranks':all_memory}
  Path(f'outputs/cp1_recipe/full_memory_{a.dtype}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
