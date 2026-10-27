"""Compare HF and native verl FSDP2 on identical real SQL-token fixtures."""
import argparse, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Gemma4ForConditionalGeneration

MODEL='/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'


def fixture(model_path, lengths, row_indices=None):
    if 'tiny' in model_path:
        g=torch.Generator().manual_seed(128)
        return [(torch.randint(3,128,(n,),generator=g), torch.arange(n)>=n//2) for n in lengths]
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    from src.nl2sql_verl.prepare_data import TOOLS
    tok=AutoTokenizer.from_pretrained(model_path)
    table=pq.read_table('outputs/cp1_recipe/data/train.parquet')
    records=(table.take(row_indices) if row_indices is not None else table.slice(0,len(lengths))).to_pylist()
    assert len(records)==len(lengths)
    rows=[]
    for record,n in zip(records,lengths):
        prompt=tok.apply_chat_template(record['prompt'],tools=TOOLS,tokenize=True,add_generation_prompt=True,return_dict=False)
        answer=tok.encode('```sql\n'+json.loads(record['reward_model']['ground_truth'])['gold_sql']+'\n```',add_special_tokens=False)
        ids=(prompt+answer)[-n:]
        mask=[False]*(len(ids)-len(answer))+[True]*len(answer)
        rows.append((torch.tensor(ids),torch.tensor(mask)))
    return rows


def selected(name):
    return 'language_model' in name and any(f'.layers.{i}.' in name for i in (0,1,5,59)) and any(x in name for x in ('q_proj','k_proj','o_proj','down_proj','input_layernorm'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hf','native'],required=True)
    p.add_argument('--model',default=MODEL);p.add_argument('--lengths',type=int,nargs='+',default=[257,385])
    p.add_argument('--micro',type=int,default=1)
    p.add_argument('--fp32-residual',action='store_true')
    p.add_argument('--fp32-output',action='store_true')
    p.add_argument('--row-indices',type=int,nargs='+')
    p.add_argument('--bucket',type=int,default=1);p.add_argument('--plain-loss',action='store_true');p.add_argument('--compat',action='store_true');p.add_argument('--sp',type=int,default=1);p.add_argument('--output',required=True);p.add_argument('--reference');p.add_argument('--backend',default='sdpa')
    a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if a.fp32_output:
        a.fp32_residual=True
        os.environ['NL2SQL_ATTENTION_BF16_IO']='1'
    if a.fp32_residual:torch.set_autocast_cache_enabled(False)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.preferred_blas_library('cublaslt')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
    rank=0
    if a.mode=='native':
        dist.init_process_group('nccl',device_id=torch.device('cuda',torch.cuda.current_device()));rank=dist.get_rank()
        from verl.workers.config import HFModelConfig,FSDPEngineConfig,FSDPOptimizerConfig
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
        from verl.trainer.config import CheckpointConfig
        if a.compat:
            from src.nl2sql_verl import fsdp_engine
        custom=a.backend=='nl2sql_flex_ulysses'
        cls=fsdp_engine.GemmaFSDPEngine if custom else FSDPEngineWithLMHead
        mc=HFModelConfig(path=a.model,load_tokenizer=False,use_remove_padding=custom,override_config={'attn_implementation':a.backend,'nl2sql_precision':'tf32' if a.fp32_output else 'amp'})
        precision={'param_dtype':'fp32' if a.fp32_residual else 'bf16','reduce_dtype':'fp32','buffer_dtype':'fp32'}
        engine=cls(mc,FSDPEngineConfig(strategy='fsdp2',mixed_precision=precision,use_torch_compile=False,ulysses_sequence_parallel_size=a.sp,pad_to_length=a.bucket>1,pad_to_length_bucket=a.bucket),FSDPOptimizerConfig(lr=1e-6,total_training_steps=2),CheckpointConfig())
        engine.initialize();model=engine.module
        if a.fp32_residual:engine._autocast_dtype=torch.bfloat16
        if a.fp32_output:engine._autocast_dtype=torch.float32
    else:
        if a.backend=='flex_attention' and a.fp32_residual and os.environ.get('NL2SQL_ATTENTION_FP32')!='1':
            from transformers import AttentionInterface
            from transformers.integrations.flex_attention import flex_attention_forward
            def aligned_attention(module,q,k,v,mask,**kwargs):
                if a.fp32_output:
                    out,weights=flex_attention_forward(module,q.bfloat16(),k.bfloat16(),v.bfloat16(),mask,**kwargs)
                    return out.to(q.dtype),weights
                return flex_attention_forward(module,q.to(v.dtype),k.to(v.dtype),v,mask,**kwargs)
            AttentionInterface.register('flex_attention',aligned_attention)
        if a.backend=='flex_attention' and os.environ.get('NL2SQL_ATTENTION_FP32')=='1':
            from transformers import AttentionInterface
            from transformers.integrations.flex_attention import flex_attention_forward
            def precise_attention(module,q,k,v,mask,**kwargs):
                dtype=q.dtype
                options=dict(kwargs.pop('kernel_options',{}))
                options.update(BLOCK_M=16,BLOCK_N=16,FLOAT32_PRECISION="'tf32x3'")
                with torch.autocast('cuda',enabled=False):
                    output,weights=flex_attention_forward(module,q.float(),k.float(),v.float(),mask,kernel_options=options,**kwargs)
                return output.to(dtype),weights
            AttentionInterface.register('flex_attention',precise_attention)
        model=Gemma4ForConditionalGeneration.from_pretrained(a.model,dtype=torch.float32 if a.fp32_residual else torch.bfloat16,device_map={'':'cuda'},attn_implementation=a.backend)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if a.fp32_output:torch.set_float32_matmul_precision('medium')
    for name,param in model.named_parameters():
        if any(x in name for x in ('vision_tower','embed_vision','multi_modal_projector','audio_tower','embed_audio')):param.requires_grad_(False)
    offloaded={}
    if a.mode=='hf' and a.fp32_residual:
        # Compute every parameter gradient, but release it as soon as accumulated
        # so an independent FP32-weight HF reference fits one B200. Retain the
        # same selected tensors on CPU across the two backward passes.
        for name,param in model.named_parameters():
            if not param.requires_grad:continue
            def release_gradient(p,name=name):
                if selected(name):
                    grad=p.grad.detach().float().cpu()
                    if name in offloaded:offloaded[name].add_(grad)
                    else:offloaded[name]=grad
                p.grad=None
            param.register_post_accumulate_grad_hook(release_gradient)
    model.train();rows=fixture(a.model,a.lengths,a.row_indices);count=sum(int(mask[1:].sum()) for ids,mask in rows)
    loss_sum=0.;lps=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    assert a.micro==1 or a.backend=='nl2sql_flex_ulysses'
    for offset in range(0,len(rows),a.micro):
        group=rows[offset:offset+a.micro]
        ids,mask=group[0]
        ids=ids.cuda()[None];mask=mask.cuda()[None,1:];pos=torch.arange(ids.shape[1],device='cuda')[None]
        if a.mode=='hf' and a.bucket>1:
            pad=(-ids.shape[-1])%a.bucket
            if pad:
                ids=F.pad(ids,(0,pad))
                mask=F.pad(mask,(0,pad))
                pos=torch.cat([pos,torch.arange(pad,device='cuda')[None]],-1)
        from contextlib import nullcontext
        ctx=engine.train_mode(disable_auto_offload=True,zero_grad_on_exit=False) if a.mode=='native' else nullcontext()
        with ctx,sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.fp32_output):
            if a.backend=='nl2sql_flex_ulysses':
                from tensordict import TensorDict
                from verl.utils import tensordict_utils as tu
                nested=torch.nested.as_nested_tensor([x.cuda() for x,m in group],layout=torch.jagged)
                positions=torch.nested.as_nested_tensor([torch.arange(len(x),device='cuda') for x,m in group],layout=torch.jagged)
                batch=TensorDict({'input_ids':nested,'position_ids':positions,'temperature':torch.ones(len(group),device='cuda')},batch_size=[len(group)])
                tu.assign_non_tensor(batch,use_remove_padding=True,use_fused_kernels=False)
                inputs,output_args=engine.prepare_model_inputs(batch)
                output=model(**inputs,use_cache=False)
                logits=output.logits
                if a.plain_loss:
                    assert a.sp==1 and a.micro==1
                    lp=-F.cross_entropy(logits[:,:-1][mask].float(),ids[:,1:][mask],reduction='none')
                else:
                    native_lp=engine.prepare_model_outputs(output,output_args,batch,None)['log_probs'].values()
                    with torch.no_grad():
                        local_direct=-F.cross_entropy(logits.detach().squeeze(0).float(),output_args['input_ids_rmpad_rolled'],reduction='none')
                        direct=engine._gather_and_unpad_packed(local_direct,output_args['pad_size'])
                        error=float((native_lp.detach()-direct).abs().max())
                        if rank==0:print(json.dumps({'logprob_kernel_max_error':error}),flush=True)
                        assert error<.01,error
                    response_mask=torch.cat([F.pad(m[1:],(0,1),value=False) for x,m in group]).cuda()
                    lp=native_lp[response_mask]
            else:
                extra={}
                if a.backend=='flex_attention':
                    from src.nl2sql_verl.fsdp_attention import OPTIONS
                    extra['kernel_options']=OPTIONS
                logits=model(input_ids=ids,position_ids=pos,use_cache=False,**extra).logits
                lp=-F.cross_entropy(logits[:,:-1][mask].float(),ids[:,1:][mask],reduction='none')
            loss=-lp.sum()/count
            loss.backward()
        loss_sum+=float(loss.detach());lps.append(lp.detach().cpu());del logits,lp,loss
    gradients=dict(offloaded)
    for name,param in model.named_parameters():
        if selected(name) and param.grad is not None:
            grad=param.grad.full_tensor() if hasattr(param.grad,'full_tensor') else param.grad
            if rank==0:gradients[name.replace('._checkpoint_wrapped_module','').replace('._fsdp_wrapped_module','')]=grad.detach().float().cpu()
    if rank==0:
        assert gradients, 'No gradients captured'
        result={'model':a.model,'row_indices':a.row_indices,'cp':a.sp if a.mode=='native' else 1,'dp':dist.get_world_size()//a.sp if a.mode=='native' else 1,'fp32_output':a.fp32_output,'fp32_residual':a.fp32_residual,'microbatch':a.micro,'bucket':a.bucket,'attention_fp32':os.environ.get('NL2SQL_ATTENTION_FP32')=='1','plain_loss':a.plain_loss,'mode':a.mode,'backend':a.backend,'loss':loss_sum,'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'lengths':[len(ids) for ids,mask in rows],'selected_gradients':len(gradients)}
        saved={'gradients':gradients,'logprobs':torch.cat(lps),'result':result}
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        if a.reference:
            ref=torch.load(a.reference,map_location='cpu',weights_only=False);stats=[]
            for n,x in ref['gradients'].items():
                x=x.cuda();y=gradients[n].cuda();norm=float(x.norm());delta=float((x-y).norm())
                stats.append({'name':n,'relative_l2':delta/max(norm,1e-12),'cosine':float(F.cosine_similarity(x.flatten().double(),y.flatten().double(),dim=0)),'finite':bool(y.isfinite().all())})
            result.update(rows=stats,loss_abs_difference=abs(loss_sum-ref['result']['loss']),max_logprob_abs_difference=float((saved['logprobs']-ref['logprobs']).abs().max()),pass_gradient=all(s['finite'] and s['relative_l2']<.03 and s['cosine']>.999 for s in stats))
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path.with_suffix('.pt'));path.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        if a.reference:
            assert result['pass_gradient'] and result['max_logprob_abs_difference']<.1 and result['loss_abs_difference']<.01,result
    if a.mode=='native':dist.destroy_process_group()

if __name__=='__main__':main()
