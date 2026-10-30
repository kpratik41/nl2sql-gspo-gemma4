"""Exercise the real engine with a local tiny Gemma checkpoint, TP2/CP2 and FSDP2."""
import json
import os
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from transformers import AutoConfig
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import AutomodelOptimizerConfig

from src.nl2sql_verl.engine import Gemma4CPEngine, Gemma4CPEngineConfig


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cp',type=int,choices=[1,2],default=2)
    args=parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    dist.init_process_group('nccl')
    path = str(Path('outputs/verl_recipe_checks/tiny_gemma').resolve())
    if dist.get_rank() == 0 and not Path(path,'config.json').exists():
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration
        torch.manual_seed(17)
        cfg = Gemma4Config.from_json_file(str(Path(__file__).with_name('tiny_config.json')))
        Gemma4ForConditionalGeneration(cfg).save_pretrained(path)
    dist.barrier()
    model_config = SimpleNamespace(path=path, hf_config=AutoConfig.from_pretrained(path))
    engine = Gemma4CPEngine(model_config,
        Gemma4CPEngineConfig(tp_size=2, cp_size=args.cp, activation_checkpointing=True, model_dtype='bf16'),
        AutomodelOptimizerConfig(lr=1e-5, total_training_steps=2, lr_scheduler_type='constant'), CheckpointConfig())
    engine.initialize()
    torch.manual_seed(5)
    ids = torch.randint(3,128,(32,),device='cuda')
    nested = torch.nested.as_nested_tensor([ids],layout=torch.jagged)
    positions = torch.nested.nested_tensor_from_jagged(torch.arange(32,device='cuda'), nested.offsets())
    mask_values = torch.ones(32,device='cuda')
    mask_values[:16] = 0
    mask = torch.nested.nested_tensor_from_jagged(mask_values,nested.offsets())
    batch = TensorDict({'input_ids': nested, 'position_ids': positions,
        'loss_mask':mask,'temperature':torch.tensor([.8],device='cuda')},batch_size=[1])
    tu.assign_non_tensor(batch, calculate_entropy=False, use_fused_kernels=False,
                         use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    def loss_function(model_output,data,dp_group):
        lp=model_output['log_probs'].values()
        shifted=data['loss_mask'].values().roll(-1)
        return -(lp*shifted).sum()/shifted.sum(), {}
    # Exercise the same batch interface verl's TrainingWorker calls.
    output = engine.train_batch(batch,loss_function)
    norm = output['metrics']['grad_norm'] if engine.is_mp_src_rank_with_outputs() else 1.
    assert norm > 0
    assert all((p.to_local() if hasattr(p,'to_local') else p).device.type == 'cpu'
               for p in engine.module.parameters())
    checkpoint=str(Path(f'outputs/verl_recipe_checks/tiny_checkpoint_cp{args.cp}').resolve())
    before_local = {name:p.to_local().detach().clone() for name,p in engine.module.named_parameters()}
    before_optim = {(id(p),key):value.to_local().detach().clone()
                    for p,state in engine.optimizer.state.items() for key,value in state.items()
                    if hasattr(value,'to_local')}
    engine.save_checkpoint(checkpoint,global_step=1)
    values,_=engine.get_per_tensor_param()
    before={name:value.cpu().clone() for name,value in values}
    with torch.no_grad():
        for p in engine.module.parameters():
            p.add_(.1)
        for state in engine.optimizer.state.values():
            for value in state.values():
                if hasattr(value,'to_local'):
                    value.add_(1.)
    engine.load_checkpoint(checkpoint)
    for name,p in engine.module.named_parameters():
        torch.testing.assert_close(p.to_local(),before_local[name],atol=0,rtol=0)
    for p,state in engine.optimizer.state.items():
        for key,value in state.items():
            if hasattr(value,'to_local'):
                torch.testing.assert_close(value.to_local(),before_optim[id(p),key],atol=0,rtol=0)
    restored,_=engine.get_per_tensor_param()
    for name,value in restored:
        torch.testing.assert_close(value.cpu(),before[name],atol=0,rtol=0)
    result={'status':'pass','tp':2,'cp':args.cp,'gradient_norm':float(norm),
            'checkpoint_restored':True,'weight_export':'BF16 HF tensors from CPU FSDP/TP shards',
            'scope':'tiny two-layer Gemma train_batch, CPU-master optimizer, checkpoint restore and weight export; not 31B or rollout'}
    if dist.get_rank()==0:
        Path(f'outputs/verl_recipe_checks/tiny_engine_cp{args.cp}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
