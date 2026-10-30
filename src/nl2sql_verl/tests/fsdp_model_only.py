"""Exercise a real FSDP2 update and BF16 HF model-only export/reload."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import Gemma4ForConditionalGeneration
from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig

from src.nl2sql_verl.fsdp_attention import BACKEND
from src.nl2sql_verl.fsdp_engine import GemmaFSDPEngine
from src.nl2sql_verl.model_checkpoints import publish_model_only
from src.nl2sql_verl.tests.fsdp_memory_probe import make_batch, token_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='outputs/h200_checks/tiny_gemma')
    parser.add_argument('--root', default='/opt/dlami/nvme/nl2sql/checkpoints/tiny_model_only')
    parser.add_argument('--backup', default='/home/ubuntu/nl2sql-checkpoints/tiny_model_only')
    a = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    engine = GemmaFSDPEngine(
        HFModelConfig(path=a.model, load_tokenizer=False, use_remove_padding=True,
                      override_config={'attn_implementation': BACKEND,
                                       'nl2sql_precision': 'ieee',
                                       'nl2sql_attention_precision': 'ieee',
                                       'nl2sql_static_shapes': True}),
        FSDPEngineConfig(strategy='fsdp2', mixed_precision={
            'param_dtype': 'fp32', 'reduce_dtype': 'fp32', 'buffer_dtype': 'fp32'},
            use_torch_compile=False, ulysses_sequence_parallel_size=2,
            pad_to_length=True, pad_to_length_bucket=256,
            use_no_sync_for_gradient_accumulation=False),
        FSDPOptimizerConfig(lr=1e-6, total_training_steps=10),
        CheckpointConfig(save_contents=['hf_model'], load_contents=[]))
    engine.initialize()
    batch = make_batch([257, 385], 42)
    with engine.train_mode(disable_auto_offload=True):
        engine.train_batch(batch, token_loss)
    actor = Path(a.root)/'global_step_5'/'actor'
    engine.save_checkpoint(str(actor), global_step=5)
    expected = {}
    tensors, _ = engine.get_per_tensor_param()
    for name, value in tensors:
        if dist.get_rank() == 0:
            expected[name] = value.detach().cpu().clone()
    if dist.get_rank() == 0:
        assert not list(actor.glob('*world_size*'))
        path = Path(publish_model_only(a.root, 5, a.backup))
        index = json.loads((path/'model.safetensors.index.json').read_text())
        checked = 0
        dtypes = set()
        for shard in set(index['weight_map'].values()):
            for name, tensor in load_file(path/shard).items():
                torch.testing.assert_close(tensor, expected[name].to(tensor.dtype), rtol=0, atol=0)
                dtypes.add(str(tensor.dtype))
                checked += 1
        loaded = Gemma4ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        assert checked and loaded.config.text_config.dtype == torch.bfloat16
        report = {'status': 'pass', 'tensors_checked': checked, 'stored_dtypes': sorted(dtypes),
                  'hf_reload': True, 'optimizer_shards_saved': False,
                  'export': str(path), 'world_size': dist.get_world_size()}
        Path('outputs/h200_checks/model_only_export.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
