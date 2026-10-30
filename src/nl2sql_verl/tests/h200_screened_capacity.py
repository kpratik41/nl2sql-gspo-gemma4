"""Actual selected 8K prompt+response capacity on the 16-rank CP2 actor."""
import argparse
import json
import os
from pathlib import Path


def main():
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    from tensordict import TensorDict
    from transformers import AutoTokenizer
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig
    from verl.trainer.config import CheckpointConfig
    from src.nl2sql_verl.fsdp_engine import GemmaFSDPEngine
    from src.nl2sql_verl.fsdp_attention import BACKEND
    from src.nl2sql_verl.tool_profiles import load_tool_profile
    from src.nl2sql_verl.tests import fsdp_memory_probe as memory
    from src.nl2sql_verl.tests.probe_metadata import record_probe
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record_probe(args)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    rows = pq.read_table(args.data).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    profile = load_tool_profile('gemma4_native_v1')
    # Tokenize once on rank0, broadcast the exact worst selected prompt to all
    # ranks; the original6601-row longest example need not be selected.
    payload = [None]
    if dist.get_rank() == 0:
        encoded = [tokenizer.apply_chat_template(r['prompt'], tools=profile.declarations(),
            tokenize=True, add_generation_prompt=True, return_dict=False) for r in rows]
        index = max(range(len(rows)), key=lambda i: len(encoded[i]))
        payload[0] = (index, encoded[index])
    dist.broadcast_object_list(payload, src=0)
    index, prompt = payload[0]
    answer = tokenizer.encode('<final_answer><sql_code>' + json.loads(rows[index]['reward_model']['ground_truth'])['gold_sql'] +
                              '</sql_code></final_answer>', add_special_tokens=False)
    response = (answer * ((8192 + len(answer) - 1) // len(answer)))[:8192]
    tokens = torch.tensor(prompt + response, device='cuda')
    length = ((len(tokens) + 255) // 256) * 256

    def batch(unused_lengths, unused_seed):
        ids = torch.nested.as_nested_tensor([tokens], layout=torch.jagged)
        positions = torch.nested.as_nested_tensor([torch.arange(len(tokens), device='cuda')], layout=torch.jagged)
        mask = torch.nested.as_nested_tensor([(torch.arange(len(tokens), device='cuda') >= len(prompt)).float()], layout=torch.jagged)
        result = TensorDict({'input_ids': ids, 'position_ids': positions, 'loss_mask': mask,
                             'temperature': torch.ones(1, device='cuda')}, batch_size=[1])
        tu.assign_non_tensor(result, use_remove_padding=True, use_fused_kernels=False, calculate_entropy=False,
                             use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
        return result

    memory.make_batch = batch
    args.sp = 2; args.micro = 1; args.accum = 1; args.lengths = [length]
    args.fp32_output = True; args.fp32_residual = True; args.checkpoint = None
    torch.set_autocast_cache_enabled(False)
    engine = GemmaFSDPEngine(
        HFModelConfig(path=args.model, load_tokenizer=False, use_remove_padding=True, override_config={
            'attn_implementation': BACKEND, 'nl2sql_precision': 'ieee', 'nl2sql_attention_precision': 'ieee',
            'nl2sql_static_shapes': True, 'nl2sql_gather_global_key_projection': True}),
        FSDPEngineConfig(strategy='fsdp2', model_dtype='fp32', mixed_precision={
            'param_dtype': 'fp32', 'reduce_dtype': 'fp32', 'buffer_dtype': 'fp32'}, use_torch_compile=False,
            ulysses_sequence_parallel_size=2, use_no_sync_for_gradient_accumulation=False,
            pad_to_length=True, pad_to_length_bucket=256),
        FSDPOptimizerConfig(lr=1e-6, weight_decay=0., total_training_steps=100), CheckpointConfig())
    engine.initialize()
    memory.exercise_engine(engine, args)
    if dist.get_rank() == 0:
        Path(args.output + '.scope.json').write_text(json.dumps({
            'world_size': dist.get_world_size(), 'cp': 2, 'dp': dist.get_world_size() // 2,
            'training_prompt_row': index, 'source_reference_index': rows[index]['extra_info']['reference_index'],
            'prompt_tokens': len(prompt), 'response_loss_tokens': 8192,
            'actual_sequence_tokens': len(tokens), 'padded_tokens': length,
            'scope': 'Two actual full-model forward/backward/Adam updates on longest selected prompt plus full8K response'}, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
