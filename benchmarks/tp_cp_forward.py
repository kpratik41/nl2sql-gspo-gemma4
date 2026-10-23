"""TP2 x CP1/CP2 BF16 forward, all-token vocabulary-sharded logits; no training."""
import argparse
import gc
import hashlib
import importlib
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cp', type=int, choices=[1, 2], required=True)
    p.add_argument('--lengths', type=int, nargs='+', required=True)
    p.add_argument('--output', default='benchmark_results/tp2_capacity')
    a = p.parse_args()
    rank, lr = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    # A length sweep legitimately specializes multiple causal/window mask shapes.
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 1024
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch.backends.cuda.preferred_blas_library('cublaslt')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    assert dist.get_world_size() == 2*a.cp
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh('cuda', (a.cp, 2), mesh_dim_names=('cp', 'tp'))
    cp_rank, tp_rank = mesh['cp'].get_local_rank(), mesh['tp'].get_local_rank()
    root = Path(a.output)/f'tp2_cp{a.cp}'
    root.mkdir(parents=True, exist_ok=True)
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.gemma4_moe.parallelization import PARALLELIZER, _gemma4_tp_plan
    from nemo_automodel.components.distributed.parallelizer import _update_attention_head_counts_for_tp
    from nemo_automodel.shared.tied_weights import ensure_tied_lm_head
    from torch.distributed.tensor.parallel import parallelize_module
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices
    from transformers import AutoTokenizer
    snapshot = Path.home()/'.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'
    print(f'[{rank}] load checkpoint', flush=True)
    model = NeMoAutoModelForImageTextToText.from_pretrained(str(snapshot), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', use_liger_kernel=False,
        use_sdpa_patching=False, backend=BackendConfig(linear='torch', attn='sdpa', rms_norm='torch',
        experts='torch', dispatcher='torch', enable_hf_state_dict_adapter=False),
        text_config={'use_cache': False, 'cp_full_attn_backend': 'flex', 'cp_sliding_attn_backend': 'flex'}).eval()
    PARALLELIZER._validate_tp_mesh(model, mesh['tp'])
    model._gemma4_tp_enabled, model._gemma4_tp_size, model._gemma4_tp_mesh = True, 2, mesh['tp']
    parallelize_module(model, mesh['tp'], _gemma4_tp_plan(model, sequence_parallel=False))
    ensure_tied_lm_head(model)
    _update_attention_head_counts_for_tp(model, 2)
    from cp_flex_options import configure_flex_tiles
    for m in model.modules():
        if hasattr(m, '_cp_manual_metadata_keys'):
            configure_flex_tiles(m)
    # Standard compiled BlockMask constructor avoids eager quadratic temporaries.
    flex_module = importlib.import_module('torch.nn.attention.flex_attention')
    original_create_mask = flex_module.create_block_mask
    compiled_mask_builder = torch.compile(original_create_mask, dynamic=False)
    def compiled_create_mask(*args, **kwargs):
        kwargs.pop('_compile', None)
        return compiled_mask_builder(*args, **kwargs)
    flex_module.create_block_mask = compiled_create_mask
    from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa
    ring_shapes = set()
    ring_original = cpa._run_gemma4_cp_ring_attention
    def record_ring(module, ctx):
        ring_shapes.add((ctx.cp_size, ctx.query.shape[1], ctx.key.shape[1], ctx.query.shape[-1]))
        return ring_original(module, ctx)
    cpa._run_gemma4_cp_ring_attention = record_ring
    gc.collect()
    torch.cuda.empty_cache()
    local_parameter_bytes = sum((x.to_local() if isinstance(x, DTensor) else x).numel()*x.element_size() for x in model.parameters())
    manifest = {'rank': rank, 'tp_rank': tp_rank, 'cp_rank': cp_rank,
        'tp_group': dist.get_process_group_ranks(mesh['tp'].get_group()),
        'cp_group': dist.get_process_group_ranks(mesh['cp'].get_group()),
        'local_parameter_GiB': local_parameter_bytes/2**30,
        'dtensor_parameters': sum(isinstance(x, DTensor) for x in model.parameters()),
        'max_position_embeddings': model.config.text_config.max_position_embeddings,
        'dtype': 'bfloat16', 'attention': 'ring Flex BF16', 'compiled_masks': True,
        'matmul_policy': 'cuBLASLt, BF16 reduced reduction=False, split-K=False',
        'fsdp': False, 'gradients': False, 'kv_cache': False, 'batch_size': 1,
        'logits': 'all sequence positions; vocabulary sharded over TP2'}
    (root/f'manifest_rank{rank}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    unit = tok.encode(Path('benchmark_results/model_parity/fixture.txt').read_text(), add_special_tokens=False)
    for length in a.lengths:
        assert length <= model.config.text_config.max_position_embeddings
        assert length % (2*a.cp) == 0
        try:
            for m in model.modules():
                if hasattr(m, '_gemma4_tp_output_holder'):
                    m._gemma4_tp_output_holder[0] = None
            gc.collect()
            torch.cuda.empty_cache()
            ids = torch.tensor(([tok.bos_token_id]+unit*(length//len(unit)+1))[:length], device='cuda')[None]
            batch = {'input_ids': ids, 'labels': ids.clone(), 'position_ids': torch.arange(length, device='cuda')[None]}
            if a.cp == 1:
                sharder = ContextParallelSharder(shard_batch=model._cp_shard_batch_aux_only,
                    local_token_global_indices=contiguous_local_indices, device_mesh=mesh)
            else:
                sharder = ContextParallelSharder(model, mesh, batch)
            ctx, batch = sharder.shard(batch)
            batch.pop('labels')
            # The ring constructs the actual causal/window/document masks. Skip
            # redundant HF dense masks that the installed attention hooks discard.
            batch['attention_mask'] = {'full_attention': None, 'sliding_attention': None}
            idx = contiguous_local_indices(mesh['cp'], length, device=ids.device)
            dist.barrier()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            print(f'[{rank}] forward {length} local={idx.numel()}', flush=True)
            start = time.perf_counter()
            with torch.no_grad(), ctx():
                logits = model(**batch, use_cache=False, logits_to_keep=0).logits
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)*1000
            assert isinstance(logits, DTensor), type(logits)
            local = logits.to_local()
            assert local.shape[1] == length//a.cp
            assert local.shape[2] == model.config.text_config.vocab_size//2
            peak = torch.cuda.max_memory_allocated()
            row = {'status': 'ok', 'tokens': length, 'tp': 2, 'cp': a.cp, 'rank': rank,
                'forward_ms_including_cold_shape': elapsed, 'baseline_GiB': baseline/2**30,
                'peak_allocated_GiB': peak/2**30, 'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
                'incremental_peak_MiB': (peak-baseline)/2**20, 'local_logits_shape': list(local.shape),
                'ring_shapes': [list(s) for s in sorted(ring_shapes)],
                'input_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
            if length <= 2048:
                warm_times = []
                for _ in range(3):
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad(), ctx():
                        measured = model(**batch, use_cache=False, logits_to_keep=0).logits
                    torch.cuda.synchronize()
                    warm_times.append((time.perf_counter()-start)*1000)
                    del measured
                row['warm_median_ms'] = statistics.median(warm_times)
                row['warm_times_ms'] = warm_times
            assert torch.isfinite(local[0, torch.linspace(0, local.shape[1]-1, 8, device='cuda').long()]).all()
            if length <= 2048:
                # TP-global target normalization, chunked to bound FP32 temporaries.
                logprobs = []
                vstart = tp_rank*local.shape[-1]
                for offset in range(0, local.shape[1], 32):
                    values = local[0, offset:offset+32].float()
                    maximum = values.max(-1).values
                    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=mesh['tp'].get_group())
                    denominator = (values-maximum[:, None]).exp().sum(-1)
                    dist.all_reduce(denominator, group=mesh['tp'].get_group())
                    targets = ids[0, (idx[offset:offset+32]+1).clamp_max(length-1)]
                    owned = (targets >= vstart) & (targets < vstart+local.shape[-1])
                    selected = values.gather(1, (targets-vstart).clamp(0, local.shape[-1]-1)[:, None]).squeeze(1)
                    selected = torch.where(owned, selected, 0.)
                    dist.all_reduce(selected, group=mesh['tp'].get_group())
                    logprobs.append((selected-maximum-denominator.log()).cpu())
                positions = torch.linspace(0, length-1, 8, device='cuda').long().unique()
                keep = (idx[:, None] == positions).any(-1)
                sample = local[0, keep].contiguous()
                parts = [torch.empty_like(sample) for _ in range(2)]
                dist.all_gather(parts, sample, group=mesh['tp'].get_group())
                distributions = torch.cat(parts, -1).float().log_softmax(-1).cpu()
                if tp_rank == 0:
                    torch.save({'indices': idx.cpu(), 'lp': torch.cat(logprobs),
                        'sample_indices': idx[keep].cpu(), 'sample_logdist': distributions,
                        'input_sha256': row['input_sha256']}, root/f'probabilities_cp_rank{cp_rank}_{length}.pt')
                del values, maximum, denominator, selected, sample, parts, distributions, logprobs
            (root/f'length{length}_rank{rank}.json').write_text(json.dumps(row, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            del logits, local, batch, ids, sharder
            dist.barrier()
        except Exception as exc:
            failure = {'status': 'oom' if isinstance(exc, torch.OutOfMemoryError) else 'error',
                       'tokens': length, 'rank': rank, 'error': repr(exc)}
            (root/f'length{length}_rank{rank}_failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            print(json.dumps(failure), flush=True)
            raise
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
