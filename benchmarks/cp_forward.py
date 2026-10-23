"""Real checkpoint CP1 Flex vs CP2 model-owned ring Flex forward screening.

Forward only, replicated BF16 weights, no FSDP/optimizer/cache. Run CP1 first,
then torchrun CP2 with the same output directory. JSON records failures too.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cp-size', type=int, choices=[1, 2], required=True)
    p.add_argument('--ring-cp1', action='store_true', help='One-rank model-owned ring control; use torchrun with one worker and a separate output directory')
    p.add_argument('--trace', action='store_true', help='Save read-only layer/operation tensors from the first forward')
    p.add_argument('--no-compare', action='store_true', help='Collect CP2 diagnostics without requiring CP1 artifacts in this output directory')
    p.add_argument('--matmul-policy', choices=['default', 'no-reduced', 'cublaslt-full'], default='default',
                   help='Opt-in process-local BF16 reduction policy; weights and model code remain unchanged')
    p.add_argument('--attention-precision', choices=['bf16', 'fp32'], default='bf16',
                   help='Ring-only diagnostic: upcast QKV inside Flex and merge FP32 chunk outputs before final BF16 cast')
    p.add_argument('--lengths', nargs='+', type=int, default=[128, 1024])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--fixture', default='benchmark_results/model_parity/fixture.txt')
    p.add_argument('--output', default='benchmark_results/cp_forward')
    p.add_argument('--model-path', default=str(Path.home()/'.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'))
    a = p.parse_args()
    assert not a.ring_cp1 or a.cp_size == 1
    use_ring = a.cp_size > 1 or a.ring_cp1
    assert a.attention_precision == 'bf16' or use_ring, 'FP32 attention comparison uses a ring CP1 reference'
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    if a.matmul_policy == 'no-reduced':
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    elif a.matmul_policy == 'cublaslt-full':
        torch.backends.cuda.preferred_blas_library('cublaslt')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    mesh = None
    if use_ring:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
        assert dist.get_world_size() == a.cp_size
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh('cuda', (a.cp_size,), mesh_dim_names=('cp',))

    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from transformers import AutoTokenizer
    backend = BackendConfig(linear='torch', attn='sdpa', rms_norm='torch',
                            experts='torch', dispatcher='torch',
                            enable_hf_state_dict_adapter=False)
    print(f'[rank {rank}] loading checkpoint CP={a.cp_size}', flush=True)
    start = time.perf_counter()
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        a.model_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa' if use_ring else 'flex_attention',
        use_liger_kernel=False, use_sdpa_patching=False, backend=backend,
        text_config={'use_cache': False, 'cp_full_attn_backend': 'flex',
                     'cp_sliding_attn_backend': 'flex'})
    model.eval()
    from cp_flex_options import configure_flex_tiles, OPTIONS
    if use_ring:
        for module in model.modules():
            if hasattr(module, '_cp_manual_metadata_keys'):
                configure_flex_tiles(module, attention_precision=a.attention_precision)
    ring_calls = []
    if use_ring:
        from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa
        original_ring = cpa._run_gemma4_cp_ring_attention
        def record_ring(module, ring_ctx):
            ring_calls.append((ring_ctx.cp_size, ring_ctx.seq_local))
            return original_ring(module, ring_ctx)
        cpa._run_gemma4_cp_ring_attention = record_ring
    assert 'nemo_automodel' in type(model).__module__, type(model)
    assert next(model.parameters()).device.type == 'cuda'
    assert next(model.parameters()).dtype == torch.bfloat16
    # Verify shared checkpoint identity without copying all 31B weights to CPU.
    fingerprint = hashlib.sha256(model.get_input_embeddings().weight[:16].detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    print(f'[rank {rank}] loaded {type(model).__name__} in {time.perf_counter()-start:.1f}s', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    fixture = Path(a.fixture).read_text()
    unit = tokenizer.encode(fixture, add_special_tokens=False)
    results = []
    for length in a.lengths:
        ids = torch.tensor(([tokenizer.bos_token_id] + unit*(length//len(unit)+1))[:length], device='cuda')[None]
        digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
        batch = {'input_ids': ids, 'position_ids': torch.arange(length, device='cuda')[None]}
        sharder = None
        if use_ring:
            batch['labels'] = ids.clone()
            if a.ring_cp1:
                from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices
                sharder = ContextParallelSharder(shard_batch=model._cp_shard_batch_aux_only,
                    local_token_global_indices=contiguous_local_indices, device_mesh=mesh)
            else:
                sharder = ContextParallelSharder(model, mesh, batch)
            ctx, batch = sharder.shard(batch)
            batch.pop('labels')
            assert '_packed_seq_ids' in batch
            # Required HF entry point: Gemma's installed hooks replace SDPA with ring Flex.
            assert model.config.text_config._attn_implementation == 'sdpa'
            indices = sharder.local_token_global_indices(mesh['cp'], sharder.shard_layout.padded_seq_len, device=ids.device)
        else:
            from contextlib import nullcontext
            ctx = nullcontext
            indices = torch.arange(length, device='cuda')
        kwargs = {} if use_ring else {'kernel_options': {'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 1}}
        def forward():
            with torch.no_grad(), ctx():
                return model(**batch, use_cache=False, logits_to_keep=0, **kwargs).logits
        print(f'[rank {rank}] warmup length={length}, local_positions={indices.numel()}', flush=True)
        if a.trace:
            from cp_trace import TraceCapture
            trace = TraceCapture(model)
        logits = forward()
        if a.trace:
            trace.close()
            torch.save({'values': trace.values, 'indices': indices.cpu(),
                        'input_ids': ids.cpu(), 'weight_fingerprint': fingerprint},
                       root/f'trace_cp{a.cp_size}_rank{rank}_{length}.pt')
            del trace
        torch.cuda.synchronize()
        assert logits.shape[1] == indices.numel(), (logits.shape, indices.shape)
        # All target logprobs plus eight full vocabulary distributions for KL checks.
        valid = indices < length - 1
        lp_parts = []
        for offset in range(0, logits.shape[1], 16):
            idx = indices[offset:offset+16]
            chunk = logits[0, offset:offset+16].float()
            targets = ids[0, (idx+1).clamp_max(length-1)]
            lp_parts.append((chunk.gather(1, targets[:, None]).squeeze(1) - chunk.logsumexp(-1)).cpu())
        sampled_positions = torch.linspace(0, length-1, 8, device='cuda').long().unique()
        sampled_mask = (indices[:, None] == sampled_positions).any(-1)
        payload = {'indices': indices.cpu(), 'lp': torch.cat(lp_parts),
                   'sample_indices': indices[sampled_mask].cpu(),
                   'sample_logdist': logits[0, sampled_mask].float().log_softmax(-1).cpu(),
                   'input_sha256': digest, 'weight_fingerprint': fingerprint}
        del logits
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(a.repeats):
            if a.cp_size > 1:
                dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = forward()
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
            del logits
        row = {'cp_size': a.cp_size, 'rank': rank, 'tokens': length,
               'matmul_policy': a.matmul_policy,
               'attention_precision': a.attention_precision,
               'fixture': a.fixture,
               'local_tokens': indices.numel(), 'input_sha256': digest,
               'weight_fingerprint': fingerprint, 'model_class': str(type(model)),
               'backend': 'model-owned ring Flex' if use_ring else 'native Flex',
               'median_ms': statistics.median(times), 'times_ms': times,
               'baseline_GiB': baseline/2**30,
               'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
               'incremental_peak_MiB': (torch.cuda.max_memory_allocated()-baseline)/2**20,
               'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
               'ring_dispatch_calls': len(ring_calls),
               'ring_cp_sizes_observed': sorted(set(x[0] for x in ring_calls)),
               'ring_kernel_options': ({**OPTIONS, **({'BLOCK_M': 16, 'BLOCK_N': 16} if a.attention_precision == 'fp32' else {})} if use_ring else None),
               'scope': 'forward only; replicated BF16 weights; no FSDP, optimizer, packing or cache'}
        torch.save(payload, root/f'cp{a.cp_size}_rank{rank}_{length}.pt')
        if a.cp_size > 1:
            assert ring_calls and all(x[0] == a.cp_size for x in ring_calls)
            all_rows = [None]*a.cp_size
            dist.all_gather_object(all_rows, row)
            dist.barrier()
            if rank == 0 and not a.no_compare:
                pieces = [torch.load(root/f'cp2_rank{r}_{length}.pt', weights_only=True) for r in range(a.cp_size)]
                ref = torch.load(root/f'cp1_rank0_{length}.pt', weights_only=True)
                assert all(x['input_sha256'] == ref['input_sha256'] and x['weight_fingerprint'] == ref['weight_fingerprint'] for x in pieces)
                idx = torch.cat([x['indices'] for x in pieces])
                assert torch.equal(idx, torch.arange(length)), idx
                lp = torch.cat([x['lp'] for x in pieces])[:-1]
                ref_lp = ref['lp'][:-1]
                current = torch.cat([x['sample_logdist'] for x in pieces])
                reference = ref['sample_logdist']
                assert torch.equal(torch.cat([x['sample_indices'] for x in pieces]), ref['sample_indices'])
                kl = (reference.exp()*(reference-current)).sum(-1)
                comparison = {'nll_cp1': -ref_lp.mean().item(), 'nll_cp2': -lp.mean().item(),
                    'nll_delta': (ref_lp.mean()-lp.mean()).item(),
                    'max_target_logprob_error': (lp-ref_lp).abs().max().item(),
                    'mean_target_logprob_error': (lp-ref_lp).abs().mean().item(),
                    'max_sampled_kl': kl.max().item(),
                    'sampled_top1_agreement': (reference.argmax(-1)==current.argmax(-1)).float().mean().item()}
                comparison['screening_pass'] = abs(comparison['nll_delta']) <= .01 and comparison['max_target_logprob_error'] <= .1 and comparison['max_sampled_kl'] <= .001
                row = {'tokens': length, 'ranks': all_rows, 'comparison': comparison}
                print(json.dumps(row), flush=True)
        else:
            print(json.dumps(row), flush=True)
        results.append(row)
        (root/f'cp{a.cp_size}_rank{rank}_results.json').write_text(json.dumps(results, indent=2)+'\n')
        del payload
    if use_ring:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
