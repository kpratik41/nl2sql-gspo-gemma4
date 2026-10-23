"""Summarize native-Flex, one-rank ring, and two-rank ring forward diagnostics."""
import json
from pathlib import Path
import torch

root = Path('benchmark_results/cp_forward')
summary = []
for row in json.loads((root/'cp2_rank0_results.json').read_text()):
    length = row['tokens']
    native = torch.load(root/f'cp1_rank0_{length}.pt', weights_only=True)
    ring1 = torch.load(root/f'ring_cp1/cp1_rank0_{length}.pt', weights_only=True)
    parts = [torch.load(root/f'cp2_rank{r}_{length}.pt', weights_only=True) for r in range(2)]
    ring2 = {key: torch.cat([part[key] for part in parts]) for key in
             ('indices', 'lp', 'sample_indices', 'sample_logdist')}
    assert all(part['input_sha256'] == native['input_sha256'] == ring1['input_sha256'] for part in parts)
    assert all(part['weight_fingerprint'] == native['weight_fingerprint'] == ring1['weight_fingerprint'] for part in parts)
    comparisons = {}
    for label, ref, got in [('native_to_ring1', native, ring1), ('native_to_ring2', native, ring2), ('ring1_to_ring2', ring1, ring2)]:
        assert torch.equal(ref['indices'], got['indices'])
        assert torch.equal(ref['sample_indices'], got['sample_indices'])
        expected, actual = ref['lp'][:-1], got['lp'][:-1]
        error = (actual-expected).abs()
        kl = (ref['sample_logdist'].exp()*(ref['sample_logdist']-got['sample_logdist'])).sum(-1)
        comparison = {'reference_nll': -expected.mean().item(), 'candidate_nll': -actual.mean().item(),
            'nll_delta': (expected.mean()-actual.mean()).item(),
            'mean_target_logprob_error': error.mean().item(),
            'max_target_logprob_error': error.max().item(),
            'worst_query_position': error.argmax().item(),
            'target_logprob_error_p50_p95_p99': torch.quantile(error, torch.tensor([.5,.95,.99])).tolist(),
            'max_sampled_kl': kl.max().item(),
            'sampled_top1_agreement': (ref['sample_logdist'].argmax(-1)==got['sample_logdist'].argmax(-1)).float().mean().item()}
        comparison['screening_pass'] = abs(comparison['nll_delta']) <= .01 and comparison['max_target_logprob_error'] <= .1 and comparison['max_sampled_kl'] <= .001
        comparisons[label] = comparison
    cp1 = next(r for r in json.loads((root/'cp1_rank0_results.json').read_text()) if r['tokens'] == length)
    ring_cp1 = next(r for r in json.loads((root/'ring_cp1/cp1_rank0_results.json').read_text()) if r['tokens'] == length)
    item = {'tokens': length, 'comparisons': comparisons,
            'native_cp1_median_ms': cp1['median_ms'], 'ring_cp1_median_ms': ring_cp1['median_ms'],
            'cp2_worst_rank_median_ms': max(r['median_ms'] for r in row['ranks']),
            'native_cp1_peak_allocated_GiB': cp1['peak_allocated_GiB'],
            'cp2_worst_rank_peak_allocated_GiB': max(r['peak_allocated_GiB'] for r in row['ranks'])}
    summary.append(item)
    print(json.dumps(item), flush=True)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
