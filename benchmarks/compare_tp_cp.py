"""Exact TP2 CP1/CP2 probability and memory summary from measured artifacts."""
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(4)
root = Path('benchmark_results/tp2_capacity')
rows = []
for length in (128, 1024):
    ref = torch.load(root/f'tp2_cp1/probabilities_cp_rank0_{length}.pt', weights_only=True)
    parts = [torch.load(root/f'tp2_cp2/probabilities_cp_rank{r}_{length}.pt', weights_only=True) for r in range(2)]
    assert all(p['input_sha256'] == ref['input_sha256'] for p in parts)
    assert torch.equal(torch.cat([p['indices'] for p in parts]), ref['indices'])
    assert torch.equal(torch.cat([p['sample_indices'] for p in parts]), ref['sample_indices'])
    lp = torch.cat([p['lp'] for p in parts])[:-1]
    expected = ref['lp'][:-1]
    distributions = torch.cat([p['sample_logdist'] for p in parts])
    kl = (ref['sample_logdist'].exp()*(ref['sample_logdist']-distributions)).sum(-1)
    row = {'tokens': length, 'tp': 2, 'cp1_nll': -expected.mean().item(),
           'cp2_nll': -lp.mean().item(), 'nll_delta': (expected.mean()-lp.mean()).item(),
           'max_target_logprob_error': (lp-expected).abs().max().item(),
           'mean_target_logprob_error': (lp-expected).abs().mean().item(),
           'max_sampled_kl': kl.max().item(),
           'sampled_top1_agreement': (ref['sample_logdist'].argmax(-1)==distributions.argmax(-1)).float().mean().item(),
           'input_sha256': ref['input_sha256']}
    worst = int((lp-expected).abs().argmax())
    row['worst_token_example'] = {'query_position': worst,
        'cp1_target_probability': math.exp(float(expected[worst])),
        'cp2_target_probability': math.exp(float(lp[worst])),
        'cp2_to_cp1_probability_ratio': math.exp(float(lp[worst]-expected[worst]))}
    for cp in (1, 2):
        ranks = [json.loads((root/f'tp2_cp{cp}/length{length}_rank{r}.json').read_text()) for r in range(cp*2)]
        row[f'cp{cp}_peak_allocated_GiB_per_rank'] = [r['peak_allocated_GiB'] for r in ranks]
        row[f'cp{cp}_worst_rank_warm_median_ms'] = max(r.get('warm_median_ms', float('nan')) for r in ranks)
    row['screening_pass'] = abs(row['nll_delta']) <= .01 and row['max_target_logprob_error'] <= .1 and row['max_sampled_kl'] <= .001
    rows.append(row)
    print(json.dumps(row), flush=True)
(root/'probability_comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
