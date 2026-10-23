"""Combine the measured TP2 forward capacity and short probability comparisons."""
import csv
import json
from pathlib import Path

root = Path('benchmark_results/tp2_capacity')
capacity = json.loads((root/'capacity_summary.json').read_text())
report = {'scope': json.loads((root/'environment.json').read_text()),
          'short_comparison': json.loads((root/'probability_comparison.json').read_text()),
          'capacity': []}
points = []
for cp in (1, 2):
    entry = dict(capacity[f'tp2_cp{cp}'])
    length = entry['largest_success_tokens']
    ranks = [json.loads((root/f'tp2_cp{cp}/length{length}_rank{r}.json').read_text()) for r in range(cp*2)]
    entry['gpu_count'] = cp*2
    entry['peak_allocated_GiB_per_rank_at_largest_success'] = [r['peak_allocated_GiB'] for r in ranks]
    entry['peak_reserved_GiB_per_rank_at_largest_success'] = [r['peak_reserved_GiB'] for r in ranks]
    upper = entry.get('first_failed_upper_tokens')
    if upper is not None:
        entry['first_failed_bound_details'] = [json.loads(p.read_text()) for p in
            sorted((root/f'tp2_cp{cp}').glob(f'length{upper}_rank*_failure.json'))]
    report['capacity'].append(entry)
    for path in (root/f'tp2_cp{cp}').glob('length*_rank0.json'):
        row = json.loads(path.read_text())
        values = [json.loads((path.parent/f'length{row["tokens"]}_rank{r}.json').read_text())
                  for r in range(cp*2) if (path.parent/f'length{row["tokens"]}_rank{r}.json').exists()]
        if len(values) != cp*2:
            continue
        points.append({'tp': 2, 'cp': cp, 'tokens': row['tokens'],
            'worst_rank_peak_allocated_GiB': max(v['peak_allocated_GiB'] for v in values),
            'worst_rank_peak_reserved_GiB': max(v['peak_reserved_GiB'] for v in values),
            'worst_rank_cold_forward_seconds': max(v['forward_ms_including_cold_shape']/1000 for v in values)})
    print(f'TP2 CP{cp}: {length} fits; {entry.get("first_failed_upper_tokens")} first failed bound; peaks {[round(v["peak_allocated_GiB"],3) for v in ranks]}')
(root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
with (root/'memory_curve.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(points[0]))
    writer.writeheader()
    writer.writerows(sorted(points, key=lambda x:(x['cp'], x['tokens'])))
