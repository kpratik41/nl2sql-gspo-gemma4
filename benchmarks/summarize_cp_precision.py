"""Collect the precision investigation results without changing screening gates."""
import json
from pathlib import Path

root = Path('benchmark_results/cp_forward')
policies = {'original': root, 'matmul_only': root/'debug/cublaslt',
            'fp32_attention': root/'debug/fp32_attention',
            'inventory_fp32': root/'debug/sql_inventory_fp32'}
summary = {}
for label, path in policies.items():
    result = path/'cp2_rank0_results.json'
    if not result.exists():
        continue
    reference_rows = json.loads((path/'cp1_rank0_results.json').read_text())
    summary[label] = [{'tokens': row['tokens'], 'comparison': row['comparison'],
                       'reference_backend': reference_rows[0]['backend']}
                      for row in json.loads(result.read_text())]
(root/'debug/precision_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for label, rows in summary.items():
    print(label, [(r['tokens'], r['comparison']['max_target_logprob_error'],
                    r['comparison']['screening_pass']) for r in rows])
