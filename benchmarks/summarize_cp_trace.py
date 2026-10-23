"""Compare traces by global token position; retain operation and layer errors."""
import argparse
import json
from pathlib import Path
import torch
from cp_trace import errors

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference')
p.add_argument('candidate')
p.add_argument('--output', required=True)
a = p.parse_args()
torch.set_num_threads(4)
ref = torch.load(a.reference, weights_only=True)
got = torch.load(a.candidate, weights_only=True)
assert ref['weight_fingerprint'] == got['weight_fingerprint']
rows = []
for key, value in got['values'].items():
    if key not in ref['values']:
        continue
    expected = ref['values'][key]
    if key == 'embedding':
        n = min(value.shape[1], expected.shape[1])
        value, expected = value[:, :n], expected[:, :n]
    else:
        idx = got['indices']
        ref_idx = ref['indices']
        common = idx[torch.isin(idx, ref_idx)]
        gi = torch.searchsorted(idx, common)
        ri = torch.searchsorted(ref_idx, common)
        value, expected = value[:, gi], expected[:, ri]
    if value.shape != expected.shape:
        raise ValueError((key, value.shape, expected.shape))
    rows.append({'operation': key, **errors(value, expected)})
result = {'reference': a.reference, 'candidate': a.candidate,
          'first_nonidentical': next((r for r in rows if r['max_abs'] != 0), None), 'operations': rows}
Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'first_nonidentical': result['first_nonidentical'],
                  'first_layer_operations': [r for r in rows if r['operation'].startswith('layer_00')],
                  'layer_outputs': [r for r in rows if r['operation'].endswith('.output')]}), flush=True)
