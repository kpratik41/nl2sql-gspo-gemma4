"""Compare saved full-model probes with stable CPU FP64 statistics.

Useful for cross-precision diagnostics: a matching CP reference alone cannot
establish agreement with a more accurate arithmetic policy.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def compare(reference, candidate):
    assert reference['gradients'].keys() == candidate['gradients'].keys()
    rows = []
    for name, tensor in reference['gradients'].items():
        other = candidate['gradients'][name]
        assert tensor.shape == other.shape, name
        xx = yy = xy = dd = 0.
        finite = True
        for x, y in zip(tensor.flatten().split(1048576), other.flatten().split(1048576)):
            x, y = x.double(), y.double()
            finite &= bool(x.isfinite().all() and y.isfinite().all())
            xx += float(x.square().sum()); yy += float(y.square().sum())
            xy += float((x*y).sum()); dd += float((x-y).square().sum())
        rows.append(dict(name=name, relative_l2=math.sqrt(dd/max(xx, 1e-24)),
                         cosine=xy/max(math.sqrt(xx*yy), 1e-24), finite=finite))
    return dict(reference=reference['result'], candidate=candidate['result'], rows=rows,
                loss_abs_difference=abs(reference['result']['loss']-candidate['result']['loss']),
                max_logprob_abs_difference=float((reference['logprobs']-candidate['logprobs']).abs().max()),
                pass_gradient=all(r['finite'] and r['relative_l2'] < .03 and r['cosine'] > .999 for r in rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reference'); p.add_argument('candidate'); p.add_argument('--output', required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    result = compare(*(torch.load(f, map_location='cpu', weights_only=False) for f in (a.reference, a.candidate)))
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'reference', 'candidate')}))
    print('worst_gradient_relative_l2:', max(r['relative_l2'] for r in result['rows']))


if __name__ == '__main__':
    main()
