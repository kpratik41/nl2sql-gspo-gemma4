"""Render historical decoder replays, accepting both recorded result schemas."""
import argparse
import json
from pathlib import Path

ALIASES = {'sdpa_math_fp32': 'fp32_sdpa_math', 'eager_bf16': 'eager',
           'sdpa_bf16': 'sdpa', 'flex_bf16': 'flex_attention', 'ffpa_bf16': 'replay_ffpa'}
STAGES = ('attention', 'projected', 'decoder')


def normalize_results(data):
    if isinstance(data, list):
        rows = data
        semantics = []
        for row in rows:
            for key in ('causal_prefix_max_abs', 'outside_window_last_query_max_abs'):
                if key in row:
                    value = row[key]
                    semantics.append({'kind': key, 'max_abs': value, 'pass': value <= 1e-5})
    elif isinstance(data, dict) and isinstance(data.get('replays'), list) and isinstance(data.get('semantics'), list):
        rows, semantics = data['replays'], data['semantics']
    else:
        raise ValueError('Expected a replay list or a replays/semantics object')
    if not rows:
        raise ValueError('No replay measurements')
    normalized = []
    hashes = {}
    for row in rows:
        for stage in STAGES:
            if stage not in row['errors']:
                raise ValueError(f'Missing {stage} measurements')
        key = (row['length'], row['layer'])
        if row.get('input_sha256'):
            hashes.setdefault(key, set()).add(row['input_sha256'])
        normalized.append({**row, 'backend': ALIASES.get(row['backend'], row['backend'])})
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError('Backends did not receive identical captured inputs')
    return normalized, semantics


def render_report(data):
    rows, semantics = normalize_results(data)
    failures = sum(not s['pass'] for s in semantics)
    lines = ['# Gemma decoder-layer replay results', '',
        '## Objective and scope', '',
        'Enable Gemma 4 31B CP=2 with validated efficient attention for full-parameter '
        'NL2SQL RL in verl. The current training path is native FSDP2, Ulysses and '
        'FlexAttention; FFPA remains optional and gated on correctness. These historical '
        'single-GPU replays isolate numerical and masking behavior before distributed '
        'training. See [the current recipe](src/nl2sql_verl/README.md) and '
        '[chronological work log](WORK_LOG.md) for subsequent results.', '',
        '## Method', '',
        'Each backend replays captured decoder inputs from the same BF16 checkpoint. '
        'The reference upcasts those quantized weights and inputs to FP32; it is not '
        'a full-FP32 model reference or a higher-precision pretrained checkpoint. '
        'Attention, projected attention, and decoder outputs are compared. The stored '
        'measurements are descriptive; no new BF16 acceptance threshold is imposed.', '',
        'The result schemas record different harness versions. This report preserves '
        'the supplied measurements and semantic checks without inventing missing '
        'reference rows, input hashes, or checks. Consult the corresponding execution '
        'log for mask-construction details.', '',
        '## Maximum relative L2 error', '',
        '| Backend | Cases | Attention | Projected attention | Decoder |',
        '|---|---:|---:|---:|---:|']
    for backend in sorted({r['backend'] for r in rows}):
        selected = [r for r in rows if r['backend'] == backend]
        values = [max(r['errors'][s]['relative_l2'] for r in selected) for s in STAGES]
        lines.append(f'| {backend} | {len(selected)} | ' + ' | '.join(f'{v:.8f}' for v in values) + ' |')
    lines += ['', '## Semantic checks', '',
        f'Recorded replay cases: {len(rows)}. Recorded semantic checks: {len(semantics)}. '
        f'Semantic failures: {failures}.']
    if semantics:
        lines += ['', f"Maximum recorded semantic difference: {max(s['max_abs'] for s in semantics):.8f}."]
    else:
        lines += ['', 'No semantic checks were recorded; none are claimed as passing.']
    reference = [r for r in rows if r['backend'] == 'fp32_sdpa_math']
    if reference:
        exact = all(r['errors'][s]['max_abs'] == 0 for r in reference for s in STAGES)
        lines += ['', f'FP32 SDPA-math reference comparisons: {len(reference)}; exact agreement at all measured stages: {exact}.']
    lines += ['', '## Decoder error by case', '',
        '| Tokens | Layer | Backend | Relative L2 | Maximum absolute |',
        '|---:|---:|---|---:|---:|']
    for row in rows:
        err = row['errors']['decoder']
        lines.append(f"| {row['length']} | {row['layer']} | {row['backend']} | {err['relative_l2']:.8f} | {err['max_abs']:.8f} |")
    lines += ['', '## Interpretation', '',
        'Local replay does not resolve whole-model log-probability disagreement. '
        'Passing these controls does not validate every mask, cached decoding, '
        'packing, backward, or CP. Later native FSDP2/CP2 validation and the current '
        'H200 training results are recorded separately in WORK_LOG.md.', '',
        '## Reproduction', '', '```bash',
        'python benchmarks/summarize_replay.py --input benchmark_results/layer_replay/results.json',
        '```', '',
        'Both summarize_replay.py and the older summarize_layer_replay.py entry '
        'point use this one renderer. Neither command reruns GPU experiments.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('benchmark_results/layer_replay/results.json'))
    parser.add_argument('--output', type=Path, default=Path('GEMMA4_LAYER_REPLAY_RESULTS.md'))
    args = parser.parse_args()
    args.output.write_text(render_report(json.loads(args.input.read_text())))


if __name__ == '__main__':
    main()
