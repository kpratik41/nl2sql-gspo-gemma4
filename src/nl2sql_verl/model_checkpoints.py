"""Publish portable model-only HF exports to EBS, retaining completed exports."""
import json
import os
from pathlib import Path
import re
import shutil

from .checkpoint_retention import atomic_write

MARKER = 'nl2sql_model_only_complete.json'
FORMAT = 'nl2sql-hf-model-only-v1'


def _completed(root):
    result = []
    for path in Path(root).glob('global_step_*'):
        match = re.fullmatch(r'global_step_(\d+)', path.name)
        if not match or path.is_symlink() or not path.is_dir():
            continue
        try:
            meta = json.loads((path/MARKER).read_text())
        except (ValueError, OSError):
            continue
        if meta.get('format') == FORMAT and meta.get('step') == int(match[1]):
            result.append((int(match[1]), path))
    return sorted(result)


def publish_model_only(root, step, destination, keep=2):
    if keep < 0:
        raise ValueError('Use zero to retain all completed model exports, or a positive retention count')
    root, destination = Path(root), Path(destination)
    source = root/f'global_step_{step}'/'actor'/'huggingface'
    index = json.loads((source/'model.safetensors.index.json').read_text())
    if not (source/'config.json').is_file() or not index.get('weight_map'):
        raise ValueError('Incomplete HF model export')
    names = set(index['weight_map'].values())
    if any(Path(name).name != name for name in names):
        raise ValueError('Unexpected weight shard path')
    from safetensors import safe_open
    total = 0
    for name in names:
        path = source/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing model shard: {name}')
        total += path.stat().st_size
        with safe_open(path, framework='pt', device='cpu') as weights:
            actual = set(weights.keys())
        expected = {k for k,v in index['weight_map'].items() if v == name}
        if actual != expected:
            raise ValueError(f'Weight index mismatch: {name}')
    if total < index['metadata']['total_size']:
        raise ValueError('Truncated HF weight export')
    if list((source.parent).glob('optim_world_size_*.pt')):
        raise ValueError('Model-only save unexpectedly wrote optimizer shards')
    destination.mkdir(parents=True, exist_ok=True)
    # Leave room for a temporary export plus the previously completed exports.
    if shutil.disk_usage(destination).free < total * 1.05 + 10 * 1024**3:
        raise OSError('Insufficient EBS space to publish model without deleting prior exports')
    temporary = destination/f'.global_step_{step}.partial'
    target = destination/f'global_step_{step}'
    if temporary.exists() or target.exists():
        raise FileExistsError(f'Model export already exists: {target}')
    def durable_copy(src, dst):
        result = shutil.copy2(src, dst)
        with open(dst, 'rb') as stream:
            os.fsync(stream.fileno())
        return result
    shutil.copytree(source, temporary, copy_function=durable_copy)
    metadata = {'format': FORMAT, 'step': step, 'weight_bytes': total,
                'contents': 'HF model/config/tokenizer; no optimizer or RNG',
                'training_resume': 'model initialization only'}
    for name in names:
        if (temporary/name).stat().st_size != (source/name).stat().st_size:
            raise ValueError(f'Model shard copy size mismatch: {name}')
    atomic_write(temporary/MARKER, json.dumps(metadata, indent=2)+'\n')
    os.replace(temporary, target)
    atomic_write(destination/'latest_model.txt', target.name+'\n')
    atomic_write(root/f'global_step_{step}'/MARKER, json.dumps(metadata)+'\n')
    # Only prune model-only exports bearing our own completion marker.
    if keep:
        for base in (destination, root):
            for old_step, path in _completed(base)[:-keep]:
                if old_step < step:
                    shutil.rmtree(path)
    return str(target)
