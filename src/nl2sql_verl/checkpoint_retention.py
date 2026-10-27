"""Publish completed trainer checkpoints before pruning older owned saves."""
import json
import os
from pathlib import Path
import re
import shutil

MARKER = 'nl2sql_complete.json'
FORMAT = 'nl2sql-full-training-v1'


def atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def completed_steps(root):
    result = []
    for directory in Path(root).glob('global_step_*'):
        match = re.fullmatch(r'global_step_(\d+)', directory.name)
        if directory.is_symlink() or not directory.is_dir() or not match:
            continue
        try:
            metadata = json.loads((directory/MARKER).read_text())
        except (OSError, ValueError):
            continue
        step = int(match.group(1))
        if metadata.get('format') == FORMAT and metadata.get('step') == step:
            result.append((step, directory))
    return sorted(result)


def complete_and_prune(root, step, keep):
    """Called on the controller AFTER actor, optimizer, RNG and data save return.

    Retain at least the latest two checkpoints; never delete unmarked directories.
    Publishing the new checkpoint precedes deletion so interruption preserves a
    resumable checkpoint. Local NVMe is persistent across process exits, but is
    ephemeral instance storage and does not survive instance termination.
    """
    if keep < 2:
        raise ValueError('Keep at least two completed checkpoints')
    root = Path(root)
    directory = root/f'global_step_{step}'
    if directory.is_symlink():
        raise ValueError('Checkpoint directory must not be a symlink')
    actor = directory/'actor'
    automodel = (all((actor/name).exists() for name in ('model', 'optim'))
                 and bool(list(actor.glob('rng_rank*.pt'))))
    native_files = list(actor.glob('model_world_size_*_rank_0.pt'))
    native = False
    if len(native_files) == 1:
        match = re.fullmatch(r'model_world_size_(\d+)_rank_0.pt', native_files[0].name)
        world = int(match.group(1)) if match else 0
        native = world > 0 and all(
            (actor/f'{kind}_world_size_{world}_rank_{rank}.pt').is_file()
            for rank in range(world) for kind in ('model', 'optim', 'extra_state'))
    if not (directory/'data.pt').is_file() or not (automodel or native):
        raise ValueError('Cannot publish incomplete full training checkpoint')
    atomic_write(directory/MARKER, json.dumps({'format': FORMAT, 'step': step})+'\n')
    atomic_write(root/'latest_checkpointed_iteration.txt', str(step))
    completed = completed_steps(root)
    for old_step, old_path in completed[:-keep]:
        if old_step < step:
            shutil.rmtree(old_path)


def repair_resume_pointer(root):
    """Ignore a partially written new save after an interrupted trainer save."""
    completed = completed_steps(root)
    pointer = Path(root)/'latest_checkpointed_iteration.txt'
    if completed:
        atomic_write(pointer, str(completed[-1][0]))
    elif pointer.exists():
        raise ValueError('Resume pointer exists but no completed NL2SQL checkpoint was found')
