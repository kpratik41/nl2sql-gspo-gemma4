import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.nl2sql_verl.model_checkpoints import MARKER, publish_model_only


def export(root, step):
    path = root/f'global_step_{step}'/'actor'/'huggingface'
    path.mkdir(parents=True)
    save_file({'weight': torch.ones(4, dtype=torch.bfloat16)}, path/'model.safetensors')
    (path/'config.json').write_text('{"dtype":"bfloat16"}')
    (path/'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': 8}, 'weight_map': {'weight': 'model.safetensors'}}))
    return path


def test_model_only_publication_is_loadable_and_prunes_only_completed_owned_exports(tmp_path):
    source, backup = tmp_path/'nvme', tmp_path/'ebs'
    export(source, 5)
    target = Path(publish_model_only(source, 5, backup, keep=2))
    assert (target/MARKER).is_file()
    assert not list(target.glob('optim*'))
    with __import__('safetensors').safe_open(target/'model.safetensors', framework='pt') as weights:
        torch.testing.assert_close(weights.get_tensor('weight'), torch.ones(4, dtype=torch.bfloat16))
    (backup/'global_step_1').mkdir()
    for step in (10, 15):
        export(source, step)
        publish_model_only(source, step, backup, keep=2)
    assert not (backup/'global_step_5').exists()
    assert not (source/'global_step_5').exists()
    assert (backup/'global_step_1').exists()
    assert (backup/'latest_model.txt').read_text().strip() == 'global_step_15'


def test_missing_shard_cannot_publish_or_remove_previous_model(tmp_path):
    source, backup = tmp_path/'nvme', tmp_path/'ebs'
    export(source, 5)
    publish_model_only(source, 5, backup)
    path = export(source, 10)
    (path/'model.safetensors').unlink()
    with pytest.raises(ValueError, match='Missing model shard'):
        publish_model_only(source, 10, backup)
    assert (backup/'global_step_5'/MARKER).exists()
    assert not (backup/'global_step_10').exists()


def test_zero_retention_limit_preserves_every_completed_export(tmp_path):
    source, backup = tmp_path/'nvme', tmp_path/'published'
    for step in (5, 10, 15, 20):
        export(source, step)
        publish_model_only(source, step, backup, keep=0)
    for step in (5, 10, 15, 20):
        assert (source/f'global_step_{step}'/MARKER).is_file()
        assert (backup/f'global_step_{step}'/MARKER).is_file()
