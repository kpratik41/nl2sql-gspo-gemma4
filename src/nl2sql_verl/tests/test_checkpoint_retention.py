import json
from pathlib import Path
import pytest
from src.nl2sql_verl.checkpoint_retention import complete_and_prune, repair_resume_pointer


def save(root, step):
    directory = root/f'global_step_{step}'
    (directory/'actor/model').mkdir(parents=True)
    (directory/'actor/optim').mkdir()
    (directory/'actor/rng_rank0.pt').touch()
    (directory/'data.pt').touch()
    complete_and_prune(root, step, keep=2)
    return directory


def test_publish_then_prune_preserves_two_complete_and_unowned(tmp_path):
    unowned = tmp_path/'global_step_0'
    unowned.mkdir()
    old = save(tmp_path, 1)
    second = save(tmp_path, 2)
    third = save(tmp_path, 3)
    assert not old.exists()
    assert second.exists() and third.exists() and unowned.exists()
    assert (tmp_path/'latest_checkpointed_iteration.txt').read_text() == '3'


def test_incomplete_save_cannot_evict_previous_or_become_resume_target(tmp_path):
    save(tmp_path, 1)
    save(tmp_path, 2)
    (tmp_path/'global_step_3').mkdir()
    with pytest.raises(ValueError, match='incomplete'):
        complete_and_prune(tmp_path, 3, keep=2)
    # Simulate upstream's pointer write before our completion marker.
    (tmp_path/'latest_checkpointed_iteration.txt').write_text('3')
    repair_resume_pointer(tmp_path)
    assert (tmp_path/'latest_checkpointed_iteration.txt').read_text() == '2'
    assert (tmp_path/'global_step_1').exists()


def test_retention_does_not_follow_checkpoint_symlinks(tmp_path):
    outside = tmp_path/'outside'
    outside.mkdir()
    (tmp_path/'global_step_0').symlink_to(outside, target_is_directory=True)
    for step in range(1, 4):
        save(tmp_path, step)
    assert outside.exists() and (tmp_path/'global_step_0').is_symlink()


def test_native_fsdp_requires_all_model_optimizer_and_rng_shards(tmp_path):
    directory=tmp_path/'global_step_1'
    actor=directory/'actor';actor.mkdir(parents=True)
    (directory/'data.pt').touch()
    for rank in range(2):
        for kind in ('model','optim','extra_state'):
            (actor/f'{kind}_world_size_2_rank_{rank}.pt').touch()
    missing=actor/'extra_state_world_size_2_rank_1.pt';missing.unlink()
    with pytest.raises(ValueError,match='incomplete'):
        complete_and_prune(tmp_path,1,2)
    missing.touch()
    complete_and_prune(tmp_path,1,2)
    assert (directory/'nl2sql_complete.json').exists()
