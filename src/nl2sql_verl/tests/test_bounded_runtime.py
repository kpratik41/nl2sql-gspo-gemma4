"""Check that the wall clock stops at a completed update, preserving final eval."""
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf


def test_duration_stop_preserves_current_step_and_weight_sync(tmp_path, monkeypatch):
    from src.nl2sql_verl import bounded_runtime as runtime
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    clock = [100.0]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    synced = []
    trainer = SimpleNamespace(
        config=OmegaConf.create({'trainer': {'nl2sql_training_seconds': 7200,
            'save_freq': 1, 'test_freq': 1, 'default_local_dir': str(tmp_path)}}),
        trainer_mode='sync', total_training_steps=10000, global_steps=1, timing_raw={},
        checkpoint_manager=SimpleNamespace(update_weights=lambda step: synced.append(step)),
    )
    PPOTrainer.on_train_begin(trainer)
    clock[0] = 7299.0
    PPOTrainerSync.on_step_end(trainer)
    assert trainer.total_training_steps == 10000
    trainer.global_steps = 2
    clock[0] = 7300.0
    PPOTrainerSync.on_step_end(trainer)
    assert trainer.total_training_steps == 2
    assert synced == [1, 2]
    assert (tmp_path/'training_clock.json').exists()


def test_reward_metrics_accept_transferqueue_linked_list_and_ignore_padding():
    from tensordict import TensorDict
    from tensordict.utils import LinkedList
    from src.nl2sql_verl.bounded_runtime import record_reward_metrics
    from src.nl2sql_verl.task import REWARD_NAMES
    owner = TensorDict({}, batch_size=[])
    rows = LinkedList([{'reward_extra_info': dict.fromkeys((*REWARD_NAMES, 'acc'), v)}
                       for v in (0., 1.)] + [{}], td=owner)
    metrics = {}
    record_reward_metrics(rows, [{}, {}, {'is_padding': True}], metrics)
    assert metrics['reward/trajectory_count'] == 2
    for name in (*REWARD_NAMES, 'acc'):
        assert metrics[f'reward/{name}/mean'] == .5
        assert metrics[f'reward/{name}/std'] == .5


def test_resume_duration_excludes_downtime(tmp_path, monkeypatch):
    import json
    from src.nl2sql_verl import bounded_runtime as runtime
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    (tmp_path/'training_clock.json').write_text(json.dumps({'completed_update': 1, 'elapsed_seconds': 1000}))
    clock = [20000.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    trainer = SimpleNamespace(
        config=OmegaConf.create({'trainer': {'nl2sql_training_seconds': 7200,
            'save_freq': 1, 'test_freq': 1, 'default_local_dir': str(tmp_path), 'resume_mode': 'auto'}}),
        trainer_mode='sync', total_training_steps=10000, global_steps=2, timing_raw={},
        checkpoint_manager=SimpleNamespace(update_weights=lambda step: None),
    )
    PPOTrainer.on_train_begin(trainer)
    clock[0] += 6200
    PPOTrainerSync.on_step_end(trainer)
    assert trainer.total_training_steps == 2
    saved = json.loads((tmp_path/'training_clock.json').read_text())
    assert saved['elapsed_seconds'] == 7200
    assert saved['duration_reached']


def test_five_step_cadence_stops_only_after_saved_update(tmp_path, monkeypatch):
    from src.nl2sql_verl import bounded_runtime as runtime
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    clock = [100.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    trainer = SimpleNamespace(
        config=OmegaConf.create({'trainer': {'nl2sql_training_seconds': 7200,
            'save_freq': 5, 'test_freq': 5, 'default_local_dir': str(tmp_path)}}),
        trainer_mode='sync', total_training_steps=100, global_steps=4, timing_raw={},
        checkpoint_manager=SimpleNamespace(update_weights=lambda step: None))
    PPOTrainer.on_train_begin(trainer)
    clock[0] = 7400.
    PPOTrainerSync.on_step_end(trainer)
    assert trainer.total_training_steps == 100
    trainer.global_steps = 5
    PPOTrainerSync.on_step_end(trainer)
    assert trainer.total_training_steps == 5
