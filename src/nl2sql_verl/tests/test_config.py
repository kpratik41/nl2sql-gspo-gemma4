from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


@pytest.mark.parametrize('name', ['nl2sql', 'smoke', 'gspo'])
def test_hydra_actor_and_engine_configuration(name):
    from src.nl2sql_verl.engine import Gemma4CPEngine, Gemma4CPEngineConfig
    from verl.workers.engine.base import EngineRegistry
    directory = Path(__file__).resolve().parents[1]/'configs'
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name=name)
        actor = instantiate(cfg.actor_rollout_ref.actor, _convert_='partial')
        assert isinstance(actor.engine, Gemma4CPEngineConfig)
        assert actor.engine.tp_size == actor.engine.cp_size == 2
        assert actor.policy_loss.loss_mode == ('gspo' if name == 'gspo' else 'vanilla')
        assert cfg.algorithm.adv_estimator == 'nl2sql_batch_grpo'
        assert actor.loss_agg_mode == 'token-mean'
        assert actor.ppo_micro_batch_size_per_gpu == 1
        assert EngineRegistry.get_engine_cls('language_model', 'automodel') is Gemma4CPEngine
        assert cfg.actor_rollout_ref.rollout.multi_turn.format == 'nl2sql_gemma4'


def test_ray_runtime_environment_is_resolved_before_serialization(monkeypatch):
    from omegaconf import OmegaConf
    from src.nl2sql_verl.preflight import load_config
    from src.nl2sql_verl.train import resolve_runtime_env
    monkeypatch.setenv('NL2SQL_DATABASE_DIR', '/tmp/example-databases')
    monkeypatch.setenv('NL2SQL_EXEC_TIMEOUT_S', '15')
    cfg = load_config('smoke')
    resolve_runtime_env(cfg)
    # Mirror pinned verl's serialization, which deliberately omits resolve=True.
    env = OmegaConf.to_container(cfg.ray_kwargs.ray_init)['runtime_env']['env_vars']
    assert env['NL2SQL_DATABASE_DIR'] == '/tmp/example-databases'
    assert env['NL2SQL_EXEC_TIMEOUT_S'] == '15'
    assert all(isinstance(v, str) and '${' not in v for v in env.values())


def test_b200_layout_state_and_checkpoint_configuration():
    from src.nl2sql_verl.preflight import load_config, validate
    config = load_config('b200')
    result = validate(config, inputs=False)
    assert (result['tp'], result['cp'], result['data_parallel_size']) == (4, 2, 1)
    assert result['master_weights_and_optimizer'] == 'GPU FP32'
    assert result['compute_dtype'] == 'fp32'
    assert config.trainer.save_freq == config.trainer.test_freq == 1
    assert config.trainer.max_actor_ckpt_to_keep == 2
    assert config.trainer.nl2sql_training_seconds == 7200


def test_cp1_full_data_and_batch_geometry():
    from src.nl2sql_verl.preflight import load_config, validate
    cfg=load_config('cp1')
    report=validate(cfg,inputs=False)
    assert (report['tp'],report['cp'],report['data_parallel_size'])==(2,1,4)
    assert report['compute_dtype']=='bf16'
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==4
    assert cfg.data.expected_train_rows==6601
    assert cfg.data.expected_val_rows==640
    assert not cfg.data.filter_overlong_prompts
    assert cfg.data.train_batch_size==cfg.actor_rollout_ref.actor.ppo_mini_batch_size==32
    assert cfg.actor_rollout_ref.rollout.n==16
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==2
    assert 32*16//(4*2)==64
    assert cfg.algorithm.adv_estimator=='nl2sql_batch_grpo'
    assert cfg.algorithm.filter_groups.enable
    assert cfg.algorithm.filter_groups.metric=='result'
    assert cfg.algorithm.filter_groups.max_inflight_gen_batches==2

    assert cfg.trainer.save_freq==5
    assert cfg.trainer.test_freq==10
