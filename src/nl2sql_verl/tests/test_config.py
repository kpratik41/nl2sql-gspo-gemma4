from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


@pytest.mark.parametrize('name', ['nl2sql', 'smoke', 'gspo'])
def test_hydra_actor_and_engine_configuration(name):
    pytest.importorskip('nemo_automodel', reason='Legacy AutoModel configs require the optional NeMo environment')
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
