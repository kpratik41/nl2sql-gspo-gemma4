from pathlib import Path
from hydra import initialize_config_dir,compose


def test_fsdp_recipe_keeps_agreed_data_and_grpo_batch():
    with initialize_config_dir(config_dir=str(Path('src/nl2sql_verl/configs').resolve()),version_base=None):
        cfg=compose(config_name='fsdp_cp2')
    actor=cfg.actor_rollout_ref.actor
    assert actor.strategy=='fsdp2'
    assert actor.engine.ulysses_sequence_parallel_size==2
    assert actor.engine.mixed_precision.param_dtype=='fp32'
    assert actor.engine.mixed_precision.reduce_dtype=='fp32'
    assert cfg.data.expected_train_rows==6601 and cfg.data.expected_val_rows==640
    assert not cfg.data.filter_overlong_prompts
    assert cfg.data.train_batch_size*cfg.actor_rollout_ref.rollout.n==512
    assert cfg.algorithm.adv_estimator=='nl2sql_batch_grpo'
    assert cfg.algorithm.filter_groups.enable and cfg.algorithm.filter_groups.metric=='result'
    assert cfg.trainer.save_freq==5 and cfg.trainer.test_freq==10
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==4
