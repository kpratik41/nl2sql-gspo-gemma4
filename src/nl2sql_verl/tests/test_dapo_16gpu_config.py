"""Compose with the pinned upstream YAML; no GPU/runtime imports are needed."""
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest


@pytest.fixture
def recipe():
    upstream = os.environ.get('VERL_CONFIG_DIR')
    if not upstream:
        pytest.skip('Set VERL_CONFIG_DIR to the pinned verl/trainer/config directory')
    overrides = [f'hydra.searchpath=[file://{Path(upstream).resolve()}]']
    with initialize_config_dir(config_dir=str(Path('src/nl2sql_verl/configs').resolve()), version_base=None):
        return compose(config_name='dapo_fsdp_cp2_16gpu', overrides=overrides)


def test_16gpu_batch_and_parallelism_are_compatible(recipe):
    actor = recipe.actor_rollout_ref.actor
    world = recipe.trainer.nnodes * recipe.trainer.n_gpus_per_node
    cp = actor.engine.ulysses_sequence_parallel_size
    assert world == 16 and cp == 2
    dp = world // cp
    trajectories = recipe.data.train_batch_size * recipe.actor_rollout_ref.rollout.n
    mini_batch = actor.ppo_mini_batch_size * recipe.actor_rollout_ref.rollout.n
    micro = actor.ppo_micro_batch_size_per_gpu
    assert trajectories == mini_batch == 512
    assert mini_batch % (dp * micro) == 0
    assert mini_batch // (dp * micro) == 64
    tp_rollout = recipe.actor_rollout_ref.rollout.tensor_model_parallel_size
    assert tp_rollout == 4
    assert recipe.trainer.n_gpus_per_node % tp_rollout == 0
    assert world // tp_rollout == 4
    assert actor.strategy == actor.engine.strategy == 'fsdp2'
    assert actor.engine.param_offload is actor.engine.optimizer_offload is False
    assert actor.engine.model_dtype == 'fp32'
    assert recipe.actor_rollout_ref.model.override_config.nl2sql_gather_global_key_projection is True


def test_4k_responses_and_context_cover_saved_prompts(recipe):
    rollout = recipe.actor_rollout_ref.rollout
    actor = recipe.actor_rollout_ref.actor
    assert recipe.data.max_prompt_length == 30720
    assert recipe.data.max_response_length == 4096
    context = recipe.data.max_prompt_length + recipe.data.max_response_length
    assert context == rollout.max_model_len == 34816
    assert rollout.max_num_batched_tokens >= context
    assert rollout.log_prob_max_token_len_per_gpu >= context
    assert actor.ppo_max_token_len_per_gpu >= context
    assert recipe.data.filter_overlong_prompts is False
    for split in ('train', 'validation', 'dev_full'):
        path = Path(f'outputs/gemma4_native_v1/data/{split}_prompt_lengths.json')
        if path.exists():
            assert max(json.loads(path.read_text())) <= recipe.data.max_prompt_length


def test_dapo_uses_group_advantages_dynamic_sampling_and_token_loss(recipe):
    actor = recipe.actor_rollout_ref.actor
    assert recipe.algorithm.adv_estimator == 'grpo'
    assert recipe.algorithm.norm_adv_by_std_in_grpo is True
    assert recipe.algorithm.filter_groups.enable is True
    assert recipe.algorithm.filter_groups.metric == 'result'
    assert recipe.algorithm.filter_groups.max_inflight_gen_batches * recipe.data.train_batch_size == 64
    assert recipe.data.gen_batch_size == 1
    assert actor.policy_loss.loss_mode == 'vanilla'
    assert actor.loss_agg_mode == 'token-mean'
    assert (actor.clip_ratio_low, actor.clip_ratio_high, actor.clip_ratio_c) == (0.2, 0.28, 10.0)
    assert actor.use_kl_loss is recipe.algorithm.use_kl_in_reward is False
    env = recipe.ray_kwargs.ray_init.runtime_env.env_vars
    assert (env.NL2SQL_LENGTH_MAX, env.NL2SQL_LENGTH_BUFFER) == ('4096', '512')
    assert recipe.ray_kwargs.ray_init.address == 'auto'
    assert recipe.trainer.save_freq == recipe.trainer.test_freq == 5
    assert recipe.trainer.nl2sql_model_only_checkpoints is True
    assert recipe.trainer.nl2sql_rolling_checkpoints is False
    assert actor.checkpoint.save_contents == ['hf_model']
    assert actor.checkpoint.load_contents == []
    assert recipe.trainer.resume_mode == 'disable'


def test_future_recipe_uses_versioned_native_data_and_observed_tools(recipe):
    assert recipe.actor_rollout_ref.model.external_lib == 'src.nl2sql_verl.native_runtime'
    assert recipe.data.train_files == 'outputs/gemma4_native_v1/data/train.parquet'
    assert recipe.data.val_files == 'outputs/gemma4_native_v1/data/validation.parquet'
    rollout = recipe.actor_rollout_ref.rollout
    assert rollout.agent.default_agent_loop == 'nl2sql_native_tool_agent'
    assert rollout.agent.agent_loop_config_path.endswith('/agent_native.yaml')
    assert rollout.multi_turn.format == 'nl2sql_gemma4'
    assert rollout.multi_turn.tool_config_path.endswith('/tools_native.yaml')
    assert recipe.reward.custom_reward_function.path.endswith('/reward_entry_native.py')


def test_initial_and_periodic_evaluations_are_greedy_without_changing_training(recipe):
    assert recipe.trainer.val_before_train is True
    assert recipe.trainer.test_freq == 5
    rollout = recipe.actor_rollout_ref.rollout
    assert rollout.mode == 'async'
    assert (rollout.temperature, rollout.top_p, rollout.n) == (1.2, 0.95, 16)
    assert rollout.val_kwargs.do_sample is False
    assert (rollout.val_kwargs.temperature, rollout.val_kwargs.top_p,
            rollout.val_kwargs.top_k, rollout.val_kwargs.n) == (0., 1., -1, 1)


def test_original_fsdp_entry_point_still_composes():
    upstream = os.environ.get('VERL_CONFIG_DIR')
    if not upstream:
        pytest.skip('Set VERL_CONFIG_DIR to the pinned verl/trainer/config directory')
    with initialize_config_dir(config_dir=str(Path('src/nl2sql_verl/configs').resolve()), version_base=None):
        cfg = compose(config_name='fsdp_cp2', overrides=[f'hydra.searchpath=[file://{Path(upstream).resolve()}]'])
    assert cfg.trainer.nnodes == 1 and cfg.trainer.n_gpus_per_node == 8
    assert cfg.algorithm.adv_estimator == 'nl2sql_batch_grpo'
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768


def test_screened_8k_recipe_has_consistent_context_and_retains_all_checkpoints():
    upstream = os.environ.get('VERL_CONFIG_DIR')
    if not upstream:
        pytest.skip('Set VERL_CONFIG_DIR to the pinned verl/trainer/config directory')
    with initialize_config_dir(config_dir=str(Path('src/nl2sql_verl/configs').resolve()), version_base=None):
        cfg = compose(config_name='dapo_screened_8k_16gpu', overrides=[f'hydra.searchpath=[file://{Path(upstream).resolve()}]'])
    assert cfg.data.expected_train_rows == 1039
    assert cfg.data.max_response_length == 8192
    assert cfg.data.max_prompt_length + cfg.data.max_response_length == 38912
    assert cfg.actor_rollout_ref.rollout.max_model_len == cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 38912
    assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == 0
    assert cfg.trainer.val_before_train is True and cfg.trainer.test_freq == 5
    assert cfg.trainer.max_actor_ckpt_to_keep == 0
    assert cfg.trainer.nl2sql_model_backup_dir.startswith('/opt/dlami/nvme/')
    assert cfg.actor_rollout_ref.actor.checkpoint.save_contents == ['hf_model']
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.NL2SQL_LENGTH_MAX == '8192'
    assert cfg.algorithm.filter_groups.enable and cfg.algorithm.filter_groups.metric == 'result'
