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
    assert cfg.actor_rollout_ref.model.override_config.nl2sql_precision=='ieee'
    assert cfg.actor_rollout_ref.model.override_config.nl2sql_static_shapes is True
    assert cfg.actor_rollout_ref.model.override_config.nl2sql_attention_precision=='ieee'
    assert actor.ppo_micro_batch_size_per_gpu==1
    assert cfg.data.expected_train_rows==6601 and cfg.data.expected_val_rows==640
    assert not cfg.data.filter_overlong_prompts
    assert cfg.data.train_batch_size*cfg.actor_rollout_ref.rollout.n==512
    assert cfg.data.gen_batch_size==1
    assert cfg.algorithm.filter_groups.max_inflight_gen_batches==2
    assert cfg.algorithm.adv_estimator=='nl2sql_batch_grpo'
    assert cfg.algorithm.filter_groups.enable and cfg.algorithm.filter_groups.metric=='result'
    assert cfg.trainer.save_freq==5 and cfg.trainer.test_freq==10
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==4


def test_memory_probe_loss_accepts_native_metadata_and_normalizes_gradients():
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from src.nl2sql_verl.tests.fsdp_memory_probe import token_loss
    lp=torch.tensor([-1.,-2.,-3.],requires_grad=True)
    data=TensorDict({'loss_mask':torch.nested.as_nested_tensor([torch.tensor([0.,1.,1.])],layout=torch.jagged)},batch_size=[1])
    tu.assign_non_tensor(data,dp_size=2,batch_num_tokens=4)
    loss,_=token_loss({'log_probs':torch.nested.as_nested_tensor([lp],layout=torch.jagged)},data,None)
    assert loss.item()==1.5
    loss.backward()
    torch.testing.assert_close(lp.grad,torch.tensor([-.5,-.5,0.]))


def test_bounded_pilot_keeps_batch_and_preserves_final_checkpoint_eval_contract():
    from src.nl2sql_verl.run_fsdp import command
    args=command(Path('/tmp/run'),Path('/tmp/checkpoint'),smoke=False,pilot=True)
    assert 'trainer.nl2sql_training_seconds=7200' in args
    assert 'trainer.save_freq=1' in args and 'trainer.test_freq=1' in args
    assert not any(x.startswith('data.train_batch_size=') for x in args)
    production=command(Path('/tmp/run'),Path('/tmp/checkpoint'),smoke=False)
    assert not any(x.startswith(('trainer.save_freq=','trainer.test_freq=','trainer.nl2sql_training_seconds=')) for x in production)
