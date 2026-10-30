"""Resolve task environment values before the pinned verl launcher serializes Ray kwargs."""
import hydra
import os
from pathlib import Path
from omegaconf import OmegaConf, open_dict


def resolve_runtime_env(config):
    # Pinned main_ppo uses to_container(resolve=False) for ray.init. Otherwise
    # workers receive literal '${oc.env:...}' database paths and timeout values.
    OmegaConf.resolve(config.ray_kwargs.ray_init.runtime_env)
    # Connecting to an existing cluster does not inherit the driver's env.
    # Forward logging controls explicitly, while keeping credentials out of Ray.
    env = config.ray_kwargs.ray_init.runtime_env.env_vars
    with open_dict(env):
        for key in ('WANDB_MODE', 'WANDB_DIR', 'WANDB_ENTITY'):
            if key in os.environ:
                env[key] = os.environ[key]


def resolve_task_paths(config):
    """Ray workers may start outside the repository on either node."""
    root = Path(__file__).resolve().parents[2]
    def absolute(value):
        if isinstance(value, str):
            if '://' in value or Path(value).is_absolute():
                return value
            return str(root/value)
        return [absolute(item) for item in value]
    for key in ('data.train_files', 'data.val_files',
                'actor_rollout_ref.rollout.agent.agent_loop_config_path',
                'actor_rollout_ref.rollout.multi_turn.tool_config_path',
                'reward.custom_reward_function.path'):
        value = OmegaConf.select(config, key)
        if value:
            OmegaConf.update(config, key, absolute(value))


@hydra.main(config_path='configs', config_name='smoke', version_base=None)
def main(config):
    from verl.trainer.main_ppo import main as verl_main
    resolve_runtime_env(config)
    resolve_task_paths(config)
    controller = config.trainer.get('nl2sql_controller_node_ip')
    if controller:
        from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
        if not config.trainer.use_v1 or config.global_profiler.tool == 'nsys':
            raise ValueError('Pinned NL2SQL controller requires V1 without the nsys controller wrapper')
        runner = TaskRunnerV1.options(resources={f'node:{controller}': 0.001})
        return run_ppo(config, task_runner_class=runner)
    verl_main(config)


if __name__ == '__main__':
    main()
