"""Resolve task environment values before the pinned verl launcher serializes Ray kwargs."""
import hydra
from omegaconf import OmegaConf


def resolve_runtime_env(config):
    # Pinned main_ppo uses to_container(resolve=False) for ray.init. Otherwise
    # workers receive literal '${oc.env:...}' database paths and timeout values.
    OmegaConf.resolve(config.ray_kwargs.ray_init.runtime_env)


@hydra.main(config_path='configs', config_name='smoke', version_base=None)
def main(config):
    from verl.trainer.main_ppo import main as verl_main
    resolve_runtime_env(config)
    verl_main(config)


if __name__ == '__main__':
    main()
