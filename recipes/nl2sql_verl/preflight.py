"""Validate the recipe configuration and local inputs before starting Ray."""
import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .prepare_data import normalize_record
from .task import database_path


def load_config(name='smoke', overrides=()):
    directory = Path(__file__).resolve().parent/'configs'
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def validate(config, inputs=True):
    from .engine import Gemma4CPEngineConfig
    actor = instantiate(config.actor_rollout_ref.actor, _convert_='partial')
    rollout = config.actor_rollout_ref.rollout
    if not isinstance(actor.engine, Gemma4CPEngineConfig):
        raise ValueError('Expected the recipe-local Gemma4 CP engine')
    world = config.trainer.nnodes*config.trainer.n_gpus_per_node
    if world % (actor.engine.tp_size*actor.engine.cp_size):
        raise ValueError('GPU count must be divisible by TP × CP')
    if actor.ppo_micro_batch_size_per_gpu != 1 or actor.use_dynamic_bsz:
        raise ValueError('Use one example per microbatch and disable packing/dynamic batching')
    if actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        raise ValueError('This recipe uses consensus beta=0; a reference CP engine must be configured before enabling KL')
    if rollout.max_model_len < config.data.max_prompt_length+config.data.max_response_length:
        raise ValueError('rollout.max_model_len must cover prompt plus full multi-turn response budget')
    if not rollout.calculate_log_probs or config.algorithm.rollout_correction.bypass_mode:
        raise ValueError('Retain rollout logprobs and recompute old actor logprobs to measure serving/training mismatch')
    actor.validate(world, config.data.train_batch_size, config.actor_rollout_ref.model)
    if inputs:
        import pyarrow.parquet as pq
        for split in ('train_files','val_files'):
            paths = config.data[split]
            paths = [paths] if isinstance(paths, str) else paths
            for file in paths:
                if not Path(file).is_file():
                    raise FileNotFoundError(f'Create {file} before training; expected parquet prompts and gold metadata')
                count = 0
                for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        normalized = normalize_record(row, count)
                        database_path(json.loads(normalized['reward_model']['ground_truth'])['db_id'])
                        count += 1
                if count == 0:
                    raise ValueError(f'{file}: empty parquet')
    return {'status':'pass','config_only':not inputs,'tp':actor.engine.tp_size,'cp':actor.engine.cp_size,
            'data_parallel_size':world//(actor.engine.tp_size*actor.engine.cp_size),
            'policy_loss':actor.policy_loss.loss_mode,'loss_aggregation':actor.loss_agg_mode,
            'compute_dtype':'bfloat16','master_weights_and_optimizer':'CPU FP32',
            'model':config.actor_rollout_ref.model.path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=['nl2sql','smoke','gspo'], default='smoke')
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--dump', action='store_true')
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    print(json.dumps(validate(config, inputs=not args.config_only),indent=2))
    if args.dump:
        print(OmegaConf.to_yaml(config,resolve=True))


if __name__ == '__main__':
    main()
