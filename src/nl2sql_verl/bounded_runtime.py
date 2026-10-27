"""Opt-in hooks for the synchronous B200 run; imported in the Ray controller.

The two-hour clock starts after initial validation. A step finishes, saves its
checkpoint, syncs weights, evaluates and logs before the loop stops.
"""
import json
import math
from pathlib import Path
import statistics
import time

from . import engine  # noqa: F401; register recipe engine in controller/workers
from .checkpoint_retention import atomic_write, complete_and_prune, repair_resume_pointer


def record_reward_metrics(extra, tags, metrics):
    """TransferQueue may return a TensorDict LinkedList or a non-tensor stack."""
    from .task import REWARD_NAMES
    extra = extra.tolist() if hasattr(extra, 'tolist') else list(extra)
    rewards = [item.get('reward_extra_info', {}) for item, tag in zip(extra, tags, strict=True)
               if not tag.get('is_padding', False)]
    for name in (*REWARD_NAMES, 'acc'):
        values = [float(r[name]) for r in rewards if name in r and r[name] is not None]
        if len(values) != len(rewards) or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Missing or nonfinite reward component: {name}')
        if values:
            metrics[f'reward/{name}/mean'] = statistics.mean(values)
            metrics[f'reward/{name}/std'] = statistics.pstdev(values)
    metrics['reward/trajectory_count'] = len(rewards)


def register():
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    if getattr(PPOTrainer, '_nl2sql_bounded_runtime', False):
        return
    original_save = PPOTrainer._save_checkpoint
    original_load = PPOTrainer._load_checkpoint
    original_begin = PPOTrainer.on_train_begin
    original_end_step = PPOTrainerSync.on_step_end
    original_metrics = PPOTrainer._compute_metrics

    def save(self):
        original_save(self)
        if self.config.trainer.get('nl2sql_rolling_checkpoints', False):
            complete_and_prune(self.config.trainer.default_local_dir, self.global_steps,
                               int(self.config.trainer.max_actor_ckpt_to_keep))

    def load(self):
        if (self.config.trainer.get('nl2sql_rolling_checkpoints', False)
                and self.config.trainer.resume_mode == 'auto'):
            repair_resume_pointer(self.config.trainer.default_local_dir)
        return original_load(self)

    def begin(self):
        duration = float(self.config.trainer.get('nl2sql_training_seconds', 0))
        if duration:
            if self.trainer_mode != 'sync' or self.config.trainer.save_freq != 1 or self.config.trainer.test_freq != 1:
                raise ValueError('Bounded run requires sync mode, save_freq=1 and test_freq=1')
            self._nl2sql_start = time.monotonic()
            self._nl2sql_started_epoch = time.time()
            self._nl2sql_prior_elapsed = 0.0
            previous = Path(self.config.trainer.default_local_dir)/'training_clock.json'
            if self.config.trainer.get('resume_mode') == 'auto' and previous.exists():
                saved = json.loads(previous.read_text())
                if saved['completed_update'] < self.global_steps:
                    self._nl2sql_prior_elapsed = float(saved['elapsed_seconds'])
        return original_begin(self)

    def end_step(self):
        result = original_end_step(self)
        duration = float(self.config.trainer.get('nl2sql_training_seconds', 0))
        if duration:
            elapsed = self._nl2sql_prior_elapsed + time.monotonic()-self._nl2sql_start
            reached = elapsed >= duration
            if reached:
                # Current step has already checkpointed. Validation and metric
                # logging still run (test_freq=1), then the loop condition ends.
                self.total_training_steps = self.global_steps
            root = Path(self.config.trainer.default_local_dir)
            root.mkdir(parents=True, exist_ok=True)
            atomic_write(root/'training_clock.json', json.dumps({
                'started_epoch': self._nl2sql_started_epoch, 'elapsed_seconds': elapsed,
                'target_seconds': duration, 'completed_update': self.global_steps,
                'resumed_elapsed_seconds': self._nl2sql_prior_elapsed,
                'duration_reached': reached,
            }, indent=2)+'\n')
        return result

    def metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        original_metrics(self, batch, metrics, timing_raw, global_steps, epoch)
        if not self.config.trainer.get('nl2sql_rolling_checkpoints', False):
            return
        from verl.trainer.ppo.v1.trainer_base import tq
        extra = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=['extra_fields'])['extra_fields']
        record_reward_metrics(extra, batch.tags, metrics)

    PPOTrainer._save_checkpoint = save
    PPOTrainer._load_checkpoint = load
    PPOTrainer.on_train_begin = begin
    PPOTrainerSync.on_step_end = end_step
    PPOTrainer._compute_metrics = metrics
    PPOTrainer._nl2sql_bounded_runtime = True


register()
