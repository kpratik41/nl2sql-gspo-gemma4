"""The active consensus formula: group centering, batch-wide reward std."""
from collections import defaultdict

import torch
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est('nl2sql_batch_grpo')
def batch_grpo_advantage(token_level_rewards, response_mask, index, config=None, **kwargs):
    with torch.no_grad():
        scores = token_level_rewards.sum(-1).float()
        groups = defaultdict(list)
        for position, uid in enumerate(index):
            groups[uid].append(position)
        centered = torch.empty_like(scores)
        for positions in groups.values():
            centered[positions] = scores[positions]-scores[positions].mean()
        deviation = scores.std() if scores.numel() > 1 else scores.new_zeros(())
        advantages = (centered/(deviation+1e-4))[:, None]*response_mask
        return advantages, advantages
