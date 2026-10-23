import numpy as np
import torch

from recipes.nl2sql_verl.advantages import batch_grpo_advantage


def test_group_centering_and_batch_reward_std_match_consensus():
    scores = torch.tensor([0.,1.,2.,6.])
    rewards = torch.zeros(4,5)
    rewards[:,-1] = scores
    mask = torch.ones_like(rewards)
    mask[:,2] = 0
    actual, returns = batch_grpo_advantage(rewards,mask,np.array(['a','a','b','b']))
    expected = (scores-scores.reshape(2,2).mean(-1).repeat_interleave(2))/(scores.std()+1e-4)
    torch.testing.assert_close(actual,expected[:,None]*mask)
    torch.testing.assert_close(actual,returns)


def test_flat_reward_groups_have_zero_advantage():
    rewards = torch.ones(4,2)
    mask = torch.ones_like(rewards)
    actual,_ = batch_grpo_advantage(rewards,mask,np.array(['a','a','b','b']))
    assert torch.isfinite(actual).all() and not actual.any()
