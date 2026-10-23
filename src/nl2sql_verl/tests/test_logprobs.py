import torch

from src.nl2sql_verl.parallel_logprobs import vocab_logprobs


def test_logprobs_entropy_and_gradients_match_full_vocabulary():
    torch.manual_seed(9)
    logits = torch.randn(1, 19, 37, requires_grad=True)
    targets = torch.randint(37, (1, 19))
    temperature = 0.8
    lp, entropy = vocab_logprobs(logits, targets, temperature, chunk_size=4, entropy=True)
    reference = (logits/temperature).log_softmax(-1)
    expected_lp = reference.gather(-1, targets[..., None]).squeeze(-1)
    expected_entropy = -(reference.exp()*reference).sum(-1)
    torch.testing.assert_close(lp, expected_lp, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, atol=1e-6, rtol=1e-6)
    weights = torch.randn_like(lp)
    actual_grad, = torch.autograd.grad((lp*weights + .03*entropy).sum(), logits, retain_graph=True)
    expected_grad, = torch.autograd.grad((expected_lp*weights + .03*expected_entropy).sum(), logits)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
