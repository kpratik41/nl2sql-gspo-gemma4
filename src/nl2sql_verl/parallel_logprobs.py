"""Differentiable TP-vocabulary logprobs without gathering the vocabulary."""
import torch
import torch.distributed as dist


def _reduce(value, group, op=dist.ReduceOp.SUM):
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(value, op=op, group=group)
    return value


class _VocabLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, temperature, group, vocab_start, chunk_size, entropy):
        shape = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])
        targets = labels.reshape(-1)
        temps = torch.broadcast_to(temperature, shape).reshape(-1).float()
        if not bool((temps > 0).all()):
            raise ValueError('Temperature must be positive')
        lp = torch.empty(flat.shape[0], device=flat.device, dtype=torch.float32)
        normalizers, entropies = torch.empty_like(lp), torch.zeros_like(lp)
        for begin in range(0, flat.shape[0], chunk_size):
            end = min(begin+chunk_size, flat.shape[0])
            values = flat[begin:end].float()/temps[begin:end, None]
            maximum = _reduce(values.max(-1).values, group, dist.ReduceOp.MAX)
            shifted = values-maximum[:, None]
            weights = shifted.exp()
            denominator = _reduce(weights.sum(-1), group)
            local_target = targets[begin:end]-vocab_start
            selected = values.gather(-1, local_target.clamp(0, flat.shape[-1]-1)[:, None]).squeeze(-1)
            selected = torch.where((local_target >= 0) & (local_target < flat.shape[-1]), selected, 0.)
            selected = _reduce(selected, group)
            normalizers[begin:end] = maximum+denominator.log()
            lp[begin:end] = selected-normalizers[begin:end]
            if entropy:
                weighted = _reduce((weights*shifted).sum(-1), group)
                entropies[begin:end] = denominator.log()-weighted/denominator
        ctx.save_for_backward(logits, labels, temps, normalizers, entropies)
        ctx.vocab_start, ctx.chunk_size = vocab_start, chunk_size
        ctx.set_materialize_grads(False)
        return lp.reshape(shape), entropies.reshape(shape)

    @staticmethod
    def backward(ctx, lp_grad, entropy_grad):
        logits, labels, temps, normalizers, entropies = ctx.saved_tensors
        flat, targets = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        gradient = torch.empty_like(flat)
        lp_grad = lp_grad.reshape(-1) if lp_grad is not None else None
        entropy_grad = entropy_grad.reshape(-1) if entropy_grad is not None else None
        for begin in range(0, flat.shape[0], ctx.chunk_size):
            end = min(begin+ctx.chunk_size, flat.shape[0])
            logp = flat[begin:end].float()/temps[begin:end, None]-normalizers[begin:end, None]
            probability = logp.exp()
            grad = torch.zeros_like(probability)
            if lp_grad is not None:
                grad = -probability*lp_grad[begin:end, None]
                indices = targets[begin:end]-ctx.vocab_start
                owned = (indices >= 0) & (indices < flat.shape[-1])
                rows = owned.nonzero().squeeze(-1)
                grad[rows, indices[rows]] += lp_grad[begin:end][rows]
            if entropy_grad is not None:
                grad -= probability*(logp+entropies[begin:end, None])*entropy_grad[begin:end, None]
            gradient[begin:end] = (grad/temps[begin:end, None]).to(logits.dtype)
        return gradient.reshape(logits.shape), None, None, None, None, None, None


def vocab_logprobs(logits, labels, temperature=1., group=None, vocab_start=0, chunk_size=128, entropy=False):
    temperature = torch.as_tensor(temperature, device=logits.device, dtype=torch.float32)
    return _VocabLogprobs.apply(logits, labels, temperature, group, vocab_start, chunk_size, entropy)
