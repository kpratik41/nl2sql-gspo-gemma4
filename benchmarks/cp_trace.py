"""Opt-in read-only tensor hooks for short-context CP diagnosis."""
import torch


class TraceCapture:
    def __init__(self, model):
        self.values = {}
        self.handles = []
        def capture(name, value):
            if isinstance(value, (tuple, list)):
                value = value[0]
            if isinstance(value, torch.Tensor):
                self.values[name] = value.detach().cpu().clone()
        self.handles.append(model.get_input_embeddings().register_forward_hook(
            lambda m, args, out: capture('embedding', out)))
        for module in model.modules():
            if module.__class__.__name__ != 'Gemma4TextDecoderLayer':
                continue
            i = module.layer_idx
            def before(m, args, kwargs, i=i):
                capture(f'layer_{i:02d}.input', args[0] if args else kwargs['hidden_states'])
            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(
                lambda m, args, out, i=i: capture(f'layer_{i:02d}.output', out)))
            # Each operation is observed without altering its inputs or output.
            for name, child in module.named_modules():
                if not name or len(list(child.children())):
                    continue
                self.handles.append(child.register_forward_hook(
                    lambda m, args, out, i=i, name=name: capture(f'layer_{i:02d}.{name}', out)))
            self.handles.append(module.self_attn.o_proj.register_forward_pre_hook(
                lambda m, args, i=i: capture(f'layer_{i:02d}.attention_before_o_proj', args[0])))

    def close(self):
        for handle in self.handles:
            handle.remove()


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    delta = a-b
    return {'max_abs': delta.abs().max().item(),
            'relative_l2': (delta.norm()/b.norm().clamp_min(1e-12)).item(),
            'equal_fraction': (a == b).float().mean().item(),
            'position1_max_abs': delta[:, 1].abs().max().item() if delta.shape[1] > 1 else None}
