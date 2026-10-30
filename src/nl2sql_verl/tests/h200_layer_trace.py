"""Locate the first full-model CP2 forward/adjoint divergence by decoder layer."""
from pathlib import Path
import torch
from src.nl2sql_verl.tests import fsdp_cp_probe as probe

original = probe.run_comparison


def traced(args, model, engine, rank, offloaded):
    captured = {}
    handles = []
    if rank < 2:
        core = model.model.language_model

        def attach(label, module):
            def forward(module, inputs, output):
                if label in captured:
                    return
                assert isinstance(output, torch.Tensor)
                entry = {'output': output.detach().cpu()}
                captured[label] = entry

                def backward(gradient):
                    entry['gradient'] = gradient.detach().cpu()
                    return gradient
                output.register_hook(backward)
            handles.append(module.register_forward_hook(forward))

        attach('embedding', core.embed_tokens)
        for i, layer in enumerate(core.layers):
            attach(f'layer{i:02}', layer)
        attach('final_norm', core.norm)
    original(args, model, engine, rank, offloaded)
    if rank < 2:
        root = Path('/opt/dlami/nvme/nl2sql/checkpoints/layer_trace')/f'{args.mode}_cp{args.sp}_rank{rank}'
        root.mkdir(parents=True, exist_ok=True)
        for label, entry in captured.items():
            assert 'gradient' in entry, label
            torch.save(entry, root/f'{label}.pt')
        print(f'Saved {len(captured)} layer traces to {root}', flush=True)
    for h in handles:
        h.remove()


if __name__=='__main__':
    probe.run_comparison = traced
    probe.main()
