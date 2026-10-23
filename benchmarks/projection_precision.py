"""Replay the first differing projection with checkpoint weights and saved inputs."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from cp_trace import errors

torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
root = Path('benchmark_results/cp_forward/debug')
trace = torch.load(root/'prefix/trace_cp1_rank0_128.pt', weights_only=True)
x = trace['values']['layer_00.input_layernorm'].cuda()
snapshot = Path.home()/'.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'
key = 'model.language_model.layers.0.self_attn.q_proj.weight'
shard = json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map'][key]
with safe_open(snapshot/shard, framework='pt', device='cuda') as f:
    weight = f.get_tensor(key)
reference = torch.nn.functional.linear(x.float(), weight.float())
rows = []
for name, setting in [('default', True), ('no_reduced_reduction', False), ('cublaslt_no_reduced_no_splitk', (False, False))]:
    if name.startswith('cublaslt'):
        torch.backends.cuda.preferred_blas_library('cublaslt')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = setting
    full = torch.nn.functional.linear(x, weight)
    prefix = torch.nn.functional.linear(x[:, :64].contiguous(), weight)
    row = {'mode': name, 'shape_error': errors(prefix, full[:, :64]),
           'full_vs_fp32': errors(full, reference),
           'prefix_vs_fp32': errors(prefix, reference[:, :64])}
    rows.append(row)
    print(json.dumps(row), flush=True)
full_fp32 = torch.nn.functional.linear(x.float(), weight.float()).to(x.dtype)
prefix_fp32 = torch.nn.functional.linear(x[:, :64].float(), weight.float()).to(x.dtype)
rows.append({'mode': 'fp32_linear_cast_bf16', 'shape_error': errors(prefix_fp32, full_fp32[:, :64])})
(root/'projection_precision.json').write_text(json.dumps(rows, indent=2)+'\n')
