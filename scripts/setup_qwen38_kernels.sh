#!/usr/bin/env bash
# Install the optimized attention kernels for Qwen3.8-27B (Qwen3.5 architecture).
#
# Qwen3.8-27B is a HYBRID model: config.json layer_types has 48 `linear_attention`
# layers and 16 `full_attention` layers (full_attention_interval=4, 64 layers).
# The two paths need different kernels:
#
#   full_attention (16 layers)   -> flash-attn      (attn_implementation="flash_attention_2")
#   linear_attention (48 layers) -> causal-conv1d + flash-linear-attention (fla)
#
# transformers/models/qwen3_5/modeling_qwen3_5.py gates these behind
# is_causal_conv1d_available() and is_flash_linear_attention_available(); when
# either is missing it SILENTLY falls back to torch_causal_conv1d_update /
# torch_chunk_gated_delta_rule, which are far slower and use more memory.
#
# fla is pinned to 0.4.2: fla 0.5.0 produces "!!!!" gibberish via the
# gated-delta-rule kernels (fla-org/flash-linear-attention#792). That report is
# inference-only and multimodal-only, so this text-only workload is unlikely to
# hit it, but 0.4.2 is the version confirmed good in that thread.
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
PIP="$VENV/bin/pip"

# flash-attn and causal-conv1d ship as sdists only, so they compile here.
# Cap parallelism: nvcc jobs are memory-hungry and 192 concurrent jobs OOM.
export MAX_JOBS="${MAX_JOBS:-32}"

echo "### 1/3 flash-linear-attention (has wheels, fast) ###"
"$PIP" install "flash-linear-attention==0.4.2"

echo "### 2/3 causal-conv1d (compiles) ###"
"$PIP" install --no-build-isolation "causal-conv1d==1.6.2.post1"

echo "### 3/3 flash-attn (compiles, slowest) ###"
"$PIP" install --no-build-isolation "flash-attn==2.8.3.post1"

"$PIP" check || echo "WARNING: pip check reported conflicts"
echo "KERNELS_DONE"
