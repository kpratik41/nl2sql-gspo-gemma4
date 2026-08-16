#!/usr/bin/env bash
# OPTIONAL: build the flash-attn package. You almost certainly do NOT need this.
#
# ---------------------------------------------------------------------------
# Read this before running: FA2 is already in use without this package.
# ---------------------------------------------------------------------------
#
# 1. vLLM ROLLOUTS (eval + RL generation) use vLLM's own bundled FA2 kernels.
#    Confirmed in the worker logs:
#        Using FLASH_ATTN attention backend out of potential backends:
#          ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
#        Using FlashAttention version 2
#    This is independent of the pip `flash-attn` package. Installing or not
#    installing flash-attn changes nothing for rollouts.
#
# 2. TRANSFORMERS TRAINING (forward/backward) uses ATTN_IMPLEMENTATION=sdpa.
#    torch SDPA dispatches to its OWN FlashAttention-2 backend. Verified on
#    this box at Qwen3.8-27B's exact shapes (24 q heads / 4 kv heads /
#    head_dim=256, H200):
#        FLASH          OK
#        MEM_EFFICIENT  OK
#        MATH           OK
#    head_dim=256 is the FA2 limit and it passes, so SDPA gives FA2 kernels.
#
# 3. Either way this only affects the 16 `full_attention` layers. The other 48
#    are linear_attention (Gated DeltaNet) and never use FlashAttention at all
#    -- they run Triton/FLA GDN kernels from causal-conv1d + flash-linear-attention.
#    Those are the ones that actually matter here; see setup_qwen38_kernels.sh.
#
# ---------------------------------------------------------------------------
# Why a plain `pip install flash-attn` fails on this machine
# ---------------------------------------------------------------------------
#
#   RuntimeError: The detected CUDA version (13.2) mismatches the version that
#   was used to compile PyTorch (12.8). Please make sure to use the same CUDA
#   versions.
#
# Host toolkit is /usr/local/cuda-13.2 while torch is 2.10.0+cu128, and torch's
# cpp_extension._check_cuda_version aborts the build.
#
# DEAD END (do not retry): `pip install nvidia-cuda-nvcc-cu12==12.8.93` does NOT
# fix this. That wheel ships only `ptxas` -- there is no `nvcc` driver binary in
# it, so pointing CUDA_HOME at the wheel tree fails with
# "nvcc: No such file or directory". Verified.
#
# ---------------------------------------------------------------------------
# The only route that works: a real CUDA 12.8 toolkit
# ---------------------------------------------------------------------------
#
# Install one (needs root), then re-run this script with CUDA_HOME set:
#
#   wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
#   sudo sh cuda_12.8.1_570.124.06_linux.run --toolkit --silent --toolkitpath=/usr/local/cuda-12.8
#   CUDA_HOME=/usr/local/cuda-12.8 bash scripts/setup_flash_attn.sh
#
# Then opt in for training with ATTN_IMPLEMENTATION=flash_attention_2.
# Expect a long compile (tens of minutes) even with MAX_JOBS parallelism.
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
PIP="$VENV/bin/pip"

if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "ERROR: CUDA_HOME is not set." >&2
  echo "flash-attn needs a CUDA toolkit matching torch's build (cu128)." >&2
  echo "Read the header of this script -- you probably do not need flash-attn." >&2
  exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "ERROR: no nvcc at ${CUDA_HOME}/bin/nvcc" >&2
  echo "The pip nvidia-cuda-nvcc-cu12 wheel ships only ptxas and will not work." >&2
  exit 1
fi

TORCH_CUDA="$("$VENV/bin/python" -c 'import torch; print(torch.version.cuda)')"
NVCC_CUDA="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
echo "[flash-attn] torch CUDA=${TORCH_CUDA}  nvcc CUDA=${NVCC_CUDA}"
if [[ "${TORCH_CUDA}" != "${NVCC_CUDA}"* ]]; then
  echo "ERROR: nvcc ${NVCC_CUDA} does not match torch ${TORCH_CUDA}; the build will abort." >&2
  exit 1
fi

export PATH="${CUDA_HOME}/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-32}"          # nvcc jobs are memory-hungry; 192 would OOM
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H200 = sm_90 only

"$PIP" install --no-build-isolation "flash-attn==2.8.3.post1"

"$VENV/bin/python" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "FLASH_ATTN_DONE  -- opt in with ATTN_IMPLEMENTATION=flash_attention_2"
