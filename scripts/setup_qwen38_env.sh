#!/usr/bin/env bash
# Build the Qwen3.8-27B RL training environment.
#
# Version choices follow env.md (the proven replication snapshot: torch 2.10.0 +
# transformers 5.12.1 + trl 1.4.0 + vllm 0.19.1 + triton 3.6.0 + flashinfer 0.6.6,
# which passed `pip check`), not the stale requirements.txt pins.
#
# Qwen3.8-27B constraints:
#   - transformers >= 5.8.0 (config.json was written by 5.8.0.dev0)
#   - vllm >= 0.17.0       (per recipes.vllm.ai/Qwen/Qwen3.8-27B)
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"

python3.12 -m venv "$VENV" 2>/dev/null || true
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# Install torch first so vllm resolves against the pinned build rather than
# dragging in its own.
"$VENV/bin/pip" install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"

"$VENV/bin/pip" install \
  "transformers==5.12.1" \
  "trl==1.4.0" \
  "datasets==5.0.0" \
  "accelerate==1.14.0" \
  "deepspeed==0.19.2" \
  "vllm==0.19.1" \
  "wandb==0.28.0" \
  "sentencepiece==0.2.1" \
  "protobuf==6.33.6" \
  "rank-bm25==0.2.2"

"$VENV/bin/pip" check || echo "WARNING: pip check reported conflicts (see above)"
echo "SETUP_DONE"
