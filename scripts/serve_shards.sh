#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8000 bash scripts/serve.sh > logs/server-shard0.log 2>&1 &
shard0_pid=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=8001 bash scripts/serve.sh > logs/server-shard1.log 2>&1 &
shard1_pid=$!
printf '%s\n' "$shard0_pid" "$shard1_pid" > logs/server-shards.pid
trap 'kill -TERM "$shard0_pid" "$shard1_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
wait -n "$shard0_pid" "$shard1_pid"
