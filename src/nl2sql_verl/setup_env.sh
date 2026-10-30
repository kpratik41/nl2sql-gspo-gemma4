#!/usr/bin/env bash
# Compatibility entry point for the historical AutoModel path.
# Current native environment: setup_fsdp_env.sh; screened launcher: run_screened_training.py.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/legacy/automodel/setup_env.sh" "$@"
