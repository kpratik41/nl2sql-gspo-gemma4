#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
.venv/bin/python - <<'PY'
import time
import urllib.request
deadline = time.monotonic() + 900
for port in (8000, 8001):
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).close()
            print(f'Shard on port {port} ready', flush=True)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise SystemExit('Server readiness timed out; check logs/server-shard*.log')
            time.sleep(2)
PY
.venv/bin/python -m bird_inference.run --limit 2 --output outputs/smoke_tp4x2
.venv/bin/python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('outputs/smoke_tp4x2/summary.json').read_text())
if summary['completed'] != 2 or summary['tool_calls'] < 2:
    raise SystemExit('Smoke test did not complete native tool loops on both shards; inspect traces')
PY
.venv/bin/python -m bird_inference.run
.venv/bin/python -m bird_inference.evaluate
