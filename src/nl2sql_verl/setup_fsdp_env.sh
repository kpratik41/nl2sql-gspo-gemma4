#!/usr/bin/env bash
# Replicate the ongoing native FSDP2 environment in a separate virtualenv.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_python="${VERL_BASE_PYTHON:-python3.12}"
runtime_venv="${VERL_VENV:-$repo_root/.venv-h200-replica}"
source_root="${VERL_SOURCE:-$repo_root/outputs/verl_runtime/verl-d430157}"
revision=d430157d15516d89e0cc853fd10fcd21d9234c44
check_only=false
case "${1:-}" in
  --check-only) check_only=true ;;
  --help|-h)
    echo "Usage: VERL_VENV=/new/venv VERL_SOURCE=/path/to/verl bash $0 [--check-only]"
    echo "Defaults: a separate .venv-h200-replica and outputs/verl_runtime/verl-d430157."
    exit 0 ;;
  '') ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
cd "$repo_root"
if ! "$check_only"; then
  "$runtime_python" -c 'import platform,sys; assert sys.version_info[:2] == (3,12); assert platform.system()=="Linux" and platform.machine()=="x86_64"'
  if [[ ! -x "$runtime_venv/bin/python" ]]; then
    "$runtime_python" -m venv "$runtime_venv"
  fi
  "$runtime_venv/bin/python" -m pip install 'pip==26.2.2'
  "$runtime_venv/bin/python" -m pip install -r "$repo_root/requirements.txt"
  if [[ ! -d "$source_root/.git" ]]; then
    if [[ -e "$source_root" ]]; then
      echo "Choose an unused VERL_SOURCE directory; $source_root already exists without a git checkout." >&2
      exit 1
    fi
    mkdir -p "$source_root"
    git -C "$source_root" init
    git -C "$source_root" remote add origin https://github.com/verl-project/verl.git
    git -C "$source_root" fetch --depth 1 origin "$revision"
    git -C "$source_root" checkout --detach "$revision"
  fi
fi
if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$revision" ]]; then
  echo "Expected clean verl revision $revision at $source_root" >&2
  exit 1
fi
if [[ -n "$(git -C "$source_root" status --porcelain)" ]]; then
  echo "The verl source checkout has local changes; preserve them and choose a clean VERL_SOURCE." >&2
  exit 1
fi
if ! "$check_only"; then
  "$runtime_venv/bin/python" - "$source_root" "$repo_root" <<'PY'
from pathlib import Path
import site,sys
Path(site.getsitepackages()[0], 'nl2sql-verl-source.pth').write_text(
    str(Path(sys.argv[1]).resolve())+'\n'+str(Path(sys.argv[2]).resolve())+'\n')
PY
fi
"$runtime_venv/bin/python" -m pip check
VERL_PYTHON="$runtime_venv/bin/python" "$repo_root/src/nl2sql_verl/worker_python.sh" - "$repo_root" "$source_root" <<'PY'
from pathlib import Path
from importlib import metadata
import json,sys,sysconfig
from packaging.requirements import Requirement
from urllib.parse import urldefrag,unquote

root, source = map(Path,sys.argv[1:])
assert sys.version_info[:2] == (3,12)
checked = 0
for raw in (root/'requirements.txt').read_text().splitlines():
    line=raw.strip()
    if not line or line.startswith(('#','--')):
        continue
    requirement=Requirement(line)
    dist=metadata.distribution(requirement.name)
    if not requirement.specifier.contains(dist.version, prereleases=True):
        if requirement.name == 'pip':
            import pip
            print(f'Installer metadata: pip {dist.version}; imported pip {pip.__version__}; canonical snapshot 26.2.2')
        else:
            raise RuntimeError(f'{requirement.name}: installed {dist.version}, expected {requirement.specifier}')
    if requirement.url:
        installed=json.loads(dist.read_text('direct_url.json') or '{}')
        url,fragment=urldefrag(requirement.url)
        assert unquote(installed.get('url','')) == unquote(url), requirement.name
        if fragment.startswith('sha256='):
            assert installed['archive_info']['hashes']['sha256'] == fragment.split('=',1)[1]
    checked += 1
pth=Path(sysconfig.get_paths()['purelib'])/'nl2sql-verl-source.pth'
assert {source.resolve(), root.resolve()} <= {Path(p).resolve() for p in pth.read_text().splitlines()}
import torch
import vllm._C
from vllm import LLM
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
import src.nl2sql_verl.fsdp_engine
assert torch.__version__ == '2.10.0+cu130' and torch.version.cuda == '13.0'
assert metadata.version('vllm') == '0.19.1+cu130'
print(f'Verified {checked} package versions, source paths, native FSDP2/vLLM imports and CUDA 13.0 build.')
PY
echo "Native FSDP2 environment ready: $runtime_venv"
echo "Run GPU commands with VERL_PYTHON=$runtime_venv/bin/python src/nl2sql_verl/worker_python.sh ..."
