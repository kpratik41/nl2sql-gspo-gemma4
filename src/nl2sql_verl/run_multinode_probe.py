"""Run an owned torchrun probe on the two H200 nodes, collecting both logs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
PYTHON = '/opt/dlami/nvme/nl2sql/runtime/venv/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--port', type=int, default=29610)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--module', required=True)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    if not a.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Probe name must be alphanumeric with hyphens/underscores')
    root = REPO/'outputs/h200_checks'/a.name
    root.mkdir(parents=True, exist_ok=False)
    if a.args[:1] == ['--']:
        a.args = a.args[1:]
    env = {'OMP_NUM_THREADS': '2', 'TORCHINDUCTOR_COMPILE_THREADS': '2',
           'NCCL_DEBUG': 'INFO', 'NCCL_SOCKET_IFNAME': 'enp71s0',
           'GLOO_SOCKET_IFNAME': 'enp71s0',
           'NL2SQL_ATTENTION_FP32': '1', 'NL2SQL_FLEX_FP32_PRECISION': 'ieee',
           'NL2SQL_FLEX_STATIC': '1', 'NL2SQL_FLEX_CANONICAL_LAYOUT': '0',
           'TORCHINDUCTOR_CACHE_DIR': '/opt/dlami/nvme/nl2sql/caches/inductor',
           'TRITON_CACHE_DIR': '/opt/dlami/nvme/nl2sql/caches/triton',
           'NL2SQL_DATABASE_DIR': '/opt/dlami/nvme/nl2sql/databases',
           'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    processes = []
    streams = []
    start = time.monotonic()
    try:
        for node in range(2):
            command = [str(REPO/'src/nl2sql_verl/worker_python.sh'), '-m', 'torch.distributed.run', '--nnodes=2',
                       '--nproc-per-node=8', f'--node-rank={node}',
                       '--master-addr=172.31.11.186', f'--master-port={a.port}',
                       '-m', a.module, *a.args]
            stream = (root/f'node{node}.log').open('w', buffering=1)
            streams.append(stream)
            if node == 0:
                proc = subprocess.Popen(command, cwd=REPO, env=os.environ | env,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            else:
                pidfile = f'/tmp/nl2sql-{a.name}.pid'
                script = (f'cd {shlex.quote(str(REPO))} && echo $$ > {shlex.quote(pidfile)} && '
                          'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+ ' '+shlex.join(command))
                proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'h200-node1',
                                         'setsid --wait bash -c '+shlex.quote(script)],
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            processes.append(proc)
        while any(p.poll() is None for p in processes):
            codes = [p.poll() for p in processes]
            (root/'heartbeat.json').write_text(json.dumps({'elapsed_seconds': time.monotonic()-start,
                                                          'node_exit_codes': codes})+'\n')
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f'Probe worker failed: {codes}; see {root}')
            if time.monotonic()-start > a.timeout:
                raise TimeoutError(f'Probe exceeded {a.timeout}s')
            time.sleep(5)
        codes = [p.returncode for p in processes]
        if any(codes):
            raise RuntimeError(f'Probe failed: {codes}')
        (root/'status.json').write_text(json.dumps({'status': 'pass', 'node_exit_codes': codes,
                                                   'elapsed_seconds': time.monotonic()-start}, indent=2)+'\n')
        print(json.dumps({'status': 'pass', 'logs': str(root)}), flush=True)
    finally:
        if any(p.poll() is None for p in processes):
            subprocess.run(['ssh', 'h200-node1',
                            f'test ! -f /tmp/nl2sql-{a.name}.pid || kill -TERM -- -$(cat /tmp/nl2sql-{a.name}.pid)'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
