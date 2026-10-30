"""Check all 16 GPUs, node-local CP pairs, and inter-node NCCL reductions."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10),
                            device_id=torch.device('cuda', local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    assert len(set(hosts)) == 2
    assert all(hosts[i] == hosts[i+1] for i in range(0, world, 2))
    cp_group = None
    for start in range(0, world, 2):
        group = dist.new_group([start, start+1], backend='nccl')
        if rank in (start, start+1):
            cp_group = group
    x = torch.full((1024 * 1024,), float(rank), device='cuda')
    dist.all_reduce(x)
    assert bool((x == sum(range(world))).all())
    send = torch.full((2, 1024), float(rank), device='cuda')
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=cp_group)
    assert bool((receive[0] == rank//2*2).all())
    assert bool((receive[1] == rank//2*2+1).all())
    x = torch.ones(4 * 1024 * 1024, device='cuda')
    for _ in range(3):
        dist.all_reduce(x)
        x.fill_(1.)
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(10):
        dist.all_reduce(x)
        x.fill_(1.)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    if rank == 0:
        report = {'status': 'pass', 'world_size': world, 'rank_hosts': hosts,
                  'node_local_cp_pairs': True, 'all_reduce': True, 'all_to_all': True,
                  'ten_16MiB_reductions_seconds': elapsed}
        path = Path('outputs/h200_checks/collectives.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
