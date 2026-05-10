#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_command(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {"ok": False, "stdout": "", "stderr": f"command not found: {command[0]}"}

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def parse_visible_gpus(gpus_arg):
    device_count = torch.cuda.device_count()
    if gpus_arg:
        selected = [int(part.strip()) for part in gpus_arg.split(",") if part.strip()]
    else:
        selected = list(range(device_count))

    invalid = [index for index in selected if index < 0 or index >= device_count]
    if invalid:
        raise ValueError(f"GPU indices must be logical indices within 0..{device_count - 1}; got {invalid}")

    return selected


def summarize_environment(selected_gpus):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    nccl_env = {
        key: os.environ.get(key, "<unset>")
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "NCCL_DEBUG",
            "TORCH_NCCL_TIMEOUT_MS",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "NCCL_SOCKET_IFNAME",
        ]
    }
    gpu_details = []
    peer_matrix = []
    for gpu_index in selected_gpus:
        props = torch.cuda.get_device_properties(gpu_index)
        gpu_details.append(
            {
                "index": gpu_index,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            }
        )

    for source in selected_gpus:
        row = {"gpu": source, "peer_access": {}}
        for target in selected_gpus:
            if source == target:
                row["peer_access"][str(target)] = True
                continue
            try:
                row["peer_access"][str(target)] = torch.cuda.can_device_access_peer(source, target)
            except RuntimeError as exc:
                row["peer_access"][str(target)] = f"error: {exc}"
        peer_matrix.append(row)

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": cuda_visible,
        "selected_gpus": selected_gpus,
        "torch_gpu_count": torch.cuda.device_count(),
        "nccl_version": getattr(torch.cuda.nccl, "version", lambda: None)(),
        "env": nccl_env,
        "gpus": gpu_details,
        "peer_access": peer_matrix,
    }


def write_result(result_dir, local_rank, payload):
    output_path = os.path.join(result_dir, f"rank_{local_rank}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def nccl_worker(local_rank, world_size, port, timeout_seconds, result_dir):
    device = torch.device(f"cuda:{local_rank}")
    start = time.time()
    try:
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=local_rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )

        all_reduce_input = torch.tensor([float(local_rank + 1)], device=device)
        dist.all_reduce(all_reduce_input, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(1, world_size + 1))
        if abs(all_reduce_input.item() - expected_sum) > 1e-5:
            raise RuntimeError(
                f"all_reduce mismatch on rank {local_rank}: got {all_reduce_input.item()} expected {expected_sum}"
            )

        broadcast_input = torch.tensor([local_rank], device=device, dtype=torch.int64)
        if local_rank == 0:
            broadcast_input.fill_(4242)
        dist.broadcast(broadcast_input, src=0)
        if int(broadcast_input.item()) != 4242:
            raise RuntimeError(
                f"broadcast mismatch on rank {local_rank}: got {int(broadcast_input.item())} expected 4242"
            )

        gather_input = torch.tensor([local_rank], device=device, dtype=torch.int64)
        gathered = [torch.empty_like(gather_input) for _ in range(world_size)]
        dist.all_gather(gathered, gather_input)
        gathered_values = [int(item.item()) for item in gathered]
        expected_values = list(range(world_size))
        if gathered_values != expected_values:
            raise RuntimeError(
                f"all_gather mismatch on rank {local_rank}: got {gathered_values} expected {expected_values}"
            )

        dist.barrier()
        write_result(
            result_dir,
            local_rank,
            {
                "rank": local_rank,
                "ok": True,
                "seconds": round(time.time() - start, 3),
                "device": torch.cuda.get_device_name(local_rank),
            },
        )
    except Exception as exc:
        write_result(
            result_dir,
            local_rank,
            {
                "rank": local_rank,
                "ok": False,
                "seconds": round(time.time() - start, 3),
                "error": repr(exc),
            },
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_nccl_smoke(world_size, timeout_seconds):
    if world_size < 2:
        return {"ok": True, "skipped": True, "reason": "need at least 2 visible GPUs for NCCL smoke test"}

    port = pick_free_port()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="nccl_health_") as result_dir:
        try:
            mp.spawn(
                nccl_worker,
                args=(world_size, port, timeout_seconds, result_dir),
                nprocs=world_size,
                join=True,
            )
            results = []
            for rank in range(world_size):
                result_path = os.path.join(result_dir, f"rank_{rank}.json")
                with open(result_path, "r", encoding="utf-8") as handle:
                    results.append(json.load(handle))
            return {
                "ok": True,
                "skipped": False,
                "port": port,
                "seconds": round(time.time() - started, 3),
                "results": sorted(results, key=lambda item: item["rank"]),
            }
        except Exception as exc:
            results = []
            for rank in range(world_size):
                result_path = os.path.join(result_dir, f"rank_{rank}.json")
                if not os.path.exists(result_path):
                    continue
                with open(result_path, "r", encoding="utf-8") as handle:
                    results.append(json.load(handle))
            return {
                "ok": False,
                "skipped": False,
                "port": port,
                "seconds": round(time.time() - started, 3),
                "error": repr(exc),
                "results": sorted(results, key=lambda item: item["rank"]),
            }


def main():
    parser = argparse.ArgumentParser(description="Check local GPU, NCCL, and launcher health.")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU indices to test. Defaults to all visible GPUs.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Per-process-group timeout for the NCCL smoke test.")
    parser.add_argument("--skip-nccl", action="store_true", help="Skip the NCCL collective smoke test.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available in this environment.", file=sys.stderr)
        return 1

    try:
        selected_gpus = parse_visible_gpus(args.gpus)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not selected_gpus:
        print("No GPUs selected for health checks.", file=sys.stderr)
        return 1

    summary = summarize_environment(selected_gpus)
    summary["nvidia_smi"] = {
        "summary": run_command([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader",
        ]),
        "topology": run_command(["nvidia-smi", "topo", "-m"]),
    }

    if shutil.which("df"):
        summary["disk"] = run_command(["df", "-h", "/", "/dev/shm"])

    if not args.skip_nccl:
        summary["nccl_smoke"] = run_nccl_smoke(len(selected_gpus), args.timeout_seconds)
    else:
        summary["nccl_smoke"] = {"ok": True, "skipped": True, "reason": "requested by --skip-nccl"}

    summary["overall_ok"] = summary["nvidia_smi"]["summary"]["ok"] and summary["nccl_smoke"]["ok"]

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Host: {summary['host']}")
        print(f"Python/Torch/CUDA: {summary['python']} / {summary['torch']} / {summary['cuda_runtime']}")
        print(f"Visible GPUs: {summary['cuda_visible_devices']}")
        print(f"Selected GPUs: {summary['selected_gpus']}")
        print(f"NCCL version: {summary['nccl_version']}")
        print("GPU inventory:")
        for gpu in summary["gpus"]:
            print(
                f"  GPU {gpu['index']}: {gpu['name']} | {gpu['total_memory_gib']} GiB | "
                f"SMs={gpu['multi_processor_count']} | cc={gpu['major']}.{gpu['minor']}"
            )
        print("nvidia-smi summary:")
        print(summary["nvidia_smi"]["summary"]["stdout"] or summary["nvidia_smi"]["summary"]["stderr"])
        print("Peer access:")
        for row in summary["peer_access"]:
            print(f"  GPU {row['gpu']}: {row['peer_access']}")
        print("NCCL smoke:")
        print(json.dumps(summary["nccl_smoke"], indent=2, sort_keys=True))
        if summary["nvidia_smi"]["topology"]["stdout"]:
            print("nvidia-smi topo -m:")
            print(summary["nvidia_smi"]["topology"]["stdout"])

    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())