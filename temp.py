#!/usr/bin/env python3
import argparse
import math
import multiprocessing as mp
import os
import signal
import sys
import time


def worker(stop_event, duty_cycle, period_s):
    busy_s = period_s * duty_cycle
    sleep_s = max(0.0, period_s - busy_s)

    while not stop_event.is_set():
        start = time.perf_counter()
        while (time.perf_counter() - start) < busy_s:
            pass
        if sleep_s > 0:
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-percent", type=float, default=2.0,
                        help="Target overall CPUUtilization percent across all vCPUs.")
    parser.add_argument("--period", type=float, default=1.0,
                        help="Duty-cycle control period in seconds.")
    args = parser.parse_args()

    vcpus = os.cpu_count() or 1
    target = max(0.0, min(args.target_percent, 100.0))
    busy_core_equiv = (target / 100.0) * vcpus

    if busy_core_equiv <= 0:
        print("Target percent is 0, nothing to do.")
        return

    workers = max(1, math.ceil(busy_core_equiv))
    duty_cycle = min(1.0, busy_core_equiv / workers)

    print(f"Detected vCPUs: {vcpus}")
    print(f"Target overall CPUUtilization: {target:.2f}%")
    print(f"Approx busy-core equivalent: {busy_core_equiv:.2f}")
    print(f"Launching workers: {workers}")
    print(f"Per-worker duty cycle: {duty_cycle:.3f}")
    print("Press Ctrl+C to stop.")

    stop_event = mp.Event()
    processes = []

    def shutdown(signum=None, frame=None):
        stop_event.set()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for _ in range(workers):
        process = mp.Process(target=worker, args=(stop_event, duty_cycle, args.period), daemon=True)
        process.start()
        processes.append(process)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()