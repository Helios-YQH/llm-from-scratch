"""§5.1 all-reduce benchmark: single-node, multi-process (PDF 5.1).

Measures all-reduce latency across data sizes and world sizes.
Supports gloo (CPU, local dev) and nccl (GPU).

Usage:
    uv run python -m lm_systems.distributed_communication_single_node \
        --backend gloo --sizes-mb 1 10 100 1000 --world-sizes 2 4 6
"""
import argparse
import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run_allreduce_benchmark(rank, args):
    # rank 0 prints results; others just execute
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    if rank == 0:
        print(f"=== backend={args.backend} world_size={args.world_size} ===")

    results = {}  # size_mb -> mean_ms (gathered on rank 0)
    for size_mb in args.sizes_mb:
        n = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        data = torch.ones(n, dtype=torch.float32)
        if args.backend == "nccl":
            torch.cuda.set_device(rank % torch.cuda.device_count())
            data = data.cuda()

        # warmup
        for _ in range(args.warmup):
            dist.all_reduce(data, async_op=False)

        # measure
        times = []
        for _ in range(args.iters):
            t0 = timeit.default_timer()
            dist.all_reduce(data, async_op=False)
            if args.backend == "nccl":
                torch.cuda.synchronize()
            times.append(timeit.default_timer() - t0)
        mean_ms = statistics.fmean(times) * 1e3
        results[size_mb] = mean_ms

        if rank == 0:
            print(f"  {size_mb:>6} MB: {mean_ms:8.3f} ms")

    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    ap.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 10, 100, 1000])
    ap.add_argument("--world-sizes", nargs="+", type=int, default=[2, 4, 6])
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    for ws in args.world_sizes:
        args.world_size = ws
        mp.spawn(fn=_run_allreduce_benchmark, args=(args,), nprocs=ws, join=True)


if __name__ == "__main__":
    main()
