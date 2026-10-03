"""§5.2 naive_ddp_benchmarking: measure per-step time and gradient-comm share.

PDF spec: 1 node x 2 GPUs, xl model. We use medium (xl OOMs on A6000 per
Section 2 findings) and note it. Compares:
  - no DDP (baseline single-process, full batch on one GPU)
  - naive DDP (per-parameter all-reduce after backward)

Run on server with 2+ GPUs via nccl. Usage:
    mp.spawn via torchrun or uv run python -m lm_systems.naive_ddp_benchmark
"""
import argparse
import math
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from lm_basics.model import BasicsTransformerLM
from lm_basics.nn_utils import cross_entropy
from lm_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def setup(rank, world_size):
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _bench_rank(rank, args):
    setup(rank, args.world_size)
    device = f"cuda:{rank}"

    cfg = MODEL_CONFIGS[args.model_size]
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.ctx,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)
    optimizer = AdamW(model.parameters())

    # each rank gets local_bs = batch/world_size examples (DP sharding)
    local_bs = args.batch // args.world_size
    inputs = torch.randint(0, 10000, (local_bs, args.ctx), device=device)
    targets = torch.randint(0, 10000, (local_bs, args.ctx), device=device)

    def step(with_comm):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        if with_comm:
            t0 = timeit.default_timer()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(args.world_size)
            torch.cuda.synchronize()
            comm_ms = (timeit.default_timer() - t0) * 1e3
        else:
            comm_ms = 0.0
        optimizer.step()
        torch.cuda.synchronize()
        return comm_ms

    # warmup
    for _ in range(args.warmup):
        step(True)

    # measure full step (with comm) and comm-only
    full_times, comm_times = [], []
    for _ in range(args.iters):
        t0 = timeit.default_timer()
        c = step(True)
        full_times.append((timeit.default_timer() - t0) * 1e3)
        comm_times.append(c)

    # gather across ranks
    if rank == 0:
        full_mean = statistics.fmean(full_times)
        comm_mean = statistics.fmean(comm_times)
        print(f"=== {args.model_size}, world={args.world_size}, batch={args.batch} ===")
        print(f"per-step total:  {full_mean:8.1f} ms")
        print(f"comm (all-reduce): {comm_mean:8.1f} ms")
        print(f"comm share:       {comm_mean/full_mean*100:5.1f}%")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="medium")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    mp.spawn(_bench_rank, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
