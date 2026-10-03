"""Official PyTorch DDP / ZeRO-1 / FSDP benchmark (Section 7 cross-check).

Compares the three official PyTorch distributed training methods on memory
(steady + peak) and per-step time, for the same model. Companion to the
from-scratch FSDP in lm_systems/fsdp.py.

Run (2 GPUs, same NUMA): torchrun --nproc_per_node=2 -m ... --method ddp|zero|fsdp
"""
import argparse
import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_basics.model import BasicsTransformerLM, Linear, Embedding
from lm_basics.nn_utils import cross_entropy
from lm_basics.optimizer import AdamW as RefAdamW

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


def build_model(args, device):
    cfg = MODEL_CONFIGS[args.model_size]
    return BasicsTransformerLM(
        vocab_size=10000, context_length=args.ctx,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ddp", "zero", "fsdp"], required=True)
    ap.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="small")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)
    if args.backend == "nccl":
        torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    def mem_mib():
        return torch.cuda.memory_allocated() / 1024**2

    # ---- build model + wrap per method ----
    torch.manual_seed(0)
    model = build_model(args, device)

    if args.method == "ddp":
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model)
        opt = RefAdamW(model.parameters())
    elif args.method == "zero":
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(model.parameters(), RefAdamW)
    elif args.method == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

        def wrap_policy(module, recurse, nonwrapped_numel):
            return isinstance(module, (Linear, Embedding))

        model = FSDP(model, auto_wrap_policy=wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD)
        opt = RefAdamW(model.parameters())
    else:
        raise ValueError(args.method)

    # ---- steady memory after init (model + optimizer state) ----
    torch.cuda.synchronize()
    steady = mem_mib()

    local_bs = args.batch // world_size
    inputs = torch.randint(0, 10000, (local_bs, args.ctx), device=device)
    targets = torch.randint(0, 10000, (local_bs, args.ctx), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        step()
    peak = torch.cuda.max_memory_allocated() / 1024**2

    times = []
    for _ in range(5):
        t0 = timeit.default_timer()
        step()
        times.append((timeit.default_timer() - t0) * 1e3)
    per_step = statistics.fmean(times)

    if rank == 0:
        print(f"=== official {args.method}: {args.model_size} world={world_size} ===")
        print(f"steady (MiB): {steady:8.1f}")
        print(f"peak   (MiB): {peak:8.1f}")
        print(f"per-step (ms): {per_step:8.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
