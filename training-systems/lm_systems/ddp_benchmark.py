"""§5.3 DDP benchmarking: naive vs flat vs overlap (per-step time + comm share).

PDF spec: 1 node x 2 GPUs, xl model. We use medium (xl OOMs on A6000 per
Section 2 findings). Three DDP strategies:
  - naive:   per-parameter all-reduce after backward
  - flat:    single all-reduce over a flat buffer after backward
  - overlap: async per-parameter all-reduce during backward (hooks)

Run on server with 2+ GPUs via nccl:
    uv run python -m lm_systems.ddp_benchmark --variant naive|flat|overlap
"""
import argparse
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lm_basics.model import BasicsTransformerLM
from lm_basics.nn_utils import cross_entropy
from lm_basics.optimizer import AdamW
from lm_systems.ddp import DDP, FlatDDP, OverlapDDP

MODEL_CONFIGS = {
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
VARIANT_CLS = {"naive": DDP, "flat": FlatDDP, "overlap": OverlapDDP}


def setup(rank, world_size):
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _bench_rank(rank, args):
    setup(rank, args.world_size)
    device = f"cuda:{rank}"

    cfg = MODEL_CONFIGS[args.model_size]
    torch.manual_seed(0)
    base = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.ctx,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)
    model = VARIANT_CLS[args.variant](base)
    optimizer = AdamW(model.parameters())

    local_bs = args.batch // args.world_size
    inputs = torch.randint(0, 10000, (local_bs, args.ctx), device=device)
    targets = torch.randint(0, 10000, (local_bs, args.ctx), device=device)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        t0 = timeit.default_timer()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        sync_ms = (timeit.default_timer() - t0) * 1e3
        optimizer.step()
        torch.cuda.synchronize()
        return sync_ms

    for _ in range(args.warmup):
        one_step()

    full_times, sync_times = [], []
    for _ in range(args.iters):
        t0 = timeit.default_timer()
        s = one_step()
        full_times.append((timeit.default_timer() - t0) * 1e3)
        sync_times.append(s)

    if rank == 0:
        full_mean = statistics.fmean(full_times)
        sync_mean = statistics.fmean(sync_times)
        print(f"=== {args.variant}: {args.model_size}, world={args.world_size}, batch={args.batch} ===")
        print(f"per-step total:  {full_mean:8.1f} ms")
        print(f"sync/comm time:  {sync_mean:8.1f} ms")
        print(f"comm share:      {sync_mean/full_mean*100:5.1f}%")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_CLS.keys()), required=True)
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="medium")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    mp.spawn(_bench_rank, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
