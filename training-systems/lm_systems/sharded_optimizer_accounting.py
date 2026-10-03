"""§6 optimizer_state_sharding_accounting — non-sharded vs sharded.

non-sharded runs independently on each rank (no collectives, same result
everywhere): full AdamW state on every rank. sharded runs collectively:
each rank keeps only 1/world_size of Adam state. Compares memory and time.
"""
import argparse
import os
import statistics
import timeit

import torch
import torch.distributed as dist

from lm_basics.model import BasicsTransformerLM
from lm_basics.nn_utils import cross_entropy
from lm_basics.optimizer import AdamW
from lm_systems.sharded_optimizer import ShardedOptimizer

MODEL_CONFIGS = {
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def measure(rank, world_size, args, use_sharded):
    device = f"cuda:{rank}"
    cfg = MODEL_CONFIGS[args.model_size]
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=10000, context_length=args.ctx,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
    ).to(device)

    opt = ShardedOptimizer(model.parameters(), AdamW) if use_sharded else AdamW(model.parameters())
    torch.cuda.synchronize()
    after_init = torch.cuda.memory_allocated() / 1024**2

    local_bs = args.batch // world_size
    inputs = torch.randint(0, 10000, (local_bs, args.ctx), device=device)
    targets = torch.randint(0, 10000, (local_bs, args.ctx), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    step()
    before_step = torch.cuda.memory_allocated() / 1024**2
    step()
    after_step = torch.cuda.memory_allocated() / 1024**2

    for _ in range(3):
        step()
    times = []
    for _ in range(5):
        t0 = timeit.default_timer()
        step()
        times.append((timeit.default_timer() - t0) * 1e3)
    return after_init, before_step, after_step, statistics.fmean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="medium")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Non-sharded: no collectives, run on every rank (results identical).
    ns = measure(rank, world_size, args, use_sharded=False)

    # Sharded: collective, must be run on all ranks together.
    sh = measure(rank, world_size, args, use_sharded=True)

    if rank == 0:
        a, b, c, t_ns = ns
        a2, b2, c2, t_sh = sh
        print(f"=== {args.model_size} world={world_size} ===")
        print(f"{'':12} {'non-sharded':>12} {'sharded':>12}")
        print(f"{'after init':12} {a:12.1f} {a2:12.1f} MiB")
        print(f"{'before step':12} {b:12.1f} {b2:12.1f} MiB")
        print(f"{'after step':12} {c:12.1f} {c2:12.1f} MiB")
        print(f"{'per-step':12} {t_ns:12.1f} {t_sh:12.1f} ms")
        print(f"\nmem savings (after step): {(c - c2) / c * 100:5.1f}%")
        print(f"time overhead: {(t_sh / t_ns - 1) * 100:5.1f}%")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
