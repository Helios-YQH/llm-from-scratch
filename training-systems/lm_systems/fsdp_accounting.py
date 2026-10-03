"""§7 fsdp_accounting: peak memory of FSDP vs non-sharded (DDP-style).

Each rank measures steady-state memory (params + optimizer state) for a
full model vs the FSDP-sharded model, plus peak memory during one step.
Uses large (xl OOMs on A6000 per Section 2).

Run: torchrun --nproc_per_node=N -m lm_systems.fsdp_accounting
"""
import os

import torch
import torch.distributed as dist

from lm_basics.model import BasicsTransformerLM
from lm_basics.nn_utils import cross_entropy
from lm_basics.optimizer import AdamW
from lm_systems.fsdp import FSDP

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def build_model(args, device):
    cfg = MODEL_CONFIGS[args.model_size]
    return BasicsTransformerLM(
        vocab_size=10000, context_length=args.ctx,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
    ).to(device)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="large")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    def mem_mib():
        return torch.cuda.memory_allocated() / 1024**2

    # --- non-sharded baseline (full model + full optimizer state) ---
    torch.manual_seed(0)
    model_full = build_model(args, device)
    opt_full = AdamW(model_full.parameters())
    torch.cuda.synchronize()
    full_steady = mem_mib()

    local_bs = args.batch // world_size
    inputs = torch.randint(0, 10000, (local_bs, args.ctx), device=device)
    targets = torch.randint(0, 10000, (local_bs, args.ctx), device=device)

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    step(model_full, opt_full)
    full_peak = torch.cuda.max_memory_allocated() / 1024**2

    del model_full, opt_full
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- FSDP sharded model ---
    torch.manual_seed(0)
    model_fsdp = FSDP(build_model(args, device))
    opt_fsdp = AdamW(model_fsdp.parameters())
    # Release the full params that FSDP sharded away (they're dropped by GC;
    # empty_cache returns the freed blocks to the CUDA allocator).
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    fsdp_steady = mem_mib()

    torch.cuda.reset_peak_memory_stats()
    step(model_fsdp, opt_fsdp)
    fsdp_peak = torch.cuda.max_memory_allocated() / 1024**2

    if rank == 0:
        print(f"=== {args.model_size} world={world_size} ===")
        print(f"{'':16} {'non-sharded':>13} {'FSDP':>13}")
        print(f"{'steady (MiB)':16} {full_steady:13.1f} {fsdp_steady:13.1f}")
        print(f"{'peak (MiB)':16} {full_peak:13.1f} {fsdp_peak:13.1f}")
        print(f"\nsteady savings: {(full_steady - fsdp_steady) / full_steady * 100:5.1f}%")
        print(f"peak savings:   {(full_peak - fsdp_peak) / full_peak * 100:5.1f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
