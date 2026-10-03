"""训练步速基准: 用与 scripts/train.py 完全一致的模型/优化器/精度配置测每步耗时。

不 import train.py(它硬依赖 wandb/modal), 但复刻其训练循环: bf16 autocast、
torch.compile、fused AdamW、梯度裁剪、相同的 get_batch 取数方式。单卡直接跑,
多卡用 torchrun(走和 train.py 一样的 nccl DDP 路径)。

    # 单卡
    python scripts/benchmark_train_step.py --train-bin /tmp/speedtest/train.bin --steps 40

    # 双卡
    torchrun --standalone --nproc_per_node=2 scripts/benchmark_train_step.py \\
        --train-bin /tmp/speedtest/train.bin --steps 40

输出: 预热后的每步耗时(均值/中位数)、tokens/s, 并给出 16384 步(官方配置)的预计时间。
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from lm_basics.data import get_batch
from lm_basics.model import BasicsTransformerLM
from lm_basics.train_config import ModelConfig, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--steps", type=int, default=40, help="计时步数(不含预热)")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128, help="每卡 batch(官方配置 128)")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        from torch.distributed import init_process_group

        init_process_group(backend="nccl")
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world_size, device = 0, 1, "cuda"

    torch.manual_seed(train_cfg.seed + rank)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    model = BasicsTransformerLM(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        rope_theta=model_cfg.rope_theta,
    ).to(device)

    if not args.no_compile:
        model = torch.compile(model)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank])

    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {"params": [p for p in param_dict.values() if p.dim() >= 2], "weight_decay": train_cfg.weight_decay},
        {"params": [p for p in param_dict.values() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_cfg.lr, betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                                  eps=train_cfg.adam_eps, fused=True)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    batch_x, batch_y = get_batch(train_data, batch_size=args.batch_size,
                                 context_length=model_cfg.context_length, device=device)

    def one_step() -> None:
        with amp_ctx:
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if rank == 0:
        print(f"world_size={world_size} batch/device={args.batch_size} ctx={model_cfg.context_length} "
              f"compile={not args.no_compile}", flush=True)

    for i in range(args.warmup_steps):
        t0 = time.perf_counter()
        one_step()
        if rank == 0:
            print(f"  warmup {i}: {time.perf_counter() - t0:.2f}s", flush=True)

    times = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        one_step()
        dt = time.perf_counter() - t0
        times.append(dt)
        if rank == 0 and (i < 3 or i % 10 == 0):
            print(f"  step {i}: {dt:.3f}s", flush=True)

    if rank == 0:
        tokens_per_step = args.batch_size * model_cfg.context_length * world_size
        median = statistics.median(times)
        mean = statistics.mean(times)
        print(f"\n=== 结果 (world_size={world_size}) ===")
        print(f"每步耗时: 中位数 {median:.3f}s / 均值 {mean:.3f}s (n={len(times)})")
        print(f"吞吐: {tokens_per_step / median:.0f} tokens/s ({tokens_per_step / median * 3600 / 1e6:.1f}M tokens/h)")
        print(f"官方 16384 步预计: {median * 16384 / 3600:.1f} 小时 (产出 {tokens_per_step * 16384 / 1e9:.2f}B tokens)")
        print(f"对齐官方 8.6B tokens 需 {8.59e9 / tokens_per_step:.0f} 步: {8.59e9 / tokens_per_step * median / 3600:.1f} 小时")
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
