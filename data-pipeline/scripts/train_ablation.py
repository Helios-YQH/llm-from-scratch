"""数据消融用的训练脚本: 同模型/优化器/超参, 比较不同数据版本。

与 `scripts/train.py` 的差异(都是为在 48GB 卡上跑 + 便于对照实验):
  - 支持梯度累积(`--batch-size 32 --accum 4` 等效官方 batch 128/卡)
  - 验证用**固定种子的批次**, 两条对照曲线看到的是同一批验证数据, 可比
  - 定期把 train/val loss 记到 CSV; `--max-minutes` 控制时间成本(到点就停并保存)
模型结构、lr、cosine/warmup、AdamW 参数、梯度裁剪与 train.py 完全一致。

单卡:

    .venv-a4/bin/python scripts/train_ablation.py \\
        --train-bin shared-data/train.bin --valid-bin shared-data/paloma.bin \\
        --tag filtered --steps 400 --out-csv shared-data/ablation_filtered.csv

多卡(相同命令前加 torchrun):

    torchrun --standalone --nproc_per_node=2 scripts/train_ablation.py ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_basics.data import get_batch  # noqa: E402
from lm_basics.model import BasicsTransformerLM  # noqa: E402
from lm_basics.optimizer import get_cosine_lr  # noqa: E402
from lm_basics.train_config import ModelConfig, TrainingConfig  # noqa: E402

VAL_SEED = 1234


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--tag", required=True, help="实验名, 写进 CSV")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--steps", type=int, default=400, help="优化器步数")
    parser.add_argument("--batch-size", type=int, default=32, help="每卡 micro-batch")
    parser.add_argument("--accum", type=int, default=4, help="梯度累积步数(32x4 等效官方 128)")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=100, help="每卡验证批次数")
    parser.add_argument("--max-minutes", type=float, default=None, help="到点停止(控制时间成本)")
    parser.add_argument(
        "--preset",
        choices=["official", "small"],
        default="small",
        help="official=作业的 430M 配置; small≈30M(d_model=256/4 层), 给数据规模只有几亿 token 的消融用",
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    if args.preset == "small":
        # 用作业同款架构但缩小规模: 数据只有 ~3 亿 token 时, 430M 模型会直接背下来
        # (train loss -> 0 而 val loss 升到比均匀分布还差), 小模型才处在"学泛化特征"的区间。
        model_cfg.d_model = 256
        model_cfg.num_layers = 4
        model_cfg.num_heads = 4
        model_cfg.d_ff = 1024
    train_cfg = TrainingConfig()

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        from torch.distributed import init_process_group

        init_process_group(backend="nccl")
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank, world_size, device = 0, 1, "cuda"
        is_master = True

    torch.manual_seed(train_cfg.seed + rank)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    valid_data = np.memmap(args.valid_bin, dtype=np.uint16, mode="r")
    model = BasicsTransformerLM(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        rope_theta=model_cfg.rope_theta,
    ).to(device)
    model = torch.compile(model)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank])

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params.values() if p.dim() >= 2], "weight_decay": train_cfg.weight_decay},
            {"params": [p for p in params.values() if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        eps=train_cfg.adam_eps,
        fused=True,
    )
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def estimate_val_loss() -> float:
        model.eval()
        torch.manual_seed(VAL_SEED)  # 固定验证批次, 保证各条曲线可比
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(valid_data, batch_size=args.batch_size,
                             context_length=model_cfg.context_length, device=device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
        model.train()
        return sum(losses) / len(losses)

    batch_x, batch_y = get_batch(train_data, batch_size=args.batch_size,
                                 context_length=model_cfg.context_length, device=device)
    tokens_per_step = args.batch_size * model_cfg.context_length * args.accum * world_size

    writer = None
    csv_file = None
    if is_master:
        csv_file = open(args.out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["tag", "step", "tokens_seen", "train_loss", "val_loss", "lr", "elapsed_s"])
        print(f"[{args.tag}] steps={args.steps} batch={args.batch_size}x{args.accum} "
              f"tokens/step={tokens_per_step} val_iters={args.eval_iters}", flush=True)

    started = time.perf_counter()
    tokens_seen = 0
    for step in range(args.steps):
        lr = get_cosine_lr(step, max_learning_rate=train_cfg.lr, min_learning_rate=train_cfg.lr * 0.1,
                           warmup_iters=int(args.steps * train_cfg.warmup_ratio), cosine_cycle_iters=args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for micro in range(args.accum):
            if is_ddp:
                model.require_backward_grad_sync = micro == args.accum - 1
            with amp_ctx:
                logits = model(batch_x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1)) / args.accum
            loss.backward()
            batch_x, batch_y = get_batch(train_data, batch_size=args.batch_size,
                                         context_length=model_cfg.context_length, device=device)

        if train_cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if is_master and (step % args.eval_every == 0 or step == args.steps - 1):
            val_loss = estimate_val_loss()
            elapsed = time.perf_counter() - started
            writer.writerow([args.tag, step, tokens_seen, loss.item() * args.accum, val_loss, lr, round(elapsed, 1)])
            csv_file.flush()
            print(f"  step {step:5d} | train {loss.item() * args.accum:.4f} | val {val_loss:.4f} "
                  f"| {elapsed:.0f}s | {tokens_seen / max(elapsed, 1e-9):,.0f} tok/s", flush=True)

        if args.max_minutes and time.perf_counter() - started > args.max_minutes * 60:
            if is_master:
                print(f"[{args.tag}] 到达 --max-minutes={args.max_minutes}, 在 step {step} 停止", flush=True)
            break

    if is_master:
        val_loss = estimate_val_loss()
        writer.writerow([args.tag, "final", tokens_seen, "", val_loss, "", round(time.perf_counter() - started, 1)])
        csv_file.close()
        print(f"[{args.tag}] 最终 val loss {val_loss:.4f}, 用时 {(time.perf_counter() - started) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
