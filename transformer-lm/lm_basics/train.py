import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from .dataloader import run_get_batch
from .transformer import TransformerLM
from .loss import cross_entropy
from .optimizer import AdamW, clip_gradient_norm, get_lr_cosine_schedule
from .checkpoint import run_save_checkpoint, run_load_checkpoint


def evaluate(model, data, batch_size, context_length, device, num_batches=10):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = run_get_batch(data, batch_size, context_length, device)
            logits = model(x)
            total_loss += cross_entropy(logits, y).item()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--csv_log", type=str, default=None)
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--transformer_variant", type=str, default=None,
                        help="Ablation variant: 4a, 4b, 4c, or 4d (imports transformer_{variant}.py)")

    args = parser.parse_args()

    if args.log_interval is None:
        args.log_interval = max(1, args.total_iters // 50)

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    # Model with ablation variant support
    if args.transformer_variant:
        import importlib
        variant_mod = importlib.import_module(f"lm_basics.transformer_{args.transformer_variant}")
        TransformerLM = variant_mod.TransformerLM
    else:
        from .transformer import TransformerLM

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    schedule = lambda it: get_lr_cosine_schedule(
        it, args.lr, args.min_lr, args.warmup_iters, args.total_iters,
    )

    start_iter = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_iter = run_load_checkpoint(args.checkpoint_path, model, optimizer)

    if args.csv_log:
        csv_file = open(args.csv_log, "a")
        if start_iter == 0:
            csv_file.write("step,train_loss,val_loss,lr\n")
    else:
        csv_file = None

    iterator = range(start_iter, args.total_iters)
    if not args.no_progress:
        iterator = tqdm(iterator, total=args.total_iters, initial=start_iter,
                        file=sys.stderr, dynamic_ncols=True, unit="step")

    for it in iterator:
        model.train()
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        clip_gradient_norm(model.parameters(), args.grad_clip)
        lr = schedule(it)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()

        if it % args.log_interval == 0 or it == args.total_iters - 1:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device)
            msg = f"step {it:6d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  lr={lr:.6f}"
            if args.no_progress:
                print(msg)
            else:
                tqdm.write(msg)

            if csv_file:
                csv_file.write(f"{it},{loss.item():.4f},{val_loss:.4f},{lr:.6f}\n")
                csv_file.flush()

        if not args.no_progress:
            iterator.set_postfix(train=f"{loss.item():.3f}")

        if not args.no_checkpoint and args.checkpoint_path and (it % args.save_interval == 0 or it == args.total_iters - 1):
            ckpt_path = f"{args.checkpoint_path}_step{it}"
            run_save_checkpoint(model, optimizer, it, ckpt_path)
            if args.no_progress:
                print(f"  checkpoint saved to {ckpt_path}")
            else:
                tqdm.write(f"  checkpoint saved to {ckpt_path}")

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
