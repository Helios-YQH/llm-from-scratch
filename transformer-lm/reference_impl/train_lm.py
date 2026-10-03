"""
使用现成库 (transformers + tokenizers + torch) 的完整训练脚本 ——
对照我们 from-scratch 实现的每个组件。

架构: GPT2Config → GPT2LMHeadModel
  GPT-2 是 post-norm + GELU + learnable absolute position embeddings,
  而我们的 Transformer 是 pre-norm + SwiGLU + RoPE。
  虽然架构不同,但训练流程 (分词 → 编码 → 批数据 → 前向 → loss → backward → clip → 优化)
  完全相同,且所有组件都是 from-scratch 版本的"现成库替代"。

分词器: HuggingFace tokenizers 训练 byte-level BPE (Rust 后端,极快)
优化器: torch.optim.AdamW (内置)
调度器: torch.optim.lr_scheduler.LinearLR + CosineAnnealingLR → SequentialLR

用法:
  $env:PYTHONUTF8 = "1"
  uv run python reference_impl/train_lm.py --help
"""

from __future__ import annotations

import argparse, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════════
#  1.  用 HuggingFace tokenizers 训练 byte-level BPE
# ══════════════════════════════════════════════════════════════════════════════
def build_tokenizer(data_path: str, vocab_size: int, special_tokens: list[str]) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tok.train([data_path], trainer)
    return tok


# ══════════════════════════════════════════════════════════════════════════════
#  2.  流式数据集
# ══════════════════════════════════════════════════════════════════════════════
class TokenDataset(IterableDataset):
    """Iterable dataset: yields (input, target) windows from a 1-D token array."""

    def __init__(self, token_ids: np.ndarray, context_length: int, shuffle: bool = True):
        self.token_ids = token_ids
        self.context_length = context_length
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.token_ids) - self.context_length
        indices = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in indices:
            i = int(i)
            x = torch.from_numpy(self.token_ids[i:i + self.context_length].astype(np.int64))
            y = torch.from_numpy(self.token_ids[i + 1:i + 1 + self.context_length].astype(np.int64))
            yield x, y


def collate_batch(batch):
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  构建模型 (GPT-2,对应 from-scratch 的 TransformerLM)
# ══════════════════════════════════════════════════════════════════════════════
def build_model(vocab_size: int, context_length: int, d_model: int,
                num_layers: int, num_heads: int) -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_embd=d_model,
        n_layer=num_layers,
        n_head=num_heads,
        bos_token_id=None,
        eos_token_id=None,
    )
    return GPT2LMHeadModel(config)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  评估
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, max_batches=10):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).logits
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item() * x.size(0)
        count += x.size(0)
        if count >= max_batches * x.size(0):
            break
    return total / count


# ══════════════════════════════════════════════════════════════════════════════
#  5.  主训练函数
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", type=str, required=True)
    parser.add_argument("--val_txt", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--total_iters", type=int, default=1250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # --- 1. 训练 BPE 分词器 ---
    print("Training BPE tokenizer ...")
    tok = build_tokenizer(args.train_txt, args.vocab_size, ["<|endoftext|>"])
    print(f"  vocab size: {tok.get_vocab_size()}")

    # --- 2. 编码文本 → uint16 .npy ---
    print("Encoding datasets ...")
    for split, path in [("train", args.train_txt), ("val", args.val_txt)]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        ids = tok.encode(text).ids
        out_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"{split}_{args.vocab_size}.npy"
        np.save(str(out_path), np.array(ids, dtype=np.uint16))
        print(f"  {split}: {len(ids)} tokens → {out_path}")

    # --- 3. 加载数据 (内存映射) ---
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    train_np = np.load(str(tmp / f"train_{args.vocab_size}.npy"), mmap_mode="r")
    val_np   = np.load(str(tmp / f"val_{args.vocab_size}.npy"), mmap_mode="r")
    train_ds = TokenDataset(train_np, args.context_length, shuffle=True)
    val_ds   = TokenDataset(val_np, args.context_length, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_batch)

    # --- 4. 模型 + 优化器 + 调度器 ---
    model = build_model(args.vocab_size, args.context_length, args.d_model,
                        args.num_layers, args.num_heads).to(args.device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,.0f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.0, total_iters=args.warmup_iters)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_iters - args.warmup_iters,
                                                         eta_min=args.min_lr)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                   milestones=[args.warmup_iters])

    # --- 5. 训练循环 ---
    train_iter = iter(train_loader)
    t0 = time.time()
    for it in range(args.total_iters):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        logits = model(x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if it % args.log_interval == 0 or it == args.total_iters - 1:
            val_loss = evaluate(model, val_loader, args.device)
            elapsed = time.time() - t0
            print(f"step {it:6d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.6f}  time={elapsed:.0f}s")

    # --- 6. 保存 ---
    out = tmp / "ots_lm"
    out.mkdir(exist_ok=True)
    model.save_pretrained(str(out))
    tok.save(str(out / "tokenizer.json"))
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()
