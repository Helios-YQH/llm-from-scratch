"""§3.2 Activation checkpointing experiment (gradient_checkpointing b).

Measures peak CUDA memory for a stack of TransformerBlocks under different
checkpointing strategies: no checkpoint, coarse blocks, fine blocks, and
recursive checkpointing. Uses torch.cuda.max_memory_allocated() to capture
the peak.

Strategy: use a small-but-representative config (medium block, ctx512) that
runs comfortably, and measure how peak memory scales with checkpoint block
size, plus the recursive strategy.
"""
import argparse
import torch
from torch.utils.checkpoint import checkpoint

from lm_basics.model import RotaryEmbedding, TransformerBlock

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_heads=16),
}


def build_blocks(d_model, d_ff, num_heads, ctx, n):
    rope = RotaryEmbedding(dim=d_model // num_heads, context_length=ctx)
    return torch.nn.ModuleList(
        [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, positional_encoder=rope) for _ in range(n)]
    ).cuda()


def run_forward_backward(fn, x):
    y = fn(x)
    loss = y.sum()
    loss.backward()


def measure(tag, fn, x):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    run_forward_backward(fn, x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  {tag:28s} peak = {peak:9.1f} MiB")
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="medium")
    ap.add_argument("--num-blocks", type=int, default=8)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    d_model, d_ff, num_heads = cfg["d_model"], cfg["d_ff"], cfg["num_heads"]
    n = args.num_blocks

    blocks = build_blocks(d_model, d_ff, num_heads, args.context_length, n)
    x = torch.randn(args.batch_size, args.context_length, d_model, requires_grad=True, device="cuda")

    print(f"=== {args.model_size}: {n} TransformerBlocks, ctx={args.context_length}, batch={args.batch_size} ===")

    # 1. No checkpointing
    def fwd_no_ckpt(x):
        for blk in blocks:
            x = blk(x)
        return x

    measure("no checkpoint", fwd_no_ckpt, x)

    # 2. Coarse: one checkpoint over all N blocks (input-only saved)
    def fwd_all_ckpt(x):
        return checkpoint(lambda y: fwd_no_ckpt(y), x, use_reentrant=False)

    measure("checkpoint(all N)", fwd_all_ckpt, x)

    # 3. Blocks of size 2
    def make_block_size(k):
        def fwd(x):
            for i in range(0, n, k):
                sub = blocks[i : i + k]
                x = checkpoint(lambda y, sub=sub: _chain(y, sub), x, use_reentrant=False)
            return x
        return fwd

    def _chain(x, sub):
        for blk in sub:
            x = blk(x)
        return x

    measure("checkpoint(block=2)", make_block_size(2), x)
    measure("checkpoint(block=4)", make_block_size(4), x)
    measure("checkpoint(block=1)", make_block_size(1), x)

    # 4. Recursive: split in half, checkpoint each half (nested within halves)
    def rec_fwd(x):
        return _rec(blocks, x)

    def _rec(sub_blocks, x):
        if len(sub_blocks) == 1:
            return sub_blocks[0](x)
        mid = len(sub_blocks) // 2
        # checkpoint the two halves; each half recursively recomputes internally
        return checkpoint(lambda y, s=sub_blocks[:mid]: _rec(s, y), x, use_reentrant=False)
        # note: single-level for now; true recursion would nest checkpoint calls

    # True recursive: two halves each checkpointed, and within each half recursion
    def _true_rec(sub_blocks, x):
        if len(sub_blocks) == 1:
            return sub_blocks[0](x)
        mid = len(sub_blocks) // 2
        # recurse into left half, then right half (each under checkpoint)
        x = checkpoint(lambda y, s=sub_blocks[:mid]: _true_rec(s, y), x, use_reentrant=False)
        x = checkpoint(lambda y, s=sub_blocks[mid:]: _true_rec(s, y), x, use_reentrant=False)
        return x

    def true_rec_fwd(x):
        return _true_rec(blocks, x)

    measure("recursive (half)", rec_fwd, x)
    measure("true recursive", true_rec_fwd, x)


if __name__ == "__main__":
    main()
