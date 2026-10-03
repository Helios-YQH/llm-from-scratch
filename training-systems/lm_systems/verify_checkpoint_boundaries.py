"""Verify how many boundary activations PyTorch nested checkpoint actually keeps.

Key question: does recursive checkpointing (nested checkpoint calls) keep
O(log N) boundary activations (as optimal checkpoint scheduling would), or
O(N) (one per layer, same as block=1)?

Method: measure peak allocated memory for FORWARD ONLY (no backward). With
checkpointing, forward only allocates the saved boundary activations + params
(no recompute happens during forward). So comparing block=1 vs recursive
forward-only peak directly reveals how many boundaries each keeps alive.
"""
import torch
from torch.utils.checkpoint import checkpoint

from lm_basics.model import RotaryEmbedding, TransformerBlock


def build_blocks(d_model, d_ff, num_heads, ctx, n):
    rope = RotaryEmbedding(dim=d_model // num_heads, context_length=ctx)
    return torch.nn.ModuleList(
        [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, positional_encoder=rope) for _ in range(n)]
    ).cuda()


def fwd_peak(tag, fn, x):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  {tag:28s} forward-only peak = {peak:8.1f} MiB")
    return peak


def main():
    torch.manual_seed(0)
    d_model, d_ff, num_heads, ctx = 1024, 4096, 16, 512
    N = 16
    blocks = build_blocks(d_model, d_ff, num_heads, ctx, N)
    x = torch.randn(4, ctx, d_model, requires_grad=True, device="cuda")
    print(f"=== {N} blocks, medium config, FORWARD ONLY ===")

    # no checkpoint: forward saves ALL residuals -> highest
    def fwd_no(x):
        for b in blocks:
            x = b(x)
        return x
    fwd_peak("no checkpoint", fwd_no, x)

    # block=1: each layer checkpointed, keeps N inputs
    def fwd_b1(x):
        for b in blocks:
            x = checkpoint(lambda y, b=b: b(y), x, use_reentrant=False)
        return x
    fwd_peak("block=1", fwd_b1, x)

    # recursive: nested checkpoint, left then right
    def rec(sub, x):
        if len(sub) == 1:
            return sub[0](x)
        mid = len(sub) // 2
        x = checkpoint(lambda y, s=sub[:mid]: rec(s, y), x, use_reentrant=False)
        x = checkpoint(lambda y, s=sub[mid:]: rec(s, y), x, use_reentrant=False)
        return x
    fwd_peak("recursive (2 halves)", lambda x: rec(blocks, x), x)


if __name__ == "__main__":
    main()
