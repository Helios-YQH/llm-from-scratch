"""§4.1.1 + §4.2: benchmark naive PyTorch attention vs torch.compile.

Matches PDF: batch=8, no head dim (head_dim IS the embedding dim). Sweeps
head_dim ∈ {16,32,64,128} × seq ∈ {256,1024,4096,8192,16384}.

Timing: forward measured alone; full (forward+backward) measured alone;
backward = full - forward (same subtraction as §2.1.3). Memory: allocated
right before backward starts, and peak during full step. OOMs are reported.

Usage:
    uv run python -m lm_systems.attention_benchmark [--head-dims ...] [--seqs ...]
"""
import argparse
import itertools
import math

import torch
import torch.nn.functional as F
import triton.testing

from lm_basics.nn_utils import softmax


def naive_attention(Q, K, V):
    """Plain scaled-dot-product attention with causal mask (PDF Eq 1)."""
    d_k = K.shape[-1]
    scores = torch.einsum("bqd,bkd->bqk", Q, K) / math.sqrt(d_k)
    S = Q.shape[1]
    iota = torch.arange(S, device=Q.device)
    mask = iota[:, None] >= iota[None, :]
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return torch.einsum("bqk,bkd->bqd", weights, V)


def bench_forward(fn, Q, K, V):
    return triton.testing.do_bench(lambda: fn(Q, K, V))


def bench_full(fn, Q, K, V):
    def full():
        out = fn(Q, K, V)
        out.sum().backward()
    return triton.testing.do_bench(full)


def measure_memory(fn, Q, K, V):
    """Return (mem_before_backward_MiB, peak_during_full_MiB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn(Q, K, V)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    out.sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return mem_before, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dims", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--seqs", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    compiled_naive = torch.compile(naive_attention)
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)

    print(f"{'head':>4} {'seq':>6} | {'fwd':>7} {'bwd':>7} | {'mem_before':>10} {'peak':>9} | {'impl'}")
    print("-" * 60)

    for head_dim, seq in itertools.product(args.head_dims, args.seqs):
        Q = torch.randn(args.batch, seq, head_dim, device="cuda", requires_grad=True)
        K = torch.randn(args.batch, seq, head_dim, device="cuda", requires_grad=True)
        V = torch.randn(args.batch, seq, head_dim, device="cuda", requires_grad=True)

        for tag, fn in [("naive", naive_attention), ("compile", compiled_naive), ("sdpa", sdpa)]:
            try:
                fwd = bench_forward(fn, Q, K, V)
                full = bench_full(fn, Q, K, V)
                bwd = full - fwd
                mb, pk = measure_memory(fn, Q, K, V)
                print(f"{head_dim:>4} {seq:>6} | {fwd:6.2f} {bwd:6.2f} | {mb:9.0f} {pk:8.0f} | {tag}")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{head_dim:>4} {seq:>6} |   OOM  |   OOM  |    OOM     {tag}")
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"{head_dim:>4} {seq:>6} |   ERR  |   ERR  |    ERR     {tag} ({type(e).__name__})")


if __name__ == "__main__":
    main()
