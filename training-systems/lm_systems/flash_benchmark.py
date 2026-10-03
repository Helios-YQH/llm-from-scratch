"""§4.2 flash_benchmarking: FlashAttention-2 vs naive PyTorch attention.

PDF spec: batch=1, causal=True, sweep seq ∈ powers of 2 from 128 to 65536,
embedding dim ∈ powers of 2 from 16 to 128, precisions bf16 and fp32.
Reports forward, backward, and end-to-end latencies for both implementations
using triton.testing.do_bench. OOMs are reported.

Usage:
    uv run python -m lm_systems.flash_benchmark [--seqs ...] [--dims ...] [--dtypes ...]
"""
import argparse
import itertools
import math

import torch
import triton.testing

from lm_systems.flash_attention import FlashAttentionTritonBwd


def naive_attention(Q, K, V):
    """Plain scaled-dot-product attention with causal mask (matches lm_basics)."""
    d_k = K.shape[-1]
    S = torch.einsum("bqd,bkd->bqk", Q, K) / math.sqrt(d_k)
    nq = S.shape[-2]
    iota = torch.arange(nq, device=S.device)
    mask = iota[:, None] >= iota[None, :]
    S = torch.where(mask, S, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bqk,bkd->bqd", P, V)


def bench(fn, setup=None):
    """do_bench on fn; setup is called once before the benchmark (e.g. to make inputs)."""
    if setup is not None:
        setup()
    return triton.testing.do_bench(fn)


def bench_one_impl(tag, impl_fwd, Q, K, V, do):
    """Benchmark forward, full (fwd+bwd), report (fwd_ms, bwd_ms, full_ms) or OOM."""
    try:
        fwd_ms = triton.testing.do_bench(lambda: impl_fwd(Q, K, V))

        def full():
            out = impl_fwd(Q, K, V)
            out.backward(do)
        full_ms = triton.testing.do_bench(full)
        return fwd_ms, full_ms - fwd_ms, full_ms
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--dtypes", nargs="+", type=str, default=["bf16", "fp32"])
    args = ap.parse_args()

    flash = FlashAttentionTritonBwd.apply
    print(f"{'dtype':>5} {'d':>4} {'seq':>6} | {'flash f':>9} {'flash b':>9} {'flash full':>10} | {'naive f':>9} {'naive b':>9} {'naive full':>10}")
    print("-" * 84)

    for dtype_name, d, seq in itertools.product(args.dtypes, args.dims, args.seqs):
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        Q = torch.randn(1, seq, d, device="cuda", dtype=dtype, requires_grad=True)
        K = torch.randn(1, seq, d, device="cuda", dtype=dtype, requires_grad=True)
        V = torch.randn(1, seq, d, device="cuda", dtype=dtype, requires_grad=True)
        do = torch.randn(1, seq, d, device="cuda", dtype=dtype)

        # Flash impl: forward uses custom Function; do is the upstream grad.
        ff, fb, ff_full = bench_one_impl("flash", lambda q,k,v: flash(q, k, v, True), Q, K, V, do)
        nf, nb, nf_full = bench_one_impl("naive", naive_attention, Q, K, V, do)

        fmt = lambda x: "     OOM" if x is None else f"{x:9.2f}"
        print(f"{dtype_name:>5} {d:>4} {seq:>6} | {fmt(ff)} {fmt(fb)} {fmt(ff_full)} | {fmt(nf)} {fmt(nb)} {fmt(nf_full)}")


if __name__ == "__main__":
    main()
