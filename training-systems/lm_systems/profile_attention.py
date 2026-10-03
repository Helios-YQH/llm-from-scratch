"""Profile self-attention internals with NVTX (2.1.4e).

Splits causal scaled-dot-product attention into QK^T / mask / softmax / AV
so Nsight Systems can attribute GPU time to each stage. Run under `nsys profile`.

Config defaults match the small model's attention (d_model=768, heads=12,
d_head=64), which is what the 2.1.4 profiles use.
"""
import argparse
import math
import timeit

import torch
import torch.cuda.nvtx as nvtx

from lm_basics.nn_utils import softmax


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("qk_matmul"):
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
    if mask is not None:
        with nvtx.range("mask"):
            scores = torch.where(mask, scores, float("-inf"))
    with nvtx.range("softmax"):
        weights = softmax(scores, dim=-1)
    with nvtx.range("av_matmul"):
        out = torch.einsum("...qk,...kd->...qd", weights, V)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--d-head", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=10)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    B, H, S, D = args.batch, args.heads, args.seq, args.d_head
    Q = torch.randn(B, H, S, D, device=device)
    K = torch.randn(B, H, S, D, device=device)
    V = torch.randn(B, H, S, D, device=device)

    mask = None
    if args.causal:
        iota = torch.arange(S, device=device)
        mask = (iota[:, None] >= iota[None, :])[None, None, :, :]  # (1,1,S,S)

    for _ in range(args.warmup):
        annotated_scaled_dot_product_attention(Q, K, V, mask)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.measure):
        t0 = timeit.default_timer()
        annotated_scaled_dot_product_attention(Q, K, V, mask)
        torch.cuda.synchronize()
        times.append(timeit.default_timer() - t0)
    mean_ms = sum(times) / len(times) * 1e3
    print(f"attention forward mean: {mean_ms:.3f} ms")


if __name__ == "__main__":
    main()
