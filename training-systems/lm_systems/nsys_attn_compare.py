"""nsys profile target: one forward+backward of flash vs naive attention.

Run under nsys: `nsys profile -o out --trace=cuda,nvtx python -m ... naive|flash`
"""
import sys
import torch

from lm_systems.flash_attention import FlashAttentionTritonBwd


def naive_attention(Q, K, V):
    import math
    d_k = K.shape[-1]
    S = torch.einsum("bqd,bkd->bqk", Q, K) / math.sqrt(d_k)
    nq = S.shape[-2]
    iota = torch.arange(nq, device=S.device)
    mask = iota[:, None] >= iota[None, :]
    S = torch.where(mask, S, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bqk,bkd->bqd", P, V)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "flash"
    B, S, D = 1, 2048, 64
    dtype = torch.bfloat16
    q = torch.randn(B, S, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, D, device="cuda", dtype=dtype, requires_grad=True)
    do = torch.randn(B, S, D, device="cuda", dtype=dtype)

    fwd = FlashAttentionTritonBwd.apply if which == "flash" else naive_attention

    # warmup
    for _ in range(3):
        o = fwd(q, k, v)
        o.backward(do)
    torch.cuda.synchronize()

    # profiled runs
    for _ in range(10):
        o = fwd(q, k, v)
        o.backward(do)
    torch.cuda.synchronize()
    print(f"done {which}")


if __name__ == "__main__":
    main()
