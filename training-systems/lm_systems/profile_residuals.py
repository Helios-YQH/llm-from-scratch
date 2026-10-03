"""Measure residuals saved by a single TransformerBlock (2.1.6f / 3.1).

Uses torch.autograd.graph.saved_tensors_hooks to count exactly how many bytes
a single TransformerBlock saves for backward, grouped by tensor shape/dtype,
then compares against the gradient tensors produced during backward.
"""
import argparse
from collections import defaultdict

import torch

from lm_basics.model import RotaryEmbedding, TransformerBlock

MODEL_CONFIGS = {
    "medium": dict(d_model=1024, d_ff=4096, num_heads=16),
    "xl": dict(d_model=2560, d_ff=10240, num_heads=32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="medium")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    d_model, d_ff, num_heads = cfg["d_model"], cfg["d_ff"], cfg["num_heads"]

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=args.context_length),
    ).cuda()

    x = torch.randn((args.batch_size, args.context_length, d_model), requires_grad=True, device="cuda")

    total_bytes = 0
    by_shape = defaultdict(int)  # (shape, dtype) -> total bytes

    def pack_hook(t):
        nonlocal total_bytes
        if isinstance(t, torch.nn.Parameter):
            return t  # skip parameters (counted separately as model weights)
        nbytes = t.numel() * t.element_size()
        total_bytes += nbytes
        by_shape[(tuple(t.shape), str(t.dtype))] += nbytes
        return t

    def unpack_hook(t):
        return t

    # Warm up allocator
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = block(x)
    torch.cuda.synchronize()

    # Real measurement
    total_bytes = 0
    by_shape = defaultdict(int)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = block(x)
    torch.cuda.synchronize()

    print(f"=== {args.model_size} TransformerBlock (ctx={args.context_length}, batch={args.batch_size}) ===")
    print(f"total residuals saved for backward: {total_bytes / 1024**2:.2f} MiB")

    print("\ntop 5 residual contributions (by shape):")
    ranked = sorted(by_shape.items(), key=lambda kv: -kv[1])
    for (shape, dtype), nb in ranked[:5]:
        print(f"  {nb/1024**2:8.2f} MiB  shape={shape}  dtype={dtype}")

    # Gradient tensors: every parameter with requires_grad gets a .grad of the
    # same size during backward.
    n_grad_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    grad_bytes = n_grad_params * 4  # FP32 gradients
    print(f"\ngradient tensors for this block: {grad_bytes / 1024**2:.2f} MiB")
    print(f"  (params: {n_grad_params} x 4 bytes FP32)")

    # Also count the input x gradient (dL/dx), same shape as x
    input_grad_bytes = x.numel() * 4
    print(f"  input activation grad (dL/dx, shape {tuple(x.shape)}): {input_grad_bytes / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
