"""§3.1 Autograd Residuals experiment.

Shows how many tensors autograd saves for the backward pass through a plain
RMSNorm (PDF §3.1), and how torch.compile fuses the op to save far fewer.
Runs on CPU — no GPU needed.
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


def make_hooks():
    saved = []

    def pack_hook(t):
        if isinstance(t, torch.nn.Parameter):
            return t  # skip params
        saved.append((tuple(t.shape), str(t.dtype)))
        return t

    def unpack_hook(t):
        return t

    return saved, pack_hook, unpack_hook


def main():
    print("========== plain RMSNorm (no compile) ==========")
    x = torch.randn((4, 512, 2560), requires_grad=True)
    ln = RMSNorm(x.shape[-1])

    saved, pack, unpack = make_hooks()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = ln(x)
        y.sum().backward()
    print(f"tensors saved for backward: {len(saved)}")
    for shape, dtype in saved:
        nbytes = torch.tensor([], dtype=torch.float32).numel()  # noop
        print(f"  shape={str(shape):24s} dtype={dtype}")

    print("\n========== RMSNorm compiled with torch.compile ==========")
    try:
        ln_c = torch.compile(RMSNorm(x.shape[-1]))
        saved2, pack2, unpack2 = make_hooks()
        with torch.autograd.graph.saved_tensors_hooks(pack2, unpack2):
            y2 = ln_c(x)
            y2.sum().backward()
        print(f"tensors saved for backward: {len(saved2)}")
        for shape, dtype in saved2:
            print(f"  shape={str(shape):24s} dtype={dtype}")
    except Exception as e:
        print(f"torch.compile not available here ({type(e).__name__}: {str(e)[:80]})")
        print("-> run on a GPU server where torch.compile works")


if __name__ == "__main__":
    main()
