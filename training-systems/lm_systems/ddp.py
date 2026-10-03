"""§5.2 Distributed Data Parallel training.

Implements a minimal DDP wrapper:
- broadcasts rank-0 parameters to all ranks on construction
- synchronizes gradients after backward by all-reducing each parameter's
  gradient individually (naive) or as a single flat buffer (flat).

The `DDP` class here is the "naive" variant used by test_ddp.py via
adapters.get_ddp / ddp_on_after_backward.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Naive DDP: broadcast params, all-reduce per-parameter gradients.

    After each backward pass, call `finish_gradient_synchronization()` (or
    the adapter `ddp_on_after_backward`) before optimizer.step() so every
    rank holds the averaged gradients.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # Broadcast parameters from rank 0 so all ranks start identical.
        for p in self.module.parameters():
            dist.broadcast(p, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Average gradients across ranks (all-reduce each param, then divide)."""
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)


def get_ddp(module: nn.Module) -> nn.Module:
    return DDP(module)


def ddp_on_after_backward(ddp_model: nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()


# ---------------------------------------------------------------------------
# §5.3.1 FlatDDP: single all-reduce over a flat buffer of all gradients.
# ---------------------------------------------------------------------------

class FlatDDP(nn.Module):
    """DDP that all-reduces all gradients in one flat concatenated buffer.

    Reduces the number of communication calls from N parameters to 1, but
    communication is still synchronous (waits for the full backward).
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list = []
        for p in self.module.parameters():
            dist.broadcast(p, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _flatten_grads(self):
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads])
        return flat, grads

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        flat, grads = self._flatten_grads()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        # unflatten back into each param's grad
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(flat[offset : offset + numel].reshape(g.shape))
            offset += numel


# ---------------------------------------------------------------------------
# §5.3.2 OverlapDDP: async per-parameter all-reduce overlapping with backward.
# ---------------------------------------------------------------------------

class OverlapDDP(nn.Module):
    """DDP that overlaps gradient communication with backward computation.

    Uses register_post_accumulate_grad_hook to all-reduce each gradient as
    soon as it becomes ready during backward (async_op=True), then waits for
    all handles in finish_gradient_synchronization().
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list = []
        for p in self.module.parameters():
            dist.broadcast(p, src=0)
            p.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param):
        """Called when param.grad is ready during backward. Async all-reduce."""
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)
        return None

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for h in self._handles:
            h.wait()
        self._handles.clear()
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(world_size)

