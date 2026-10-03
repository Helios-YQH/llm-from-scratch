"""§7 Fully-Sharded Data Parallel (simplified, forward-patch + custom Function).

Shards Linear/Embedding weights across ranks IN PLACE (mod.weight becomes a
row-shard Parameter; module type stays Linear/Embedding so tests' isinstance
checks work). Each module's forward is patched to all-gather the full weight
and compute through a custom autograd.Function whose backward reduce-scatters
the gradient back to this rank's shard. Norm layers are not sharded; their
gradients are all-reduced.

Interface (matches tests/adapters.py + tests/test_fsdp.py):
    get_fsdp(module, compute_dtype=None) -> FSDP
    fsdp_gather_full_params(fsdp_model) -> dict[name, full tensor]
    fsdp_on_after_backward(fsdp_model, optimizer)
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from lm_basics.model import Embedding, Linear


def _shard_rows(tensor, rank, world_size):
    rows = tensor.shape[0]
    assert rows % world_size == 0, f"rows {rows} not divisible by {world_size}"
    s = rows // world_size
    # .clone() (not .contiguous()) — a row slice is already contiguous, so
    # .contiguous() would return a view sharing the full weight's storage and
    # the sharded parameter would keep the full weight alive in memory.
    return tensor[rank * s : (rank + 1) * s].clone()


def _all_gather_rows(shard, world_size):
    # Gather into a single preallocated contiguous tensor (avoids a separate
    # torch.cat, whose cross-stream read of NCCL output deadlocks on PCIe
    # without NVLink). The explicit synchronize further forces the default
    # stream to see NCCL's internal-stream results.
    out_shape = (world_size * shard.shape[0], *shard.shape[1:])
    full = torch.empty(out_shape, dtype=shard.dtype, device=shard.device)
    # no_grad: gloo's allgather splits the output into views and copies the
    # (requires-grad) shard into them in-place; with grad mode on this trips
    # autograd's "view created in no_grad mode" check. NCCL writes via CUDA
    # kernels so it is unaffected, but the tests run under gloo even on GPU.
    with torch.no_grad():
        dist.all_gather_into_tensor(full, shard)
    if shard.is_cuda:
        torch.cuda.synchronize()
    return full


def _reduce_scatter_rows(full_grad, rank, world_size):
    """Sum of row-shards of full_grad, return this rank's shard.

    Uses a true reduce-scatter (each rank's output is the sum over all ranks'
    corresponding shards) via all_reduce then slice — same result, simpler.
    Averages (div world_size) because each rank's loss is a local-mean over
    its data shard, so the global mean is the mean of the per-rank grads.
    """
    g = full_grad.clone()
    with torch.no_grad():
        dist.all_reduce(g)
    if g.is_cuda:
        torch.cuda.synchronize()
    g.div_(world_size)
    s = full_grad.shape[0] // world_size
    return g[rank * s : (rank + 1) * s].clone()


class _FSDPLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard_w, rank, world_size, dtype):
        full_w = _all_gather_rows(shard_w, world_size)
        if dtype is not None:
            full_w = full_w.to(dtype)
        ctx.save_for_backward(x, full_w)
        ctx.rank = rank
        ctx.world_size = world_size
        return F.linear(x, full_w)

    @staticmethod
    def backward(ctx, grad_out):
        x, full_w = ctx.saved_tensors
        x2 = x.reshape(-1, full_w.shape[1])
        go2 = grad_out.reshape(-1, full_w.shape[0])
        grad_full_w = go2.t() @ x2
        grad_shard = _reduce_scatter_rows(grad_full_w, ctx.rank, ctx.world_size)
        grad_x = grad_out @ full_w
        return grad_x, grad_shard, None, None, None


class _FSDPEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard_w, rank, world_size, dtype):
        full_w = _all_gather_rows(shard_w, world_size)
        if dtype is not None:
            full_w = full_w.to(dtype)
        ctx.save_for_backward(x, full_w)
        ctx.rank = rank
        ctx.world_size = world_size
        return F.embedding(x, full_w)

    @staticmethod
    def backward(ctx, grad_out):
        x, full_w = ctx.saved_tensors
        grad_full_w = torch.zeros_like(full_w)
        grad_full_w.index_add_(0, x.reshape(-1), grad_out.reshape(-1, full_w.shape[1]))
        grad_shard = _reduce_scatter_rows(grad_full_w, ctx.rank, ctx.world_size)
        return None, grad_shard, None, None, None


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        self._patch_linear_embeddings(module)

        # Non-sharded params (norms etc.): all-reduce grads so ranks stay synced.
        for name, p in module.named_parameters():
            if not getattr(p, "_fsdp_shard", False):
                p.register_post_accumulate_grad_hook(self._norm_grad_hook)

    def _norm_grad_hook(self, param):
        if param.grad is not None:
            with torch.no_grad():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            if param.grad.is_cuda:
                torch.cuda.synchronize()
            param.grad.div_(self._world_size)

    def _patch_linear_embeddings(self, root):
        rank, ws = self._rank, self._world_size
        dtype = self.compute_dtype
        for name, mod in root.named_modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue
            full_w = mod.weight.data.detach()
            mod.weight = nn.Parameter(_shard_rows(full_w, rank, ws))
            mod.weight._fsdp_shard = True

            if isinstance(mod, Linear):
                def fwd(x, m=mod):
                    return _FSDPLinearFn.apply(x, m.weight, rank, ws, dtype)
            else:
                def fwd(x, m=mod):
                    return _FSDPEmbeddingFn.apply(x, m.weight, rank, ws, dtype)
            mod.forward = fwd

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        pass  # all collectives synchronous within autograd


def get_fsdp(module: nn.Module, compute_dtype: torch.dtype | None = None) -> nn.Module:
    return FSDP(module, compute_dtype)


def fsdp_on_after_backward(fsdp_model: nn.Module, optimizer: torch.optim.Optimizer):
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, mod in fsdp_model.module.named_modules():
        if isinstance(mod, (Linear, Embedding)):
            out[name + ".weight"] = _all_gather_rows(mod.weight, fsdp_model._world_size)
    for name, p in fsdp_model.module.named_parameters():
        if name not in out:
            out[name] = p.data
    return out
