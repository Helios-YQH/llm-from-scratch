"""§6 Optimizer State Sharding (simplified ZeRO stage 1).

Wraps a torch.optim.Optimizer so that each rank maintains optimizer state
(m/v for Adam) for only ~1/world_size of the parameters. After each step,
the rank that owns a parameter shard broadcasts its updated value so all
ranks stay in sync.

Design (matches test_sharded_optimizer.py):
  - all ranks run the same forward/backward (full gradients everywhere)
  - params are sharded by index: rank r owns params[r], params[r+ws], ...
  - each rank builds a real optimizer (optimizer_cls) over ONLY its shard
    -> optimizer state (m/v) lives on the owning rank only
  - after step(), the owner broadcasts the updated parameter to all ranks
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        params = list(params)
        self._full_params = params
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        # Shard by index (deterministic across ranks): rank r owns params[r],
        # params[r+ws], ... Owner is the INDEX in the param list, which is the
        # same on every rank (they build the model identically). Do NOT use
        # id(p) — ids are process-local and differ across ranks.
        self._local_params = [p for i, p in enumerate(params) if i % self._world_size == self._rank]
        self._owner_of = {i: i % self._world_size for i in range(len(params))}

        # Real optimizer over just the local shard. Its state (m/v) exists
        # only on this rank -> per-rank memory is ~1/world_size.
        self._local_optimizer = optimizer_cls(self._local_params, **kwargs)

        # Call the Optimizer superclass constructor with the full params so
        # that param_groups / zero_grad behave like a normal optimizer.
        super().__init__(params, self._local_optimizer.defaults)

    def add_param_group(self, param_group: dict):
        """Append a param group (called by super().__init__ / during training)."""
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        param_group = {**param_group, "params": params}
        self.param_groups.append(param_group)

    def step(self, closure=None):
        # 1. Each rank updates only its shard's parameters.
        self._local_optimizer.step(closure)
        # 2. Owner of each parameter broadcasts its updated value to all ranks.
        #    The broadcast is a value sync, not a differentiable op -> no_grad.
        with torch.no_grad():
            for i, p in enumerate(self._full_params):
                owner = self._owner_of[i]
                dist.broadcast(p, src=owner)

    def zero_grad(self, set_to_none=True):
        self._local_optimizer.zero_grad(set_to_none)
        # ensure the full model grads are zeroed too (they're on all ranks)
        for p in self._full_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


def get_sharded_optimizer(params, optimizer_cls, **kwargs):
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
