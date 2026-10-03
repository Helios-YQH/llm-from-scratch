import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).

    Uses the same parameter-update formula as torch.optim.AdamW so that
    the test can match it at atol=1e-4.

    Constructor signature mirrors torch.optim.AdamW:
        AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 1.  Step counter (per-parameter, starts at 0, incremented before use)
                if "step" not in state:
                    state["step"] = torch.tensor(0, dtype=torch.float32, device=p.device)
                    # moment buffers, same shape/dtype/device as parameter
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 2.  Decoupled weight decay:  θ ← θ · (1 − lr·λ)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 3.  Moment estimates
                # m_t = β₁·m_{t−1} + (1−β₁)·g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = β₂·v_{t−1} + (1−β₂)·g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4.  Bias correction
                bias_correction1 = 1 - beta1 ** step.item()
                bias_correction2 = 1 - beta2 ** step.item()

                # 5.  θ ← θ − (lr / (1−β₁ᵗ)) · m / (√(v/(1−β₂ᵗ)) + ε)
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ──────────────────────────────────────────────────────────────────
#  Cosine learning-rate schedule with linear warmup
# ──────────────────────────────────────────────────────────────────

def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    """Return the learning rate at iteration `it` under cosine annealing with warmup.

    - Warmup (t < T_w):        α_t = (t / T_w) · α_max
    - Cosine (T_w ≤ t ≤ T_c):  α_t = α_min + ½·(1 + cos(π·(t−T_w)/(T_c−T_w)))·(α_max−α_min)
    - Post (t > T_c):          α_t = α_min
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


# ──────────────────────────────────────────────────────────────────
#  Gradient clipping
# ──────────────────────────────────────────────────────────────────

def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip the gradients of `parameters` so their combined ℓ₂-norm ≤ max_l2_norm.

    Gradients are modified in-place.  Parameters with `grad is None` (e.g.
    frozen parameters with `requires_grad=False`) are skipped.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    total_norm = torch.tensor(0.0, dtype=torch.float32, device=params_with_grad[0].device)
    for p in params_with_grad:
        grad = p.grad.data
        total_norm += grad.norm() ** 2
    total_norm = total_norm.sqrt()

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(scale)
