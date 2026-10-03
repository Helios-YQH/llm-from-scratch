import math

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the average cross-entropy loss over the batch.

    Numerically stable implementation: subtracts the maximum logit from each
    row before computing log-sum-exp (the softmax denominator), then gathers
    the logit of the correct class.  This corresponds to

        ℓ = −log softmax(logits)[target] = lse(logits) − logits[target]

    where lse(logits) = log(Σ exp(logits − max)).  The log and exp cancel
    exactly, avoiding the instability of computing softmax first.

    Args:
        logits:  (..., vocab_size)  unnormalised predictions
        targets: (...,)             ground-truth class indices (int)

    Returns:
        scalar — mean loss over all batch dimensions
    """
    x_max = logits.max(dim=-1, keepdim=True).values          # (..., 1)
    shifted = logits - x_max                                  # (..., V)
    lse = shifted.exp().sum(dim=-1).log()                     # (...,)
    target_logit = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (...,)
    return (lse - target_logit).mean()


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """exp of the average cross-entropy over the batch (per-token perplexity)."""
    return torch.exp(cross_entropy(logits, targets))
