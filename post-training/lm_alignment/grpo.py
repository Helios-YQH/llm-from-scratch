"""Core components of on-policy and off-policy GRPO."""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from lm_alignment.tokenization import tokenize_prompt_and_output


def _token_entropy(logits: Tensor, chunk_size: int = 4096) -> Tensor:
    """Per-token entropy of the next-token distribution, streamed over the vocab.

    `-(p * log p).sum()` needs a full (..., vocab) fp32 tensor for `p` and a second
    one for the product. At 256 sequences x 1041 tokens x 50304 vocab each of those
    is 26GB, which is most of why a GRPO step peaks around 45GB. Summing in vocab
    chunks keeps only (..., chunk_size) alive; the value is identical.

    Two passes, both streaming: one to get logsumexp (with a running max, so a
    chunk that is much hotter than the ones before it cannot overflow), one to
    accumulate `sum(z * exp(z - lse))`.
    """
    logits = logits.float()
    shape = logits.shape[:-1]
    vocab = logits.shape[-1]

    running_max = torch.full(shape, float("-inf"), device=logits.device)
    running_sum = torch.zeros(shape, device=logits.device)
    for start in range(0, vocab, chunk_size):
        chunk = logits[..., start : start + chunk_size]
        new_max = torch.maximum(running_max, chunk.amax(dim=-1))
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
            chunk - new_max.unsqueeze(-1)
        ).sum(dim=-1)
        running_max = new_max
    logsumexp = running_max + torch.log(running_sum)

    expected = torch.zeros(shape, device=logits.device)
    for start in range(0, vocab, chunk_size):
        chunk = logits[..., start : start + chunk_size]
        expected = expected + (chunk * torch.exp(chunk - logsumexp.unsqueeze(-1))).sum(dim=-1)
    return logsumexp - expected


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Per-token log p(x_t | x_<t) under a causal LM, plus entropy when asked."""
    logits = model(input_ids).logits
    log_probs_all = F.log_softmax(logits.float(), dim=-1)
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    output = {"log_probs": log_probs}
    if return_token_entropy:
        # Detached by every caller -- it is a diagnostic, not part of the objective
        # -- so it does not need to sit in the autograd graph, and dropping out of
        # it is what lets the sum be streamed.
        with torch.no_grad():
            output["token_entropy"] = _token_entropy(logits)
    return output


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    """Score rollouts; returns raw rewards plus per-component means for logging."""
    scores = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)
    n = len(scores)
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": sum(s["format_reward"] for s in scores) / n,
        "answer_reward_mean": sum(s["answer_reward"] for s in scores) / n,
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    """Subtract a per-group baseline and rescale (variants: writeup sections 4-5).

    Rewards are grouped as consecutive runs of `group_size`.
    """
    if advantage_normalizer not in ("std", "none", "mean"):
        raise NotImplementedError(f"advantage_normalizer={advantage_normalizer!r}")
    if baseline not in ("mean", "none"):
        raise NotImplementedError(f"baseline={baseline!r}")

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    if baseline == "mean":
        advantages = grouped_rewards - group_means
    else:
        advantages = grouped_rewards

    if advantage_normalizer == "std":
        # torch.std defaults to the bias-corrected (n - 1) estimate, which is
        # the convention used here.
        advantages = advantages / (grouped_rewards.std(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (group_means + advantage_eps)

    metadata = {
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }
    return advantages.reshape(-1), metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-token surrogate loss, negated because optimizers descend.

    The off-policy branches reweight by pi_theta/pi_old, which already contains
    the log-prob, so `policy_log_probs` never appears in them.
    """
    if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
        raise NotImplementedError(f"importance_reweighting_method={importance_reweighting_method!r}")

    # (batch_size,) or (batch_size, 1) -> (batch_size, 1), broadcast over tokens.
    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    if importance_reweighting_method == "none":
        return -advantages * policy_log_probs, {}

    if old_log_probs is None:
        raise ValueError("old_log_probs is required for off-policy importance reweighting.")

    log_ratio = policy_log_probs - old_log_probs

    if importance_reweighting_method == "noclip":
        ratio = log_ratio.exp()
        return -advantages * ratio, {"importance_ratio_mean": ratio.mean().detach()}

    if cliprange is None:
        raise ValueError("cliprange is required for grpo/gspo clipping.")

    if importance_reweighting_method == "grpo":
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
        objective = torch.minimum(advantages * ratio, advantages * clipped_ratio)
        clipped = (advantages >= 0) & (ratio > 1.0 + cliprange)
        clipped |= (advantages < 0) & (ratio < 1.0 - cliprange)
        metadata = {
            "clip_fraction": clipped.float().mean().detach(),
            "importance_ratio_mean": ratio.mean().detach(),
        }
        return -objective, metadata

    # gspo: one ratio per sequence, the geometric mean over response tokens.
    if response_mask is None:
        raise ValueError("response_mask is required for gspo.")
    mask = response_mask.to(log_ratio.dtype)
    n_response_tokens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    log_sequence_ratio = (log_ratio * mask).sum(dim=1, keepdim=True) / n_response_tokens
    sequence_ratio = log_sequence_ratio.exp()
    clipped_ratio = sequence_ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
    objective = torch.minimum(advantages * sequence_ratio, advantages * clipped_ratio)
    clipped = (advantages >= 0) & (sequence_ratio > 1.0 + cliprange)
    clipped |= (advantages < 0) & (sequence_ratio < 1.0 - cliprange)
    metadata = {
        "clip_fraction": clipped.float().mean().detach(),
        "sequence_ratio_mean": sequence_ratio.mean().detach(),
    }
    return -objective.expand_as(policy_log_probs), metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    """Reduce per-token losses to a scalar; "sequence" averages within then across."""
    mask = mask.to(per_token_policy_gradient_loss.dtype)
    masked_loss = per_token_policy_gradient_loss * mask

    if loss_normalization == "sequence":
        n_response_tokens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        per_sequence_loss = masked_loss.sum(dim=1, keepdim=True) / n_response_tokens
        return per_sequence_loss.mean()

    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for constant normalization.")
        return masked_loss.sum() / normalization_constant

    raise NotImplementedError(f"loss_normalization={loss_normalization!r}")


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """One optimizer step over a batch of rollouts.

    Zero-advantage sequences are dropped, but the accumulation still divides by
    the original batch size.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")

    device = next(model.parameters()).device

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    n_sequences = input_ids.shape[0]
    keep = (advantages != 0).nonzero(as_tuple=True)[0]
    microbatch_size = max(1, len(keep) // gradient_accumulation_steps)

    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    entropy_tokens = 0
    step_metadata: dict[str, Tensor] = {}

    for start in range(0, len(keep), microbatch_size):
        indices = keep[start : start + microbatch_size]
        micro_mask = response_mask[indices]

        scores = get_response_log_probs(
            model=model,
            input_ids=input_ids[indices],
            labels=labels[indices],
            return_token_entropy=True,
        )
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages[indices],
            policy_log_probs=scores["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=None if old_log_probs is None else old_log_probs[indices],
            cliprange=cliprange,
            response_mask=micro_mask,
        )
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=micro_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        # "constant" already divides by a batch-independent constant, so only
        # the per-sequence mean needs the microbatches re-weighted.
        if loss_normalization == "sequence":
            microbatch_loss = microbatch_loss * (len(indices) / n_sequences)

        microbatch_loss.backward()
        total_loss = total_loss + microbatch_loss.detach()
        step_metadata.update(loss_metadata)
        entropy_sum = entropy_sum + (scores["token_entropy"].detach() * micro_mask).sum()
        entropy_tokens += int(micro_mask.sum().item())

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    optimizer.step()
    optimizer.zero_grad()

    metadata: dict[str, Tensor | float] = {
        "loss": total_loss,
        "grad_norm": grad_norm,
        "token_entropy": entropy_sum / max(entropy_tokens, 1),
        "reward_mean": reward_metadata["reward_mean"],
        "format_reward_mean": reward_metadata["format_reward_mean"],
        "answer_reward_mean": reward_metadata["answer_reward_mean"],
        # Log all of it; n_sequences_kept is the one that matters most.
        **{k: float(v) for k, v in advantage_metadata.items()},
        "n_sequences_kept": float(len(keep)),
        **step_metadata,
    }
    return total_loss, metadata
