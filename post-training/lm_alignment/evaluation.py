"""Grading rollouts and summarizing them into metrics."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch


def grade_rollouts(
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    num_workers: int = 1,
) -> list[dict[str, float]]:
    """Score responses, optionally in parallel.

    Only the default `fast=True` path is thread-safe: `fast=False` calls
    signal.alarm, which needs the main thread.
    """
    if len(responses) != len(ground_truths):
        # zip() truncates silently, and the result of this call is the accuracy
        # the report is built on.
        raise ValueError(f"got {len(responses)} responses for {len(ground_truths)} ground truths")
    if num_workers <= 1:
        return [reward_fn(response, truth) for response, truth in zip(responses, ground_truths)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(reward_fn, responses, ground_truths))


def summarize_rollouts(scores: list[dict[str, float]]) -> dict[str, float]:
    """Count how rollouts split across the (format, answer) reward quadrants."""
    n = len(scores)
    if n == 0:
        return {}
    return {
        "n": n,
        "accuracy": sum(1 for s in scores if s["answer_reward"] == 1) / n,
        "format_rate": sum(1 for s in scores if s["format_reward"] == 1) / n,
        "correct_and_formatted": sum(
            1 for s in scores if s["format_reward"] == 1 and s["answer_reward"] == 1
        )
        / n,
        "formatted_only": sum(
            1 for s in scores if s["format_reward"] == 1 and s["answer_reward"] == 0
        )
        / n,
        "unformatted": sum(
            1 for s in scores if s["format_reward"] == 0 and s["answer_reward"] == 0
        )
        / n,
        "reward_mean": sum(s["reward"] for s in scores) / n,
    }


def compute_group_pass_at_k(raw_rewards, group_size: int) -> dict[str, float]:
    """Fraction of prompts solved by at least one of the first k rollouts."""
    grouped = torch.as_tensor(raw_rewards, dtype=torch.float32).reshape(-1, group_size)
    return {
        f"pass@{k}": (grouped[:, :k].max(dim=1).values > 0).float().mean().item()
        for k in (1, 2, 4, 8, 16, group_size)
        if k <= group_size
    }


# Ordered by how much the prompt asks for it; the bare-letter fallback trades
# precision for recall.
_MMLU_PATTERNS = (
    r"\\boxed\{\s*\(?([ABCD])\)?\s*\}",
    r"answer\s*(?:is|:)\s*\(?([ABCD])\b",
    r"\b([ABCD])\b",
)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """Extract the predicted option letter, or None if there isn't one."""
    del mmlu_example
    for pattern in _MMLU_PATTERNS:
        match = re.search(pattern, model_output)
        if match:
            return match.group(1)
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """Extract the last number in the output, or None if there is none."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    return numbers[-1] if numbers else None
