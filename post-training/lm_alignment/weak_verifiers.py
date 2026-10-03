"""Weak reward functions: verifiers that are correlated with correctness but not equal to it.

These exist to ask how a policy degrades when the verifier is imperfect. They keep
the `(response, ground_truth) -> {"reward", "format_reward", "answer_reward"}`
signature of `drgrpo_grader`, so they are drop-in replacements for `reward_fn` in
the training loop and for the scorer in best-of-n selection.

`self-verify` is not here yet: it needs a call back into an inference engine, so it
gets its own hook rather than pretending to be a pure function.
"""

from __future__ import annotations

import hashlib
from typing import Callable

RewardFn = Callable[..., dict[str, float]]


def format_only_reward_fn(response: str, ground_truth: str, fast: bool = True) -> dict[str, float]:
    """Reward exactly what the r1_zero prompt asks for structurally, never the answer.

    This is the classic proxy: the model is paid for producing the tags, so the
    format rate goes to 1 and then the group's reward variance goes to 0 -- at
    which point no gradient reaches the model at all.
    """
    del ground_truth, fast
    formatted = "</think> <answer>" in response and "</answer>" in response
    return {
        "format_reward": float(formatted),
        # Always 0: this verifier never looks at the answer, and saying otherwise
        # would make the logged answer rate uninterpretable.
        "answer_reward": 0.0,
        "reward": float(formatted),
    }


def _uniform_from(*parts: object) -> float:
    """A deterministic [0, 1) draw from the arguments."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def make_noisy_reward_fn(exact_fn: RewardFn, epsilon: float, seed: int = 0) -> RewardFn:
    """Flip the exact verdict with probability epsilon.

    With probability `epsilon` a correct response is scored 0 and an incorrect one
    is scored 1. For a binary reward that makes the expected reward
    `(1 - 2*epsilon) * eta + epsilon`, so the gradient is the true gradient scaled
    by `(1 - 2*epsilon)`: RL still climbs the same objective, just slower, and the
    optimum does not move. At epsilon = 0.5 there is no signal; above it, the model
    is trained the wrong way.

    The noise comes from a hash of the inputs, NOT from a global RNG: drawing from
    `random` here would consume the generator that seeds prompt sampling and
    rollout decoding, so the noisy and clean arms would explore differently for a
    reason that has nothing to do with the verifier.
    """

    def noisy_reward_fn(response: str, ground_truth: str, fast: bool = True) -> dict[str, float]:
        scores = exact_fn(response, ground_truth, fast=fast)
        flip = _uniform_from(seed, epsilon, response, ground_truth) < epsilon
        correct = scores["reward"] == 1.0
        if flip:
            correct = not correct
        return {
            "format_reward": scores["format_reward"],
            "answer_reward": float(correct),
            "reward": float(correct),
        }

    return noisy_reward_fn
