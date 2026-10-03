"""Repeated-sampling metrics, and the on-disk format the selection pass reads back.

One record per question:

    {"question", "ground_truth", "responses": [...], "rewards": [...], "n_tokens": [...]}

Keeping the raw responses is the point: selection with a different verifier is then
a pure re-read, so the same rollouts can be scored by an exact grader, a format-only
one and a noisy one without resampling anything.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from math import comb
from pathlib import Path
from typing import Callable, Iterable

from lm_alignment.drgrpo_grader import _normalize


def extract_final_answer(response: str) -> str | None:
    """The answer a response claims, normalized, or None if it doesn't state one.

    Handles both prompt conventions: r1_zero puts the answer inside `<answer>`, and
    the question_only prompt asks for `\\boxed{}`. Normalization goes through the
    grader's own `_normalize`, so "72", "72." and "$72$" do not count as three
    different votes in majority voting.
    """
    if "<answer>" in response and "</answer>" in response:
        inner = response.split("<answer>")[-1].split("</answer>")[0]
    elif "\\boxed" in response:
        from lm_alignment.drgrpo_grader import extract_boxed_answer

        inner = extract_boxed_answer(response)
        if inner is None:
            return None
    else:
        return None
    normalized = _normalize(inner)
    return normalized or None


def pass_at_k(records: Iterable[dict], k: int) -> float:
    """Fraction of questions solved by at least one of k samples.

    This is the oracle-selection ceiling: what you would get if the verifier were
    perfect. Also the upper bound for any best-of-n.

    Uses the unbiased estimator from Chen et al. 2021, which draws on all n samples
    per question. The naive `any(rewards[:k])` is unbiased too, but at k < n it
    ignores n-k samples: with n=32, the k=1 estimate would rest on a single draw
    (~5.7x the standard error). The cost of switching is that this no longer equals
    `best_of_n_accuracy` under a perfect verifier exactly -- that one still scores
    prefixes, so the two now differ by roughly one standard error.
    """
    records = list(records)
    total = 0.0
    for record in records:
        n = len(record["rewards"])
        correct = sum(1 for reward in record["rewards"] if reward > 0)
        total += 1.0 - comb(n - correct, k) / comb(n, k)
    return total / len(records)


def majority_at_k(records: Iterable[dict], k: int) -> float:
    """Fraction of questions where the most common answer among the first k is right.

    The verifier-free baseline: it needs no judgement at all, only agreement. It
    plateaus early because a rare correct answer cannot outvote a common wrong one.
    """
    records = list(records)
    hits = 0
    for record in records:
        votes = Counter(
            answer
            for answer in (
                extract_final_answer(response) for response in record["responses"][:k]
            )
            if answer is not None
        )
        if not votes:
            continue
        mode = votes.most_common(1)[0][0]
        if mode == _normalize(record["ground_truth"]):
            hits += 1
    return hits / len(records)


def answer_diversity(records: Iterable[dict], k: int) -> float:
    """Mean number of distinct answers among the first k samples.

    The quantity the sampling budget actually buys. If doubling the samples does
    not raise this, it does not raise coverage either.
    """
    records = list(records)
    counts = []
    for record in records:
        answers = {
            answer
            for answer in (extract_final_answer(r) for r in record["responses"][:k])
            if answer is not None
        }
        counts.append(len(answers))
    return sum(counts) / len(counts)


def best_of_n_accuracy(
    records: Iterable[dict],
    k: int,
    scorer: Callable[[str, str], float],
    tie_break_seed: int = 0,
) -> tuple[float, float, float]:
    """Best-of-n under a given verifier. Returns (accuracy, tie_fraction, identical_fraction).

    `scorer(response, ground_truth) -> float` stands in for a verifier; with a
    perfect one this equals `pass_at_k`, and the gap is what the verifier costs.

    Ties are broken uniformly at random rather than by taking the first maximum: a
    verifier that cannot discriminate would otherwise always win with candidate 0,
    which measures the tie-breaking rule instead of the verifier. The randomness is
    a hash of the question, not a global RNG, so re-runs agree.

    The two fractions describe the score distribution, and **neither reads out
    whether the verifier is any good** -- only the accuracy gap against the oracle
    does. They are easy to confuse, so:

    * `tie_fraction` -- questions where **more than one** candidate shares the top
      score. With 32 candidates this is ~1.0 even for a verifier that discriminates
      well (a noisy one leaves a top set of ~9); it is the share of questions whose
      winner had to be chosen arbitrarily.
    * `identical_fraction` -- questions where **every** candidate scored the same.
      A strictly blind verifier sits at 1.0, but a useless verifier need not: a
      format-only check returns 1 for formatted and 0 for unformatted, so it
      varies, and lands near 0.24 -- about where the *perfect* verifier does
      (0.23), on the questions the model gets entirely right or entirely wrong.
      Format-only's n=32 accuracy is nevertheless the base rate, because being
      formatted says nothing about being correct.

    Measured on the base model's 512x32 samples: `perfect` scores 0.7715 at n=32
    and `format-only` 0.2129, with near-identical tie/identical profiles. Read the
    accuracy columns; these two are diagnostics of *how* the choice was made.
    """
    records = list(records)
    hits = 0
    tied = 0
    identical = 0
    for record in records:
        candidates = record["responses"][:k]
        if not candidates:
            continue
        scores = [scorer(response, record["ground_truth"]) for response in candidates]
        top = [i for i, score in enumerate(scores) if score == max(scores)]
        if len(set(scores)) == 1:
            identical += 1
        if len(top) > 1:
            tied += 1
            digest = hashlib.sha256(f"{tie_break_seed}|{record['question']}".encode()).hexdigest()
            pick = top[int(digest[:16], 16) % len(top)]
        else:
            pick = top[0]
        hits += int(bool(record["rewards"][pick]))
    n = len(records)
    return hits / n, tied / n, identical / n


def write_samples(path: str | Path, records: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_samples(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
