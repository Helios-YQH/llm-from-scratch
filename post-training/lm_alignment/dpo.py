"""Direct Preference Optimization loss."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from lm_alignment.grpo import get_response_log_probs
from lm_alignment.tokenization import tokenize_prompt_and_output


def response_log_prob(model: torch.nn.Module, tokenizer, prompt: str, response: str) -> torch.Tensor:
    """Sum of the response tokens' conditional log-probabilities under `model`."""
    tokenized = tokenize_prompt_and_output([prompt], [response], tokenizer)
    scores = get_response_log_probs(
        model=model,
        input_ids=tokenized["input_ids"],
        labels=tokenized["labels"],
        return_token_entropy=False,
    )
    return scores["log_probs"][tokenized["response_mask"]].sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """DPO loss for one pair:

        -log sigmoid( beta * [ (log pi_c - log pi_ref_c) - (log pi_r - log pi_ref_r) ] )
    """
    chosen_logratio = response_log_prob(lm, tokenizer, prompt, response_chosen) - response_log_prob(
        lm_ref, tokenizer, prompt, response_chosen
    )
    rejected_logratio = response_log_prob(lm, tokenizer, prompt, response_rejected) - response_log_prob(
        lm_ref, tokenizer, prompt, response_rejected
    )
    return -F.logsigmoid(beta * (chosen_logratio - rejected_logratio))


def batch_response_log_probs(
    model: torch.nn.Module, tokenizer, prompts: list[str], responses: list[str]
) -> torch.Tensor:
    """Per-example response log-probabilities in one forward over the batch.

    Computes `log p(label)` as `logit[label] - logsumexp(logits)` rather than via an
    explicit `log_softmax`, which materialises a (batch, tokens, 128256) fp32
    tensor -- 4.1GB at 16 pairs and ~500 tokens, and precisely the allocation that
    OOMed a DPO launch on a shared card (wanting 4.17GB with 1.66GB free). The
    reduction in `logsumexp` runs over the vocabulary as it goes, so only
    (batch, tokens) is ever held. `logsumexp` accumulates in fp32 internally, so
    the value matches `log_softmax` to float precision -- `test_dpo_batch.py`
    checks this against the per-instance loss, which still uses `log_softmax`.
    """
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    # Tokenization always produces CPU tensors; the model may be on a GPU.
    device = next(model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    logits = model(input_ids).logits
    picked = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs = picked.float() - torch.logsumexp(logits, dim=-1)
    return (log_probs * tokenized["response_mask"].to(device)).sum(dim=1)


def batch_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompts: list[str],
    responses_chosen: list[str],
    responses_rejected: list[str],
) -> torch.Tensor:
    """Mean DPO loss over a batch.

    Same objective as `compute_per_instance_dpo_loss`, evaluated for many pairs at
    once. The per-instance version issues four separate forwards per pair, which
    on batch size 1 leaves the GPU idle between kernel launches; this issues four
    forwards per *batch*. `tests/test_dpo.py` checks the two agree.
    """
    chosen_logratio = batch_response_log_probs(lm, tokenizer, prompts, responses_chosen) - batch_response_log_probs(
        lm_ref, tokenizer, prompts, responses_chosen
    )
    rejected_logratio = batch_response_log_probs(lm, tokenizer, prompts, responses_rejected) - batch_response_log_probs(
        lm_ref, tokenizer, prompts, responses_rejected
    )
    return -F.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()


HUMAN_TAG = "\n\nHuman: "
ASSISTANT_TAG = "\n\nAssistant: "


def _single_turn(conversation: str) -> tuple[str, str] | None:
    if conversation.count(HUMAN_TAG) != 1 or conversation.count(ASSISTANT_TAG) != 1:
        return None
    _, after_human = conversation.split(HUMAN_TAG)
    prompt, response = after_human.split(ASSISTANT_TAG)
    return prompt.strip(), response.strip()


def load_hh_pairs(path: str | Path) -> list[dict[str, str]]:
    """Single-turn `(prompt, chosen, rejected)` triples from an Anthropic HH file.

    Multi-turn conversations are dropped, as the single-turn recipe requires: only the first
    exchange starts from the same prompt in both branches, so a later turn is not
    a comparison of two responses to one question. About 29% of the rows survive
    (helpful-base: 12,822 of 43,835).
    """
    pairs: list[dict[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chosen = _single_turn(row["chosen"])
            rejected = _single_turn(row["rejected"])
            if chosen is None or rejected is None or chosen[0] != rejected[0]:
                continue
            pairs.append({"prompt": chosen[0], "chosen": chosen[1], "rejected": rejected[1]})
    return pairs
