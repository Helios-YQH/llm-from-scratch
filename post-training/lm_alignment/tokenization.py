"""Turning prompt/response strings into the tensors the RL loop trains on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Shift prompt+response into causal-LM inputs.

    `response_mask` aligns with `labels`, not `input_ids`.
    """
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    full_ids = [p + o for p, o in zip(prompt_ids, output_ids)]
    batch_size = len(full_ids)
    max_len = max(len(ids) for ids in full_ids)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)

    for i, (prompt, output, ids) in enumerate(zip(prompt_ids, output_ids, full_ids)):
        padded = torch.full((max_len,), pad_id, dtype=torch.long)
        padded[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        # The two views drop different ends; the response starts at len(prompt)-1.
        input_ids[i] = padded[:-1]
        labels[i] = padded[1:]
        response_mask[i, len(prompt) - 1 : len(ids) - 1] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
