"""The batched DPO loss must agree with the per-instance one it stands in for.

`scripts/dpo_train.py` calls the batched form because the per-instance version
issues four separate forwards per pair and leaves the GPU idle on batch size 1.
This pins the two together, so a padding or response-mask mistake cannot make DPO
quietly optimise something other than the objective the graded function defines.
"""

import torch
from transformers import AutoModelForCausalLM

from lm_alignment.dpo import batch_dpo_loss, compute_per_instance_dpo_loss

from .common import FIXTURES_PATH
from .test_dpo import _tokenizer


def test_batched_dpo_loss_matches_per_instance():
    tokenizer = _tokenizer()
    model = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2").eval()
    model_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2-ref").eval()

    prompts = ["The quick brown fox jumps over", "The lazy dog jumps over"]
    chosen = ["the lazy dog.", "the quick fox."]
    rejected = ["their crazy frog.", "their crazy frog."]
    beta = 0.5

    batched = batch_dpo_loss(model, model_ref, tokenizer, beta, prompts, chosen, rejected)
    per_instance = torch.stack(
        [
            compute_per_instance_dpo_loss(
                lm=model,
                lm_ref=model_ref,
                tokenizer=tokenizer,
                beta=beta,
                prompt=prompt,
                response_chosen=good,
                response_rejected=bad,
            )
            for prompt, good, bad in zip(prompts, chosen, rejected)
        ]
    ).mean()

    assert torch.isclose(batched, per_instance, atol=1e-4), (batched.item(), per_instance.item())
