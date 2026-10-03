"""Loss of a (base, +adapter) pair on held-out instruction data, split by segment.

The training loss is a single number averaged over a packed stream, and in that
stream ~34% of the tokens are the *user's* instruction -- which nothing can
predict well. A flat total loss therefore says very little on its own: it may be
sitting on the prompt floor while the response half keeps improving, or it may
mean the adapter is not learning at all. This separates the two.

    python scripts/eval_sft_loss.py --adapter runs/sft/checkpoints/step002000
"""

from __future__ import annotations

import gzip
import json

import torch
import torch.nn.functional as F
import typer

from lm_alignment.checkpoint import get_model_and_tokenizer
from lm_alignment.sft import load_sft_prompt_template
from lm_alignment.tokenization import tokenize_prompt_and_output

app = typer.Typer(add_completion=False)


@torch.no_grad()
def segment_losses(model, tokenizer, template, examples, batch_size, device) -> dict[str, float]:
    """Mean cross-entropy over prompt tokens and over response tokens."""
    totals = {"prompt": 0.0, "response": 0.0}
    counts = {"prompt": 0, "response": 0}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [template.format(instruction=e["prompt"], response="") for e in batch]
        responses = [e["response"] for e in batch]
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        mask = tokenized["response_mask"].to(device)

        # `tokenize_prompt_and_output` already returns next-token targets: labels[t]
        # is the token after input_ids[t]. Shifting again here (the obvious-looking
        # `logits[:, :-1]` / `labels[:, 1:]`) trains the comparison on the wrong
        # token and reports losses around 12 nats.
        logits = model(input_ids).logits
        targets = labels
        mask = tokenized["response_mask"].to(device)
        # `ignore_index` keeps padding out of the mean; the two masks then pick
        # out the halves we want to report separately.
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view(targets.shape)

        valid = targets.ne(pad_id)
        for name, in_segment in (("prompt", ~mask), ("response", mask)):
            keep = valid & in_segment
            totals[name] += float(token_loss[keep].sum())
            counts[name] += int(keep.sum())

    return {"prompt": totals["prompt"] / counts["prompt"], "response": totals["response"] / counts["response"]}


@app.command()
def main(
    model: str = typer.Option("/mnt/14T/houyi/models/Meta-Llama-3.1-8B"),
    adapter: str = typer.Option("", help="LoRA dir; empty evaluates the base model."),
    data: str = typer.Option("data/sft/test.jsonl.gz"),
    n_examples: int = typer.Option(200),
    batch_size: int = typer.Option(4),
    device: str = typer.Option("cuda:0"),
) -> None:
    template = load_sft_prompt_template()
    examples = []
    with gzip.open(data, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= n_examples:
                break
            examples.append(json.loads(line))

    base, tokenizer = get_model_and_tokenizer(model, device, attn_implementation="sdpa")
    base.eval()
    label = "BASE"
    losses = segment_losses(base, tokenizer, template, examples, batch_size, device)
    typer.echo(f"  {label:<22} prompt {losses['prompt']:.4f}   response {losses['response']:.4f}")

    if adapter:
        from peft import PeftModel

        tuned = PeftModel.from_pretrained(base, adapter).eval()
        losses = segment_losses(tuned, tokenizer, template, examples, batch_size, device)
        typer.echo(f"  {'BASE + ' + adapter.split('/')[-1]:<22} prompt {losses['prompt']:.4f}   response {losses['response']:.4f}")


if __name__ == "__main__":
    app()
