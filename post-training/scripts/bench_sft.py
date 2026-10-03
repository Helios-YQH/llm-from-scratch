"""Sweep micro-batch size for the 8B SFT run and report throughput.

The full epoch is 6,836 optimizer steps, so the gap between 25s and 40s a step is
most of a day. Sequence length is fixed at 512 and packing means every sample is a
full block, so the *data* does not affect speed -- this measures on a small slice
and the answer carries over to the real run.

Loads the model once and reuses it across sizes, so the sweep costs a couple of
minutes rather than one model load per configuration.

    python scripts/bench_sft.py --micro-batch-sizes 8,16,24,32
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import typer
from torch.optim import AdamW

from lm_alignment.checkpoint import get_model_and_tokenizer
from lm_alignment.sft import get_packed_sft_dataset, load_sft_prompt_template

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option("/mnt/14T/houyi/models/Meta-Llama-3.1-8B"),
    data: str = typer.Option("data/sft/smoke.jsonl.gz"),
    micro_batch_sizes: str = typer.Option("8,16,24,32"),
    warmup_batches: int = typer.Option(2, help="Discarded: the first step pays for cuda warmup."),
    timed_batches: int = typer.Option(6),
    seq_length: int = typer.Option(512),
    device: str = typer.Option("cuda:0"),
) -> None:
    from peft import LoraConfig, get_peft_model

    base, tokenizer = get_model_and_tokenizer(model, device, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    policy = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    trainable = [p for p in policy.parameters() if p.requires_grad]
    vocab = policy.config.vocab_size

    # One big enough tensor to slice per configuration; tokenizing once avoids
    # paying `get_packed_sft_dataset` again for every size.
    dataset = get_packed_sft_dataset(tokenizer, data, seq_length, shuffle=False)
    blocks = torch.stack([dataset[i]["input_ids"] for i in range(len(dataset))])
    labels = torch.stack([dataset[i]["labels"] for i in range(len(dataset))])

    typer.echo(f"  blocks available: {len(blocks)}")
    typer.echo(f"  {'micro':>6} {'s/micro':>9} {'tok/s':>9} {'peak GB':>9}")
    for size in [int(s) for s in micro_batch_sizes.split(",") if s.strip()]:
        needed = size * (warmup_batches + timed_batches)
        if len(blocks) < needed:
            typer.echo(f"  {size:>6} -- not enough blocks ({needed} needed)")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer = AdamW(trainable, lr=1e-5)
        policy.train()

        def one_batch(start: int, *, step: bool):
            input_ids = blocks[start : start + size].to(device)
            out = policy(input_ids).logits
            loss = F.cross_entropy(out.reshape(-1, vocab), labels[start : start + size].to(device).reshape(-1))
            (loss / 4).backward()
            if step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        try:
            for i in range(warmup_batches):
                one_batch(i * size, step=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(warmup_batches, warmup_batches + timed_batches):
                one_batch(i * size, step=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        except torch.cuda.OutOfMemoryError:
            typer.echo(f"  {size:>6}     OOM")
            optimizer.zero_grad(set_to_none=True)
            continue

        per_batch = elapsed / timed_batches
        peak = torch.cuda.max_memory_allocated() / 1e9
        typer.echo(f"  {size:>6} {per_batch:>8.2f}s {size * seq_length / per_batch:>8.0f} {peak:>8.1f}")


if __name__ == "__main__":
    app()
