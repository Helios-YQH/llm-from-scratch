"""Supervised fine-tuning of Llama-3.1-8B on the packed single-turn SFT data.

LoRA, not full fine-tuning: 8B with fp32 Adam needs ~96GB and a 48GB A6000 cannot
hold it. LoRA keeps the resident footprint near 25GB. It does *not* save compute --
the forward and backward still run through the whole model.

**The loss is a plain cross-entropy, not HF's `labels=` argument.**
`get_packed_sft_dataset` already returns next-token targets, globally shifted so
that the last position of every block is supervised by the first token of the next
one. HF's loss would shift them a second time and train the model to predict the
token before the right one.

Every token is supervised, prompts included; there is no instruction masking,
because the packed stream `sft.py` builds carries no mask.

    python scripts/sft_train.py --data data/sft/train.jsonl.gz --output-dir runs/sft \\
        --model /mnt/14T/houyi/models/Meta-Llama-3.1-8B
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from torch.optim import AdamW

from lm_alignment.checkpoint import get_model_and_tokenizer
from lm_alignment.run_logging import RunLogger, build_config
from lm_alignment.sft import get_packed_sft_dataset, run_iterate_batches

app = typer.Typer(add_completion=False)


def build_lora(model, r: int, alpha: int, dropout: float, targets: str):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[name.strip() for name in targets.split(",") if name.strip()],
    )
    return get_peft_model(model, config)


@app.command()
def main(
    model: str = typer.Option("/mnt/14T/houyi/models/Meta-Llama-3.1-8B"),
    data: str = typer.Option("data/sft/train.jsonl.gz"),
    output_dir: str = typer.Option("runs/sft"),
    seq_length: int = typer.Option(512, help="The reference recipe fixes this at 512."),
    batch_size: int = typer.Option(32, help="Sequences per gradient step -- the reference number."),
    micro_batch_size: int = typer.Option(
        8,
        help="Sequences per forward. `batch_size // micro_batch_size` micro-batches are "
             "accumulated into one gradient step: the same effective batch, and a fraction of "
             "the activation and logit memory, which is what makes this fit on a shared card. "
             "Set equal to --batch-size to disable accumulation.",
    ),
    learning_rate: float = typer.Option(2e-4, help="LoRA wants a much larger step than full FT."),
    epochs: int = typer.Option(1),
    max_steps: int = typer.Option(0, help="Stop early; 0 runs the whole epoch. For smoke tests."),
    lora_r: int = typer.Option(16),
    lora_alpha: int = typer.Option(32),
    lora_dropout: float = typer.Option(0.05),
    lora_targets: str = typer.Option(
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names.",
    ),
    max_grad_norm: float = typer.Option(1.0),
    warmup_steps: int = typer.Option(20),
    log_every: int = typer.Option(10),
    save_every: int = typer.Option(1000, help="Checkpoint cadence in steps; 0 disables."),
    device: str = typer.Option("cuda:0"),
    tensorboard: bool = typer.Option(True),
    wandb_project: str | None = typer.Option("reasoning-rl"),
    wandb_entity: str | None = typer.Option("houyi"),
    overwrite_run: bool = typer.Option(False),
) -> None:
    out_dir = Path(output_dir)
    logger = RunLogger(
        out_dir,
        build_config(locals()),
        tensorboard=tensorboard,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        overwrite=overwrite_run,
    )
    start_time = time.time()

    base, tokenizer = get_model_and_tokenizer(model, device, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    policy = build_lora(base, lora_r, lora_alpha, lora_dropout, lora_targets)
    policy.print_trainable_parameters()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=learning_rate)

    dataset = get_packed_sft_dataset(tokenizer, data, seq_length, shuffle=True)
    accumulation = max(1, batch_size // micro_batch_size)
    batches_per_epoch = len(dataset) // batch_size
    total_steps = batches_per_epoch * epochs
    if max_steps:
        # The schedule has to be built for the run we are actually going to do,
        # or a smoke test never leaves the warmup.
        total_steps = min(total_steps, max_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.03,
    )
    typer.echo(
        f"train={len(dataset)} blocks  effective batch={batch_size} "
        f"({micro_batch_size} x {accumulation})  {batches_per_epoch} steps/epoch  total={total_steps}"
    )

    vocab = policy.config.vocab_size
    policy.train()
    step = 0
    # Accumulated since the last record. `micro_batches` is what makes the first
    # record honest: it only covers `accumulation` micro-batches, not `log_every`.
    running_loss, running_tokens, micro_batches = 0.0, 0, 0
    last_log_time = start_time
    for epoch in range(epochs):
        for index, batch in enumerate(run_iterate_batches(dataset, micro_batch_size, shuffle=True)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = policy(input_ids).logits
            loss = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1))
            # Scale so the accumulated gradient is the mean over the full batch.
            (loss / accumulation).backward()

            running_loss += loss.item()
            running_tokens += input_ids.numel()
            micro_batches += 1
            if (index + 1) % accumulation:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            if step < total_steps - 1:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0 or step == 1:
                now = time.time()
                record = {
                    # Mean over the micro-batches since the last record. The old
                    # form divided by `log_every` unconditionally, so the first
                    # record -- which covers only `accumulation` of them -- read
                    # 0.18 when the true loss was 1.79.
                    "loss": running_loss / max(micro_batches, 1),
                    "grad_norm": float(grad_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                    # Over the interval, not since the run began: the old form put
                    # a per-interval token count over the whole-run elapsed time,
                    # so it decayed towards zero as the run went on.
                    "tokens_per_s": running_tokens / max(now - last_log_time, 1e-9),
                    "epoch": epoch,
                }
                logger.log(step, record)
                typer.echo(
                    f"step {step}/{total_steps}  loss {record['loss']:.4f}  "
                    f"lr {record['learning_rate']:.2e}  {record['tokens_per_s']:.0f} tok/s"
                )
                running_loss, running_tokens, micro_batches = 0.0, 0, 0
                last_log_time = now

            if save_every and step % save_every == 0:
                save(policy, tokenizer, logger.checkpoint_dir(f"step{step:06d}"), merged=False)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            typer.echo(f"stopping early at {step} steps (--max-steps)")
            break

    typer.echo("merging LoRA and saving the final policy")
    save(policy, tokenizer, logger.checkpoint_dir("final"), merged=True)
    logger.close()
    typer.echo(f"done in {time.time() - start_time:.0f}s; artifacts in {out_dir}")


def save(policy, tokenizer, path: Path, *, merged: bool) -> None:
    """`merged` writes a plain HF checkpoint; otherwise just the adapter.

    The merge runs on the **host**: in place on the GPU it needs a second copy of
    the weights beside the ones already resident, and on a shared card that is the
    allocation that fails -- DPO hit exactly this, wanting 11.4GB with 8.6GB free
    after every training step had already succeeded.
    """
    path.mkdir(parents=True, exist_ok=True)
    if merged:
        policy.to("cpu")
        torch.cuda.empty_cache()
        policy.merge_and_unload().save_pretrained(str(path))
    else:
        policy.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


if __name__ == "__main__":
    app()
