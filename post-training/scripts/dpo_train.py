"""DPO on Anthropic HH, starting from the SFT model.

The policy gets a **fresh** LoRA on top of the SFT weights; the reference is that
same module with `disable_adapter()`, so the frozen copy costs no extra memory and
is exactly the model DPO is supposed to regularise towards.

Training set is all four HH files combined, multi-turn conversations dropped --
49,326 pairs survive of 160,800. The loss is the batched form from `dpo.py`; the
per-instance version issues four forwards per *pair* and leaves the GPU idle.

    python scripts/dpo_train.py --model runs/sft/checkpoints/final \\
        --output-dir runs/dpo
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from torch.optim import AdamW

from lm_alignment.checkpoint import get_model_and_tokenizer
from lm_alignment.dpo import batch_dpo_loss, batch_response_log_probs, load_hh_pairs
from lm_alignment.run_logging import RunLogger, build_config

app = typer.Typer(add_completion=False)

HH_FILES = (
    "data/hh/helpful-base.jsonl.gz",
    "data/hh/harmless-base.jsonl.gz",
    "data/hh/helpful-online.jsonl.gz",
    "data/hh/helpful-rejection-sampled.jsonl.gz",
)


class FrozenReference(torch.nn.Module):
    """The policy with its adapter switched off, as a module DPO can be handed."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, *args, **kwargs):
        with torch.no_grad(), self.policy.disable_adapter():
            return self.policy(*args, **kwargs)


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def evaluate(policy, reference, tokenizer, beta, pairs, batch_size, device) -> float:
    """Fraction of validation pairs whose implicit reward prefers `chosen`."""
    policy.eval()
    correct = 0
    for batch in batches(pairs, batch_size):
        prompts = [p["prompt"] for p in batch]
        chosen = [p["chosen"] for p in batch]
        rejected = [p["rejected"] for p in batch]
        margin = batch_response_log_probs(policy, tokenizer, prompts, chosen) - batch_response_log_probs(
            reference, tokenizer, prompts, chosen
        ) - (
            batch_response_log_probs(policy, tokenizer, prompts, rejected)
            - batch_response_log_probs(reference, tokenizer, prompts, rejected)
        )
        correct += int((margin > 0).sum())
    policy.train()
    return correct / len(pairs)


@app.command()
def main(
    model: str = typer.Option("runs/sft/checkpoints/final", help="The instruction-tuned policy."),
    output_dir: str = typer.Option("runs/dpo"),
    beta: float = typer.Option(0.1),
    batch_size: int = typer.Option(16, help="Pairs per gradient step."),
    micro_batch_size: int = typer.Option(
        8,
        help="Pairs per forward. `batch_size // micro_batch_size` micro-batches accumulate "
             "into one gradient step: the same effective batch at half the activation peak. "
             "Batch 16 OOMed at step 150 on a card shared with one other job, wanting 5.3GB "
             "with 4.3GB free. Set equal to --batch-size to disable.",
    ),
    learning_rate: float = typer.Option(1e-5),
    epochs: int = typer.Option(1),
    val_fraction: float = typer.Option(0.02),
    max_steps: int = typer.Option(0, help="Stop early; 0 runs the whole epoch."),
    lora_r: int = typer.Option(16),
    lora_alpha: int = typer.Option(32),
    lora_dropout: float = typer.Option(0.05),
    lora_targets: str = typer.Option("q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
    max_grad_norm: float = typer.Option(1.0),
    eval_every: int = typer.Option(250),
    checkpoints_every: int = typer.Option(
        250,
        help="Steps between resumable checkpoints; 0 disables them and with it --resume. "
             "The adapter is ~50MB, so keeping every one is cheap.",
    ),
    resume: bool = typer.Option(
        False, help="Continue the run in --output-dir from its newest checkpoint."
    ),
    log_every: int = typer.Option(25),
    device: str = typer.Option("cuda:0"),
    seed: int = typer.Option(0),
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
        resume=resume,
    )
    start_time = time.time()

    pairs: list[dict[str, str]] = []
    for path in HH_FILES:
        loaded = load_hh_pairs(path)
        typer.echo(f"  {Path(path).name:<34} {len(loaded):>6} single-turn pairs")
        for pair in loaded:
            pairs.append({**pair, "source": Path(path).name})
    random.Random(seed).shuffle(pairs)

    n_val = int(len(pairs) * val_fraction)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    typer.echo(f"train {len(train_pairs)}  val {len(val_pairs)}")

    from peft import LoraConfig, get_peft_model

    base, tokenizer = get_model_and_tokenizer(model, device, attn_implementation="sdpa")
    base.config.use_cache = False
    policy = get_peft_model(
        base,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[name.strip() for name in lora_targets.split(",") if name.strip()],
        ),
    )
    # After `get_peft_model`: peft rewrites the forward path, and enabling it on the
    # bare base beforehand can be undone by the wrapping.
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.print_trainable_parameters()
    reference = FrozenReference(policy)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=learning_rate)
    accumulation = max(1, batch_size // micro_batch_size)
    batches_per_epoch = len(train_pairs) // batch_size

    resume_checkpoint = logger.latest_checkpoint() if resume else None
    if resume and resume_checkpoint is None:
        # RunLogger's overwrite guard is bypassed by resume, so this is the only
        # thing stopping it from appending a fresh run onto the old metrics.
        raise typer.BadParameter(
            f"--resume given but {out_dir}/checkpoints holds no usable checkpoint. "
            f"Pass --overwrite-run to start over, or delete {out_dir}."
        )
    start_step = 0
    if resume_checkpoint is not None:
        policy.load_adapter(str(resume_checkpoint), adapter_name="default")
        state = torch.load(resume_checkpoint / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        typer.echo(f"resumed from {resume_checkpoint.name}; continuing at step {start_step}")
    total_steps = batches_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    typer.echo(f"{batches_per_epoch} steps/epoch, total {total_steps}")

    policy.train()
    step = start_step
    best_accuracy = -1.0
    running_loss = 0.0
    micro_batches = 0
    for epoch in range(epochs):
        order = list(range(len(train_pairs)))
        random.Random(seed + epoch).shuffle(order)
        # The data order is a pure function of (seed, epoch), so a resumed run
        # replays the same shuffle and skips to where it stopped -- the sampler
        # does not need serialising.
        if epoch < start_step // batches_per_epoch:
            continue
        skip = (start_step % batches_per_epoch) * accumulation if epoch == start_step // batches_per_epoch else 0
        for index, group in enumerate(batches(order, micro_batch_size)):
            if index < skip:
                continue
            batch = [train_pairs[i] for i in group]
            loss = batch_dpo_loss(
                policy,
                reference,
                tokenizer,
                beta,
                [p["prompt"] for p in batch],
                [p["chosen"] for p in batch],
                [p["rejected"] for p in batch],
            )
            # Scale so the accumulated gradient is the mean over the full batch.
            (loss / accumulation).backward()
            running_loss += loss.item()
            micro_batches += 1
            if (index + 1) % accumulation:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0 or step == 1:
                record = {
                    # Mean over the micro-batches since the last record -- dividing by
                    # `log_every` unconditionally would understate the first one, which
                    # only covers `accumulation` of them.
                    "loss": running_loss / max(micro_batches, 1),
                    "grad_norm": float(grad_norm),
                    "epoch": epoch,
                    "pairs_per_s": step * batch_size / (time.time() - start_time),
                    # `gpu_alloc_gb` is the one that separates "needs this much" from
                    # "grows every step"; reserved and peak only ever climb.
                    "gpu_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
                logger.log(step, record)
                typer.echo(f"step {step}/{total_steps}  loss {record['loss']:.4f}")
                running_loss, micro_batches = 0.0, 0

            if checkpoints_every and step % checkpoints_every == 0:
                # The adapter plus the optimizer state: enough to continue exactly,
                # and small enough (~50MB) that keeping every one costs nothing. The
                # step that motivated this had 150 steps thrown away by an OOM.
                path = logger.checkpoint_dir(f"step{step:06d}")
                path.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(str(path))
                tokenizer.save_pretrained(str(path))
                torch.save(
                    {"step": step, "optimizer": optimizer.state_dict()}, path / "training_state.pt"
                )

            if eval_every and step % eval_every == 0:
                accuracy = evaluate(policy, reference, tokenizer, beta, val_pairs, micro_batch_size, device)
                logger.log(step, {"val_reward_accuracy": accuracy})
                typer.echo(f"  step {step}: val reward accuracy {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save(policy, tokenizer, logger.checkpoint_dir("best"), merged=False)
                    typer.echo(f"  new best -> checkpoints/best")

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            typer.echo(f"stopping early at {step} steps (--max-steps)")
            break

    logger.log(step, {"final_val_reward_accuracy": best_accuracy})
    save(policy, tokenizer, logger.checkpoint_dir("final"), merged=True)
    logger.close()
    typer.echo(f"best val reward accuracy {best_accuracy:.4f}")
    typer.echo(f"done in {time.time() - start_time:.0f}s; artifacts in {out_dir}")


def save(policy, tokenizer, path: Path, *, merged: bool) -> None:
    """`merged` writes a plain HF checkpoint; otherwise just the adapter.

    The merge runs on the **host**. Doing it in place on the GPU needs a second
    copy of the weights next to the ones already resident, and on a card shared
    with two other jobs that allocation is what fails -- measured: 11.4GB wanted
    with 8.6GB free, after all 40 steps of a smoke run had already completed. The
    host has the memory and the merge is elementwise, so it costs seconds.
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
