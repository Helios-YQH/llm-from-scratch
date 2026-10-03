"""Launch one GRPO run for a named experiment variant.

Each variant is a set of argument overrides on top of scripts/grpo_train.py, so
a sweep is reproducible instead of a pile of hand-typed flags.

    python scripts/run_experiments.py --variant dr_grpo --seed 0 \
        --train-gpu 3 --vllm-gpu 5
    python scripts/run_experiments.py --list
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

# Variant -> overrides for scripts/grpo_train.py.
VARIANTS: dict[str, dict] = {
    "grpo_sequence": {},
    "grpo_constant": {"loss_normalization": "constant"},
    "dr_grpo": {"loss_normalization": "constant", "advantage_normalizer": "none"},
    "rft": {
        "loss_normalization": "constant",
        "advantage_normalizer": "none",
        "baseline": "none",
    },
    "maxrl": {"loss_normalization": "constant", "advantage_normalizer": "mean"},
    # 32x off-policy: one training step per 1/32nd of the rollout batch.
    "offpolicy_naive": {"train_batch_size": 8, "gradient_accumulation_steps": 1},
    "offpolicy_noclip": {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "importance_reweighting_method": "noclip",
    },
    "offpolicy_clip": {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "importance_reweighting_method": "grpo",
        "cliprange": 0.2,
    },
    "offpolicy_gspo": {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "importance_reweighting_method": "gspo",
        "cliprange": 3e-4,
    },
    # track2 (V2): the standard run with a weak verifier swapped in for the reward.
    # Every other hyperparameter is identical, so the `exact` arm of this sweep is
    # literally the standard run truncated to the same number of steps.
    "veritas_format_only": {"reward_fn": "format-only"},
    "veritas_noisy0.1": {"reward_fn": "noisy:0.1"},
    "veritas_noisy0.3": {"reward_fn": "noisy:0.3"},
    # Raises on purpose: self-verify needs a call back into an inference engine.
    "veritas_self_verify": {"reward_fn": "self-verify"},
}


@app.command()
def main(
    variant: str = typer.Option("", help="Variant name; see --list."),
    list_variants: bool = typer.Option(False, "--list", help="Show the available variants."),
    seed: int = typer.Option(0),
    train_gpu: int = typer.Option(0, help="Physical GPU for the trainable policy."),
    rollout_backend: str = typer.Option(
        "vllm", help="'vllm' needs two GPUs (see --vllm-gpu); 'hf' runs on one."
    ),
    vllm_gpu: int = typer.Option(1, help="Physical GPU for the vLLM server; ignored for 'hf'."),
    vllm_port: int = typer.Option(
        0, help="0 picks a distinct port per (variant, seed) so a sweep cannot collide."
    ),
    model: str = typer.Option("/mnt/14T/houyi/models/OLMo-2-0425-1B"),
    prompt: str = typer.Option("r1_zero"),
    num_rollout_steps: int = typer.Option(200),
    rollout_batch_size: int = typer.Option(256),
    group_size: int = typer.Option(8),
    learning_rate: float = typer.Option(1e-5),
    n_train: int = typer.Option(6400),
    n_val: int = typer.Option(1024),
    eval_every: int = typer.Option(10),
    runs_dir: str = typer.Option("runs"),
    output_dir: str = typer.Option("", help="Override the derived runs/<variant>_seed<N> path."),
    wandb_project: str = typer.Option("", help="Empty disables wandb; metrics.jsonl is always written."),
    wandb_entity: str = typer.Option("", help="Required when the wandb account has no default entity."),
    extra_args: str = typer.Option("", help="Extra flags, passed through verbatim."),
    dry_run: bool = typer.Option(False, help="Print the command instead of running it."),
) -> None:
    if list_variants:
        for name, overrides in VARIANTS.items():
            shown = overrides or {"(defaults)": "sequence normalization, group-mean baseline"}
            typer.echo(f"{name:<20} {shown}")
        return
    if variant not in VARIANTS:
        raise typer.BadParameter(f"Unknown variant {variant!r}. Known: {', '.join(VARIANTS)}")

    if vllm_port == 0:
        # The vLLM server now refuses to start on an occupied port rather than
        # killing whatever is there, so give every run its own.
        vllm_port = 8100 + zlib.crc32(f"{variant}_{seed}".encode()) % 800

    overrides = dict(VARIANTS[variant])
    normalization_constant = rollout_batch_size * 512
    if overrides.get("loss_normalization") == "constant":
        overrides["normalization_constant"] = normalization_constant

    output_dir = Path(output_dir) if output_dir else Path(runs_dir) / f"{variant}_seed{seed}"
    command = [
        sys.executable,
        str(Path(__file__).parent / "grpo_train.py"),
        "--model", model,
        "--prompt", prompt,
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--num-rollout-steps", str(num_rollout_steps),
        "--rollout-batch-size", str(rollout_batch_size),
        "--group-size", str(group_size),
        "--learning-rate", str(learning_rate),
        "--n-train", str(n_train),
        "--n-val", str(n_val),
        "--eval-every", str(eval_every),
        "--policy-device", "cuda:0",
        "--rollout-backend", rollout_backend,
        "--vllm-gpu", str(vllm_gpu),
        "--vllm-port", str(vllm_port),
    ]
    for key, value in overrides.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    if wandb_project:
        command += ["--wandb-project", wandb_project]
    if wandb_entity:
        command += ["--wandb-entity", wandb_entity]
    if extra_args:
        command += extra_args.split()

    typer.echo(f"variant={variant} seed={seed}")
    typer.echo(f"CUDA_VISIBLE_DEVICES={train_gpu} " + " ".join(command))
    if dry_run:
        return

    # Pin the training process to one GPU; the vLLM subprocess re-sets
    # CUDA_VISIBLE_DEVICES to `vllm_gpu`, which is a physical index.
    subprocess.run(
        command,
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(train_gpu)},
    )


if __name__ == "__main__":
    app()
