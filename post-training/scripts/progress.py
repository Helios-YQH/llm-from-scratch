"""One-line progress for a run directory.

    python scripts/progress.py runs/e3_3shot_hf_seed0

Reads metrics.jsonl, which is the durable record; wandb/TensorBoard are views of
the same numbers but can lag or be down. Tolerates a torn final line, since a
killed process leaves one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

SHOWN = [
    "loss",
    "grad_norm",
    "token_entropy",
    "n_sequences_kept",
    "train_accuracy",
    "response_clipped_ratio",
    "val_accuracy",
    "val_sampled_accuracy",
    "time_generate_s",
    "time_train_s",
]


@app.command()
def main(run_dir: str = typer.Argument(..., help="e.g. runs/e3_3shot_hf_seed0")) -> None:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        typer.echo(f"no metrics.jsonl in {run_dir} — the run may still be starting up")
        raise typer.Exit(1)

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line from a kill
    if not records:
        typer.echo(f"{path} is empty — no step has finished yet")
        raise typer.Exit(1)

    last = records[-1]
    total = None
    config_path = Path(run_dir) / "config.json"
    if config_path.is_file():
        try:
            total = json.loads(config_path.read_text(encoding="utf-8"))["hyperparameters"][
                "num_rollout_steps"
            ]
        except (KeyError, json.JSONDecodeError):
            pass

    done = last.get("step", len(records) - 1) + 1
    done = max(done, len(records))
    progress = f"{done}/{total}" if total else str(done)
    typer.echo(f"{run_dir}  step {progress}")

    # Validation only lands on evaluation steps, so between them the val numbers
    # would vanish from the readout. Carry the most recent ones forward and say
    # which step they came from, rather than showing nothing.
    last_val: dict = {}
    last_val_step = None
    for record in reversed(records):
        if any(key.startswith("val_") for key in record):
            last_val, last_val_step = record, record.get("step")
            break

    for key in SHOWN:
        source = last_val if key.startswith("val_") else last
        if key not in source:
            continue
        value = source[key]
        text = f"{value:>12.4f}" if isinstance(value, float) else f"{value:>12}"
        stale = key.startswith("val_") and last_val_step != last.get("step")
        suffix = f"   (step {last_val_step})" if stale else ""
        typer.echo(f"  {key:<24} {text}{suffix}")

    # ETA from the median of recent step durations. `elapsed_s / done` would be
    # dominated by one-off startup costs (model load, server boot) for the first
    # few steps and would overstate the ETA by hours. The median also keeps a slow
    # evaluation step from swinging it.
    elapsed = last.get("elapsed_s")
    if total and done and elapsed:
        deltas = [
            b["elapsed_s"] - a["elapsed_s"]
            for a, b in zip(records, records[1:])
            if "elapsed_s" in a and "elapsed_s" in b
        ]
        per_step = sorted(deltas)[len(deltas) // 2] if deltas else elapsed / done
        if per_step > 0:
            remaining_min = per_step * (total - done) / 60
            typer.echo(
                f"  {'per step':<24} {per_step:>11.0f}s"
                f"   eta {remaining_min / 60:.1f}h"
            )

    for name in ("error.txt",):
        if (Path(run_dir) / name).is_file():
            typer.echo(f"\n!! {run_dir}/{name} exists — the run died:")
            typer.echo((Path(run_dir) / name).read_text(encoding="utf-8").strip()[-500:])


if __name__ == "__main__":
    sys.exit(app())
