"""Repair the two logging bugs that `sft_train.py` shipped with.

Both are recoverable from what is already on disk, so the ~2,400 records an
unattended run has already written do not have to be thrown away:

1. **`tokens_per_s` decayed towards zero.** It divided a per-interval token count
   by the elapsed time since the run began. Recompute it as
   `tokens_in_interval / (wall_time - previous wall_time)`; the interval's token
   count is `(step - previous step) * effective_batch * seq_length`.
2. **The first record's `loss` is 10x too small.** It divided by `log_every`
   although only `accumulation` micro-batches had accumulated, so step 1 read
   0.179 when the real loss was 1.79. Multiply that one record back.

Refuses to touch a file written to in the last few minutes: the run is probably
still appending to it.

    python scripts/repair_sft_metrics.py --run-dir runs/sft --batch-size 32 --seq-length 512
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_dir: str = typer.Option("runs/sft"),
    batch_size: int = typer.Option(32, help="Effective batch: the tokens one step covers."),
    seq_length: int = typer.Option(512),
    log_every: int = typer.Option(10, help="Only used to undo the first record's division."),
    quiet_minutes: float = typer.Option(3.0, help="Refuse if written to more recently than this."),
    dry_run: bool = typer.Option(False),
) -> None:
    metrics = Path(run_dir) / "metrics.jsonl"
    if not metrics.exists():
        raise typer.BadParameter(f"{metrics} does not exist")
    age_minutes = (time.time() - metrics.stat().st_mtime) / 60
    if age_minutes < quiet_minutes:
        raise typer.BadParameter(
            f"{metrics} was written to {age_minutes:.1f} min ago; the run is still going. "
            f"Stop it first, or lower --quiet-minutes."
        )

    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]

    # The first record's interval starts at the run's own start, which is what
    # config.json recorded before the model was loaded.
    config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    started = datetime.fromisoformat(config["started_at"]).timestamp()

    tokens_per_step = batch_size * seq_length
    changed_rate = changed_first = 0
    previous_step, previous_wall = 0, started
    for row in rows:
        if "tokens_per_s" in row:
            interval = row["wall_time"] - previous_wall
            if interval > 0:
                row["tokens_per_s"] = (row["step"] - previous_step) * tokens_per_step / interval
                changed_rate += 1
        if row["step"] == 1 and "loss" in row:
            row["loss"] *= log_every
            changed_first += 1
        previous_step, previous_wall = row["step"], row["wall_time"]

    typer.echo(f"  {len(rows)} records; recomputed tokens_per_s on {changed_rate}, "
               f"rescaled {changed_first} first-record loss")
    if rows and "tokens_per_s" in rows[-1]:
        typer.echo(f"  last record: step {rows[-1]['step']}  loss {rows[-1]['loss']:.4f}  "
                   f"tokens/s {rows[-1]['tokens_per_s']:.0f}")
    if dry_run:
        typer.echo("  (dry run, nothing written)")
        return

    backup = metrics.with_suffix(".jsonl.broken")
    shutil.copy2(metrics, backup)
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    typer.echo(f"  wrote {metrics}; original kept at {backup}")


if __name__ == "__main__":
    app()
