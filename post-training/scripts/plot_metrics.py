"""Plot training metrics across seeds.

Reads runs/<variant>_seed<N>/metrics.jsonl and, for each metric, plots every
seed plus the mean with a 95% confidence band (mu +/- 1.96 * sigma / sqrt(n)).
RL runs vary a lot between seeds, so the band is the honest way to read these
curves -- a single run is not evidence.

Usage:
    python scripts/plot_metrics.py --runs-dir runs --out-dir analysis_plots
    python scripts/plot_metrics.py --runs-dir runs --metrics val_accuracy,loss
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

app = typer.Typer(add_completion=False)

# metric -> (title, ylabel)
METRICS: dict[str, tuple[str, str]] = {
    # validation
    "val_accuracy": ("Validation accuracy (greedy)", "accuracy"),
    "val_reward_mean": ("Validation reward (greedy)", "reward"),
    "val_format_rate": ("Validation format rate (greedy)", "fraction"),
    "val_response_len_mean": ("Validation response length (greedy)", "tokens"),
    # sampled validation, for comparing against the prompting baselines
    "val_sampled_accuracy": ("Validation accuracy (sampled)", "accuracy"),
    "val_sampled_reward_mean": ("Validation reward (sampled)", "reward"),
    "val_sampled_format_rate": ("Validation format rate (sampled)", "fraction"),
    "val_sampled_response_len_mean": ("Validation response length (sampled)", "tokens"),
    # training rollouts
    "reward_mean": ("Training reward", "reward"),
    "format_reward_mean": ("Training format rate", "fraction"),
    "answer_reward_mean": ("Training answer reward", "reward"),
    "train_accuracy": ("Training accuracy", "accuracy"),
    "train_response_len_mean": ("Training response length", "tokens"),
    "pass@1": ("pass@1", "fraction"),
    "pass@8": ("pass@8", "fraction"),
    # optimisation health
    "loss": ("Loss", "loss"),
    "grad_norm": ("Gradient norm (before clipping)", "norm"),
    "token_entropy": ("Token entropy", "nats"),
    # the diagnostics that say whether a batch carried any signal at all
    "n_sequences_kept": ("Sequences kept after advantage pruning", "count"),
    "advantage_mean": ("Advantage mean", "advantage"),
    "advantage_std": ("Advantage std", "advantage"),
    "reward_max": ("Max reward in batch", "reward"),
    "reward_min": ("Min reward in batch", "reward"),
    # response-length health: a clipped response cannot score for reasons that
    # have nothing to do with the policy
    "response_clipped_ratio": ("Responses hitting max_tokens", "fraction"),
    "train_response_len_max": ("Longest training response", "tokens"),
    # the only metric that catches a stale inference engine
    "sampling_logprob_gap_mean": ("Sampling vs trainer log-prob gap (mean)", "nats"),
    "sampling_logprob_gap_max": ("Sampling vs trainer log-prob gap (max)", "nats"),
    # off-policy
    "clip_fraction": ("Clip fraction", "fraction"),
    "importance_ratio_mean": ("Importance ratio mean", "ratio"),
    # where the wall clock goes
    "time_generate_s": ("Generation time per step", "seconds"),
    "time_grade_s": ("Grading time per step", "seconds"),
    "time_train_s": ("Gradient step time per step", "seconds"),
}

RUN_DIR_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)$")


def load_runs(runs_dir: Path) -> dict[str, list[list[dict]]]:
    """variant -> list of per-seed metric records, sorted by step."""
    runs: dict[str, list[list[dict]]] = defaultdict(list)
    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        match = RUN_DIR_RE.match(run_dir.name)
        variant = match.group("variant") if match else run_dir.name
        records = []
        for line_number, line in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A run killed mid-write leaves a torn final line, and the first
                # figure should not be the one that reports it.
                print(f"skipping malformed line {line_number} of {metrics_path}")
        records.sort(key=lambda r: r["step"])
        runs[variant].append(records)
    return runs


def mean_and_ci(
    seed_records: list[list[dict]], metric: str
) -> tuple[list[int], list[float], list[float | None]]:
    """Mean and 95% CI half-width per step.

    The half-width is None where fewer than two seeds reached that step: a
    zero-width band would draw as a hard line and read as certainty.
    """
    by_step: dict[int, list[float]] = defaultdict(list)
    for records in seed_records:
        for record in records:
            if metric in record:
                by_step[record["step"]].append(record[metric])
    steps = sorted(by_step)
    means: list[float] = []
    half_widths: list[float | None] = []
    for step in steps:
        values = by_step[step]
        mean = sum(values) / len(values)
        means.append(mean)
        if len(values) < 2:
            half_widths.append(None)
            continue
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        half_widths.append(1.96 * (variance**0.5) / (len(values) ** 0.5))
    return steps, means, half_widths


def available_metrics(runs: dict[str, list[list[dict]]]) -> list[str]:
    present: set[str] = set()
    for per_seed_records in runs.values():
        for records in per_seed_records:
            for record in records:
                present.update(record)
    return [metric for metric in METRICS if metric in present]


@app.command()
def main(
    runs_dir: str = typer.Option("runs"),
    out_dir: str = typer.Option("analysis_plots"),
    metrics: str = typer.Option("", help="Comma-separated subset; default is all available."),
    formats: str = typer.Option("png"),
) -> None:
    runs_root = Path(runs_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_root)
    if not runs:
        raise typer.BadParameter(f"No runs with metrics.jsonl under {runs_root}.")

    available = available_metrics(runs)
    wanted = [m.strip() for m in metrics.split(",") if m.strip()] or available

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for metric in wanted:
        title, ylabel = METRICS.get(metric, (metric, metric))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (variant, seed_records) in enumerate(sorted(runs.items())):
            color = colors[i % len(colors)]
            for records in seed_records:
                steps = [r["step"] for r in records if metric in r]
                values = [r[metric] for r in records if metric in r]
                ax.plot(steps, values, color=color, alpha=0.25, linewidth=1)
            steps, means, half = mean_and_ci(seed_records, metric)
            n_seeds = len(seed_records)
            ax.plot(steps, means, color=color, linewidth=2, label=f"{variant} (n={n_seeds})")
            band = [(s, m, h) for s, m, h in zip(steps, means, half) if h]
            if band:
                band_steps, band_means, band_half = zip(*band)
                ax.fill_between(
                    band_steps,
                    [m - h for m, h in zip(band_means, band_half)],
                    [m + h for m, h in zip(band_means, band_half)],
                    color=color,
                    alpha=0.15,
                )
        ax.set_xlabel("rollout step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  (mean with 95% CI over seeds; faint lines are individual runs)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for fmt in [f.strip() for f in formats.split(",") if f.strip()]:
            fig.savefig(out_root / f"{metric}.{fmt}", dpi=150)
        plt.close(fig)
        typer.echo(f"wrote {out_root / (metric + '.png')}")

    typer.echo(f"\n{len(runs)} variants: " + ", ".join(f"{k}({len(v)} seeds)" for k, v in sorted(runs.items())))


if __name__ == "__main__":
    app()
