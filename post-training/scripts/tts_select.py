"""Experiment 3: the same rollouts, re-ranked by verifiers of differing quality.

Pure re-read of `tts_panorama.py`'s output -- no generation, no training. This is the
TTS arm of the comparison: how far does best-of-n fall short of the oracle as the
verifier degrades? The RL arm (`weak_verifiers` swapped into `grpo_train.py`) uses the
same verifiers, so the two curves share an x-axis.

Example:
    python scripts/tts_select.py --samples runs/tts_panorama/samples.jsonl
"""

from __future__ import annotations

import typer

from lm_alignment.drgrpo_grader import r1_zero_reward_fn
from lm_alignment.tts_metrics import best_of_n_accuracy, majority_at_k, pass_at_k, read_samples
from lm_alignment.weak_verifiers import format_only_reward_fn, make_noisy_reward_fn

app = typer.Typer(add_completion=False)


@app.command()
def main(
    samples: str = typer.Option("runs/tts_panorama/samples.jsonl"),
    noise_levels: str = typer.Option("0.1,0.3,0.5", help="Epsilon values for the noisy verifier."),
    seed: int = typer.Option(0),
) -> None:
    records = read_samples(samples)
    num_samples = len(records[0]["responses"])
    typer.echo(f"{len(records)} questions, {num_samples} samples each\n")

    scorers = {
        "perfect": lambda response, truth: r1_zero_reward_fn(response, truth)["reward"],
        "format-only": lambda response, truth: format_only_reward_fn(response, truth)["reward"],
    }
    for epsilon in [float(e) for e in noise_levels.split(",") if e.strip()]:
        noisy = make_noisy_reward_fn(r1_zero_reward_fn, epsilon=epsilon, seed=seed)
        scorers[f"noisy eps={epsilon}"] = lambda response, truth, fn=noisy: fn(response, truth)["reward"]

    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= num_samples]
    header = (
        f"{'scorer':<16}"
        + "".join(f"{f'n={k}':>10}" for k in ks)
        + f"{'tie rate':>10}{'identical':>10}"
    )
    typer.echo(header)

    # Oracle ceiling and the verifier-free baseline, for reference.
    typer.echo(
        f"{'pass@n (oracle)':<16}" + "".join(f"{pass_at_k(records, k):>10.4f}" for k in ks) + f"{'-':>10}{'-':>10}"
    )
    typer.echo(
        f"{'majority@n':<16}" + "".join(f"{majority_at_k(records, k):>10.4f}" for k in ks) + f"{'-':>10}{'-':>10}"
    )

    for name, scorer in scorers.items():
        accuracies, ties, identical = [], [], []
        for k in ks:
            accuracy, tie_fraction, identical_fraction = best_of_n_accuracy(
                records, k, scorer, tie_break_seed=seed
            )
            accuracies.append(accuracy)
            ties.append(tie_fraction)
            identical.append(identical_fraction)
        typer.echo(
            f"{name:<16}"
            + "".join(f"{accuracy:>10.4f}" for accuracy in accuracies)
            + f"{ties[-1]:>10.4f}{identical[-1]:>10.4f}"
        )

    typer.echo(
        "\n'tie rate' : questions where more than one candidate shares the top score."
        "\n' identical': questions where every candidate scored the same."
        "\nNeither says whether the verifier is any good -- both are near-identical for the"
        "\nperfect and the format-only scorer. The accuracy columns are the readout; these"
        "\ntwo only say how the winner was picked (ties are broken at random, so a blind"
        "\nverifier lands on the base rate)."
    )


if __name__ == "__main__":
    app()
