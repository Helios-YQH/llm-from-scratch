"""Length-controlled AlpacaEval winrate, computed by AlpacaEval's own estimator.

The simplified version in `analyze_s9.py` fits a one-feature logistic regression on
the log length ratio. That is not what the benchmark does. The official estimator
(Dubois et al., 2024) regresses on

    tanh(std_delta_len) + instruction_difficulty + not_gamed_baseline - 1

where `std_delta_len` is the standardised *raw* length difference, the instruction
difficulty is a per-item term precomputed by the AlpacaEval authors, and the fit is
regularised towards two deliberately length-gamed GPT-4 baselines. Reproducing that
by hand would be a reimplementation with its own bugs, so this calls their function
directly and only does the input plumbing.

Two deliberate choices in the plumbing:

* Each pair is judged in both orders, so the preference fed in is the *mean* of the
  two, which lands on their own scale: both orders won -> 2.0, split -> 1.5 (their
  draw), both lost -> 1.0. This removes position bias, which their single-order
  pipeline does not, and their continuous-target path supports it.
* `HF_ENDPOINT` defaults to the mirror; the estimator downloads `df_gamed.csv`.

    python scripts/lc_winrate_official.py                 # all runs x judges
    python scripts/lc_winrate_official.py --run base --judge gpt-5.6-luna
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import typer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd  # noqa: E402

from alpaca_eval.metrics.glm_winrate import get_length_controlled_winrate  # noqa: E402

app = typer.Typer(add_completion=False)

RUNS = ["base", "sft", "dpo"]
JUDGES = ["gpt-5.6-luna", "deepseek-flash", "qwen3.8-flash", "local-8b-sft"]
REFERENCE = Path("data/alpaca_eval/alpaca_eval_gpt4_turbo.json")
REFERENCE_GENERATOR = "gpt4_turbo"


def annotations_for(run: str, judge: str) -> pd.DataFrame | None:
    judged = Path(f"runs/_s9_{run}/judged/alpaca_eval.{judge}.jsonl")
    if not judged.exists():
        return None
    predictions = json.loads(Path(f"runs/_s9_{run}/alpaca_eval.json").read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    rows: dict[str, dict] = {}
    for line in judged.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[f"{row['index']}:{row['order']}"] = row

    scores: dict[int, list[float]] = defaultdict(list)
    for row in rows.values():
        if row.get("score") is not None:
            scores[row["index"]].append(row["score"])

    records = []
    for index in sorted(scores):
        mean_score = sum(scores[index]) / len(scores[index])
        records.append(
            {
                "index": index,
                # Their convention: 2 = this model wins, 1 = baseline wins, 1.5 = draw.
                "preference": 1.0 + mean_score,
                "output_1": predictions[index]["output"],
                "output_2": reference[index]["output"],
                "generator_1": predictions[index]["generator"],
                "generator_2": REFERENCE_GENERATOR,
                "annotator": judge,
                "instruction": predictions[index]["instruction"],
                "dataset": predictions[index]["dataset"],
            }
        )
    return pd.DataFrame(records)


@app.command()
def main(
    run: str = typer.Option("", help="Only this run; empty means all three."),
    judge: str = typer.Option("", help="Only this judge; empty means all four."),
    glm: str = typer.Option("length_controlled_v1"),
) -> None:
    runs = [run] if run else RUNS
    judges = [judge] if judge else JUDGES
    print(f"{'run':6} {'judge':19} {'n':>5} {'winrate%':>9} {'LC winrate%':>12} {'GLM weights'}")
    print("-" * 88)
    for name in runs:
        for j in judges:
            df = annotations_for(name, j)
            if df is None or df.empty:
                print(f"{name:6} {j:19} {'(no data)':>5}")
                continue
            metrics = get_length_controlled_winrate(
                df, glm_name=glm, save_weights_dir=None, is_warn_extreme_changes=False
            )
            print(
                f"{name:6} {j:19} {len(df):>5} {metrics['win_rate']:>9.2f} "
                f"{metrics['length_controlled_winrate']:>12.2f}"
            )
        print()


if __name__ == "__main__":
    app()
