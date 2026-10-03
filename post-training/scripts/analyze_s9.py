"""The raw AlpacaEval table for S9: winrate, parse failures, position split, and the
within-run length coefficient.

**The length-controlled winrate is not here.** It used to be -- a one-feature
logistic fit on the log length ratio -- and that estimator is wrong. It has no term
for per-item difficulty, so with answers that are all far shorter than the
reference it conflates "hard questions have long reference answers" with "the judge
likes length", and it produced a conclusion the official estimator reverses (it
said SFT's gain disappears under length control; AlpacaEval's own estimator says the
gain survives). Use `scripts/lc_winrate_official.py` for that column.

What is left is what this script can honestly say: the raw winrate, how many calls
failed or came back unparseable, the two orders separately (a judge whose orders
disagree is not measuring what the averaged number claims), and the fitted length
coefficient as a *diagnostic* -- its sign varies across runs, which is itself the
signature of the confound above and a reason not to read it as a judge property.

    python scripts/analyze_s9.py
    python scripts/analyze_s9.py --run base
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)

RUNS = ["base", "sft", "dpo"]
JUDGES = ["gpt-5.6-luna", "deepseek-flash", "qwen3.8-flash", "local-8b-sft"]
REFERENCE = Path("data/alpaca_eval/alpaca_eval_gpt4_turbo.json")


def fit_logistic(x: np.ndarray, y: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Unregularised 1-D logistic regression, as AlpacaEval's own fit uses."""
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(design @ beta)))
        weight = np.clip(p * (1.0 - p), 1e-9, None)
        gradient = design.T @ (p - y)
        hessian = design.T @ (design * weight[:, None])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def load(run: str, judge: str):
    judged_path = Path(f"runs/_s9_{run}/judged/alpaca_eval.{judge}.jsonl")
    summary_path = Path(f"runs/_s9_{run}/judged/alpaca_eval.{judge}.summary.json")
    # The summary is written last, so its absence means the run is still going.
    if not judged_path.exists() or not summary_path.exists():
        return None
    predictions = json.loads(
        Path(f"runs/_s9_{run}/alpaca_eval.json").read_text(encoding="utf-8")
    )
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    # A row can appear more than once: a failed attempt stays on disk and a rerun
    # appends a fresh row for the same key. Last wins, matching what judge_api's own
    # cache does -- otherwise the failure rate counts abandoned attempts as samples.
    rows: dict[str, dict] = {}
    for line in judged_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[f"{row['index']}:{row['order']}"] = row
    rows = list(rows.values())
    scored = [row for row in rows if row.get("score") is not None]
    length_ratio = np.array(
        [
            math.log(len(predictions[row["index"]]["output"]) + 1)
            - math.log(len(reference[row["index"]]["output"]) + 1)
            for row in scored
        ]
    )
    win = np.array([1.0 if row["score"] > 0.5 else 0.0 for row in scored])
    order = np.array([row["order"] for row in scored])
    beta = fit_logistic(length_ratio, win)
    return {
        "n": len(scored),
        # Rows whose ranking could not be parsed never get a `score`, so they are
        # missing from `scored` but still on disk -- the gap is the parse failure
        # rate. (The judge's own summary reports 0 here; it counts only the rows
        # that were scored, which cannot contain an unparsed one.)
        "parse_fail": 1.0 - len(scored) / max(len(rows), 1),
        "winrate": float(win.mean()),
        "length_coef": float(beta[1]),
        "mean_length_ratio": float(length_ratio.mean()),
        "order0": float(win[order == 0].mean()) if (order == 0).any() else float("nan"),
        "order1": float(win[order == 1].mean()) if (order == 1).any() else float("nan"),
    }


@app.command()
def main(
    run: str = typer.Option("", help="Only this run; empty means all three."),
    judges: str = typer.Option(",".join(JUDGES)),
) -> None:
    runs = [run] if run else RUNS
    header = (
        f"{'run':6} {'judge':19} {'winrate':>9} "
        f"{'len coef':>9} {'len ratio':>10} {'order0':>8} {'order1':>8} {'fail':>7} {'n':>6}"
    )
    print(header)
    print("-" * len(header))
    for name in runs:
        for judge in [j.strip() for j in judges.split(",") if j.strip()]:
            result = load(name, judge)
            if result is None:
                print(f"{name:6} {judge:19} {'(not finished)':>9}")
                continue
            print(
                f"{name:6} {judge:19} {result['winrate']:>9.4f} "
                f"{result['length_coef']:>9.3f} {result['mean_length_ratio']:>10.3f} "
                f"{result['order0']:>8.4f} {result['order1']:>8.4f} "
                f"{result['parse_fail']:>6.1%} {result['n']:>6}"
            )
        print()


if __name__ == "__main__":
    app()
