"""Figures for the alignment report, computed from the run records.

Naming: `figures/figN_*.pdf` is the deliverable (vector, embedded by LaTeX); the
`.png` beside it is a throwaway preview for an image viewer.

Everything is recomputed from `runs/` on each invocation rather than transcribed
from notes -- an earlier project on this repo had a hand-copied number that turned
out to disagree with the run it supposedly came from. The two exceptions are
marked where they occur: AlpacaEval's length-controlled winrate (produced by
`scripts/lc_winrate_official.py`, which calls the benchmark's own estimator and
needs scikit-learn plus a dataset download) is frozen as a table with its
provenance, and the training hyperparameters in the captions are read from the
run directories' own records.

    python report_track3/make_figures.py

It prints a summary of every number the text quotes, so the text and the figures
cannot drift apart silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

FULL = 5.5
HALF = 2.65

plt.rcParams.update(
    {
        # STIXGeneral, not Cambria: a .ttc collection font loses all text from the
        # second figure onwards in this matplotlib build.
        "font.family": "STIXGeneral",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Okabe-Ito, colour-blind safe.
JUDGE_COLOR = {
    "gpt-5.6-luna": "#0072B2",
    "deepseek-flash": "#D55E00",
    "qwen3.8-flash": "#009E73",
    "local-8b-sft": "#CC79A7",
}
JUDGE_LABEL = {
    "gpt-5.6-luna": "GPT-5.6-luna",
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "qwen3.8-flash": "Qwen3.8-Flash",
    "local-8b-sft": "Llama-3.1-8B-SFT (local)",
}
MODEL_COLOR = {"base": "#4C4C4C", "sft": "#0072B2", "dpo": "#D55E00"}
MODEL_LABEL = {"base": "base", "sft": "SFT", "dpo": "SFT+DPO"}

S9_RUNS = ["base", "sft", "dpo"]
S9_JUDGES = ["gpt-5.6-luna", "deepseek-flash", "qwen3.8-flash", "local-8b-sft"]
# The three judges whose scores are in the same range as each other; the local 8B
# is a contrast, not a peer (see the position-bias figure).
SERIOUS = S9_JUDGES[:3]

# AlpacaEval's headline numbers, percent, from `scripts/lc_winrate_official.py` --
# i.e. computed by the benchmark's own code, not by this script. Length control is
# their GLM: tanh(std length difference) + per-item instruction difficulty +
# regularisation towards two length-gamed GPT-4 baselines. It is frozen here rather
# than recomputed because that estimator pulls scikit-learn and an auxiliary
# dataset; our instruction order was verified against the official eval set
# (805/805), which is what makes the per-item difficulty term align.
#
# Both columns come from that one run so the two panels cannot disagree about
# their own aggregation rule. `s9_cell` below recomputes the raw winrate from the
# judged rows and the script prints both, so a divergence shows up rather than
# sitting silent.
OFFICIAL_WINRATE_PCT = {
    ("base", "gpt-5.6-luna"): 6.25, ("base", "deepseek-flash"): 3.79,
    ("base", "qwen3.8-flash"): 4.53, ("base", "local-8b-sft"): 31.87,
    ("sft", "gpt-5.6-luna"): 9.44, ("sft", "deepseek-flash"): 5.78,
    ("sft", "qwen3.8-flash"): 7.80, ("sft", "local-8b-sft"): 33.19,
    ("dpo", "gpt-5.6-luna"): 6.52, ("dpo", "deepseek-flash"): 3.86,
    ("dpo", "qwen3.8-flash"): 4.78, ("dpo", "local-8b-sft"): 26.26,
}
LC_WINRATE_PCT = {
    ("base", "gpt-5.6-luna"): 4.64, ("base", "deepseek-flash"): 2.73,
    ("base", "qwen3.8-flash"): 3.30, ("base", "local-8b-sft"): 25.90,
    ("sft", "gpt-5.6-luna"): 6.93, ("sft", "deepseek-flash"): 4.09,
    ("sft", "qwen3.8-flash"): 5.88, ("sft", "local-8b-sft"): 22.29,
    ("dpo", "gpt-5.6-luna"): 5.26, ("dpo", "deepseek-flash"): 2.95,
    ("dpo", "qwen3.8-flash"): 3.77, ("dpo", "local-8b-sft"): 20.82,
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a run killed mid-write leaves a half-line
    return rows


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def s9_cell(run: str, judge: str) -> dict | None:
    """Raw winrate over both orders, plus each order separately."""
    path = ROOT / f"runs/_s9_{run}/judged/alpaca_eval.{judge}.jsonl"
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        rows[f"{row['index']}:{row['order']}"] = row
    scored = [row for row in rows.values() if row.get("score") is not None]
    if not scored:
        return None
    win = np.array([row["score"] for row in scored])
    order = np.array([row["order"] for row in scored])
    return {
        "winrate": float(win.mean()),
        "order0": float(win[order == 0].mean()),
        "order1": float(win[order == 1].mean()),
        "n": len(scored),
    }


def sst_rate(run: str, judge: str) -> float | None:
    path = ROOT / f"runs/_s9_{run}/judged/sst.{judge}.summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["safe_rate"]


def summary_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. SFT training
# --------------------------------------------------------------------------- #
def fig_sft() -> None:
    rows = read_jsonl(ROOT / "runs/sft/metrics.jsonl")
    step = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    tps = np.array([r["tokens_per_s"] for r in rows])
    lr = np.array([r["learning_rate"] for r in rows if "learning_rate" in r])

    fig, axes = plt.subplots(2, 1, figsize=(HALF, 3.1), sharex=True, height_ratios=[1, 1])
    axes[0].plot(step, loss, lw=0.8, color=MODEL_COLOR["sft"])
    axes[0].set_ylabel("training loss")
    axes[1].plot(step, tps, lw=0.8, color="#009E73")
    axes[1].set_ylabel("tokens / s")
    axes[1].set_xlabel("step")
    save(fig, "fig1_sft_training")
    print(
        f"  SFT: {len(rows)} records, steps {step[0]}..{step[-1]}, "
        f"loss {loss[0]:.3f} -> {loss[-1]:.3f}, throughput median {np.median(tps):.0f} tok/s, "
        f"final lr {lr[-1]:.2e}"
    )


# --------------------------------------------------------------------------- #
# 2. DPO training
# --------------------------------------------------------------------------- #
def fig_dpo() -> None:
    all_rows = read_jsonl(ROOT / "runs/dpo/metrics.jsonl")
    # Evaluation records carry only (step, wall_time, val_reward_accuracy), so they
    # have to be selected before filtering down to the training rows, not after.
    rows = [r for r in all_rows if "loss" in r]
    step = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    val = [(r["step"], r["val_reward_accuracy"]) for r in all_rows if "val_reward_accuracy" in r]

    fig, axes = plt.subplots(2, 1, figsize=(HALF, 3.1), sharex=True)
    axes[0].plot(step, loss, lw=0.8, color=MODEL_COLOR["dpo"])
    axes[0].axhline(np.log(2), ls=":", lw=0.8, color="k")
    axes[0].text(step[-1] * 0.02, np.log(2) + 0.006, r"$\ln 2$", fontsize=7)
    axes[0].set_ylabel("DPO loss")
    if val:
        vx, vy = zip(*val)
        axes[1].plot(vx, vy, marker="o", ms=2.5, lw=0.8, color="#0072B2")
    axes[1].set_ylabel("val. preference acc.")
    axes[1].set_xlabel("step")
    save(fig, "fig2_dpo_training")
    peak = max(val, key=lambda p: p[1]) if val else (None, float("nan"))
    print(
        f"  DPO: steps {step[0]}..{step[-1]}, loss {loss[0]:.4f} -> {loss[-1]:.4f} "
        f"(ln2={np.log(2):.4f}), val acc first {val[0][1]:.4f} -> peak {peak[1]:.4f}@{peak[0]}"
    )


# --------------------------------------------------------------------------- #
# 3. Capability: MMLU and GSM8K under two prompting protocols
# --------------------------------------------------------------------------- #
CAPABILITY = [
    # label, mmlu summary path, gsm8k protocol note
    ("base\n(zero-shot)", "runs/zero_shot_baseline/summary.json", "unparsed"),
    ("base\n(alpaca)", "runs/_tax_base_alpaca/summary.json", "unparsed"),
    ("SFT", "runs/_tax_sft/summary.json", "unparsed"),
    ("SFT+DPO", "runs/_tax_dpo/summary.json", "unparsed"),
]


def fig_capability() -> None:
    labels = [c[0] for c in CAPABILITY]
    stats = [summary_json(c[1]) for c in CAPABILITY]
    mmlu = [s["mmlu"]["accuracy"] for s in stats]
    gsm = [s["gsm8k"]["accuracy"] for s in stats]
    mmlu_unparsed = [s["mmlu"]["unparsed_rate"] for s in stats]

    colors = ["#999999", "#4C4C4C", MODEL_COLOR["sft"], MODEL_COLOR["dpo"]]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9))
    for ax, values, title, chance in (
        (axes[0], mmlu, "MMLU (285 dev)", 0.25),
        (axes[1], gsm, "GSM8K (1319 test)", 0.0),
    ):
        bars = ax.bar(range(len(values)), values, color=colors, width=0.62)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.22)
        if chance:
            ax.axhline(chance, ls=":", lw=0.8, color="k")
            ax.text(len(values) - 0.4, chance + 0.008, "chance", fontsize=6, ha="right")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}",
                    ha="center", fontsize=6.5)
    axes[0].set_ylabel("accuracy")
    save(fig, "fig3_capability")
    for (label, _, _), s in zip(CAPABILITY, stats):
        print(f"  {label.replace(chr(10),' '):16} mmlu {s['mmlu']['accuracy']:.4f} "
              f"(unparsed {s['mmlu']['unparsed_rate']:.1%})  gsm8k {s['gsm8k']['accuracy']:.4f}")
    print(f"  MMLU unparsed rates: {[f'{u:.1%}' for u in mmlu_unparsed]}")


# --------------------------------------------------------------------------- #
# 4. AlpacaEval: raw and length-controlled winrate
# --------------------------------------------------------------------------- #
def fig_alpaca() -> None:
    from_disk = {(run, j): s9_cell(run, j)["winrate"] * 100 for run in S9_RUNS for j in S9_JUDGES}
    # The three API judges only. The local 8B scores 26-33% raw against their 4-9%,
    # and one bar eight times the height of the others flattens exactly the
    # comparison this figure exists to make; its numbers are in the caption and its
    # behaviour is the subject of the position-bias figure.
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1), sharey=True)
    width = 0.26
    for panel, table, title in (
        (axes[0], OFFICIAL_WINRATE_PCT, "raw winrate"),
        (axes[1], LC_WINRATE_PCT, "length-controlled winrate"),
    ):
        for k, judge in enumerate(SERIOUS):
            xs = [i + (k - 1) * width for i in range(len(S9_RUNS))]
            ys = [table[(run, judge)] for run in S9_RUNS]
            panel.bar(xs, ys, width=width, label=JUDGE_LABEL[judge], color=JUDGE_COLOR[judge])
        panel.set_xticks(range(len(S9_RUNS)))
        panel.set_xticklabels([MODEL_LABEL[r] for r in S9_RUNS])
        panel.set_title(title)
        panel.set_ylim(0, 12.5)
    axes[0].set_ylabel("winrate vs GPT-4 Turbo (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, loc="upper left", fontsize=6.5)
    save(fig, "fig4_alpaca_winrate")
    for run in S9_RUNS:
        raw_mean = np.mean([OFFICIAL_WINRATE_PCT[(run, j)] for j in SERIOUS])
        lc_mean = np.mean([LC_WINRATE_PCT[(run, j)] for j in SERIOUS])
        disk_mean = np.mean([from_disk[(run, j)] for j in SERIOUS])
        print(f"  {run:5} raw {raw_mean:.2f}%  LC {lc_mean:.2f}%  (mean of 3 judges); "
              f"8B raw {OFFICIAL_WINRATE_PCT[(run, 'local-8b-sft')]:.2f}%  "
              f"LC {LC_WINRATE_PCT[(run, 'local-8b-sft')]:.2f}%")
        print(f"        cross-check vs plain mean over judged rows: {disk_mean:.2f}% "
              f"(differs by {abs(disk_mean - raw_mean):.2f} pt -- draw handling)")


# --------------------------------------------------------------------------- #
# 5. Position bias, which is what separates the judges
# --------------------------------------------------------------------------- #
def fig_position() -> None:
    # Log-log: a position effect is multiplicative, and the local 8B's ratios are an
    # order of magnitude past everything else -- on linear axes it collapses the
    # three API judges into a single blob at the origin. Explicit ticks because
    # matplotlib's log minors overlap into unreadable mush at this width.
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    lo, hi = 0.02, 0.8
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="k", zorder=1)
    for run in S9_RUNS:
        for judge in S9_JUDGES:
            cell = s9_cell(run, judge)
            marker = "s" if judge == "local-8b-sft" else "o"
            ax.scatter(max(cell["order0"], lo * 1.05), max(cell["order1"], lo * 1.05),
                       s=18, marker=marker, color=JUDGE_COLOR[judge], zorder=3,
                       edgecolor="white", linewidth=0.4)
    ax.annotate("10-25$\\times$ more wins\nwhen presented first",
                xy=(0.51, 0.048), xytext=(0.25, 0.52), fontsize=6.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, shrinkA=2, shrinkB=4))
    handles = [plt.Line2D([], [], marker="s" if j == "local-8b-sft" else "o", ls="",
                          color=JUDGE_COLOR[j], label=JUDGE_LABEL[j], ms=4)
               for j in JUDGE_COLOR]
    handles.append(plt.Line2D([], [], ls="--", lw=0.8, color="k", label="no position bias"))
    ax.legend(handles=handles, frameon=False, fontsize=5.5,
              loc="upper left", handletextpad=0.35, borderaxespad=0.2)
    ax.set_xlabel("winrate when our answer is presented first")
    ax.set_ylabel("winrate when presented second")
    save(fig, "fig5_position_bias")
    for run in S9_RUNS:
        ratios = []
        for judge in S9_JUDGES:
            cell = s9_cell(run, judge)
            ratio = cell["order0"] / cell["order1"] if cell["order1"] else float("inf")
            ratios.append(f"{judge.split('-')[0]} {ratio:.2f}")
        print(f"  {run:5} order0/order1: " + ", ".join(ratios))


# --------------------------------------------------------------------------- #
# 6. Safety
# --------------------------------------------------------------------------- #
def fig_safety() -> None:
    # Unsafe rate, not the safe rate. Safe rates sit in 85-100%, so a zero-based axis
    # of them is a wall of equal-height bars and the only way to see anything is to
    # truncate the axis, which is how a percentage figure ends up lying. The
    # complement is the same information with the interesting values near zero, and
    # it reads correctly from a zero baseline.
    fig, ax = plt.subplots(figsize=(HALF, 1.9))
    width = 0.2
    for k, judge in enumerate(S9_JUDGES):
        xs = [i + (k - 1.5) * width for i in range(len(S9_RUNS))]
        ys = [(1 - sst_rate(run, judge)) * 100 for run in S9_RUNS]
        ax.bar(xs, ys, width=width, color=JUDGE_COLOR[judge], label=JUDGE_LABEL[judge])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.25, f"{y:.0f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(len(S9_RUNS)))
    ax.set_xticklabels([MODEL_LABEL[r] for r in S9_RUNS])
    ax.set_ylabel("answers judged unsafe (%)")
    ax.set_ylim(0, 22)
    # One column, hard right: two columns spread the legend across the middle of the
    # axes and land the second entry on top of the base/deepseek count.
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", ncol=1)
    save(fig, "fig6_safety")
    for run in S9_RUNS:
        rates = [f"{sst_rate(run, j)*100:.0f}" for j in S9_JUDGES]
        print(f"  {run:5} safe% by judge: {', '.join(rates)}")


if __name__ == "__main__":
    print("building figures from runs/ ...")
    fig_sft()
    fig_dpo()
    fig_capability()
    fig_alpaca()
    fig_position()
    fig_safety()
    print("done")
