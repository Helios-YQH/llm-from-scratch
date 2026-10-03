"""Regenerate the data-pipeline report figures as vector PDFs (PNG copies are
throwaway previews for eyeballing in an image viewer).

Data sources:
  * measurements/ablation_30m_{filtered,control}.csv
    4000-step data ablation produced by scripts/train_ablation.py --preset small
    on the server (two arms: full filter pipeline vs language filter only).
    (Kept out of a directory named `data/` because the project .gitignore
    excludes that name at any depth — the figure inputs must be committed.)

Sizes follow the NeurIPS layout conventions used in the other reports:
5.5in full text width, 8pt type, STIX fonts to match the Times body text.

Usage (from data-pipeline/):
    uv run --no-sync python report/make_figures.py     # needs matplotlib
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "measurements"
FIG.mkdir(exist_ok=True)

FULL_W, COL_W = 5.5, 2.65

# Okabe-Ito colour-blind-safe palette (same as the other reports)
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, GREY, LIGHTGREY = "#CC79A7", "#56B4E9", "#8C8C8C", "#BFBFBF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.axisbelow": True,
    }
)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}.pdf")


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, train, val = [], [], []
    with open(DATA / name, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["step"] == "final":
                continue
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return np.array(steps), np.array(train), np.array(val)


# ------------------------------------------------- 1. data ablation (30M) ----
def fig_ablation() -> None:
    s_f, tr_f, va_f = load("ablation_30m_filtered.csv")
    s_c, tr_c, va_c = load("ablation_30m_control.csv")
    gap = va_c - va_f

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(FULL_W, 2.15), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.28}
    )

    # (a) validation curves, with the gap shaded
    ax.fill_between(s_f, va_f, va_c, color=LIGHTGREY, alpha=0.7, linewidth=0, zorder=1)
    ax.plot(
        s_c, va_c, color=ORANGE, ls="--", marker="s", markevery=5, ms=2.6, lw=1.2, zorder=3,
        label="language filter only (1.01B tokens)",
    )
    ax.plot(
        s_f, va_f, color=BLUE, ls="-", marker="o", markevery=5, ms=2.6, lw=1.2, zorder=4,
        label="full pipeline (340M tokens)",
    )
    for steps, val in ((s_f, va_f), (s_c, va_c)):
        i = int(np.argmin(val))
        ax.plot(steps[i], val[i], marker="*", ms=6, color=RED, zorder=5, linestyle="none")
    ax.annotate(
        f"best: {va_f.min():.2f} vs {va_c.min():.2f}",
        xy=(s_f[int(np.argmin(va_f))], va_f.min()), xytext=(1650, 7.7),
        fontsize=7, color=RED,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.6, shrinkA=0, shrinkB=4),
    )
    ax.set_xlabel("optimization step")
    ax.set_ylabel("validation loss (nats)")
    ax.set_ylim(5.2, 11.2)
    ax.set_xlim(-80, 4080)
    ax.legend(loc="upper right", handlelength=2.4, borderaxespad=0.2)
    ax.set_title("(a) C4-100 validation loss", fontsize=8.5, loc="left")

    # (b) the gap itself, showing it widens with training
    ax2.fill_between(s_f, 0, gap, color=GREEN, alpha=0.18, linewidth=0, zorder=1)
    ax2.plot(s_f, gap, color=GREEN, ls="-", marker="^", markevery=5, ms=2.6, lw=1.2, zorder=3)
    ax2.set_xlabel("optimization step")
    ax2.set_ylabel(r"loss gap (control $-$ filtered, nats)")
    ax2.set_ylim(0, 0.42)
    ax2.set_xlim(-80, 4080)
    ax2.annotate(
        f"{gap[-1]:.3f}", xy=(s_f[-1], gap[-1]), xytext=(-6, -9), textcoords="offset points",
        ha="right", fontsize=7.5, color=GREEN,
    )
    ax2.set_title("(b) filtering advantage grows", fontsize=8.5, loc="left")

    save(fig, "ablation_30m")


# ---------------------------------------- 2. scale mismatch (430M vs 340M) ---
def fig_scale_mismatch() -> None:
    """同一份数据用作业的 430M 配置训练: train loss 直奔 0 而 val loss 升到均匀分布之上。"""
    s_f, tr_f, va_f = load("ablation_430m_filtered.csv")
    s_c, tr_c, va_c = load("ablation_430m_control.csv")
    uniform = float(np.log(50257))  # 均匀分布的交叉熵, 即"什么都没学到"的基线

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(FULL_W, 2.15), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.3}
    )

    # (a) validation loss: both arms climb past the uniform baseline
    ax.axhline(uniform, color=GREY, ls=":", lw=1.0, zorder=2)
    ax.plot(
        s_c, va_c, color=ORANGE, ls="--", marker="s", markevery=5, ms=2.6, lw=1.2, zorder=3,
        label="language filter only",
    )
    ax.plot(
        s_f, va_f, color=BLUE, ls="-", marker="o", markevery=5, ms=2.6, lw=1.2, zorder=4,
        label="full pipeline",
    )
    ax.annotate(
        "uniform-distribution\nbaseline", xy=(1500, uniform), xytext=(900, 9.55),
        fontsize=7, color=GREY, ha="center",
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=0, shrinkB=2),
    )
    ax.set_xlabel("optimization step")
    ax.set_ylabel("validation loss (nats)")
    ax.set_ylim(5.0, 11.6)
    ax.set_xlim(-80, 4080)
    ax.legend(loc="center right", handlelength=2.4, borderaxespad=0.2)
    ax.set_title("(a) validation loss degrades", fontsize=8.5, loc="left")

    # (b) training loss: the model memorises the corpus
    ax2.plot(
        s_c, tr_c, color=ORANGE, ls="--", marker="s", markevery=5, ms=2.6, lw=1.2, zorder=3,
        label="language filter only",
    )
    ax2.plot(
        s_f, tr_f, color=BLUE, ls="-", marker="o", markevery=5, ms=2.6, lw=1.2, zorder=4,
        label="full pipeline",
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("optimization step")
    ax2.set_ylabel("training loss (nats)")
    ax2.set_xlim(-80, 4080)
    ax2.set_ylim(5e-3, 2e1)
    ax2.annotate("", xy=(3600, 1.2e-2), xytext=(3600, 1.0), arrowprops=dict(arrowstyle="->", lw=0.8, color=GREY))
    ax2.text(3480, 1.1e-1, "memorises the\n0.6 tokens/param corpus", fontsize=7, color=GREY, ha="right")
    ax2.set_title("(b) training loss collapses", fontsize=8.5, loc="left")

    save(fig, "ablation_430m")


if __name__ == "__main__":
    fig_ablation()
    fig_scale_mismatch()
