"""Regenerate the report figures from the raw run records.

Reads the per-step logs in ../analysis_data/ (CSV where the run left one,
falling back to the plain-text log where the CSV run was interrupted) and
writes vector PDFs (for LaTeX) plus PNGs (for quick viewing) into figures/.

Sizes are chosen for the NeurIPS two-column layout: 5.5in full width, 2.65in
for a single column.

Usage (from transformer-lm/):
    uv run python report/make_figures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A1 = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = A1 / "analysis_data"
LOGS = A1 / "logs"
FIGDIR = HERE / "figures"

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, GREY, LIGHTGREY = "#CC79A7", "#56B4E9", "#8C8C8C", "#BFBFBF"

CTX = 256  # context length used in every training run
FULL_W, COL_W = 5.5, 2.65

# Colour alone fails for colour-blind readers and in greyscale print, so every
# multi-curve figure repeats the series identity in the line style as well.
STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

plt.rcParams.update({
    "font.family": "serif",
    # STIX ships with matplotlib and matches the Times body text of the
    # NeurIPS style. Cambria is a .ttc collection font and silently loses its
    # glyphs from the second figure onwards.
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
    "figure.dpi": 150,
    "svg.fonttype": "path",
})

LOG_RE = re.compile(
    r"step\s+(\d+)\s+train_loss=([\d.]+|nan)\s+val_loss=([\d.]+|nan)\s+lr=([\d.]+)"
)


class Run:
    def __init__(self, steps, train, val, lr):
        self.steps = np.asarray(steps, dtype=float)
        self.train = np.asarray(train, dtype=float)
        self.val = np.asarray(val, dtype=float)
        self.lr = np.asarray(lr, dtype=float)

    def final(self, key: str = "val") -> float:
        arr = self.val if key == "val" else self.train
        return float(arr[-1])

    def at_step(self, step: int, key: str = "val") -> float:
        """Value at the logged step closest to (and not after) `step`."""
        arr = self.val if key == "val" else self.train
        mask = self.steps <= step
        if not mask.any():
            return float("nan")
        return float(arr[np.argmax(np.where(mask, self.steps, -1))])

    def first_nan_step(self) -> int | None:
        idx = np.flatnonzero(np.isnan(self.val))
        return int(self.steps[idx[0]]) if idx.size else None

    def last_finite_step(self) -> int:
        return int(self.steps[np.flatnonzero(~np.isnan(self.val))][-1])


def _from_csv(path: Path) -> Run | None:
    steps, train, val, lr = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("step"):  # some runs appended a second header
            continue
        s, tr, va, l = line.split(",")
        steps.append(float(s))
        train.append(float(tr))
        val.append(float(va))
        lr.append(float(l))
    return Run(steps, train, val, lr) if steps else None


def _from_log(path: Path) -> Run | None:
    steps, train, val, lr = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_RE.search(line)  # tqdm progress bars do not match and are skipped
        if m:
            steps.append(float(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
            lr.append(float(m.group(4)))
    return Run(steps, train, val, lr) if steps else None


def load(name: str) -> Run:
    """Prefer the CSV; fall back to the log when the CSV run was interrupted."""
    csv = _from_csv(DATA / f"{name}.csv")
    if csv is not None and csv.steps.size > 1:
        return csv
    for folder in (DATA, LOGS):
        log = _from_log(folder / f"{name}.log")
        if log is not None:
            return log
    raise FileNotFoundError(f"no usable record for {name}")


NEW_RUNS = [
    "report_gaps_noRMSNorm_lr1e-3", "report_gaps_noRMSNorm_lr3e-4",
    "report_gaps_bs64_lr2e-2", "report_gaps_bs64_lr3e-2",
    "report_gaps_bs64_lr5e-2", "report_gaps_bs64_lr1e-1",
    "report_gaps_bs64_lr5e-1", "report_gaps_bs64_lr1.0",
    "report_gaps_bs64_lr3.0", "report_gaps_bs64_lr0.5const",
]


def load_new(runs: dict[str, Run]) -> None:
    """Load the follow-up runs, skipping any that are not on disk yet."""
    for name in NEW_RUNS:
        if (DATA / f"{name}.csv").exists():
            runs[name] = load(name)


# --------------------------------------------------------------------------- #
# Figure 1: learning-rate sensitivity
# --------------------------------------------------------------------------- #
def figure_lr_sweep(runs: dict[str, Run]) -> None:
    bs64 = {"1e-4": LIGHTGREY, "3e-4": SKY, "1e-3": BLUE, "3e-3": GREEN, "1e-2": RED}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FULL_W, 2.2))

    for i, (label, colour) in enumerate(bs64.items()):
        run = runs[f"phase1_lr_{label}"]
        ax0.plot(run.steps, run.val, color=colour, ls=STYLES[i], label=f"lr {label}")
    ax0.set_xlabel("Step")
    ax0.set_ylabel("Validation loss")
    ax0.set_title("Batch size 64, 1,000 steps")
    ax0.set_xlim(-20, 1010)
    ax0.set_ylim(1.9, 9.8)
    ax0.legend(loc="upper right", ncol=2, columnspacing=0.9, handlelength=1.5)

    xs64 = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    names64 = ["phase1_lr_1e-4", "phase1_lr_3e-4", "phase1_lr_1e-3",
               "phase1_lr_3e-3", "phase1_lr_1e-2"]
    ys64 = [runs[n].final() for n in names64]
    ax1.plot(xs64, ys64, "o-", color=BLUE, label="BS 64, 1,000 steps")

    bs256_lrs = [5e-4, 1e-3, 2e-3, 2e-3, 3e-3, 4e-3, 6e-3]
    bs256_names = ["phase2_lr_resweep_0.0005", "phase2_lr_resweep_0.001",
                   "phase2_lr_resweep_0.002", "lr_sweep_bs256_2e-3",
                   "lr_sweep_bs256_3e-3", "lr_sweep_bs256_4e-3",
                   "lr_sweep_bs256_6e-3"]
    ys256 = [runs[n].final() for n in bs256_names]
    ax1.plot(bs256_lrs[:3], ys256[:3], "s--", color=GREEN, label="BS 256, 200 steps")
    ax1.plot(bs256_lrs[3:], ys256[3:], "s--", color=GREEN)

    ax1.set_xscale("log")
    ax1.set_xlabel("Learning rate")
    ax1.set_ylabel("Final validation loss")
    ax1.set_title("The optimum depends on the batch size")
    ax1.set_xticks([1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3])
    ax1.set_xticklabels(["1e-4", "3e-4", "1e-3", "2e-3", "3e-3", "4e-3", "6e-3"],
                        fontsize=6.8, rotation=35, ha="right")
    ax1.minorticks_off()
    ax1.set_ylim(2.0, 3.5)
    ax1.legend(loc="lower left")

    fig.tight_layout(pad=0.5)
    _save(fig, "fig_lr_sweep")


# --------------------------------------------------------------------------- #
# Figure 2: batch size at a fixed token budget
# --------------------------------------------------------------------------- #
def figure_batch_size(runs: dict[str, Run]) -> None:
    sizes = [1, 32, 64, 128, 256]
    colours = {1: LIGHTGREY, 32: GREEN, 64: BLUE, 128: ORANGE, 256: RED}

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for i, bs in enumerate(sizes):
        run = runs[f"phase2_bs{bs}"]
        tokens = run.steps * bs * CTX / 1e6
        ax.plot(tokens, run.val, color=colours[bs], ls=STYLES[i], label=f"BS {bs}")

    ax.set_xlabel("Tokens seen (M)")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(0, 8.8)
    ax.legend(loc="upper right", ncol=2, columnspacing=0.7, handlelength=1.2)
    ax.annotate("BS 512: OOM", xy=(8.5, 3.5), xytext=(4.6, 4.75), fontsize=7,
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))

    fig.tight_layout(pad=0.3)
    _save(fig, "fig_batch_size")


# --------------------------------------------------------------------------- #
# Figure 3: the main run
# --------------------------------------------------------------------------- #
def figure_main(runs: dict[str, Run]) -> None:
    main = runs["phase2c_main"]
    ref = runs["phase2_main"]
    tokens = main.steps * 256 * CTX / 1e6

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.plot(tokens, ref.val, "--", color=GREY, lw=1.1, label="val, lr 1e-3")
    ax.plot(tokens, main.train, color=BLUE, lw=0.9, alpha=0.75, label="train, lr 2e-3")
    ax.plot(tokens, main.val, color=BLUE, lw=1.6, label="val, lr 2e-3")

    ax.set_xlabel("Tokens seen (M)")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 335)
    ax.set_ylim(1.3, 4.6)
    ax.legend(loc="upper right", handlelength=1.4)
    ax.annotate(f"{main.final():.4f}", xy=(300, main.final()), xytext=(240, 2.45),
                fontsize=7, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))

    fig.tight_layout(pad=0.3)
    _save(fig, "fig_main_run")


# --------------------------------------------------------------------------- #
# Figure 4: ablations
# --------------------------------------------------------------------------- #
def figure_ablations(runs: dict[str, Run]) -> None:
    baseline = runs["phase1_lr_3e-3"]
    variants = [("phase4_4b", "Post-norm", ORANGE),
                ("phase4_4c", "NoPE", PURPLE),
                ("phase4_4d", "SiLU, $d_{ff}$=2048", GREEN)]
    nonorm = runs["phase4_4a"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FULL_W, 2.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    ax0.plot(baseline.steps, baseline.val, ls=(0, (5, 2)), color=GREY, lw=1.2,
             label="pre-norm SwiGLU (baseline)")
    for i, (name, label, colour) in enumerate(variants):
        run = runs[name]
        ax0.plot(run.steps, run.val, color=colour, ls=STYLES[i + 1], label=label)
    ax0.plot(nonorm.steps, nonorm.val, color=RED, ls="-", label="no RMSNorm")
    ax0.annotate(f"NaN from step {nonorm.first_nan_step()}",
                 xy=(nonorm.last_finite_step(), 9.16), xytext=(300, 7.6),
                 fontsize=7, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
    ax0.set_xlabel("Step")
    ax0.set_ylabel("Validation loss")
    ax0.set_xlim(-30, 2050)
    ax0.set_ylim(1.7, 9.9)
    ax0.legend(loc="center right", handlelength=1.4, labelspacing=0.3)

    matched = 1000
    rows = [("baseline", baseline.at_step(matched), GREY),
            ("no RMSNorm", float("nan"), RED),
            ("post-norm", runs["phase4_4b"].at_step(matched), ORANGE),
            ("NoPE", runs["phase4_4c"].at_step(matched), PURPLE),
            ("SiLU, $d_{ff}$=2048", runs["phase4_4d"].at_step(matched), GREEN)]
    ys = np.arange(len(rows))[::-1]
    ax1.barh(ys, [0 if np.isnan(v) else v for _, v, _ in rows],
             color=[c for _, _, c in rows], height=0.6)
    for y, (label, v, _) in zip(ys, rows):
        if np.isnan(v):
            ax1.text(0.06, y, "NaN", va="center", fontsize=8, color=RED)
        else:
            ax1.text(v + 0.05, y, f"{v:.3f}", va="center", fontsize=7)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([label for label, _, _ in rows], fontsize=7.5)
    ax1.set_xlabel(f"Validation loss at step {matched}")
    ax1.set_xlim(0, 3.1)
    ax1.grid(axis="y", visible=False)

    fig.tight_layout(pad=0.5)
    _save(fig, "fig_ablations")


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(FIGDIR / f'{name}.pdf').relative_to(A1)} (+ .png preview)")


# --------------------------------------------------------------------------- #
# Figure 5: can a lower learning rate rescue the model without RMSNorm?
# --------------------------------------------------------------------------- #
def figure_norm_followup(runs: dict[str, Run]) -> None:
    baseline = runs["phase1_lr_3e-3"]
    cases = [("phase4_4a", "no RMSNorm, lr 3e-3", "-", RED),
             ("report_gaps_noRMSNorm_lr1e-3", "no RMSNorm, lr 1e-3", "-.", ORANGE),
             ("report_gaps_noRMSNorm_lr3e-4", "no RMSNorm, lr 3e-4", ":", GREEN)]

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.plot(baseline.steps, baseline.val, ls=(0, (5, 2)), color=GREY, lw=1.3,
            label="with RMSNorm (baseline)")
    for name, label, style, colour in cases:
        if name not in runs:
            continue
        run = runs[name]
        ax.plot(run.steps, run.val, style, color=colour, lw=1.5, label=label)

    nanning = runs.get("phase4_4a")
    if nanning is not None:
        ax.annotate("NaN", xy=(nanning.last_finite_step(), 9.4), xytext=(60, 8.35),
                    fontsize=7.5, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(-30, 2050)
    ax.set_ylim(1.7, 10.0)
    ax.legend(loc="center right", handlelength=1.5, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_norm_followup")


# --------------------------------------------------------------------------- #
# Figure 6: where training breaks down (log scale: the explosions are the point)
# --------------------------------------------------------------------------- #
def figure_lr_stability(runs: dict[str, Run]) -> None:
    series = [
        ("phase1_lr_3e-3", "lr 3e-3 (optimum)", GREY),
        ("phase1_lr_1e-2", "lr 1e-2", SKY),
        ("report_gaps_bs64_lr1e-1", "lr 1e-1", BLUE),
        ("report_gaps_bs64_lr1.0", "lr 1.0", ORANGE),
        ("report_gaps_bs64_lr3.0", "lr 3.0", RED),
        ("report_gaps_bs64_lr0.5const", "lr 0.5, constant", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for i, (name, label, colour) in enumerate(series):
        if name not in runs:
            continue
        run = runs[name]
        ax.plot(run.steps, run.val, color=colour, ls=STYLES[i], lw=1.2, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(-30, 1030)
    ax.set_ylim(1.4, 2e4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handlelength=1.4, labelspacing=0.25, columnspacing=1.0,
              fontsize=7)
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_lr_stability")


# --------------------------------------------------------------------------- #
def main() -> None:
    names = [
        "phase1_lr_1e-4", "phase1_lr_3e-4", "phase1_lr_1e-3", "phase1_lr_3e-3",
        "phase1_lr_1e-2", "phase1_betas095",
        "phase2_lr_resweep_0.0005", "phase2_lr_resweep_0.001",
        "phase2_lr_resweep_0.002",
        "lr_sweep_bs256_2e-3", "lr_sweep_bs256_3e-3", "lr_sweep_bs256_4e-3",
        "lr_sweep_bs256_6e-3",
        "phase2_bs1", "phase2_bs32", "phase2_bs64", "phase2_bs128", "phase2_bs256",
        "phase2_main", "phase2c_main",
        "phase4_4a", "phase4_4b", "phase4_4c", "phase4_4d",
    ]
    runs = {n: load(n) for n in names}
    load_new(runs)

    figure_lr_sweep(runs)
    figure_batch_size(runs)
    figure_main(runs)
    figure_ablations(runs)
    figure_norm_followup(runs)
    figure_lr_stability(runs)

    print("\n" + "=" * 62)
    print("numbers quoted in the report")
    print("=" * 62)
    print("\n-- LR sweep, batch size 64 (1,000 steps) --")
    for l, n in [("1e-4", "phase1_lr_1e-4"), ("3e-4", "phase1_lr_3e-4"),
                 ("1e-3", "phase1_lr_1e-3"), ("3e-3", "phase1_lr_3e-3"),
                 ("1e-2", "phase1_lr_1e-2")]:
        r = runs[n]
        print(f"  lr {l:>6}: final val {r.final():.4f}  (step {r.steps[-1]:.0f})")
    b = runs["phase1_betas095"]
    print(f"  lr 1e-3 with betas (0.9,0.95), wd 0.1: final val {b.final():.4f}")

    print("\n-- LR resweep, batch size 256 (200 steps) --")
    for n, l in [("phase2_lr_resweep_0.0005", 5e-4), ("phase2_lr_resweep_0.001", 1e-3),
                 ("phase2_lr_resweep_0.002", 2e-3), ("lr_sweep_bs256_2e-3", 2e-3),
                 ("lr_sweep_bs256_3e-3", 3e-3), ("lr_sweep_bs256_4e-3", 4e-3),
                 ("lr_sweep_bs256_6e-3", 6e-3)]:
        r = runs[n]
        print(f"  lr {l:>6}: final val {r.final():.4f}  (step {r.steps[-1]:.0f})")

    print("\n-- batch size at a fixed token budget (lr 1e-3) --")
    for bs in [1, 32, 64, 128, 256]:
        r = runs[f"phase2_bs{bs}"]
        print(f"  bs {bs:>3}: {r.steps[-1] * bs * CTX / 1e6:5.2f} M tokens, "
              f"final val {r.final():.4f}")

    print("\n-- main runs (batch size 256, 5,000 steps) --")
    for n, l in [("phase2_main", 1e-3), ("phase2c_main", 2e-3)]:
        r = runs[n]
        print(f"  lr {l:>6}: final train {r.final('train'):.4f}, "
              f"final val {r.final():.4f}")

    print("\n-- ablations (batch size 64, lr 3e-3, 2,000 steps) --")
    base = runs["phase1_lr_3e-3"]
    print(f"  baseline @1000: {base.at_step(1000):.4f}")
    for n, label in [("phase4_4a", "no RMSNorm"), ("phase4_4b", "post-norm"),
                     ("phase4_4c", "NoPE"), ("phase4_4d", "SiLU d_ff=2048")]:
        r = runs[n]
        nan = r.first_nan_step()
        extra = f", first NaN at step {nan}" if nan else ""
        print(f"  {label:>15}: @1000 {r.at_step(1000):.4f}, "
              f"@2000 {r.final():.4f}{extra}")

    if any(n.startswith("report_gaps") for n in runs):
        print("\n-- follow-up: no RMSNorm at lower learning rates (2,000 steps) --")
        for n, label in [("report_gaps_noRMSNorm_lr1e-3", "lr 1e-3"),
                         ("report_gaps_noRMSNorm_lr3e-4", "lr 3e-4")]:
            r = runs[n]
            print(f"  {label:>8}: @1000 {r.at_step(1000):.4f}, "
                  f"@2000 {r.final():.4f}, min {r.val.min():.4f}")

        print("\n-- follow-up: larger learning rates (batch 64, 1,000 steps) --")
        print(f"  {'lr':>10} {'min':>9} {'max':>10} {'final':>9}")
        for n, label in [("phase1_lr_1e-2", "1e-2"),
                         ("report_gaps_bs64_lr2e-2", "2e-2"),
                         ("report_gaps_bs64_lr3e-2", "3e-2"),
                         ("report_gaps_bs64_lr5e-2", "5e-2"),
                         ("report_gaps_bs64_lr1e-1", "1e-1"),
                         ("report_gaps_bs64_lr5e-1", "5e-1"),
                         ("report_gaps_bs64_lr1.0", "1.0"),
                         ("report_gaps_bs64_lr3.0", "3.0"),
                         ("report_gaps_bs64_lr0.5const", "0.5 const")]:
            if n not in runs:
                continue
            r = runs[n]
            print(f"  {label:>10} {r.val.min():9.4f} {r.val.max():10.2f} "
                  f"{r.final():9.4f}")


if __name__ == "__main__":
    sys.exit(main())
