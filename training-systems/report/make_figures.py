"""Regenerate the report figures as vector PDFs (PNG copies are throwaway
previews for eyeballing in an image viewer).

Data sources:
  * measurements recorded in ../experiment_results.md (hard-coded below)
  * ../memory/flash_bench_results.csv           (attention sweep, 80 configs)
  * ../nsys_reports/ddp_{naive,overlap}.sqlite  (Nsight exports, kernel spans)

Sizes follow the NeurIPS layout conventions used in the transformer-lm report:
5.5in full text width, 8pt type, STIX fonts to match the Times body text.

Usage (from training-systems/):
    uv run python report/make_figures.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SRC = ROOT.parent
NSYS = SRC / "nsys_reports"
FLASH_CSV = SRC / "memory" / "flash_bench_results.csv"

FULL_W, COL_W = 5.5, 2.65

# Okabe-Ito colour-blind-safe palette (same as the transformer-lm report)
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, GREY, LIGHTGREY = "#CC79A7", "#56B4E9", "#8C8C8C", "#BFBFBF"

plt.rcParams.update({
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
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}.pdf")


# ------------------------------------------------------- 1. mixed precision --
def fig_mixed_precision() -> None:
    models = ["small\n128M", "medium\n423M", "large\n969M", "xl\n3.4B"]
    fwd_fp32 = [0.0506, 0.1541, 0.3104, 0.9117]
    fwd_bf16 = [0.0276, 0.0732, 0.1384, 0.2868]
    fb_fp32 = [0.1575, 0.4652, 0.9307, np.nan]
    fb_bf16 = [0.0857, 0.2316, 0.4469, np.nan]

    x = np.arange(len(models))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.1))

    for ax, (a, b, title) in zip(axes, [
        (fwd_fp32, fwd_bf16, "forward"),
        (fb_fp32, fb_bf16, "forward + backward"),
    ]):
        ax.bar(x - w / 2, a, w, label="FP32", color=RED)
        ax.bar(x + w / 2, np.nan_to_num(b), w, label="BF16 autocast", color=BLUE)
        top = max(np.nanmax(a), np.nanmax(b))
        for i in range(len(models)):
            if np.isnan(b[i]):
                ax.text(x[i], 0.03, "OOM", ha="center", va="bottom",
                        fontsize=7.5, color=GREY, style="italic")
                continue
            ax.text(x[i], max(a[i], b[i]) + 0.025 * top, f"{a[i] / b[i]:.2f}$\\times$",
                    ha="center", va="bottom", fontsize=7.5, color=GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("time (s)")
        ax.set_title(title)
        ax.set_ylim(0, top * 1.18)
        ax.legend(loc="upper left")

    save(fig, "mixed_precision")


# --------------------------------------------------------- 2. memory scaling --
def fig_memory_scaling() -> None:
    ctx = [128, 256, 512, 1024, 2048]
    fwd_gib = [3285 / 1024, 5331 / 1024, 10574 / 1024, 25680 / 1024]
    full_ctx = [128, 256, 512]
    full_gib = [6701 / 1024, 8691 / 1024, 14052 / 1024]

    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    ax.plot(ctx[:4], fwd_gib, "o-", color=RED, label="forward peak")
    ax.plot(full_ctx, full_gib, "s-", color=BLUE, label="full training-step peak")

    # dashed extrapolation at the measured ~4.1x per-doubling growth rate,
    # up to the configuration where the run actually OOMs
    fwd_oom = fwd_gib[3] * 4.1
    full_oom = full_gib[2] * 4.1
    ax.plot([1024, 2048], [fwd_gib[3], fwd_oom], "--", color=RED, lw=1)
    ax.plot([512, 1024], [full_gib[2], full_oom], "--", color=BLUE, lw=1)
    ax.plot([2048], [fwd_oom], "x", color=RED, ms=7, mew=1.8)
    ax.plot([1024], [full_oom], "x", color=BLUE, ms=7, mew=1.8)
    ax.annotate("OOM", (2048, fwd_oom), textcoords="offset points",
                xytext=(-7, 5), ha="right", fontsize=7.5, color=RED, style="italic")
    ax.annotate("OOM", (1024, full_oom), textcoords="offset points",
                xytext=(-7, 5), ha="right", fontsize=7.5, color=BLUE, style="italic")

    for xi, yi in zip(ctx[:4], fwd_gib):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(-4, 5), ha="right", fontsize=7, color=RED)
    for xi, yi in zip(full_ctx, full_gib):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(5, -2), ha="left", fontsize=7, color=BLUE)

    ax.axhline(48, color=GREY, ls=":", lw=1)
    ax.text(132, 50, "A6000 limit --- 48 GiB", fontsize=7, color=GREY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx)
    ax.set_xticklabels([str(c) for c in ctx])
    ax.set_xlabel("context length")
    ax.set_ylabel("peak GPU memory (GiB)")
    ax.set_ylim(0, 122)
    ax.legend(loc="upper left")

    save(fig, "memory_scaling")


# ----------------------------------------------------- 3. checkpointing ----
def fig_checkpointing() -> None:
    labels = ["none", "all", "4-layer", "2-layer", "1-layer", "nested"]
    data = {
        "8 layers": [3609.6, 4077.6, 2611.1, 1889.8, 1553.1, 1521.1],
        "16 layers": [7070.7, 8050.9, 3651.2, 2945.9, 2641.3, 2553.3],
    }
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2), sharey=True)

    for ax, (title, vals) in zip(axes, data.items()):
        base = vals[0]
        colors = [GREY, RED, BLUE, BLUE, GREEN, GREEN]
        ax.bar(x, vals, color=colors, width=0.62)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 90, f"{v / base:.2f}$\\times$", ha="center", va="bottom",
                    fontsize=7, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 9600)

    axes[0].set_ylabel("peak memory (MiB)")
    save(fig, "checkpointing")


# ------------------------------------------------------ 4. DDP variants -----
def fig_ddp_variants() -> None:
    variants = ["naive", "flat", "overlap"]
    panels = [
        ("medium --- 2$\\times$A6000 --- batch 8",
         [345.7, 344.5, 153.9], [788.2, 790.4, 680.0]),
        ("large --- 4$\\times$A6000 --- batch 16",
         [2221.2, 2266.8, 1619.4], [3200.3, 3249.9, 2761.5]),
    ]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.3))

    for ax, (title, comm, total) in zip(axes, panels):
        compute = [t - c for t, c in zip(total, comm)]
        ax.bar(x, compute, 0.55, label="compute", color=BLUE)
        ax.bar(x, comm, 0.55, bottom=compute, label="gradient communication", color=RED)
        for xi, (c, t) in zip(x, zip(comm, total)):
            ax.text(xi, t + max(total) * 0.02, f"{100 * c / t:.0f}%",
                    ha="center", va="bottom", fontsize=8,
                    color=RED if c / t > 0.4 else GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_title(title)
        ax.set_ylim(0, max(total) * 1.12)
    axes[0].set_ylabel("time per step (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    save(fig, "ddp_variants")


# ------------------------------------------------ 5. nsys kernel timelines ---
def _kernel_spans(db: str, device: int = 0, window_s: float = 3.2):
    """Load one device's kernel spans for a `window_s` window starting at the
    first NCCL all-reduce (i.e. the moment the first gradient becomes ready)."""
    con = sqlite3.connect(NSYS / db)
    cur = con.cursor()
    t0 = cur.execute(
        """SELECT MIN(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.shortName = s.id
           WHERE k.deviceId = ? AND s.value LIKE 'ncclDevKernel_AllReduce%'""",
        (device,),
    ).fetchone()[0]
    rows = cur.execute(
        """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.shortName = s.id
           WHERE k.deviceId = ? AND k.start >= ? AND k.start < ?""",
        (device, t0, t0 + int(window_s * 1e9)),
    )
    comm, comp = [], []
    for start, end, name in rows:
        span = ((start - t0) / 1e6, max((end - start) / 1e6, 0.004))
        (comm if name.startswith("ncclDevKernel") else comp).append(span)
    con.close()
    return comm, comp


def _busy_mask(spans, window_ms: float, step: float = 1.0):
    """Boolean per-millisecond occupancy mask for a set of (start, dur) spans."""
    n = int(window_ms / step)
    mask = np.zeros(n, dtype=bool)
    for s, w in spans:
        i0 = max(0, int(s / step))
        i1 = min(n, max(i0 + 1, int(np.ceil((s + w) / step))))
        mask[i0:i1] = True
    return mask


def fig_nsys_timeline() -> None:
    window_s = 3.2
    window_ms = window_s * 1000
    panels = [
        ("ddp_naive.sqlite", "naive DDP", RED),
        ("ddp_overlap.sqlite", "overlapped DDP", GREEN),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.0), sharex=True)

    for ax, (db, label, color) in zip(axes, panels):
        comm, comp = _kernel_spans(db, window_s=window_s)
        ax.broken_barh(comp, (0.08, 0.84), facecolors=BLUE, linewidth=0)
        ax.broken_barh(comm, (1.10, 0.84), facecolors=color, linewidth=0)
        cm, xm = _busy_mask(comp, window_ms), _busy_mask(comm, window_ms)
        overlapped = (cm & xm).sum() / max(xm.sum(), 1)
        ax.set_title(f"{label}   (communication overlapped with compute: {overlapped:.0%})",
                     loc="left", fontsize=8)
        ax.set_ylim(0, 2.02)
        ax.set_yticks([0.5, 1.52])
        ax.set_yticklabels(["compute", "NCCL"], fontsize=7.5)
        ax.grid(axis="y", visible=False)

    axes[1].set_xlabel("time from the window's first all-reduce (ms)")
    axes[1].set_xlim(0, window_ms)
    save(fig, "nsys_timeline")


# --------------------------------------------------- 6. memory hierarchy ----
def fig_memory_hierarchy() -> None:
    methods = ["DDP\n(official)", "ZeRO-1\n(official)", "FSDP\n(ours)", "FSDP\n(official)"]
    steady = [982.9, 492.2, 280.8, 246.1]
    peak = [5637.6, 4652.7, 4442.4, 4894.2]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    ax = axes[0]
    ax.bar(x, steady, 0.55, color=BLUE)
    for xi, v in zip(x, steady):
        ax.text(xi, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("steady-state memory")
    ax.set_ylim(0, 1250)

    ax = axes[1]
    ax.bar(x, peak, 0.55, color=ORANGE)
    for xi, v in zip(x, peak):
        ax.text(xi, v + 90, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("peak memory")
    ax.set_ylim(0, 6900)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=7)
    axes[0].set_ylabel("memory (MiB)")
    save(fig, "memory_hierarchy")


# --------------------------------- 7. forward memory & kernel breakdown -----
def fig_breakdowns() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.1))

    sources = ["softmax ($S\\times S$)", "attention matmuls",
               "model layers\n(norm/RoPE/res.)", "attention misc", "SwiGLU"]
    vals = [5386.5, 4478.6, 3495.2, 1536.0, 768.0]
    y = np.arange(len(sources))[::-1]
    ax1.barh(y, vals, 0.6, color=[RED, RED, BLUE, GREY, GREY])
    for yi, v in zip(y, vals):
        ax1.text(v + 90, yi, f"{v / 1024:.1f}", ha="left", va="center", fontsize=7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(sources, fontsize=7)
    ax1.set_xlabel("cumulative allocation (MiB)")
    ax1.set_xlim(0, 7100)
    ax1.set_title("(a) forward allocation by source", fontsize=8.5)

    groups = ["small\nforward", "small\nfull", "large\nforward", "large\nfull"]
    matmul = [59.9, 27.2, 68.4, 21.8]
    other = [100 - m for m in matmul]
    x = np.arange(len(groups))
    ax2.bar(x, matmul, 0.55, color=RED, label="matmul")
    ax2.bar(x, other, 0.55, bottom=matmul, color=BLUE, label="everything else")
    for xi, m in zip(x, matmul):
        ax2.text(xi, 101, f"{m:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=7)
    ax2.set_ylim(0, 145)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("share of GPU kernel time (%)")
    ax2.set_title("(b) matmul share of kernel time", fontsize=8.5)
    ax2.legend(loc="upper center", ncol=2)

    save(fig, "breakdowns")


# ------------------------------------------------- 8. flash vs naive sweep --
def _load_flash() -> list[dict]:
    """Parse the attention sweep log. The naive fp32 64K cells contain the
    literal 'OOM', which any numeric parser turns into a NaN column."""
    import csv

    out = []
    with open(FLASH_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rec = {"dtype": row["dtype"], "d": int(row["d"]), "seq": int(row["seq"])}
            for key in ("flash_f", "flash_b", "flash_full",
                        "naive_f", "naive_b", "naive_full"):
                v = row[key].strip()
                rec[key] = float("nan") if v.upper() == "OOM" else float(v)
            out.append(rec)
    return out


def fig_flash_sweep() -> None:
    recs = _load_flash()
    d64 = [r for r in recs if r["d"] == 64]
    series = [
        ("flash bf16", "flash", "bf16", BLUE, "-", "o"),
        ("naive bf16", "naive", "bf16", ORANGE, "--", "s"),
        ("flash fp32", "flash", "fp32", GREEN, "-", "o"),
        ("naive fp32", "naive", "fp32", RED, "--", "s"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9))
    for ax, (kind, title) in zip(axes, [("f", "forward"), ("b", "backward"),
                                        ("full", "forward+backward")]):
        for label, prefix, prec, color, ls, mk in series:
            pts = [(r["seq"], r[f"{prefix}_{kind}"])
                   for r in d64 if r["dtype"] == prec]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls,
                    marker=mk, color=color, label=label, ms=3, lw=1.2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("seq ($S$)")
        ax.set_xticks([128, 1024, 8192, 65536])
        ax.set_xticklabels(["128", "1K", "8K", "64K"])
    axes[0].set_ylabel("time (ms)")
    axes[0].legend(loc="upper left", fontsize=7)
    save(fig, "flash_sweep_d64")


# ------------------------------------------------------------------ main ----
def main() -> None:
    fig_mixed_precision()
    fig_memory_scaling()
    fig_checkpointing()
    fig_ddp_variants()
    fig_nsys_timeline()
    fig_memory_hierarchy()
    fig_breakdowns()
    fig_flash_sweep()


if __name__ == "__main__":
    main()
