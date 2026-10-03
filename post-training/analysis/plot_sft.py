"""SFT training curve from metrics.jsonl: loss and learning rate, stacked.

Two panels, not two y-scales on one: loss and lr have nothing to do with each
other numerically, and a shared axis would invent a relationship.

Run:  .venv/Scripts/python.exe analysis/plot_sft.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
WINDOW = 25

rows = [json.loads(line) for line in Path("analysis/sft_metrics.jsonl").read_text().splitlines() if line.strip()]
step = np.array([r["step"] for r in rows], dtype=float)
loss = np.array([r["loss"] for r in rows], dtype=float)
lr = np.array([r["learning_rate"] for r in rows], dtype=float)
# The first record divided by `log_every` although only `accumulation` micro-batches
# had accumulated, so it reads 10x low. `repair_sft_metrics.py` fixes the file itself.
loss[step == 1] *= 10

mean = np.convolve(loss, np.ones(WINDOW) / WINDOW, mode="valid")
mean_step = step[WINDOW // 2 : len(mean) + WINDOW // 2]

fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[3, 1], dpi=150
)
fig.patch.set_facecolor(SURFACE)

for ax in (top, bottom):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)

top.plot(step, loss, color=BLUE, linewidth=0.7, alpha=0.30, label=f"per record (every 10 steps)")
top.plot(mean_step, mean, color=BLUE, linewidth=2.0, label=f"{WINDOW}-record mean")
top.set_ylabel("training loss (nats/token)", color=SECONDARY, fontsize=9)
top.set_title(
    "Llama-3.1-8B SFT — packed instruction data, LoRA, 1 epoch",
    color=INK, fontsize=11, loc="left", pad=10,
)
top.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="upper right")

early = mean[mean_step <= 1000].mean()
late = mean[mean_step >= 2000].mean()
top.annotate(
    f"step 250–1000 mean {early:.3f}", xy=(600, early), xytext=(600, early - 0.13),
    color=SECONDARY, fontsize=8, ha="center",
    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7),
)
top.annotate(
    f"step 2000+ mean {late:.3f}\n(Δ {late - early:+.3f})", xy=(2600, late), xytext=(2600, late - 0.20),
    color=SECONDARY, fontsize=8, ha="center",
    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7),
)

bottom.plot(step, lr, color=ORANGE, linewidth=2.0)
bottom.set_ylabel("learning rate", color=SECONDARY, fontsize=9)
bottom.set_xlabel("optimizer step  (effective batch 32 sequences = 6,836 steps/epoch)", color=SECONDARY, fontsize=9)
bottom.yaxis.set_major_formatter(lambda v, _: f"{v:.0e}")

fig.tight_layout()
out = Path("analysis/sft_loss.png")
fig.savefig(out, facecolor=SURFACE)
print(f"  wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
print(f"  records {len(rows)}  steps {int(step[-1])}/{6836}")
print(f"  step 250-1000 mean {early:.4f}   step 2000+ mean {late:.4f}   delta {late - early:+.4f}")
