"""Plot flash vs naive attention benchmark results (Section 4.2).

Parses the flash_bench.log, saves a CSV, and produces log-log scaling plots
comparing FlashAttention vs naive PyTorch attention.
"""
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "memory/flash_bench.log"
CSV = "memory/flash_bench_results.csv"

rows = []
with open(LOG) as f:
    for line in f:
        m = re.match(
            r"\s*(bf16|fp32)\s+(\d+)\s+(\d+)\s*\|\s*([\d.]+|OOM)\s+([\d.]+|OOM)\s+([\d.]+|OOM)\s*\|\s*([\d.]+|OOM)\s+([\d.]+|OOM)\s+([\d.]+|OOM)",
            line,
        )
        if m:
            dtype, d, seq = m.group(1), int(m.group(2)), int(m.group(3))
            vals = [float(x) if x != "OOM" else math.nan for x in m.groups()[3:]]
            rows.append((dtype, d, seq, vals))

with open(CSV, "w") as f:
    f.write("dtype,d,seq,flash_f,flash_b,flash_full,naive_f,naive_b,naive_full\n")
    for dtype, d, seq, vals in rows:
        f.write(f"{dtype},{d},{seq}," + ",".join(f"{v:.3f}" if v == v else "OOM" for v in vals) + "\n")
print(f"saved {len(rows)} rows to {CSV}")


def num(v):
    return v if v == v else math.nan


def plot_for_d(d_target):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, (metric, ylabel) in enumerate(
        [("f", "forward (ms)"), ("b", "backward (ms)"), ("full", "forward+backward (ms)")]
    ):
        ax = axes[idx]
        col_f = 0 if metric == "f" else 1 if metric == "b" else 2
        col_n = 3 if metric == "f" else 4 if metric == "b" else 5
        for dtype in ["bf16", "fp32"]:
            pts_f = [(r[2], r[3][col_f]) for r in rows if r[0] == dtype and r[1] == d_target and not math.isnan(r[3][col_f])]
            pts_n = [(r[2], r[3][col_n]) for r in rows if r[0] == dtype and r[1] == d_target and not math.isnan(r[3][col_n])]
            ax.loglog([p[0] for p in pts_f], [p[1] for p in pts_f], "o-", label=f"flash {dtype}")
            ax.loglog([p[0] for p in pts_n], [p[1] for p in pts_n], "s--", label=f"naive {dtype}")
        ax.set_xlabel("seq (S)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}, d={d_target}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = f"memory/flash_vs_naive_d{d_target}.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)


for d in [16, 64, 128]:
    plot_for_d(d)

# speedup summary (d=64)
print("\n=== speedup (naive/flash), d=64 ===")
print(f"{'dtype':>5} {'seq':>7} {'fwd x':>6} {'full x':>7}")
for dtype in ["bf16", "fp32"]:
    for r in rows:
        if r[0] == dtype and r[1] == 64:
            s, v = r[2], r[3]
            if not math.isnan(v[5]) and not math.isnan(v[2]):
                print(f"{dtype:>5} {s:>7} {v[5]/v[2]:>6.1f} {v[3]/v[0]:>7.1f}")
