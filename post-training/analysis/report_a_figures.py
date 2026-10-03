"""Report A's figure set: RLVR with GRPO, and verifier quality.

Figures are built at their final printed size (5.5in = the text width of a
NeurIPS-style single-column paper) so nothing gets rescaled on the way in.
Values come from `analysis/report_a/<run>/metrics.jsonl`, pulled from the server,
plus V1's stored samples (512 questions x 32 samples) for the TTS curves.

    .venv/Scripts/python.exe analysis/report_a_figures.py           # every figure
    .venv/Scripts/python.exe analysis/report_a_figures.py --only 1 2

Group sizes here are 2 (seeds), so the variant comparisons are dot plots with the
individual runs visible and a mean tick -- never a bare mean bar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer
from matplotlib.lines import Line2D

SKILL = Path(r"C:\Users\15114\.claude\skills\scipilot-figure-skill\scripts")
sys.path.insert(0, str(SKILL))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402

app = typer.Typer(add_completion=False)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "report_a"
FIGS = ROOT / "figs"

# Okabe-Ito: colour-blind safe. Every figure that separates series by colour also
# separates them by line style or marker, so the grayscale preview stays readable.
BLUE, VERM, GREEN, PINK, ORANGE, SKY, YELLOW = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
)
GREY = "0.35"
E1_THREE_SHOT = 0.1827  # E1's 3-shot prompting baseline, the same 1319 questions

_cache: dict[str, list[dict]] = {}


def rows(run: str) -> list[dict]:
    if run not in _cache:
        path = DATA / run / "metrics.jsonl"
        _cache[run] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return _cache[run]


def val_rows(run: str) -> list[dict]:
    return [row for row in rows(run) if "val_accuracy" in row]


def curve(run: str, key: str) -> tuple[list[int], list[float]]:
    return [r["step"] for r in val_rows(run)], [r[key] for r in val_rows(run)]


def final(run: str, key: str = "val_accuracy") -> float:
    return val_rows(run)[-1][key]


def style(width: float = 5.5, height: float = 2.4):
    setup_style(journal="nature", lang="en")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.figsize": (width, height),
        }
    )
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def save(fig, name: str) -> None:
    FIGS.mkdir(exist_ok=True)
    export_figure(fig, basename=str(FIGS / name), formats=["pdf", "png"], dpi=300,
                  size_inches=fig.get_size_inches(), grayscale_preview=True)
    plt.close(fig)
    print(f"  wrote {FIGS / name}.pdf / .png")


def tts_table() -> dict:
    """Best-of-n / pass@n / majority accuracy for every verifier, from V1's samples.

    The 512 x 32 generations were stored during V1, so re-ranking them here costs
    nothing but CPU: the same population of samples is what the RL arms train on.
    Cached next to the data, the pass takes about a minute.
    """
    cache = DATA / "tts_table.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sys.path.insert(0, str(ROOT.parent))
    from lm_alignment.drgrpo_grader import r1_zero_reward_fn
    from lm_alignment.tts_metrics import best_of_n_accuracy, majority_at_k, pass_at_k, read_samples
    from lm_alignment.weak_verifiers import format_only_reward_fn, make_noisy_reward_fn

    records = read_samples(str(DATA / "samples.jsonl"))
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= len(records[0]["responses"])]
    scorers = {
        "perfect": lambda response, truth: r1_zero_reward_fn(response, truth)["reward"],
        "format-only": lambda response, truth: format_only_reward_fn(response, truth)["reward"],
    }
    for epsilon in (0.1, 0.3, 0.5):
        noisy = make_noisy_reward_fn(r1_zero_reward_fn, epsilon=epsilon, seed=0)
        scorers[f"noisy {epsilon}"] = lambda r, t, fn=noisy: fn(r, t)["reward"]

    table = {"ks": ks, "questions": len(records)}
    for name, scorer in scorers.items():
        table[name] = [best_of_n_accuracy(records, k, scorer)[0] for k in ks]
    table["oracle pass@n"] = [pass_at_k(records, k) for k in ks]
    table["majority vote"] = [majority_at_k(records, k) for k in ks]
    cache.write_text(json.dumps(table, indent=2))
    print(f"  cached {cache}")
    return table


def reeval(name: str) -> float:
    """Test-set accuracy of a trained arm, from an E1-protocol re-evaluation.

    Every arm in the V2 comparison is scored on the same 1319 questions with the
    same sampling settings, because the arms' own validation numbers are not
    comparable: the format-only arms once carried their *training* reward into
    validation, and the noisy ones moved between the val and test splits.
    """
    payload = json.loads((DATA / "reeval" / f"{name}.json").read_text())
    return payload["r1_zero_three_shot_gsm8k"]["accuracy"]


def dot_groups(ax, groups: dict[str, list[float]], colors: list[str]) -> None:
    """Individual runs plus a mean tick. n per group is 2, so no bars."""
    for i, (name, vals) in enumerate(groups.items()):
        ax.scatter([i] * len(vals), vals, s=26, color=colors[i % len(colors)],
                   edgecolor="white", linewidth=0.5, zorder=3)
        mean = sum(vals) / len(vals)
        ax.hlines(mean, i - 0.16, i + 0.16, color="0.15", lw=1.4, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), rotation=12, ha="right")


@app.command()
def main(only: str = typer.Option("", help="Figure numbers to build, e.g. '1 2'. Empty = all.")) -> None:
    wanted = {int(n) for n in only.split()} if only.strip() else set(range(1, 9))

    # ---- fig 1: does RL work, and do the seeds agree? -----------------------
    if 1 in wanted:
        fig, ax = style(height=2.4)
        for seed, color in ((0, BLUE), (1, VERM)):
            x, y = curve(f"e3_3shot_seed{seed}", "val_accuracy")
            ax.plot(x, y, color=color, lw=1.3, marker="o", ms=2.4, label=f"seed {seed}, greedy")
            x, y = curve(f"e3_3shot_seed{seed}", "val_sampled_accuracy")
            ax.plot(x, y, color=color, lw=1.0, ls="--", marker="s", ms=2.0, alpha=0.8,
                    label=f"seed {seed}, sampled")
        ax.axhline(E1_THREE_SHOT, color=GREY, ls=":", lw=1.0)
        ax.annotate("3-shot prompting baseline (E1)", xy=(0, E1_THREE_SHOT), xytext=(6, 0.205),
                    color=GREY, fontsize=7.5)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("GSM8K val accuracy")
        ax.set_xlim(-4, 204)
        ax.set_ylim(0.15, 0.60)
        # upper left: the curves only reach 0.5 after step ~25, so the corner is free
        ax.legend(ncol=2, loc="upper left", frameon=False, columnspacing=1.4)
        save(fig, "fig1_e3_curves")

    # ---- fig 2: learning rate -- one diverges -------------------------------
    if 2 in wanted:
        fig, ax = style(height=2.4)
        for i, (lr, color) in enumerate((("3e-06", GREEN), ("1e-05", BLUE), ("3e-05", VERM))):
            x, y = curve(f"e4_lr{lr}_seed0", "val_accuracy")
            ax.plot(x, y, color=color, lw=1.3, ls=("-", "--", "-.")[i], marker=("o", "s", "^")[i],
                    ms=2.4, label=f"lr = {lr.replace('e-0', 'e-').replace('e-', 'e-')}")
        peak_step, peak = max(((r["step"], r["val_accuracy"]) for r in val_rows("e4_lr3e-05_seed0")),
                              key=lambda t: t[1])
        ax.annotate(f"peak {peak:.3f} at step {peak_step}", xy=(peak_step, peak),
                    xytext=(peak_step + 6, peak + 0.006), color=VERM, fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=VERM, lw=0.8))
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("GSM8K val accuracy (greedy)")
        ax.set_xlim(-2, 82)
        ax.legend(ncol=3, loc="lower left", frameon=False)
        save(fig, "fig2_e4_lr")

    # ---- fig 3: the prompt ablation, and its seed spread --------------------
    if 3 in wanted:
        fig, ax = style(height=2.4)
        for seed, color in ((0, BLUE), (1, VERM)):
            x, y = curve(f"e5_question_only_seed{seed}", "val_accuracy")
            ax.plot(x, y, color=color, lw=1.3, marker="o", ms=2.4, label=f"seed {seed}")
        ax.axhline(E1_THREE_SHOT, color=GREY, ls=":", lw=1.0)
        ax.annotate("3-shot prompting baseline (E1)", xy=(38, E1_THREE_SHOT), xytext=(40, 0.186),
                    color=GREY, fontsize=7.5)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("GSM8K val accuracy (greedy)")
        ax.set_xlim(-2, 82)
        ax.set_ylim(-0.005, 0.21)
        ax.legend(loc="upper left", frameon=False)
        save(fig, "fig3_e5_prompt")

    # ---- fig 4: one-change variants vs the standard run ---------------------
    if 4 in wanted:
        variants = {
            "grpo_constant": [final("e6_grpo_constant_seed0"), final("e6_grpo_constant_seed1")],
            "dr_grpo": [final("e6_dr_grpo_seed0"), final("e6_dr_grpo_seed1")],
            "rft": [final("e6_rft_seed0"), final("e6_rft_seed1")],
            "maxrl": [final("e6_maxrl_seed0"), final("e6_maxrl_seed1")],
        }
        standard = final("e4_lr1e-05_seed0")
        fig, ax = style(height=2.2)
        dot_groups(ax, variants, [BLUE, VERM, GREEN, PINK])
        ax.axhline(standard, color=GREY, ls="--", lw=1.0)
        ax.annotate("standard run", xy=(3.45, standard + 0.003), color=GREY, fontsize=7.5)
        ax.set_ylabel("val accuracy at 80 steps")
        ax.set_xlim(-0.5, 4.3)
        ax.set_ylim(0.46, 0.56)
        save(fig, "fig4_e6_variants")

    # ---- fig 5: off-policy -- clipping is what makes it work ----------------
    if 5 in wanted:
        variants = {
            "naive": [final("e7_offpolicy_naive_seed0"), final("e7_offpolicy_naive_seed1")],
            "noclip": [final("e7_offpolicy_noclip_seed0"), final("e7_offpolicy_noclip_seed1")],
            "clip": [final("e7_offpolicy_clip_seed0"), final("e7_offpolicy_clip_seed1")],
            "gspo": [final("e7_offpolicy_gspo_seed0"), final("e7_offpolicy_gspo_seed1")],
        }
        standard = final("e4_lr1e-05_seed0")
        fig, ax = style(height=2.2)
        dot_groups(ax, variants, [PINK, ORANGE, BLUE, GREEN])
        ax.axhline(standard, color=GREY, ls="--", lw=1.0)
        ax.annotate("on-policy reference", xy=(3.45, standard + 0.003), color=GREY, fontsize=7.5)
        ax.set_ylabel("val accuracy at 80 steps")
        ax.set_xlim(-0.5, 4.3)
        ax.set_ylim(0.44, 0.56)
        save(fig, "fig5_e7_offpolicy")

    # ---- fig 6: test-time selection collapses as the verifier degrades ------
    if 6 in wanted:
        table = tts_table()
        ks = table["ks"]
        fig, ax = style(height=2.6)
        ax.plot(ks, table["oracle pass@n"], color="0.45", ls="--", lw=1.2, marker="o", ms=2.5,
                label="oracle pass@n (coverage ceiling)")
        ax.plot(ks, table["majority vote"], color="0.45", ls=":", lw=1.2, marker="v", ms=2.5,
                label="majority vote (no verifier)")
        series = (("perfect", BLUE, "o"), ("noisy 0.1", GREEN, "s"), ("noisy 0.3", ORANGE, "^"),
                  ("noisy 0.5", VERM, "D"), ("format-only", PINK, "P"))
        for name, color, marker in series:
            ax.plot(ks, table[name], color=color, lw=1.3, marker=marker, ms=2.8,
                    label=f"best-of-n, {name} verifier")
        ax.axhline(table["perfect"][0], color="0.75", lw=0.8, zorder=0)
        ax.annotate("no selection (n = 1)", xy=(3.2, 0.172), color="0.45", fontsize=7.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("samples per question (n)")
        ax.set_ylabel(f"GSM8K accuracy ({table['questions']} questions)")
        ax.set_ylim(0.15, 0.88)
        ax.legend(ncol=2, loc="upper left", frameon=False, fontsize=6.8, columnspacing=1.2)
        save(fig, "fig6_v3_tts")

    # ---- fig 7: what a weak verifier costs during training ------------------
    if 7 in wanted:
        # whatever has been re-evaluated so far; the noisy0.3 arms arrive last
        available = lambda name: (DATA / "reeval" / f"{name}.json").exists()
        arms = {}
        for label, names in (("format-only", ("format_only_seed0", "format_only_seed1")),
                             ("noisy 0.1", ("noisy0.1_seed0", "noisy0.1_seed1")),
                             ("noisy 0.3", ("noisy0.3_seed0", "noisy0.3_seed1")),
                             ("exact", ("exact_seed0",))):
            values = [reeval(n) for n in names if available(n)]
            if values:
                arms[label] = values
        fig, ax = style(height=2.3)
        dot_groups(ax, arms, [PINK, GREEN, ORANGE, BLUE])
        ax.axhline(E1_THREE_SHOT, color=GREY, ls=":", lw=1.0)
        ax.annotate("no training (base model)", xy=(len(arms) - 0.55, E1_THREE_SHOT + 0.004),
                    color=GREY, fontsize=7.5)
        ax.set_ylabel("GSM8K test accuracy")
        ax.set_xlim(-0.5, len(arms) + 0.3)
        ax.set_ylim(0.15, 0.50)
        save(fig, "fig7_v2_training")

    # ---- fig 8: the same verifier, two ways of using it ---------------------
    if 8 in wanted:
        table = tts_table()
        n32 = table["ks"].index(32)
        base_tts, ceil_tts = table["perfect"][0], table["oracle pass@n"][n32]
        epsilons = [0.1, 0.3, 0.5]
        tts = {
            e: (table[f"noisy {e}"][n32] - base_tts) / (ceil_tts - base_tts) for e in epsilons
        }
        base_rl, ceil_rl = E1_THREE_SHOT, reeval("exact_seed0")
        rl = {0.1: [(reeval("noisy0.1_seed0") - base_rl) / (ceil_rl - base_rl),
                    (reeval("noisy0.1_seed1") - base_rl) / (ceil_rl - base_rl)]}
        try:
            rl[0.3] = [(reeval("noisy0.3_seed0") - base_rl) / (ceil_rl - base_rl),
                       (reeval("noisy0.3_seed1") - base_rl) / (ceil_rl - base_rl)]
        except FileNotFoundError:
            pass
        fig, ax = style(height=2.4)
        xs = [0.0, 0.1, 0.3, 0.5]
        ax.plot(xs, [1.0] + [1 - 2 * e for e in (0.1, 0.3, 0.5)], color=GREY, ls="--", lw=1.0,
                label="closed form for RL, $(1-2\\epsilon)$ gradient scale")
        tts_line = [1.0] + [tts[e] for e in epsilons]
        ax.plot(xs, tts_line, color=PINK, lw=1.3, marker="P", ms=3.5,
                label="test-time selection (best-of-32)")
        if 0.1 in rl:
            xs_rl = [0.1] + ([0.3] if 0.3 in rl else [])
            for i, x in enumerate(xs_rl):
                ax.scatter([x] * len(rl[x]), rl[x], s=26, color=BLUE, edgecolor="white",
                           linewidth=0.5, zorder=4, label="GRPO (one dot per seed)" if i == 0 else None)
        ax.axhline(1.0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel(r"verifier noise $\epsilon$ (verdict flipped with this probability)")
        ax.set_ylabel("fraction of the attainable gain kept")
        ax.set_xticks(xs)
        ax.set_xlim(-0.03, 0.55)
        ax.set_ylim(-0.08, 1.18)
        ax.legend(loc="lower left", frameon=False, fontsize=7)
        save(fig, "fig8_rl_vs_tts")

    if wanted - {1, 2, 3, 4, 5, 6, 7, 8}:
        print("  nothing to do")


if __name__ == "__main__":
    app()
