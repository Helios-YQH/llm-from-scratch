"""Every number the report cites, recomputed from the raw run records.

Transcribing numbers from prose into LaTeX is how a report ends up with a
misquoted result, so nothing goes into `report/tech_report.tex` until it has been
printed by this script. Data: `analysis/report_a/` (the metrics.jsonl files pulled
from the server, the E1 baseline summary, the re-evaluated checkpoints, and the
cached test-time-selection table computed by `report_a_figures.py`).

    .venv/Scripts/python.exe analysis/report_a_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "report_a"


def rows(run: str) -> list[dict]:
    text = (DATA / run / "metrics.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def val(run: str) -> list[dict]:
    return [r for r in rows(run) if "val_accuracy" in r]


def first_last_max(run: str, key: str) -> tuple[float, float, float, int]:
    vs = val(run)
    values = [r[key] for r in vs if key in r]
    steps = [r["step"] for r in vs if key in r]
    peak = max(range(len(values)), key=values.__getitem__)
    return values[0], values[-1], values[peak], steps[peak]


def reeval(name: str) -> dict:
    return json.loads((DATA / "reeval" / f"{name}.json").read_text())["r1_zero_three_shot_gsm8k"]


def line(label: str, value: object) -> None:
    print(f"{label:<44} {value}")


print("=" * 72)
print("E1 prompting baselines (1319 test questions, sampled)")
print("=" * 72)
e1 = json.loads((DATA / "summary.json").read_text())
for name, m in e1.items():
    line(f"{name}: accuracy / format", f"{m['accuracy']:.4f} / {m['format_rate']:.4f}  (len {m['response_len_mean']:.0f})")

print()
print("=" * 72)
print("E3 standard run (200 steps, 2 seeds)")
print("=" * 72)
for seed in (0, 1):
    run = f"e3_3shot_seed{seed}"
    f, l, mx, step = first_last_max(run, "val_accuracy")
    sf, sl, _, _ = first_last_max(run, "val_sampled_accuracy")
    line(f"seed{seed} greedy  first/last/max", f"{f:.4f} / {l:.4f} / {mx:.4f} @{step}")
    line(f"seed{seed} sampled first/last", f"{sf:.4f} / {sl:.4f}")
    v = val(run)
    line(f"seed{seed} format first/last", f"{v[0]['val_format_rate']:.4f} / {v[-1]['val_format_rate']:.4f}")
    line(f"seed{seed} response len first/last", f"{v[0]['val_response_len_mean']:.0f} / {v[-1]['val_response_len_mean']:.0f}")
    line(f"seed{seed} kept first/last", f"{v[0]['n_sequences_kept']:.0f} / {v[-1]['n_sequences_kept']:.0f}")
    line(f"seed{seed} entropy (train, last)", f"{rows(run)[-1]['token_entropy']:.3f}")

print()
print("=" * 72)
print("E4 learning-rate sweep (80 steps, seed 0)")
print("=" * 72)
for lr in ("3e-06", "1e-05", "3e-05"):
    run = f"e4_lr{lr}_seed0"
    f, l, mx, step = first_last_max(run, "val_accuracy")
    sf, sl, _, _ = first_last_max(run, "val_sampled_accuracy")
    line(f"lr {lr} greedy first/last/peak", f"{f:.4f} / {l:.4f} / {mx:.4f} @{step}")
    line(f"lr {lr} sampled last | entropy last", f"{sl:.4f} | {rows(run)[-1]['token_entropy']:.3f}")

print()
print("=" * 72)
print("E5 question_only ablation (80 steps, 2 seeds)")
print("=" * 72)
for seed in (0, 1):
    run = f"e5_question_only_seed{seed}"
    f, l, mx, step = first_last_max(run, "val_accuracy")
    line(f"seed{seed} greedy first/last", f"{f:.4f} / {l:.4f}")
    v = val(run)
    line(f"seed{seed} boxed-rate proxy first/last", f"{v[0]['format_reward_mean']:.3f} / {v[-1]['format_reward_mean']:.3f}")
    line(f"seed{seed} kept first/last", f"{v[0]['n_sequences_kept']:.0f} / {v[-1]['n_sequences_kept']:.0f}")

print()
print("=" * 72)
print("E6 estimator variants (80 steps, 2 seeds) -- final greedy")
print("=" * 72)
for variant in ("grpo_constant", "dr_grpo", "rft", "maxrl"):
    vals = [val(f"e6_{variant}_seed{s}")[-1]["val_accuracy"] for s in (0, 1)]
    line(variant, f"{vals[0]:.4f} / {vals[1]:.4f}")
line("standard run (e4_lr1e-05_seed0)", f"{val('e4_lr1e-05_seed0')[-1]['val_accuracy']:.4f}")

# The dispersion the paper quotes: how far the variants' two-seed means are
# apart, against how far the two seeds of one variant are apart. Both are
# computed from the same final-greedy values printed above.
e6 = {v: [val(f"e6_{v}_seed{s}")[-1]["val_accuracy"] for s in (0, 1)]
      for v in ("grpo_constant", "dr_grpo", "rft", "maxrl")}
e6_means = {v: sum(x) / 2 for v, x in e6.items()}
line("two-seed means span",
     f"{max(e6_means.values()) - min(e6_means.values()):.4f} "
     f"({min(e6_means.values()):.3f}..{max(e6_means.values()):.3f})")
e6_gaps = [abs(x[0] - x[1]) for x in e6.values()]
line("per-variant seed gaps", f"{min(e6_gaps):.4f}..{max(e6_gaps):.4f}")
line("standard run vs dr_grpo", " / ".join(
    f"{x - val('e4_lr1e-05_seed0')[-1]['val_accuracy']:+.4f}" for x in e6["dr_grpo"]))

print()
print("=" * 72)
print("E7 off-policy corrections (80 steps, 2 seeds) -- final greedy")
print("=" * 72)
for variant in ("naive", "noclip", "clip", "gspo"):
    vals = [val(f"e7_offpolicy_{variant}_seed{s}")[-1]["val_accuracy"] for s in (0, 1)]
    line(variant, f"{vals[0]:.4f} / {vals[1]:.4f}")
line("on-policy reference (e4_lr1e-05_seed0)", f"{val('e4_lr1e-05_seed0')[-1]['val_accuracy']:.4f}")

print()
print("=" * 72)
print("V1/V3 test-time selection (512 questions x 32 samples)")
print("=" * 72)
tts = json.loads((DATA / "tts_table.json").read_text())
line("ks", tts["ks"])
for key in ("oracle pass@n", "perfect", "majority vote", "noisy 0.1", "noisy 0.3", "noisy 0.5", "format-only"):
    line(key, " ".join(f"{v:.4f}" for v in tts[key]))
base, ceil = tts["perfect"][0], tts["oracle pass@n"][-1]
for eps in ("0.1", "0.3", "0.5"):
    kept = (tts[f"noisy {eps}"][-1] - base) / (ceil - base)
    line(f"retained at eps={eps} (best-of-32)", f"{kept:.3f}")
line("retained, format-only (best-of-32)", f"{(tts['format-only'][-1] - base) / (ceil - base):.3f}")

print()
print("=" * 72)
print("V2 training with weak verifiers (1319 test questions, greedy re-eval)")
print("=" * 72)
base = e1["r1_zero_three_shot_gsm8k"]["accuracy"]
ceil = reeval("exact_seed0")["accuracy"]
line("base (E1 3-shot baseline)", f"{base:.4f}")
line("exact (upper endpoint)", f"{ceil:.4f}")
for name in ("noisy0.1_seed0", "noisy0.1_seed1", "format_only_seed0", "format_only_seed1"):
    m = reeval(name)
    kept = (m["accuracy"] - base) / (ceil - base)
    line(f"{name}: acc / fmt / retained", f"{m['accuracy']:.4f} / {m['format_rate']:.4f} / {kept:.3f}")
pending = [n for n in ("noisy0.3_seed0", "noisy0.3_seed1") if (DATA / "reeval" / f"{n}.json").exists()]
if pending:
    for name in pending:
        m = reeval(name)
        kept = (m["accuracy"] - base) / (ceil - base)
        line(f"{name}: acc / fmt / retained", f"{m['accuracy']:.4f} / {m['format_rate']:.4f} / {kept:.3f}")
else:
    line("noisy0.3 (both seeds)", "PENDING -- runs finishing / re-eval queued")

print()
print("=" * 72)
print("Matched step (60) on the internal validation split, exact grader")
print("=" * 72)
# The 30%-flip arm never got its test-set re-evaluation (the cluster died), so the
# only comparison it can join is the internal validation split at a step every run
# reached -- step 60, the last validation point of one of its seeds. Same split,
# same grader, same step: a valid comparison, but a different protocol from the
# test-set table, and it says so where it is quoted.
val60 = {name: next(r["val_accuracy"] for r in val(run) if r["step"] == 60)
         for name, run in (("exact", "e4_lr1e-05_seed0"),
                           ("noisy0.1_seed1", "track2_veritas_noisy0.1_seed1"),
                           ("noisy0.3_seed0", "track2_veritas_noisy0.3_seed0"),
                           ("noisy0.3_seed1", "track2_veritas_noisy0.3_seed1"))}
base60 = val("e4_lr1e-05_seed0")[0]["val_accuracy"]  # every arm starts from the base model
for name, value in val60.items():
    line(f"val@60 {name}", f"{value:.4f}")
line("val@60 base (step 0 of the same run)", f"{base60:.4f}")
for name, value in val60.items():
    line(f"retained at step 60, {name}", f"{(value - base60) / (val60['exact'] - base60):.3f}")

print()
print("=" * 72)
print("Cross-protocol note")
print("=" * 72)
line("exact arm, internal val (1024 q)", f"{val('e4_lr1e-05_seed0')[-1]['val_accuracy']:.4f}")
line("exact arm, test (1319 q)", f"{ceil:.4f}")
