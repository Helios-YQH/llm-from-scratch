# post-training

Two post-training research tracks on small open models, each with its own
report:

- **GRPO from scratch** for reasoning RL: OLMo-2-0425-1B on GSM8K, with
  prompting baselines, a learning-rate sweep, a prompt ablation, a controlled
  estimator sweep, an off-policy correction sweep, and a verifier-quality study
  that compares RL training against test-time selection under the same degraded
  verifier.
- **SFT + DPO** (Llama-3.1-8B, LoRA) with capability evaluations (MMLU, GSM8K),
  AlpacaEval under four judges of different capability, and a safety check
  (SimpleSafetyTests).
- `modern_practices/` — engineering notes comparing the hand-rolled components
  (gradient accumulation, logging, response masking, RL infra) against how
  mature frameworks (verl, TRL, OpenRLHF) implement them.

**Reports: [GRPO](report/tech_report.pdf) · [SFT/DPO](report_track3/tech_report.pdf)**

## Layout

- `lm_alignment/` — GRPO, tokenization, DPO, SFT, the GSM8K grader,
  evaluation, run logging, vLLM rollout utilities
- `scripts/` — training drivers, the GPU-queue scheduler, evaluation and
  analysis tools (all paths are set by CLI flags)
- `analysis/` — the figure and statistics scripts behind the GRPO report
- `report/`, `report_track3/` — the two reports
- `tests/` — CPU test suite (snapshot and math tests, plus the adapter layer)

## Quickstart

```bash
# CPU tests (see the note below on the local environment):
python -m pytest tests/

# GPU workflows (vLLM rollout + NCCL weight sync, LoRA SFT/DPO) run on a CUDA
# machine; entry points: scripts/grpo_train.py, scripts/sft_train.py,
# scripts/dpo_train.py — every path (model, data, output) is a flag.
```

Environment note: the project pins specific torch / transformers / tokenizers
versions (see `uv.lock`) because the snapshot tests compare against reference
values at tight tolerances; the GPU extras (vLLM) are only needed for the
rollout path.
