# transformer-lm

A from-scratch implementation of everything needed to train a Transformer
language model: a byte-level BPE tokenizer, the model itself (RMSNorm, RoPE,
SwiGLU, causal attention), cross-entropy without materializing a softmax, AdamW
with a warmup + cosine schedule, gradient clipping, checkpointing, the data
loader, and a sampling decoder — written without the corresponding PyTorch and
HuggingFace building blocks, and cross-checked against them where a reference
exists.

**Report: [report/tech_report.pdf](report/tech_report.pdf)** — the
implementation, a learning-rate sweep and a batch-size sweep, a main run, four
ablations, generation samples, and the tokenizer's training-cost profile.

## Layout

- `lm_basics/` — the implementation: `tokenizer.py`, `transformer.py` (plus
  `transformer_4a..4d` ablation variants), `optimizer.py`, `loss.py`,
  `dataloader.py`, `checkpoint.py`, `train.py`, `decode.py`
- `tests/` — the reference test suite and the grading adapters
- `reference_impl/` — library-based cross-checks of the same components
  (tiktoken / HuggingFace `tokenizers` / `torch`), used as an independent
  oracle; see its README
- `report/` — the technical report, plus `make_figures.py` (regenerates every
  figure from the run records) and `compute_stats.py` (tokenizer statistics)

## Quickstart

```bash
uv sync
uv run pytest                       # CPU test suite
uv run python -m lm_basics.train --help
```

The tokenizer is bit-exact with the reference on the fixture corpus; a
bit-for-bit comparison against tiktoken's GPT-2 encoding is in
`reference_impl/tokenize_with_tiktoken.py` (it needs the seeded tiktoken cache
from `reference_impl/seed_tiktoken_cache.py` if the download is blocked).
