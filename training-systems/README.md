# training-systems

Systems optimizations for Transformer training, measured against the pinned
baseline model in `lm-basics/`: profiling and kernel attribution with Nsight
Systems; mixed precision; gradient checkpointing strategies; a from-scratch
Triton FlashAttention-2 (forward and backward); three DDP variants (naive,
flattened, communication/backward-overlapped); ZeRO-1 optimizer-state sharding;
FSDP; and an analytic derivation of the scaling limits of DP / FSDP / TP.

**Report: [report/tech_report.pdf](report/tech_report.pdf)**

## Layout

- `lm_systems/` — the implementations, plus their benchmark, profiling,
  and accounting scripts and cross-checks against the official PyTorch
  implementations
- `lm-basics/` — the pinned baseline Transformer (vendored, MIT-licensed)
  that the benchmarks apply to, so that measurements are tied to a fixed
  implementation
- `tests/` — correctness tests; `test_attention.py` needs CUDA + Triton, while
  the DDP / FSDP / sharded-optimizer tests run over gloo on CPU
- `report/` — the report and `make_figures.py`

## Quickstart

```bash
uv sync

# CPU (gloo) correctness tests:
uv run pytest tests/test_ddp.py tests/test_fsdp.py tests/test_sharded_optimizer.py

# GPU paths need a CUDA machine (Triton kernels, Nsight Systems, NCCL):
# uv run pytest tests/test_attention.py
# uv run python -m lm_systems.benchmark --help
```
