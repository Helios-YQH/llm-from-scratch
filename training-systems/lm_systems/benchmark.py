"""End-to-end benchmarking of the basics Transformer.

Times the forward pass, backward pass, and optimizer step on random data.
CUDA kernels are asynchronous, so we call torch.cuda.synchronize() after each
step to measure actual GPU execution time rather than CPU scheduling time.
"""
import argparse
import statistics
import timeit
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

from lm_basics.model import BasicsTransformerLM
from lm_basics.nn_utils import cross_entropy
from lm_basics.optimizer import AdamW

# PDF Table 1: vocab_size=10000, batch_size=4, context_length=512 by default.
MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def nvtx_range(name: str):
    """Emit an NVTX range on CUDA builds (for Nsight Systems), no-op otherwise.

    NVTX is CUDA-only: the CPU-only torch build raises RuntimeError when asked
    to push a range. Falling back to nullcontext keeps the script runnable
    locally for debugging.
    """
    if torch.cuda.is_available():
        return nvtx.range(name)
    return nullcontext()


def build_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    config = MODEL_CONFIGS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    model.to(device)
    return model


def get_batch(args: argparse.Namespace, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    return inputs, targets


def one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    device: str,
):
    # Only the forward pass uses autocast; backward and optimizer.step() use
    # the FP32 master weights (the standard mixed-precision recipe).
    autocast = (
        torch.autocast(device_type=device, dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
    )
    # NVTX ranges make the forward/backward/optimizer phases distinguishable
    # in Nsight Systems (e.g. to ignore warm-up steps via --nvtx-capture).
    with autocast, nvtx_range("forward"):
        logits = model(inputs)

    if args.mode == "forward":
        return

    with nvtx_range("loss+backward"):
        loss = cross_entropy(logits, targets)
        loss.backward()

    if args.mode == "full":
        with nvtx_range("optimizer"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the basics Transformer.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), default="small")
    parser.add_argument("--mode", choices=["forward", "forward_backward", "full"], default="forward")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--memory-profile", action="store_true", help="Record CUDA memory history and dump a snapshot (one step)")
    parser.add_argument("--memory-snapshot", type=str, default=None, help="Path for the memory snapshot pickle")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"device: {device} | model: {args.model_size} | mode: {args.mode} | precision: {args.precision}")

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr) if args.mode == "full" else None
    inputs, targets = get_batch(args, device)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    # Warm up: initialize CUDA context, load kernels, and let cuDNN pick
    # algorithms. These steps are not timed. The NVTX label lets nsys isolate
    # (or exclude) warm-up steps with --nvtx-capture.
    for _ in range(args.warmup_steps):
        with nvtx_range("warmup"):
            one_step(model, optimizer, inputs, targets, args, device)
    synchronize()

    # Memory profiling: record one step's allocation history and dump a
    # snapshot loadable at pytorch.org/memory_viz.
    if args.memory_profile:
        if device != "cuda":
            print("memory profiling requires CUDA")
            return
        steady = torch.cuda.memory_allocated() / (1024**2)
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        one_step(model, optimizer, inputs, targets, args, device)
        synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"steady memory (model + optimizer state): {steady:.1f} MiB")
        print(f"peak allocated during step: {peak:.1f} MiB")
        path = args.memory_snapshot or f"memory_snapshot_{args.model_size}_{args.mode}_{args.context_length}.pickle"
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"memory snapshot dumped to {path}")
        return

    times = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        one_step(model, optimizer, inputs, targets, args, device)
        synchronize()
        times.append(timeit.default_timer() - start)

    mean = statistics.fmean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"{args.mode} over {args.measure_steps} steps: {mean:.4f}s +/- {stdev:.4f}s")


if __name__ == "__main__":
    main()
