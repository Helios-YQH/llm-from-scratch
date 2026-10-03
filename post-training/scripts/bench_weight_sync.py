"""Price the policy -> vLLM weight sync against the card placement it runs on.

`NCCL_P2P_DISABLE=1` is mandatory on this box, which means the sync cannot go
card-to-card: it is staged through host memory. That puts the NUMA distance
between the training card and the vLLM card directly on the path, and the machine
has two nodes (GPU 0-3 on NUMA 0, GPU 4-5 on NUMA 1, `SYS` across).

This measures one thing: given a vLLM server on a fixed card, how much does the
sync cost when the training process sits on the other node versus the same one.
The 1B policy is pushed whole, which is what a rollout step does.

    python scripts/bench_weight_sync.py --train-gpu 3 --vllm-gpu 4 --syncs 5
"""

from __future__ import annotations

import os
import statistics
import time

import typer

from lm_alignment.checkpoint import get_model_and_tokenizer
from lm_alignment.vllm_utils import VLLMServer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    train_gpu: int = typer.Option(..., help="Physical index of the training card."),
    vllm_gpu: int = typer.Option(..., help="Physical index of the vLLM card."),
    model: str = typer.Option("/mnt/14T/houyi/models/OLMo-2-0425-1B"),
    port: int = typer.Option(8391),
    syncs: int = typer.Option(5),
    gpu_memory_utilization: float = typer.Option(0.15),
) -> None:
    # The trainer only ever sees its own card; the vLLM child re-points itself.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_gpu)

    policy, _ = get_model_and_tokenizer(model, "cuda:0", attn_implementation="sdpa")
    policy.eval()
    n_bytes = sum(p.numel() * p.element_size() for p in policy.parameters())
    typer.echo(f"policy: {n_bytes / 1e9:.2f} GB, train=GPU{train_gpu}, vLLM=GPU{vllm_gpu}")

    server = VLLMServer(
        model_id=model, port=port, gpu=vllm_gpu,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    server.start()
    try:
        t0 = time.perf_counter()
        server.init_weight_sync("cuda:0")
        init_s = time.perf_counter() - t0

        timings = []
        for _ in range(syncs):
            t0 = time.perf_counter()
            server.sync_policy_weights(policy)
            timings.append(time.perf_counter() - t0)

        # A second init would need a second server; the first is the expensive one.
        typer.echo(f"init_weight_sync   : {init_s:6.2f} s")
        typer.echo(f"sync_policy_weights: mean {statistics.mean(timings):5.2f} s  "
                   f"min {min(timings):5.2f}  max {max(timings):5.2f}  n={syncs}")
        typer.echo(f"throughput         : {n_bytes / statistics.mean(timings) / 1e9:5.2f} GB/s")
    finally:
        server.stop()


if __name__ == "__main__":
    app()
