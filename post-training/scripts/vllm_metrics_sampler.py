"""Sample a running vLLM server's Prometheus metrics into a JSONL timeline.

A single reading says almost nothing: the interesting gauges (requests running,
KV-cache usage, queue depth) are only non-zero *during* a rollout, and the
counters only mean something as differences between two readings. This samples
both, plus the GPU's own utilization, so a step's generation phase can be
correlated with what the inference engine was doing.

The question it exists to answer: sharing the vLLM card with another job took
generation from ~24s to ~67s. A one-off reading already rules out preemption
(`vllm:num_preemptions_total` stayed 0) and prefix-cache thrash (90% hit rate),
leaving raw SM contention -- which this can show as utilisation during
generation vs during training.

    python scripts/vllm_metrics_sampler.py --port 8300 --gpu 0 --duration 7200
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

# Counter/gauge series worth a column each. Names are matched as prefixes so a
# vLLM version that adds a label does not silently drop the series.
WANTED = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
)

LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)$")


def scrape(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        body = response.read().decode("utf-8", "replace")
    values: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        match = LINE.match(line.strip())
        if not match or match.group("name") not in WANTED:
            continue
        try:
            values[match.group("name")] = float(match.group("value"))
        except ValueError:
            pass
    return values


def gpu_utilization(index: int) -> float | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    for line in out.strip().splitlines():
        gpu, util = (part.strip() for part in line.split(","))
        if int(gpu) == index:
            return float(util)
    return None


@app.command()
def main(
    port: int = typer.Option(8300, help="The vLLM server's port."),
    gpu: int = typer.Option(-1, help="Physical index to sample utilisation from; -1 skips it."),
    output: str = typer.Option("runs/_vllm_metrics.jsonl"),
    interval: float = typer.Option(2.0),
    duration: float = typer.Option(3600.0, help="Seconds to sample for."),
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + duration
    samples = 0
    typer.echo(f"sampling port {port} every {interval}s for {duration:.0f}s -> {path}")
    with path.open("w", encoding="utf-8") as handle:
        while time.time() < deadline:
            try:
                record = scrape(port)
            except Exception as error:  # server gone, or a slow scrape
                record = {"error": str(error)[:120]}
            record["wall_time"] = time.time()
            if gpu >= 0:
                record["gpu_util"] = gpu_utilization(gpu)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            samples += 1
            time.sleep(interval)
    typer.echo(f"  {samples} samples")


if __name__ == "__main__":
    app()
