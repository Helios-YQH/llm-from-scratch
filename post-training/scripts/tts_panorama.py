"""Experiment 1: what the base model already reaches with repeated sampling.

No training. Samples `--num-samples` completions per test question, grades them with
the exact grader, and writes the raw responses to disk. `tts_select.py` re-reads that
file with different verifiers, so the same rollouts answer both "how much is
reachable" and "how much of it can a given verifier pick out".

Example:
    python scripts/tts_panorama.py --limit 32 --num-samples 8 --gpu 3   # smoke
    python scripts/tts_panorama.py --num-samples 32 --gpu 3             # full
"""

from __future__ import annotations

from pathlib import Path

import typer

from lm_alignment.data import build_prompts, load_gsm8k, load_prompt_template
from lm_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from lm_alignment.evaluation import grade_rollouts
from lm_alignment.tts_metrics import (
    answer_diversity,
    majority_at_k,
    pass_at_k,
    write_samples,
)
from lm_alignment.vllm_utils import VLLMServer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option("/mnt/14T/houyi/models/OLMo-2-0425-1B"),
    data_path: str = typer.Option("data/gsm8k/test.jsonl"),
    prompt: str = typer.Option("r1_zero"),
    output: str = typer.Option("runs/tts_panorama/samples.jsonl"),
    limit: int = typer.Option(512, help="Questions to sample; use a small value to smoke test."),
    num_samples: int = typer.Option(32, help="Completions per question."),
    temperature: float = typer.Option(1.0),
    max_tokens: int = typer.Option(512),
    request_batch_size: int = typer.Option(8, help="Prompts per HTTP request."),
    gpu: int = typer.Option(0, help="Physical GPU index for the vLLM server."),
    port: int = typer.Option(8020),
    gpu_memory_utilization: float = typer.Option(0.15, help="Set from free memory, not model size."),
    reward_workers: int = typer.Option(16),
    seed: int = typer.Option(0),
) -> None:
    examples = load_gsm8k(data_path, limit=limit)
    template = load_prompt_template(prompt)
    reward_fn = question_only_reward_fn if prompt == "question_only" else r1_zero_reward_fn
    stop_on_answer = prompt != "question_only"
    typer.echo(f"sampling {num_samples} completions for {len(examples)} questions")

    server = VLLMServer(
        model_id=model, port=port, gpu=gpu, seed=seed, gpu_memory_utilization=gpu_memory_utilization
    )
    server.start()

    prompts = build_prompts(template, [e["question"] for e in examples])
    sampling_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": num_samples,
        "seed": seed,
    }
    if stop_on_answer:
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    completions = server.generate_completions(
        prompts, sampling_params, batch_size=request_batch_size
    )
    responses = [c.text for c in completions]
    # The server returns prompt-major order (each prompt's n choices consecutively),
    # which is what makes the regroup below correct. Fail loudly if that ever changes.
    expected = len(prompts) * num_samples
    if len(responses) != expected:
        raise RuntimeError(f"Expected {expected} completions, got {len(responses)}.")

    ground_truths = [e["ground_truth"] for e in examples for _ in range(num_samples)]
    scores = grade_rollouts(responses, ground_truths, reward_fn, num_workers=reward_workers)
    rewards = [s["reward"] for s in scores]
    n_tokens = [len(c.token_ids) for c in completions]

    records = [
        {
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "responses": responses[i * num_samples : (i + 1) * num_samples],
            "rewards": rewards[i * num_samples : (i + 1) * num_samples],
            "n_tokens": n_tokens[i * num_samples : (i + 1) * num_samples],
        }
        for i, example in enumerate(examples)
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_samples(output_path, records)

    typer.echo(f"\n{'k':>4} {'pass@k':>9} {'majority@k':>12} {'answers/question':>17}")
    for k in [k for k in (1, 2, 4, 8, 16, 32) if k <= num_samples]:
        typer.echo(
            f"{k:>4} {pass_at_k(records, k):>9.4f} {majority_at_k(records, k):>12.4f}"
            f" {answer_diversity(records, k):>17.2f}"
        )
    typer.echo(f"\nresponses saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    typer.echo(f"mean response length: {sum(n_tokens) / len(n_tokens):.0f} tokens")


if __name__ == "__main__":
    app()
