"""Prompting baselines for GSM8K (zero-shot and few-shot templates).

Generates responses with vLLM for each prompt template and reports how the
generations split across the (format, correctness) reward quadrants. Dumps the
full generations so the category-2 and category-3 examples can be inspected by
hand -- the question of how many "wrong" answers are actually correct but
badly parsed can only be answered by reading them.

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_prompting.py \
        --n-examples 1319 --output-dir runs/prompting_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from lm_alignment.data import build_prompts, load_gsm8k, load_prompt_template
from lm_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from lm_alignment.evaluation import grade_rollouts, summarize_rollouts
from lm_alignment.vllm_utils import VLLMServer

app = typer.Typer(add_completion=False)

PROMPTS = ["question_only", "r1_zero", "r1_zero_three_shot_gsm8k"]


@app.command()
def main(
    model: str = typer.Option("allenai/OLMo-2-0425-1B"),
    data_path: str = typer.Option("data/gsm8k/test.jsonl"),
    output_dir: str = typer.Option("runs/prompting_baselines"),
    prompts: str = typer.Option(",".join(PROMPTS), help="Comma-separated prompt names."),
    n_examples: int = typer.Option(1319, help="Number of GSM8K test examples; use a small value to smoke test."),
    temperature: float = typer.Option(1.0),
    max_tokens: int = typer.Option(512),
    gpu: int = typer.Option(0, help="Physical GPU index for the vLLM server."),
    port: int = typer.Option(
        8000, help="Must differ per concurrent run: startup kills other servers on this port."
    ),
    gpu_memory_utilization: float = typer.Option(0.85),
    batch_size: int = typer.Option(64),
    reward_workers: int = typer.Option(16),
    seed: int = typer.Option(0),
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_gsm8k(data_path, limit=n_examples)
    typer.echo(f"loaded {len(examples)} examples from {data_path}")

    server = VLLMServer(
        model_id=model, port=port, gpu=gpu, seed=seed, gpu_memory_utilization=gpu_memory_utilization
    )
    server.start()

    summary = {}
    for prompt_name in [p.strip() for p in prompts.split(",") if p.strip()]:
        template = load_prompt_template(prompt_name)
        reward_fn = question_only_reward_fn if prompt_name == "question_only" else r1_zero_reward_fn
        sampling_params = {"temperature": temperature, "max_tokens": max_tokens, "n": 1, "seed": seed}
        if prompt_name != "question_only":
            # The r1_zero prompts tell the model to close with </answer>, so we
            # can stop there instead of spending the whole 512-token budget.
            sampling_params["stop"] = ["</answer>"]
            sampling_params["include_stop_str_in_output"] = True

        prompt_strs = build_prompts(template, [e["question"] for e in examples])
        completions = server.generate_completions(prompt_strs, sampling_params, batch_size=batch_size)
        if len(completions) != len(examples):
            raise RuntimeError(f"Expected {len(examples)} completions, got {len(completions)}.")
        responses = [c.text for c in completions]
        ground_truths = [e["ground_truth"] for e in examples]
        scores = grade_rollouts(responses, ground_truths, reward_fn, num_workers=reward_workers)

        metrics = summarize_rollouts(scores)
        metrics["response_len_mean"] = sum(len(c.token_ids) for c in completions) / len(completions)
        summary[prompt_name] = metrics

        with (out_dir / f"{prompt_name}.jsonl").open("w", encoding="utf-8") as f:
            for example, response, score, completion in zip(
                examples, responses, scores, completions
            ):
                f.write(
                    json.dumps(
                        {
                            "question": example["question"],
                            "ground_truth": example["ground_truth"],
                            "response": response,
                            "n_tokens": len(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            **score,
                        }
                    )
                    + "\n"
                )
        typer.echo(f"{prompt_name}: { {k: round(v, 4) for k, v in metrics.items()} }")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    typer.echo("\n=== summary ===")
    header = ["correct+f", "formatted only", "unformatted", "accuracy", "format", "len"]
    typer.echo(f"{'prompt':<30}" + "".join(f"{h:>16}" for h in header))
    for prompt_name, m in summary.items():
        row = [
            m["correct_and_formatted"],
            m["formatted_only"],
            m["unformatted"],
            m["accuracy"],
            m["format_rate"],
            m["response_len_mean"],
        ]
        typer.echo(f"{prompt_name:<30}" + "".join(f"{v:>16.4f}" for v in row))


if __name__ == "__main__":
    app()
