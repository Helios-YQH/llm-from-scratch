"""Generate the predictions that AlpacaEval and SimpleSafetyTests are judged on.

Nothing here scores anything. Both benchmarks are judged after the fact by a
Llama-3.3-70B judge -- the `alpaca_eval` CLI for the winrate, `evaluate_safety.py`
for the proportion of safe outputs -- so this script's whole job is to write the
two files those expect:

  alpaca_eval.json   JSON array, one entry per instruction   (supplement 3.3a/5.3a)
  sst.jsonl          JSON lines, one entry per instruction   (supplement 3.4a/5.4a)

Which wrapper goes around the instruction is a flag, and it is not cosmetic: the
supplement prescribes a different one per model.

  zero-shot-system  supplement 3.3/3.4, the base model. The instruction is
                    formatted into the task template and that is inserted into the
                    `# Query:` slot of `zero_shot_system_prompt`, whose code fence
                    is left open -- generation stops at the closing fence.
  alpaca-sft        supplement 5.3/5.4/5.5, the instruction-tuned models. The Alpaca
                    template SFT and DPO were trained on, trailing `### Response:`
                    and all.

    python scripts/eval_alpaca_sst.py --model /mnt/14T/houyi/models/Meta-Llama-3.1-8B \
        --generator-name llama-3.1-8b-base --output-dir runs/_alpaca_base --gpu 0

    python scripts/eval_alpaca_sst.py --model runs/sft/checkpoints/final \
        --prompt-style alpaca-sft --generator-name llama-3.1-8b-sft \
        --output-dir runs/_alpaca_sft --gpu 0
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import typer

from lm_alignment.vllm_utils import VLLMServer

app = typer.Typer(add_completion=False)

PROMPTS_DIR = Path("lm_alignment/prompts_safety")
ALPACA_REFERENCE = Path("data/alpaca_eval/alpaca_eval_gpt4_turbo.json")
SST_DATA = Path("data/simple_safety_tests/simple_safety_tests.csv")

WRAPPERS = {"zero-shot-system": "zero_shot_system_prompt", "alpaca-sft": "alpaca_sft"}
# Both wrappers end mid-document, so generation has to be cut somewhere. The
# zero-shot transcript closes with a ``` fence the model continues; the Alpaca
# template ends at "### Response:" and a runaway continues with the next turn.
STOPS = {"zero-shot-system": ["```"], "alpaca-sft": ["### Instruction:"]}


def load_alpaca_eval() -> list[dict]:
    """The instructions and their provenance, from the reference file.

    That file also holds GPT-4 Turbo's answers; they are dropped here because the
    judge reads them from the reference path itself, and copying them into the
    predictions would just be a second source of truth.
    """
    rows = json.loads(ALPACA_REFERENCE.read_text(encoding="utf-8"))
    return [{"dataset": row["dataset"], "instruction": row["instruction"]} for row in rows]


def load_sst() -> list[dict]:
    with SST_DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_prompts(instructions: list[str], task_template: str, style: str) -> list[str]:
    template = (PROMPTS_DIR / f"{task_template}.prompt").read_text(encoding="utf-8")
    filled = [template.format(instruction=text) for text in instructions]
    wrapper = (PROMPTS_DIR / f"{WRAPPERS[style]}.prompt").read_text(encoding="utf-8")
    if style == "zero-shot-system":
        return [wrapper.format(instruction=text) for text in filled]
    return [wrapper.format(instruction=text, response="") for text in filled]


# task name -> (task template, loader, write-out filename, field holding the instruction)
TASKS = {
    "alpaca_eval": ("alpaca_eval_zero_shot", load_alpaca_eval, "alpaca_eval.json", "instruction"),
    "sst": ("simple_safety_tests_zero_shot", load_sst, "sst.jsonl", "prompts_final"),
}


@app.command()
def main(
    model: str = typer.Option("/mnt/14T/houyi/models/Meta-Llama-3.1-8B"),
    prompt_style: str = typer.Option(
        "zero-shot-system",
        help="zero-shot-system for the base model (supplement 3.3/3.4); alpaca-sft for "
             "the instruction-tuned models (supplement 5.3/5.4).",
    ),
    generator_name: str = typer.Option(
        "",
        help="The `generator` tag AlpacaEval writes for every entry. Defaults to the "
             "basename of --model; give it something readable, the report quotes it.",
    ),
    tasks: str = typer.Option("alpaca_eval,sst", help="Comma-separated subset of the two."),
    output_dir: str = typer.Option("runs/alpaca_sst"),
    max_tokens: int = typer.Option(
        1024,
        help="AlpacaEval is scored against GPT-4 Turbo's answers, which run long; a "
             "tight budget would lose on length alone. The judge's own config allows 7000 "
             "tokens of context, so this is comfortably inside it.",
    ),
    max_model_len: int = typer.Option(
        4096,
        help="Context to reserve KV cache for; Llama-3.1 advertises 131072 and vLLM takes "
             "that literally. Our prompts here are a few hundred tokens.",
    ),
    gpu: int = typer.Option(0, help="Physical GPU index for the vLLM server."),
    port: int = typer.Option(8500),
    gpu_memory_utilization: float = typer.Option(0.85),
    batch_size: int = typer.Option(64),
) -> None:
    if prompt_style not in WRAPPERS:
        raise typer.BadParameter(f"--prompt-style must be one of {', '.join(WRAPPERS)}")
    names = [name.strip() for name in tasks.split(",") if name.strip()]
    for name in names:
        if name not in TASKS:
            raise typer.BadParameter(f"unknown task {name!r}; expected {', '.join(TASKS)}")
    if not generator_name:
        generator_name = Path(model.rstrip("/\\")).name
        typer.echo(f"--generator-name not given; tagging outputs as {generator_name!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    server = VLLMServer(
        model_id=model, port=port, gpu=gpu,
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
    )
    server.start()

    # Greedy, as the supplement prescribes for both benchmarks. `seed` is required
    # by the client even though sampling at temperature 0 has nothing to seed.
    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": 0,
        "stop": STOPS[prompt_style],
    }

    summary = {"model": model, "generator": generator_name, "prompt_style": prompt_style}
    for name in names:
        task_template, loader, filename, field = TASKS[name]
        examples = loader()
        prompts = build_prompts([e[field] for e in examples], task_template, prompt_style)

        started = time.perf_counter()
        completions = server.generate_completions(prompts, sampling, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        outputs = [completion.text for completion in completions]
        if len(outputs) != len(examples):
            raise RuntimeError(f"{name}: {len(outputs)} completions for {len(examples)} prompts")

        if name == "alpaca_eval":
            records = [
                {
                    "dataset": example["dataset"],
                    "instruction": example["instruction"],
                    "output": text,
                    "generator": generator_name,
                }
                for example, text in zip(examples, outputs)
            ]
            (out / filename).write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            # The judge only reads `prompts_final` and `output`, but the CSV's other
            # columns (harm_area, category) are what the error analysis slices by.
            with (out / filename).open("w", encoding="utf-8") as handle:
                for example, text in zip(examples, outputs):
                    handle.write(json.dumps({**example, "output": text}, ensure_ascii=False) + "\n")

        summary[name] = {
            "n": len(outputs),
            "elapsed_s": elapsed,
            "examples_per_s": len(outputs) / elapsed,
            "mean_output_chars": sum(len(text) for text in outputs) / len(outputs),
        }
        typer.echo(
            f"  {name:<12} n={len(outputs):<5} {len(outputs) / elapsed:.2f} ex/s "
            f"({elapsed:.0f}s)  mean {summary[name]['mean_output_chars']:.0f} chars -> {out / filename}"
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"wrote {out}/summary.json")


if __name__ == "__main__":
    app()
