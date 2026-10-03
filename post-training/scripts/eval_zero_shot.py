"""Zero-shot baselines: Llama-3.1-8B on MMLU (dev) and GSM8K.

This is the number the instruction-tuned models are measured against. Without it
there is nothing to say about whether SFT and DPO helped or hurt: both are compared
to the base model, not to each other.

Prompts come from `prompts_safety/`: the task template (`mmlu_zero_shot` or
`gsm8k_zero_shot`) is filled in and then inserted into the `{instruction}` slot of a
wrapper, and the result is passed to the model as one flat string (no chat template;
the models here take raw text). Which wrapper is a flag, because the supplement
prescribes different ones for different models:

  zero-shot-system  supplement 3.1, the base model: `zero_shot_system_prompt`,
                    a "# Query:/# Answer:" transcript whose code fence is left open,
                    so generation stops at the closing fence.
  alpaca-sft        supplement 5, the instruction-tuned models: `alpaca_sft`, the
                    template SFT and DPO were trained on -- its trailing
                    `{response}` slot is left empty and generation starts right
                    after "### Response:".

    python scripts/eval_zero_shot.py --model /mnt/14T/houyi/models/Meta-Llama-3.1-8B --gpu 0
    python scripts/eval_zero_shot.py --model runs/sft/checkpoints/final \
        --prompt-style alpaca-sft --output-dir runs/_tax_sft --gpu 3
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import typer

from lm_alignment.data import load_gsm8k
from lm_alignment.evaluation import parse_gsm8k_response, parse_mmlu_response
from lm_alignment.vllm_utils import VLLMServer

app = typer.Typer(add_completion=False)

PROMPTS_DIR = Path("lm_alignment/prompts_safety")
MMLU_DIR = Path("data/mmlu/dev")


WRAPPERS = {"zero-shot-system": "zero_shot_system_prompt", "alpaca-sft": "alpaca_sft"}


def load_templates(style: str) -> tuple[str, dict[str, str]]:
    wrapper = (PROMPTS_DIR / f"{WRAPPERS[style]}.prompt").read_text(encoding="utf-8")
    tasks = {
        name: (PROMPTS_DIR / f"{name}_zero_shot.prompt").read_text(encoding="utf-8")
        for name in ("mmlu", "gsm8k")
    }
    return wrapper, tasks


def load_mmlu(directory: Path) -> list[dict]:
    """Every dev row, with its subject taken from the file name.

    The CSVs have no header and five freely-quoted columns, so they are read with
    `csv.reader` rather than split on commas -- options like "True, True" contain
    commas of their own.
    """
    examples = []
    for path in sorted(directory.glob("*_dev.csv")):
        subject = path.stem.removesuffix("_dev").replace("_", " ")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 6:
                    continue
                question, *options, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": options[:4],
                        "answer": answer.strip(),
                    }
                )
    return examples


@app.command()
def main(
    model: str = typer.Option("/mnt/14T/houyi/models/Meta-Llama-3.1-8B"),
    prompt_style: str = typer.Option(
        "zero-shot-system",
        help="zero-shot-system for the base-model baseline (supplement 3.1); "
             "alpaca-sft for instruction-tuned models (supplement 5).",
    ),
    output_dir: str = typer.Option("runs/zero_shot_baseline"),
    n_gsm8k: int = typer.Option(1319, help="GSM8K test examples; matches the E1 evaluation."),
    max_tokens: int = typer.Option(512),
    max_model_len: int = typer.Option(
        4096,
        help="Context to reserve KV cache for. Llama-3.1 advertises 131072 and vLLM takes "
             "it literally, wanting 16GB of KV cache on top of 16GB of weights before it "
             "will serve anything; our prompts are ~2.5k tokens.",
    ),
    gpu: int = typer.Option(0, help="Physical GPU index for the vLLM server."),
    port: int = typer.Option(8400),
    gpu_memory_utilization: float = typer.Option(0.85),
    batch_size: int = typer.Option(64),
) -> None:
    wrapper, tasks = load_templates(prompt_style)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mmlu = load_mmlu(MMLU_DIR)
    # `load_gsm8k` splits the "#### 18" tail off `answer`; the raw file's key is not
    # `ground_truth`, which is what reading it directly would have assumed.
    gsm8k = load_gsm8k("data/gsm8k/test.jsonl", limit=n_gsm8k)
    typer.echo(f"MMLU {len(mmlu)} questions, GSM8K {len(gsm8k)}")

    jobs = [
        (
            "mmlu",
            [
                wrapper.format(
                    instruction=tasks["mmlu"].format(
                        subject=e["subject"], question=e["question"], options=e["options"]
                    ),
                    response="",
                )
                for e in mmlu
            ],
            mmlu,
        ),
        (
            "gsm8k",
            [
                wrapper.format(instruction=tasks["gsm8k"].format(question=e["question"]), response="")
                for e in gsm8k
            ],
            gsm8k,
        ),
    ]

    server = VLLMServer(
        model_id=model, port=port, gpu=gpu,
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
    )
    server.start()

    summary = {}
    for name, prompts, examples in jobs:
        # Both wrappers end mid-document, so generation has to be cut somewhere or a
        # base model just continues the text. The zero-shot prompt ends with an open
        # ``` fence (without the stop it answers and then repeats the whole prompt
        # back); the Alpaca template ends at "### Response:", and what a runaway
        # continues with is the next turn marker.
        stop = ["```"] if prompt_style == "zero-shot-system" else ["### Instruction:"]
        sampling = {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "n": 1,
            "seed": 0,
            "stop": stop,
        }
        started = time.perf_counter()
        completions = server.generate_completions(prompts, sampling, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        responses = [c.text for c in completions]

        if name == "mmlu":
            predicted = [parse_mmlu_response(e, r) for e, r in zip(examples, responses)]
            correct = [p is not None and p == e["answer"] for p, e in zip(predicted, examples)]
        else:
            predicted = [parse_gsm8k_response(r) for r in responses]
            correct = [p is not None and p == e["ground_truth"] for p, e in zip(predicted, examples)]

        accuracy = sum(correct) / len(correct)
        unparsed = sum(1 for p in predicted if p is None) / len(predicted)
        summary[name] = {
            "n": len(correct),
            "accuracy": accuracy,
            "unparsed_rate": unparsed,
            "elapsed_s": elapsed,
            "examples_per_s": len(prompts) / elapsed,
        }
        typer.echo(
            f"  {name:<6} n={len(correct):<6} accuracy {accuracy:.4f}  unparsed {unparsed:.4f}"
            f"  {len(prompts) / elapsed:.2f} ex/s ({elapsed:.0f}s)"
        )

        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for example, prompt, response, prediction, hit in zip(
                examples, prompts, responses, predicted, correct
            ):
                handle.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "response": response,
                            "predicted": prediction,
                            "correct": hit,
                            "ground_truth": example.get("answer") or example.get("ground_truth"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    app()
