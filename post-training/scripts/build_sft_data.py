"""Build the single-turn SFT dataset: UltraChat-200K + SafetyTunedLlamas.

The dataset used here ships pre-built with no published construction recipe,
so every rule below is **our own inference**. They are all stated here so the
report can disclose them:

1. **UltraChat**: one example per conversation, from the *first* user turn only.
   Later turns are follow-ups that read as nonsense without their history, and
   keeping the history would make the example multi-turn, contradicting the
   `single_turn` name.
2. **SafetyTunedLlamas**: `saferpaca_Instructions_{X}.json`, which is 20k Alpaca
   instructions plus X safety examples. We take X = 2000, i.e. all of the safety
   data -- "use everything available" is the least arbitrary of the six sizes the
   repo ships (100/300/500/1000/1500/2000). UltraChat outnumbers the pool ~10:1
   either way, so this knob barely moves the mixture; what it controls is how
   much safety signal the model sees.
3. **Split**: the UltraChat side uses its own official train/test split. The
   Alpaca+safety pool is shuffled with a fixed seed and 5% held out for test,
   so both splits cover both distributions.
4. **Raw text, no template.** `sft.get_packed_sft_dataset` applies
   `prompts_safety/alpaca_sft.prompt` at load time. That template has only
   `{instruction}` and `{response}`, so Alpaca's `input` column is folded into
   the prompt by hand.

Output: gzipped JSONL, one `{"prompt": ..., "response": ...}` per line.

Usage (server, from the repo root):
    .venv/bin/python scripts/build_sft_data.py
"""

from __future__ import annotations

import gzip
import json
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq
import typer

app = typer.Typer(add_completion=False)

RAW_DIR = Path("data/_raw")
STL_NAME = "safety-tuned-llamas-main"


def ultrachat_first_turns(files: list[Path]) -> list[dict[str, str]]:
    """First (user, assistant) exchange of every conversation."""
    examples: list[dict[str, str]] = []
    skipped = 0
    for path in sorted(files):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                    skipped += 1
                    continue
                prompt = messages[0]["content"].strip()
                response = messages[1]["content"].strip()
                if not prompt or not response:
                    skipped += 1
                    continue
                examples.append({"prompt": prompt, "response": response})
    if skipped:
        typer.echo(f"    skipped {skipped} conversations without a usable first turn")
    return examples


def saferpaca_rows(raw_dir: Path, safety_size: int) -> list[dict[str, str]]:
    path = raw_dir / "safety_tuned_llamas" / STL_NAME / "data" / "training" / f"saferpaca_Instructions_{safety_size}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for row in rows:
        instruction = row["instruction"].strip()
        extra = (row.get("input") or "").strip()
        response = row["output"].strip()
        if not instruction or not response:
            continue
        # Alpaca convention: `input` is extra context appended to the instruction.
        prompt = f"{instruction}\n\n{extra}" if extra else instruction
        examples.append({"prompt": prompt, "response": response})
    return examples


def write_jsonl_gz(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def describe(name: str, examples: list[dict[str, str]]) -> None:
    prompt_len = sorted(len(e["prompt"]) for e in examples)
    response_len = [len(e["response"]) for e in examples]
    typer.echo(
        f"  {name:<9} n={len(examples):>7}   "
        f"prompt chars p50={statistics.median(prompt_len):>6.0f} p95={prompt_len[int(0.95 * len(prompt_len))]:>6.0f}   "
        f"response chars p50={statistics.median(response_len):>5.0f}"
    )


@app.command()
def main(
    raw_dir: str = typer.Option(str(RAW_DIR)),
    output_dir: str = typer.Option("data/sft"),
    safety_size: int = typer.Option(2000, help="X in saferpaca_Instructions_{X}.json."),
    test_fraction: float = typer.Option(0.05, help="Fraction of the Alpaca+safety pool held out."),
    seed: int = typer.Option(0),
    examples_to_show: int = typer.Option(2, help="Print this many samples per split."),
) -> None:
    raw = Path(raw_dir)
    uc_dir = raw / "ultrachat_200k"
    uc_train_files = sorted(uc_dir.glob("train_sft-*.parquet"))
    uc_test_files = sorted(uc_dir.glob("test_sft-*.parquet"))
    if not uc_train_files or not uc_test_files:
        raise typer.BadParameter(f"no UltraChat parquet files under {uc_dir}; run the download first.")

    typer.echo("reading UltraChat-200K (first turn of each conversation)")
    uc_train = ultrachat_first_turns(uc_train_files)
    uc_test = ultrachat_first_turns(uc_test_files)

    typer.echo(f"reading SafetyTunedLlamas (saferpaca_Instructions_{safety_size})")
    pool = saferpaca_rows(raw, safety_size)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_test = int(len(pool) * test_fraction)
    stl_test, stl_train = pool[:n_test], pool[n_test:]

    train = uc_train + stl_train
    test = uc_test + stl_test
    rng.shuffle(train)

    typer.echo("\nbuilt")
    describe("uc train", uc_train)
    describe("uc test", uc_test)
    describe("stl train", stl_train)
    describe("stl test", stl_test)
    typer.echo(f"  {'TOTAL':<9} train={len(train)}  test={len(test)}")

    out = Path(output_dir)
    write_jsonl_gz(out / "train.jsonl.gz", train)
    write_jsonl_gz(out / "test.jsonl.gz", test)
    for split, data in (("train", train), ("test", test)):
        path = out / f"{split}.jsonl.gz"
        typer.echo(f"  wrote {path}  ({path.stat().st_size / 1e6:.0f} MB gzipped)")

    for split, data in (("train", train), ("test", test)):
        typer.echo(f"\n--- {split} samples ---")
        for example in data[:examples_to_show]:
            typer.echo(f"  PROMPT  : {example['prompt'][:160]!r}")
            typer.echo(f"  RESPONSE: {example['response'][:160]!r}")


if __name__ == "__main__":
    app()
