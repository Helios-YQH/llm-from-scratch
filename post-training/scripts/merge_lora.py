"""Merge a LoRA checkpoint into a plain HF model, so evaluation can point vLLM at it.

`grpo`/`sft_train.py` only write the merged form at the very end of a run. When a
run is stopped early -- which is the normal outcome for a fine-tune that has
plateaued -- the adapter is all there is, and anything downstream that wants a
normal model directory (DPO's starting point, vLLM, AlpacaEval) needs this.

    python scripts/merge_lora.py --adapter runs/sft/checkpoints/step004000 \\
        --output runs/sft/checkpoints/final
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    adapter: str = typer.Option(..., help="Directory holding the LoRA adapter."),
    output: str = typer.Option(..., help="Where to write the merged model."),
    base: str = typer.Option("/mnt/14T/houyi/models/Meta-Llama-3.1-8B"),
    device: str = typer.Option("cuda:0"),
) -> None:
    if Path(output).exists() and any(Path(output).iterdir()):
        raise typer.BadParameter(f"{output} is not empty; refusing to overwrite it.")

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    Path(output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    typer.echo(f"  merged {adapter} -> {output}")
    typer.echo(f"  {sum(f.stat().st_size for f in Path(output).rglob('*') if f.is_file()) / 1e9:.1f} GB")


if __name__ == "__main__":
    app()
