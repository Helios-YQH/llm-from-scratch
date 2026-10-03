"""Compute the tokenizer / model statistics quoted in the tech report.

Usage (from transformer-lm/):
    uv run python report/compute_stats.py

Everything here is measured locally on the Windows CPU box; GPU-side numbers
(training wall-clock) are copied from the run records and labelled as such.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

A1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A1))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is GBK

import numpy as np  # noqa: E402

from lm_basics.tokenizer import Tokenizer, _gpt2_str_to_bytes  # noqa: E402

VOCAB_PATH = A1 / "data" / "vocab.json"
MERGES_PATH = A1 / "data" / "merges.txt"
TRAIN_TXT = A1 / "data" / "TinyStories-train.txt"
VALID_TXT = A1 / "data" / "TinyStories-valid.txt"
TOKENS_TRAIN = A1 / "data" / "tokens_train.npy"
TOKENS_VALID = A1 / "data" / "tokens_valid.npy"

SPECIAL = ["<|endoftext|>"]
ENCODE_SAMPLE_BYTES = 4 * 1024 * 1024
PILE_BYTES = 825 * 1024**3


def hdr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def tokenizer_stats() -> None:
    hdr("1. Vocabulary")
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    merges = MERGES_PATH.read_text(encoding="utf-8").splitlines()
    print(f"vocab entries              : {len(vocab)}")
    print(f"merges                     : {len(merges)}")

    # vocab keys are the GPT-2 byte-to-unicode strings; recover raw bytes to
    # measure token length the way the reference defines it (bytes, not characters).
    by_len = sorted(
        ((len(_gpt2_str_to_bytes(tok)), tok) for tok in vocab),
        key=lambda kv: (-kv[0], kv[1]),
    )
    longest = by_len[0][0]
    print(f"longest token (bytes)      : {longest}")
    print("top 5 longest tokens       :")
    for n, tok in by_len[:5]:
        print(f"    {n:2d} bytes  {tok!r}  -> {_gpt2_str_to_bytes(tok)!r}")
    for threshold in (8, 10, 12):
        count = sum(1 for n, _ in by_len if n >= threshold)
        print(f"tokens with >= {threshold:2d} bytes      : {count}")


def compression_stats(tok: Tokenizer) -> None:
    hdr("2. Compression ratio (bytes per token)")

    train_bytes = TRAIN_TXT.stat().st_size
    train_tokens = TOKENS_TRAIN.stat().st_size // 2  # uint16
    valid_bytes = VALID_TXT.stat().st_size
    valid_tokens = TOKENS_VALID.stat().st_size // 2
    print(f"train  : {train_bytes / 1e9:.2f} GB -> {train_tokens / 1e6:7.1f} M tokens"
          f"  ({train_bytes / train_tokens:.3f} bytes/token)")
    print(f"valid  : {valid_bytes / 1e6:.1f} MB -> {valid_tokens / 1e6:7.1f} M tokens"
          f"  ({valid_bytes / valid_tokens:.3f} bytes/token)")

    # the reference protocol: 10 sampled documents
    text = VALID_TXT.read_text(encoding="utf-8", errors="replace")[:20 * 1024 * 1024]
    docs = [d for d in text.split("<|endoftext|>") if d.strip()][:10]
    ratios = []
    for i, doc in enumerate(docs, 1):
        n_tok = len(tok.encode(doc))
        n_bytes = len(doc.encode("utf-8"))
        ratios.append(n_bytes / n_tok)
        print(f"    doc {i:2d}: {n_bytes:7d} bytes -> {n_tok:6d} tokens  "
              f"({n_bytes / n_tok:.3f} bytes/token)")
    print(f"10 sampled docs: mean {np.mean(ratios):.3f}, "
          f"min {min(ratios):.3f}, max {max(ratios):.3f} bytes/token")


def throughput_stats(tok: Tokenizer) -> None:
    hdr("3. Tokenizer encoding throughput (local Windows CPU)")

    with open(TRAIN_TXT, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        n_bytes = 0
        for line in f:
            lines.append(line)
            n_bytes += len(line.encode("utf-8"))
            if n_bytes >= ENCODE_SAMPLE_BYTES:
                break

    t0 = time.perf_counter()
    n_tok = sum(1 for _ in tok.encode_iterable(lines))
    elapsed = time.perf_counter() - t0

    rate = n_bytes / elapsed
    print(f"encoded {n_bytes / 1e6:.1f} MB ({n_tok} tokens) in {elapsed:.1f} s")
    print(f"throughput                 : {rate / 1e6:.3f} MB/s, "
          f"{n_tok / elapsed / 1e3:.1f} K tokens/s")
    hours = PILE_BYTES / rate / 3600
    print(f"Pile (825 GB) extrapolation: {hours:.1f} h = {hours / 24:.1f} days "
          f"(single process, this machine)")


def model_stats() -> None:
    hdr("4. Model size")

    import torch

    from lm_basics.transformer import TransformerLM
    from lm_basics.transformer_4d import TransformerLM as TransformerLM_4d

    def count(model) -> int:
        return sum(p.numel() for p in model.parameters())

    base = TransformerLM(vocab_size=10000, context_length=256, d_model=512,
                         num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)
    print(f"main model (SwiGLU, d_ff=1344)     : {count(base) / 1e6:.2f} M params")

    swiglu_params = sum(p.numel() for n, p in base.named_parameters() if "ff" in n)
    print(f"  - feed-forward blocks            : {swiglu_params / 1e6:.2f} M")
    print(f"  - token embedding                : "
          f"{10000 * 512 / 1e6:.2f} M (untied from the output projection)")

    quantized = count(base) * 4 / 1024**2  # fp32 params only
    print(f"fp32 parameter memory              : {quantized:.0f} MiB")

    variant = TransformerLM_4d(vocab_size=10000, context_length=256, d_model=512,
                               num_layers=4, num_heads=16, d_ff=2048, rope_theta=10000.0)
    print(f"SiLU variant (d_ff=2048)           : {count(variant) / 1e6:.2f} M params"
          f"  (delta {abs(count(variant) - count(base)) / 1e3:.0f} K vs main model)")


def training_stats() -> None:
    hdr("5. Training throughput (from run records, 1x RTX A6000)")

    batch, ctx = 256, 256
    steps = 5000
    tokens = batch * ctx * steps
    for lr, seconds, label in [(1e-3, 1 * 3600 + 26 * 60, "lr=1e-3"),
                               (2e-3, int(1.5 * 3600), "lr=2e-3 (main run)")]:
        print(f"{label:18s}: {tokens / 1e6:.1f} M tokens in {seconds / 60:.0f} min "
              f"-> {tokens / seconds / 1e3:.1f} K tokens/s")


def dtype_note() -> None:
    hdr("6. uint16 note")
    print("vocab size 10000 < 2**16 = 65536, so every token id fits in uint16.")
    print(f"train tokens as uint16 : {TOKENS_TRAIN.stat().st_size / 1e6:.0f} MB")
    print(f"same corpus as int32   : {TOKENS_TRAIN.stat().st_size * 2 / 1e6:.0f} MB")


def main() -> None:
    tok = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=SPECIAL)
    tokenizer_stats()
    compression_stats(tok)
    throughput_stats(tok)
    model_stats()
    training_stats()
    dtype_note()


if __name__ == "__main__":
    main()
