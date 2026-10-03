"""Packing instruction-tuning data into fixed-length training sequences."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROMPTS_DIR = Path(__file__).parent / "prompts_safety"


def load_sft_prompt_template(name: str = "alpaca_sft") -> str:
    """Read the SFT template. `.strip()`ed: a trailing newline merges `.` and `\\n`."""
    return (PROMPTS_DIR / f"{name}.prompt").read_text(encoding="utf-8").strip()


class PackedSFTDataset(Dataset):
    """Fixed-length blocks cut out of one concatenated token stream."""

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


def get_packed_sft_dataset(
    tokenizer,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """Cut one concatenated token stream into `seq_length` blocks.

    `labels` is shifted globally, so it reaches one token past the last input.
    """
    path = Path(dataset_path)
    # The shipped instruction data is gzipped (`train.jsonl.gz`); the test fixture
    # is plain `.jsonl`, so dispatch on the suffix rather than assuming one.
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    examples = [json.loads(line) for line in lines if line.strip()]
    if shuffle:
        random.shuffle(examples)

    template = load_sft_prompt_template()
    stream: list[int] = []
    for example in examples:
        text = template.format(instruction=example["prompt"], response=example["response"])
        # add_special_tokens puts a BOS at the start of each document; the EOS
        # that ends it is appended by hand.
        stream += tokenizer(text, add_special_tokens=True)["input_ids"]
        stream.append(tokenizer.eos_token_id)

    n_blocks = len(stream) // seq_length
    usable = n_blocks * seq_length
    input_ids = torch.tensor(stream[:usable], dtype=torch.long).view(n_blocks, seq_length)
    # labels are the same stream shifted one left, so they reach one token past
    # the last input token -- the tail is not dropped on this side.
    labels = torch.tensor(stream[1 : usable + 1], dtype=torch.long).view(n_blocks, seq_length)
    return PackedSFTDataset(input_ids, labels)


def run_iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    """Iterable over batches covering one epoch of `dataset`."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
