"""§4.3 tokenize: JSONL 目录 → GPT-2 token 的 uint16 .bin(每篇文档末尾加 <|endoftext|>)。

    uv run python scripts/tokenize_data.py \\
        --input-dir /root/data/deduped/2_minhash_dedup \\
        --output /root/data/your_data.bin --workers 8

输出格式与 scripts/train.py 的 --train-bin 兼容(uint16 序列)。
若本机/服务器访问 HuggingFace 困难, 可先设 HF_ENDPOINT=https://hf-mirror.com。
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_data.pipeline import read_jsonl  # noqa: E402

TOKENIZER_NAME = "gpt2"


def encode_file(path_str: str) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for doc in read_jsonl(path_str):
        ids.extend(tokenizer.encode(doc["text"]))
        ids.append(eos)
    return np.asarray(ids, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件(冒烟)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.jsonl"))
    if args.limit:
        input_files = input_files[: args.limit]
    if not input_files:
        raise SystemExit(f"no *.jsonl under {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    with open(args.output, "wb") as fout:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for ids in tqdm(
                executor.map(encode_file, [str(path) for path in input_files]),
                total=len(input_files),
                desc="tokenizing",
            ):
                ids.tofile(fout)
                total_tokens += len(ids)

    print(f"wrote {total_tokens:,} tokens ({total_tokens / 1e9:.3f}B) to {args.output}")


if __name__ == "__main__":
    main()
