"""
用现成库(tiktoken)做 byte-level BPE 的 encode/decode,并和我们 from-scratch 的
Tokenizer 逐字节对比。

tiktoken 是 OpenAI 官方实现,GPT-2 的编码就是字节级 BPE。测试 test_tokenizer.py
正是拿 tiktoken.get_encoding("gpt2") 当 ground truth 的。这个脚本在真实 TinyStories
文本上验证:我们手写的 Tokenizer 和 tiktoken 产出的 token ID 完全一致。
"""
import json
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lm_basics.tokenizer import Tokenizer  # noqa: E402
from tests.common import gpt2_bytes_to_unicode  # noqa: E402


def load_ours_from_fixtures(special_tokens: list[str] | None = None) -> Tokenizer:
    """从 tests/fixtures 的 GPT-2 文件构造我们的 Tokenizer(和测试同一来源)。"""
    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(ROOT / "tests/fixtures/gpt2_vocab.json", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = {int(idx): bytes([byte_decoder[t] for t in tok]) for tok, idx in raw_vocab.items()}

    merges = []
    with open(ROOT / "tests/fixtures/gpt2_merges.txt", encoding="utf-8") as f:
        for line in f:
            t1, t2 = line.rstrip().split(" ")
            merges.append((bytes([byte_decoder[t] for t in t1]), bytes([byte_decoder[t] for t in t2])))
    return Tokenizer(vocab, merges, special_tokens)


def main() -> None:
    # 用 TinyStories 验证集开头的一段真实文本
    sample = (ROOT / "data" / "TinyStories-valid.txt").read_text(encoding="utf-8")[:2000]

    reference = tiktoken.get_encoding("gpt2")
    ours = load_ours_from_fixtures(special_tokens=["<|endoftext|>"])

    ref_ids = reference.encode(sample, allowed_special={"<|endoftext|>"})
    our_ids = ours.encode(sample)

    print(f"tiktoken ids: {len(ref_ids)}  |  ours ids: {len(our_ids)}")
    print("exact match:", ref_ids == our_ids)

    if ref_ids != our_ids:
        for i, (a, b) in enumerate(zip(ref_ids, our_ids)):
            if a != b:
                print(f"first diff at {i}: tiktoken={a} ours={b}")
                break

    # 压缩率(bytes / token)—— 讲义 tokenizer_experiments (a) 的指标
    raw_bytes = len(sample.encode("utf-8"))
    print(f"compression ratio: {raw_bytes} bytes / {len(ref_ids)} tokens = {raw_bytes / len(ref_ids):.2f} bytes/token")

    # 解码一致性
    print("decode match:", reference.decode(ref_ids) == ours.decode(our_ids))


if __name__ == "__main__":
    main()
