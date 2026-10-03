import json
from pathlib import Path
from tests.common import gpt2_bytes_to_unicode
from lm_basics.tokenizer import train_bpe


def main():
    fx = Path("tests/fixtures")
    vocab, merges = train_bpe(fx / "corpus.en", vocab_size=500, special_tokens=["<|endoftext|>"])

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(fx / "train-bpe-reference-merges.txt", encoding="utf-8") as f:
        ref_merges = [
            (bytes([gpt2_byte_decoder[t] for t in line.rstrip().split(" ")[0]]),
             bytes([gpt2_byte_decoder[t] for t in line.rstrip().split(" ")[1]]))
            for line in f
        ]
    with open(fx / "train-bpe-reference-vocab.json", encoding="utf-8") as f:
        ref_vocab = {
            int(idx): bytes([gpt2_byte_decoder[t] for t in item])
            for item, idx in json.load(f).items()
        }

    print("merges exact match:", merges == ref_merges)
    print("vocab keys match:", set(vocab.keys()) == set(ref_vocab.keys()))
    print("vocab values match:", set(vocab.values()) == set(ref_vocab.values()))
    if merges != ref_merges:
        for i, (m, r) in enumerate(zip(merges, ref_merges)):
            if m != r:
                print("first diff at", i, "mine:", m, "ref:", r)
                break
        print("len mine/ref:", len(merges), len(ref_merges))


if __name__ == "__main__":
    main()
