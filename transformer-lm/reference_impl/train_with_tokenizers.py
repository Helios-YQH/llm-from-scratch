"""
用 HuggingFace tokenizers 库(现成库)训练 byte-level BPE 并写入产物。

这是对我们 from-scratch train_bpe 的"真实世界对照":
  - models.BPE()        —— 字节级 BPE(底层是 Rust,预分词/合并都非常快)
  - pre_tokenizers.ByteLevel() —— GPT-2 式"空格前加前缀"的预分词
  - trainers.BpeTrainer(vocab_size, special_tokens) —— 训练器

产物写到 reference_impl/output/:
  - tokenizer.json —— HF 完整格式,可被 transformers 直接加载
  - vocab.json / merges.txt —— 和 save_bpe 同款 GPT-2 格式
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"

SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 1000


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1. 构造字节级 BPE + GPT-2 式预分词
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 2. 训练(用 TinyStories 验证集,19MB,够快)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train([str(ROOT / "data" / "TinyStories-valid.txt")], trainer)

    # 3. 写入产物
    tokenizer.save(str(OUT / "tokenizer.json"))

    # 4. 从 tokenizer.json 里读回 merges 和 vocab,额外导出 GPT-2 格式
    #    (tokenizer.json 的 model.merges 是 [token1, token2] 对的列表)
    import json

    saved = json.loads((OUT / "tokenizer.json").read_text(encoding="utf-8"))
    model_json = saved["model"]

    with open(OUT / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(model_json["vocab"], f, ensure_ascii=False)

    with open(OUT / "merges.txt", "w", encoding="utf-8") as f:
        for pair in model_json["merges"]:
            f.write(f"{pair[0]} {pair[1]}\n")

    print(f"vocab size: {tokenizer.get_vocab_size()}")
    print(f"wrote {OUT / 'tokenizer.json'}, {OUT / 'vocab.json'}, {OUT / 'merges.txt'}")


if __name__ == "__main__":
    main()
