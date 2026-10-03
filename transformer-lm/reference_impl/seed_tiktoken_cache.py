"""
离线播种 tiktoken 的 GPT-2 编码缓存。

tiktoken 的 get_encoding("gpt2") 首次调用会从 openaipublic.blob.core.windows.net 下载
vocab.bpe 和 encoder.json,缓存到 TIKTOKEN_CACHE_DIR(默认 %TEMP%/data-gym-cache),
缓存文件名 = sha1(下载 URL)。下载被代理/网络拦截时(如本机),get_encoding 会抛 ProxyError。

好在 tests/fixtures/ 里就有这两份数据的精确副本:
  - gpt2_vocab.json == tiktoken 的 encoder.json(哈希完全一致)
  - gpt2_merges.txt  == tiktoken vocab.bpe 去掉 "#version: 0.2" 头后的内容

本脚本把它们按 tiktoken 期望的格式写进缓存目录,并校验 SHA-256,让 get_encoding 离线可用。
"""
import hashlib
import os
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

VOCAB_BPE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
ENCODER_JSON_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
VOCAB_BPE_HASH = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
ENCODER_JSON_HASH = "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"


def cache_dir() -> str:
    return os.environ.get("TIKTOKEN_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "data-gym-cache")


def main() -> None:
    # 1) encoder.json 直接用 fixture(已验证哈希一致)
    encoder_json = (FIXTURES / "gpt2_vocab.json").read_bytes()
    assert hashlib.sha256(encoder_json).hexdigest() == ENCODER_JSON_HASH

    # 2) vocab.bpe = "#version: 0.2\n" + gpt2_merges.txt 内容
    vocab_bpe = ("#version: 0.2\n" + (FIXTURES / "gpt2_merges.txt").read_text(encoding="utf-8")).encode("utf-8")
    assert hashlib.sha256(vocab_bpe).hexdigest() == VOCAB_BPE_HASH

    os.makedirs(cache_dir(), exist_ok=True)
    for url, data in (
        (VOCAB_BPE_URL, vocab_bpe),
        (ENCODER_JSON_URL, encoder_json),
    ):
        cache_key = hashlib.sha1(url.encode()).hexdigest()
        path = os.path.join(cache_dir(), cache_key)
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {path} ({len(data)} bytes)")

    print("done")


if __name__ == "__main__":
    main()
