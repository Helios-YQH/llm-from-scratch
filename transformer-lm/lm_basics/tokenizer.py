import json
import os
import re

import multiprocessing as mp
import regex
from collections import Counter
from collections.abc import Iterable, Iterator
from tqdm import tqdm


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def find_chunk_boundaries(
    file: "os.PathLike | BinaryIO",
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    Taken from lm_basics/pretokenization_example.py.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pre_tokenize_chunk(
        start: int,
        end: int,
        data_path: str,
        special_tokens: list[str]) -> Counter:
    """
    单个进程的预分词 worker：只读 [start, end) 这一块，统计词频并返回 Counter。

    必须放在模块顶层 —— Windows 的 multiprocessing 用 spawn 启动子进程，
    子进程会重新 import 本模块，若 worker 嵌套在 train_bpe 里将无法被找到。
    """
    with open(data_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="ignore")

    # 块内仍按特殊 token 分割（硬边界，不参与 merge 计数）
    special_pattern = "|".join(re.escape(t) for t in special_tokens)
    parts = re.split(f"({special_pattern})", text)

    counter = Counter()
    for i, part in enumerate(parts):
        if i % 2 == 1 or not part:  # 奇数索引是特殊 token 本身，跳过
            continue
        for match in regex.finditer(PAT, part):
            raw = match.group().encode("utf-8")
            counter[tuple(bytes([b]) for b in raw)] += 1
    return counter


def train_bpe(
        data_path: str,
        vocab_size: int,
        special_tokens: list[str]):
    """
    训练 BPE 分词器，返回词汇表（id -> bytes）和合并操作记录（按顺序）。
    """
    # 1. 初始化词汇表：特殊 token 在前，接着是 256 个单字节
    vocab = {}
    next_id = 0
    for t in special_tokens:
        vocab[next_id] = t.encode('utf-8')
        next_id += 1
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    # 2. 分块 + 多进程预分词，直接得到词频表 word_counts。
    #    - 边界选在特殊 token 开头，块间互不干扰，结果可安全合并。
    #    - 每个 worker 只读自己的块，不把整个文件读进内存。
    #    - 多个块返回的 Counter 用 update 合并，词频天然可加。
    data_path = os.fspath(data_path)
    special_bytes = [t.encode("utf-8") for t in special_tokens]

    # 文件较小时串行更快（省去多进程启动开销）
    file_size = os.path.getsize(data_path)
    if file_size < 1_000_000:
        ranges = [(0, file_size)]
        word_counts = _pre_tokenize_chunk(0, file_size, data_path, special_tokens)
    else:
        num_processes = os.cpu_count() or 1
        with open(data_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_bytes[0])
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        with mp.Pool() as pool:
            counters = pool.starmap(
                _pre_tokenize_chunk,
                [(start, end, data_path, special_tokens) for start, end in ranges],
            )
        word_counts = Counter()
        for c in counters:
            word_counts.update(c)

    # 3. 全局 pair 计数 + 反向索引（pair -> 包含它的词）。
    #    合并 (a,b) 时，只需处理 word_index[(a,b)] 里列出的词，无需重扫全语料。
    pair_counts = {}    # (bytes, bytes) -> int
    word_index = {}     # (bytes, bytes) -> set[tuple[bytes, ...]]
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] = pair_counts.get(p, 0) + count
            word_index.setdefault(p, set()).add(word)

    # 4. 增量合并
    merges = []
    num_merges_needed = vocab_size - len(vocab)
    with tqdm(total=num_merges_needed, desc="BPE training", unit="merge") as pbar:
        while len(vocab) < vocab_size and pair_counts:
            # 频次优先；同频取字典序更大的 pair（对应讲义里 max([...]) 的例子）
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            a, b = best
            new_symbol = a + b

            vocab[next_id] = new_symbol
            next_id += 1
            merges.append(best)
            pbar.update(1)

            # 只处理"包含 best 的词"，而不是全语料。list() 拷贝一份，
            # 因为处理过程中 word_index 会被改写。
            for word in list(word_index.get(best, ())):
                count = word_counts.pop(word)  # 旧词整体移除

                # ① 旧词贡献的相邻 pair 计数全部减去，并从反向索引中摘掉该词
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    pair_counts[p] -= count
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                    if p in word_index:
                        word_index[p].discard(word)
                        if not word_index[p]:
                            del word_index[p]

                # ② 词内贪心替换 (a,b) -> new_symbol（从左到右，不重叠）
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(new_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)

                # ③ 新词贡献的相邻 pair 计数加上，并登记进反向索引
                word_counts[new_word] = word_counts.get(new_word, 0) + count
                for i in range(len(new_word) - 1):
                    q = (new_word[i], new_word[i + 1])
                    pair_counts[q] = pair_counts.get(q, 0) + count
                    word_index.setdefault(q, set()).add(new_word)

    # 5. 返回词汇表和合并记录
    return vocab, merges


def save_bpe(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        vocab_path: str,
        merges_path: str) -> None:
    """
    将训练好的 BPE 词汇表和合并记录序列化到磁盘。

    格式与 GPT-2 的官方格式一致（也是 tests/fixtures 里参考文件的格式）：
    - vocab_path: 一个 JSON 文件，形如 {"<token的可打印表示>": "<token_id>"}
    - merges_path: 一个文本文件，每行一个 merge，形如 "<token1的可打印表示> <token2的可打印表示>"
    这样保存出来的文件可以直接被 Tokenizer.from_files 读回，也能用 gpt2_bytes_to_unicode
    对照参考 fixture 进行对拍。

    注意：bytes 不能直接 JSON 序列化，所以每个字节先经 gpt2_bytes_to_unicode 映射成
    可打印字符。空格会变成 'Ġ'，不可打印字节会偏移到 256 之后的 Unicode 字符。
    """
    byte_decoder = gpt2_bytes_to_unicode()

    def bytes_to_str(bs: bytes) -> str:
        return "".join(byte_decoder[b] for b in bs)

    # 保存词汇表：{token_str: token_id}（GPT-2 encoder.json 的约定，from_files 按此读取）
    vocab_to_save = {bytes_to_str(tok_bytes): str(tok_id) for tok_id, tok_bytes in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, ensure_ascii=False)

    # 保存合并记录：每行 "<token1> <token2>"（用空格分隔，因为 token 本身不含空格字符）
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{bytes_to_str(left)} {bytes_to_str(right)}\n")


def _gpt2_str_to_bytes(s: str) -> bytes:
    """gpt2_bytes_to_unicode 的反函数：把可打印字符表示解回原始字节。"""
    unicode_to_byte = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return bytes(unicode_to_byte[ch] for ch in s)


class Tokenizer:
    """
    用训练好的 vocab + merges 把文本编码成 token ID、把 ID 解码回文本。

    编码的核心是 rank-based 贪心合并：
    - merges 列表按产生顺序排列，越靠前 rank 越小，说明训练时越早被合并（频率越高）。
    - 编码时每个 pre-token 内部反复合并"当前相邻 pair 里 rank 最小"的一对，
      直到没有可合并的 pair。这样在编码阶段精确复现训练时的优先级。
    """

    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # bytes -> id 的倒排索引（encode 查表用）
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # merges -> rank（rank 越小越优先合并）
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

        # 特殊 token 若不在 vocab 中则追加；并建立 字符串 -> id 映射
        self.special_token_to_id = {}
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            tid = self.bytes_to_id.get(tok_bytes)
            if tid is None:
                tid = len(self.vocab)
                self.vocab[tid] = tok_bytes
                self.bytes_to_id[tok_bytes] = tid
            self.special_token_to_id[tok] = tid

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str | os.PathLike,
            merges_filepath: str | os.PathLike,
            special_tokens: list[str] | None = None) -> "Tokenizer":
        """从 save_bpe 序列化的文件（GPT-2 格式）构造 Tokenizer。"""
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(idx): _gpt2_str_to_bytes(tok_str) for tok_str, idx in raw_vocab.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((_gpt2_str_to_bytes(parts[0]), _gpt2_str_to_bytes(parts[1])))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token ID 列表。特殊 token 永远保持为单个 token。"""
        if not self.special_tokens:
            return self._encode_plain(text)

        # 特殊 token 按"最长优先"切分，避免较短的特殊 token 抢在较长者之前匹配
        special_pattern = "|".join(
            re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
        parts = re.split(f"({special_pattern})", text)

        ids = []
        for part in parts:
            if part in self.special_token_to_id:
                ids.append(self.special_token_to_id[part])
            elif part:
                ids.extend(self._encode_plain(part))
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        """预分词 + rank-based 贪心合并 + 查表。"""
        ids = []
        for match in regex.finditer(PAT, text):
            raw = match.group().encode("utf-8")
            pieces = self._apply_merges([bytes([b]) for b in raw])
            ids.extend(self.bytes_to_id[p] for p in pieces)
        return ids

    def _apply_merges(self, word: list[bytes]) -> list[bytes]:
        """对单个 pre-token 反复合并当前 rank 最小的相邻 pair，直到无可合并。"""
        word = list(word)
        while True:
            best_pair, best_rank = None, float("inf")
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                r = self.merges_rank.get(p, float("inf"))
                if r < best_rank:
                    best_rank, best_pair = r, p
            if best_pair is None:
                return word
            a, b = best_pair
            new_symbol = a + b
            merged = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(new_symbol)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性逐块编码。块边界落在换行处时 token 不会跨块（见 _encode_plain）。"""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """把 token ID 列表解码回文本。坏字节用 U+FFFD 替换。"""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


def main():
    # 测试 BPE 训练
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TinyStories-train.txt")
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(data_path, vocab_size, special_tokens)

    print("Vocabulary size:", len(vocab))
    print("First 10 vocabulary entries:")
    for i in range(min(10, len(vocab))):
        print(i, vocab[i])

    print("\nFirst 10 merges:")
    for i in range(min(10, len(merges))):
        print(merges[i])

    # 演示保存
    save_bpe(vocab, merges, "vocab.json", "merges.txt")
    print("\nSaved to vocab.json / merges.txt")


if __name__ == "__main__":
    main()
