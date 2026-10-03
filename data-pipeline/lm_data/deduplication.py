"""Document deduplication: exact line dedup and MinHash+LSH fuzzy dedup.

两个文件级函数按作业要求: 每个输入文件视作一个**文档**, 输出目录里只写被保留
的文档, 文件名与输入同名。管线规模上的用法见 `lm_data.pipeline`。
"""
import re
import unicodedata
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import mmh3
import numpy as np

_MAX_HASH = 2**31 - 1


# --------------------------------------------------------------------------- #
# §3.1 精确行去重
# --------------------------------------------------------------------------- #

def exact_line_deduplication(
    input_files: list[Path], output_directory: Path
) -> None:
    """去掉在整个输入语料中出现 >=2 次的**行**, 其余行原样写回。

    做法(两趟扫描):
      1. 第一趟: 遍历所有文件, 对每一行取内容(去掉换行符)当 key 计数;
      2. 第二趟: 重写每个文件, 只保留计数 == 1 的行。
    空行(仅换行)不参与去重、总是保留 —— 它在实际数据里常充当文档分隔符,
    若也按"出现多次就删"处理会把相邻文档拼在一起。
    """
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_counts: Counter[str] = Counter()
    documents: dict[Path, list[str]] = {}

    for input_file in input_files:
        input_file = Path(input_file)
        with open(input_file, encoding="utf-8") as f:
            lines = f.readlines()  # 每行保留行尾换行符(Windows CRLF 已被转成 \n)
        documents[input_file] = lines
        for line in lines:
            key = line.rstrip("\n")
            if key != "":  # 空行不参与去重
                line_counts[key] += 1

    for input_file, lines in documents.items():
        kept = [line for line in lines if line_counts[line.rstrip("\n")] <= 1]
        out_path = out_dir / input_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(kept)


# --------------------------------------------------------------------------- #
# §3.2 MinHash + LSH 文档级模糊去重
# --------------------------------------------------------------------------- #

def normalize_text(text: str) -> str:
    """按 PDF 建议归一化: NFD → 去重音(accents) → 小写 → 去标点 → 折叠空白。

    归一化后再算 n-gram, 让"基本相同的文档"在 minhash / Jaccard 上更接近。
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_ngrams(text: str, n: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def minhash_signature(grams: Sequence[str], num_hashes: int, seed: int = 0) -> array:
    """对文档的 n-gram 集合算 minhash 签名(长度 num_hashes, 32 位整数数组)。

    朴素做法是每个 gram 调用 num_hashes 次哈希函数, 全量 CC 规模下要跑几小时。
    这里用工业界常用的等价方案: **每个 gram 只哈希一次**, 再用 k 组
    (a_j, b_j)(a_j 取奇数)做线性置换 h_j(g) = (a_j * h(g) + b_j) mod 2^31。
    奇数乘 + 加法是 2^31 上的双射, 因此 h_j 与"独立的哈希函数"一样均匀。
    全部用 numpy 向量化, 每篇文档只有一次 (gram 数 x num_hashes) 的矩阵运算。

    签名用 array('i') 而不是 list[int]: 每篇只占 4*num_hashes 字节, 百万级
    文档下这是决定性的(GB 级 vs 数十 GB)。
    """
    if not grams:
        return array("i", [_MAX_HASH]) * num_hashes

    a, b = _permutation_constants(num_hashes, seed)
    gram_hashes = np.fromiter(
        (mmh3.hash(gram, seed=seed, signed=False) for gram in grams), dtype=np.uint64, count=len(grams)
    )
    permuted = (gram_hashes[:, None] * a[None, :] + b[None, :]) & 0x7FFF_FFFF
    return array("i", permuted.min(axis=0).astype(np.int32))


@lru_cache(maxsize=None)
def _permutation_constants(num_hashes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """生成 k 组线性置换常数(a 必须为奇数才能保证 mod 2^31 上的双射)。"""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << 30, size=num_hashes, dtype=np.uint64) * 2 + 1
    b = rng.integers(0, 1 << 31, size=num_hashes, dtype=np.uint64)
    return a, b


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """两个 n-gram 集合的 Jaccard 相似度。

    两边都为空时返回 0(归一化会把非 ASCII 文本清成空串, 这种文档的 n-gram 集合
    是空的; 若按数学定义返回 1 会让它们互相判重, 这里按"没有可比的共同内容"处理)。
    """
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def lsh_candidate_pairs(signatures: Sequence[array] | np.ndarray, num_bands: int) -> set[tuple[int, int]]:
    """LSH 分桶: 签名切成 num_bands 段, 任一段相同即互为候选重复对。

    signatures 可以是 array('i') 列表或 (文档数 x num_hashes) 的 int32 矩阵。

    两处为全量语料做的内存优化:
      - band 内容用 bytes 做 key(而不是 tuple[int]), 桶数 = 文档数 x band 数,
        是全流程最占内存的结构之一;
      - 桶值先存单个 int, 只有真的发生冲突(第 2 篇同桶)才升级成 list ——
        绝大多数桶只有 1 篇, 这样省掉近亿个单元素 list 对象。
    """
    if not isinstance(signatures, np.ndarray):
        if len(signatures) == 0:
            return set()
        signatures = np.stack([np.frombuffer(sig, dtype=np.int32) for sig in signatures])

    num_hashes = signatures.shape[1]
    band_size = num_hashes // num_bands
    buckets: dict[bytes, int | list[int]] = {}
    for doc_idx in range(signatures.shape[0]):
        row = signatures[doc_idx]
        for b in range(num_bands):
            key = row[b * band_size : (b + 1) * band_size].tobytes()
            existing = buckets.get(key)
            if existing is None:
                buckets[key] = doc_idx
            elif isinstance(existing, int):
                buckets[key] = [existing, doc_idx]
            else:
                existing.append(doc_idx)

    pairs: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        if isinstance(bucket, int):
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                pairs.add((a, b) if a < b else (b, a))
    return pairs


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_keep_first(num_docs: int, pairs: set[tuple[int, int]]) -> list[bool]:
    """把重复关系按传递性聚类(并查集), 每簇保留下标最小的文档。"""
    uf = _UnionFind(num_docs)
    for a, b in pairs:
        uf.union(a, b)

    members: defaultdict[int, list[int]] = defaultdict(list)
    for doc_idx in range(num_docs):
        members[uf.find(doc_idx)].append(doc_idx)

    keep = [False] * num_docs
    for group in members.values():
        keep[min(group)] = True
    return keep


def minhash_keep_mask(
    documents: Sequence[str],
    *,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    seed: int = 336,
) -> list[bool]:
    """对一组文档做 MinHash + LSH 去重, 返回每篇是否保留。

    1. 每篇 -> 归一化 -> 词 n-gram 集合 -> minhash 签名;
    2. LSH 找候选对; 对候选对算**真实** n-gram Jaccard, 超过阈值判为重复;
    3. 重复关系传递聚类, 每簇保留第一篇。

    注意: n-gram 集合与签名都常驻内存, 适合中小规模语料(作业测试/冒烟)。
    全量 CC 规模用 pipeline.minhash_dedup_jsonl 的惰性版本。
    """
    assert num_hashes % num_bands == 0, "num_hashes 必须能被 num_bands 整除"

    gram_sets: list[set[str]] = []
    signatures: list[array] = []
    for text in documents:
        grams = word_ngrams(normalize_text(text), ngrams)
        gram_sets.append(set(grams))
        signatures.append(minhash_signature(grams, num_hashes, seed))

    pairs = {
        (a, b)
        for a, b in lsh_candidate_pairs(signatures, num_bands)
        if jaccard(gram_sets[a], gram_sets[b]) >= jaccard_threshold
    }
    return cluster_keep_first(len(documents), pairs)


def minhash_deduplication(
    input_files: list[Path],
    output_directory: Path,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
) -> None:
    """文件版: 每个文件是一个文档, 重复簇只保留一个, 其余不写入输出目录。"""
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = [Path(p) for p in input_files]
    documents = [doc_id.read_text(encoding="utf-8") for doc_id in doc_ids]
    keep = minhash_keep_mask(
        documents,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
    )

    for doc_idx, keep_this in enumerate(keep):
        if keep_this:
            (out_dir / doc_ids[doc_idx].name).write_text(documents[doc_idx], encoding="utf-8")
