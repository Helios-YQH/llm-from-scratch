"""§4 数据管线的可复用组件: 读 WET、逐文档过滤、JSONL 级去重。

CLI 入口在 scripts/: filter_data.py(并行过滤) → deduplicate_data.py(去重)
→ tokenize_data.py(tokenize)。本模块只放与 CLI 无关的逻辑, 便于单独测试。
"""
from __future__ import annotations

import gzip
import json
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import mmh3
import numpy as np
from warcio.archiveiterator import ArchiveIterator

from lm_data.deduplication import (
    cluster_keep_first,
    jaccard,
    lsh_candidate_pairs,
    minhash_signature,
    normalize_text,
    word_ngrams,
)
from lm_data.filters import (
    classify_nsfw,
    classify_quality,
    classify_toxic_speech,
    gopher_quality_filter,
    identify_language,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


@dataclass
class FilterConfig:
    langid_threshold: float = 0.7
    nsfw_threshold: float = 0.9
    toxic_threshold: float = 0.9
    quality_threshold: float = 0.5
    mask_pii: bool = True
    # 跳过的过滤器名(用于做数据消融: 例如只保留 langid 当作"未精加工的英文 CC"对照组)
    skip: frozenset[str] = frozenset()


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    pii_masks: Counter = field(default_factory=Counter)

    def merge(self, other: "FilterStats") -> None:
        self.total += other.total
        self.kept += other.kept
        self.dropped.update(other.dropped)
        self.pii_masks.update(other.pii_masks)

    def as_dict(self) -> dict:
        # 注意: 不能用 dataclasses.asdict —— 它用生成器重建 Counter, 会把
        # (key, count) 元组当成元素计数, 得到 {"('langid', 5)": 1} 这种垃圾。
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": dict(self.dropped),
            "pii_masks": dict(self.pii_masks),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FilterStats":
        return cls(
            total=data["total"],
            kept=data["kept"],
            dropped=Counter(data["dropped"]),
            pii_masks=Counter(data["pii_masks"]),
        )


def iter_wet_documents(wet_path: Path | str) -> Iterator[tuple[str, str]]:
    """逐个产出 WET 文件里的 (url, text), 跳过非 conversion 记录。"""
    with gzip.open(wet_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            url = record.rec_headers.get_header("WARC-Target-URI", "")
            text = record.content_stream().read().decode("utf-8", errors="replace")
            yield url, text


def filter_document(text: str, cfg: FilterConfig) -> tuple[str | None, str | None, Counter]:
    """过滤单篇文档, 返回 (保留的文本 | None, 丢弃原因 | None, PII 掩码计数)。

    过滤顺序按"便宜优先": langid → gopher → nsfw → toxic → quality。
    丢弃原因只记第一个命中的过滤器。PII 掩码是改写(不丢文档), 最后应用,
    分类器始终看未掩码的原文。
    """
    language, score = identify_language(text)
    if language != "en" or score < cfg.langid_threshold:
        return None, "langid", Counter()

    if "gopher" not in cfg.skip and not gopher_quality_filter(text):
        return None, "gopher", Counter()

    if "nsfw" not in cfg.skip:
        label, score = classify_nsfw(text)
        if label == "nsfw" and score >= cfg.nsfw_threshold:
            return None, "nsfw", Counter()

    if "toxic" not in cfg.skip:
        label, score = classify_toxic_speech(text)
        if label == "toxic" and score >= cfg.toxic_threshold:
            return None, "toxic", Counter()

    if "quality" not in cfg.skip:
        label, score = classify_quality(text)
        if label != "wiki" or score < cfg.quality_threshold:
            return None, "quality", Counter()

    masks: Counter = Counter()
    if cfg.mask_pii and "pii" not in cfg.skip:
        for name, mask_fn in (("email", mask_emails), ("phone", mask_phone_numbers), ("ip", mask_ips)):
            text, count = mask_fn(text)
            if count:
                masks[name] += count
    return text, None, masks


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(f, doc: dict) -> None:
    f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def exact_line_dedup_jsonl(
    input_files: Sequence[Path], output_directory: Path
) -> dict:
    """全语料行级去重(保留 JSONL 文档结构)。

    与作业的 exact_line_deduplication 同一思路(两趟扫描 + 行计数), 区别是
    以"一篇文档一行 JSON"为单位, 行数统计作用于文档内部的换行。按行内容
    的 64 位哈希计数以固定内存开销。
    """
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_counts: Counter[int] = Counter()
    for path in input_files:
        for doc in read_jsonl(path):
            for line in doc["text"].split("\n"):
                if line.strip():
                    line_counts[mmh3.hash64(line)[0]] += 1

    removed_lines = 0
    removed_docs = 0
    for path in input_files:
        with open(out_dir / Path(path).name, "w", encoding="utf-8") as fout:
            for doc in read_jsonl(path):
                lines = doc["text"].split("\n")
                kept = [ln for ln in lines if not ln.strip() or line_counts[mmh3.hash64(ln)[0]] == 1]
                removed_lines += len(lines) - len(kept)
                doc["text"] = "\n".join(kept)
                if not doc["text"].strip():
                    removed_docs += 1
                    continue
                write_jsonl(fout, doc)

    return {
        "unique_lines": len(line_counts),
        "removed_lines": removed_lines,
        "removed_empty_docs": removed_docs,
    }


def _file_signatures(task: tuple[str, int, int, int]) -> tuple[bytes, array]:
    """计算一个 JSONL 文件里各文档的 minhash 签名。

    返回 (扁平 int32 签名字节, 参与去重的文档行号)。

    排除 n-gram 为空的文档: 归一化会删掉所有非 ASCII 字符, 纯中文/日文/俄文页面
    归一化后是空串 —— 它们的签名会退化成同一个值, 在 LSH 里两两成为候选
    (候选对数量平方级爆炸), 且 Jaccard 分母为 0。这类文档直接不参与去重。
    """
    path_str, num_hashes, ngrams, seed = task
    rows = []
    line_nos = array("I")
    for line_no, doc in enumerate(read_jsonl(path_str)):
        grams = word_ngrams(normalize_text(doc["text"]), ngrams)
        if not grams:
            continue
        rows.append(minhash_signature(grams, num_hashes, seed))
        line_nos.append(line_no)
    if not rows:
        return b"", line_nos
    flat = np.concatenate([np.frombuffer(sig, dtype=np.int32) for sig in rows])
    return flat.tobytes(), line_nos


def compute_signatures(
    input_files: Sequence[Path], *, num_hashes: int, ngrams: int, seed: int, workers: int = 1
) -> tuple[np.ndarray, list[array]]:
    """返回 (签名矩阵 (参与去重的文档数 x num_hashes), 每个文件里参与去重的文档行号)。文件间并行。"""
    tasks = [(str(path), num_hashes, ngrams, seed) for path in input_files]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_file_signatures, tasks))
    else:
        results = [_file_signatures(task) for task in tasks]

    line_nos_per_file = [line_nos for _, line_nos in results]
    blocks = [
        np.frombuffer(payload, dtype=np.int32).reshape(len(line_nos), num_hashes)
        for payload, line_nos in results
        if len(line_nos)
    ]
    if not blocks:
        return np.zeros((0, num_hashes), dtype=np.int32), line_nos_per_file
    return np.concatenate(blocks), line_nos_per_file


def minhash_dedup_jsonl(
    input_files: Sequence[Path],
    output_directory: Path,
    *,
    num_hashes: int = 128,
    num_bands: int = 16,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
    seed: int = 336,
    workers: int = 1,
    file_cache_size: int = 4,
    gram_cache_docs: int = 50_000,
) -> dict:
    """文档级模糊去重(JSONL), 每簇保留一篇; 返回统计。

    全量 CC 有百万级文档, 不能像 minhash_keep_mask 那样把每篇的 n-gram 集合
    都常驻内存(那是数十 GB)。这里分三趟:
      1. 逐文件算签名(int32 矩阵常驻, workers>1 时并行)并记每篇的 (文件, 行号);
      2. LSH 出候选对, 按需重读文件算**真实** Jaccard(文件与 n-gram 集合都是
         限容缓存, 满了就淘汰重算);
      3. 传递聚类后逐文件写回保留的文档。
    """
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = [Path(p) for p in input_files]

    signature_matrix, line_nos_per_file = compute_signatures(
        input_files, num_hashes=num_hashes, ngrams=ngrams, seed=seed, workers=workers
    )
    doc_index = [
        (file_idx, line_no)
        for file_idx, line_nos in enumerate(line_nos_per_file)
        for line_no in line_nos
    ]
    assert len(doc_index) == signature_matrix.shape[0]

    candidate_pairs = lsh_candidate_pairs(signature_matrix, num_bands)

    @lru_cache(maxsize=file_cache_size)
    def file_texts(file_idx: int) -> list[str]:
        return [doc["text"] for doc in read_jsonl(input_files[file_idx])]

    gram_cache: dict[tuple[int, int], set[str]] = {}

    def doc_grams(doc_idx: int) -> set[str]:
        key = doc_index[doc_idx]
        grams = gram_cache.get(key)
        if grams is None:
            if len(gram_cache) >= gram_cache_docs:
                gram_cache.clear()
            file_idx, line_no = key
            grams = set(word_ngrams(normalize_text(file_texts(file_idx)[line_no]), ngrams))
            gram_cache[key] = grams
        return grams

    # 排序让同一文件相关的候选对相邻, 提高 file_texts 缓存命中率。
    duplicate_pairs = {
        (a, b) for a, b in sorted(candidate_pairs) if jaccard(doc_grams(a), doc_grams(b)) >= jaccard_threshold
    }
    keep = cluster_keep_first(len(doc_index), duplicate_pairs)

    kept = 0
    total_docs = 0
    cursor = 0
    for file_idx, path in enumerate(input_files):
        participating = set(line_nos_per_file[file_idx])
        with open(out_dir / path.name, "w", encoding="utf-8") as fout:
            for line_no, doc in enumerate(read_jsonl(path)):
                total_docs += 1
                if line_no in participating:
                    keep_this = keep[cursor]
                    cursor += 1
                else:
                    keep_this = True  # n-gram 为空的文档不参与去重, 总是保留
                if keep_this:
                    write_jsonl(fout, doc)
                    kept += 1

    return {
        "docs_in": total_docs,
        "docs_out": kept,
        "docs_removed": total_docs - kept,
        "docs_without_ngrams": total_docs - len(doc_index),
        "candidate_pairs": len(candidate_pairs),
        "duplicate_pairs": len(duplicate_pairs),
    }
