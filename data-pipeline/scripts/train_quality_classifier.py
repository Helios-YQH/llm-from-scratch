"""训练质量分类器(§2.7): Wikipedia 外链页(正例) vs Common Crawl 随机页(负例)。

服务器正式训练(需要 wiki URL 文件和 English WET 数据):

    uv run python scripts/train_quality_classifier.py \\
        --wiki-urls /shared-data/wiki/enwiki-20260501-extracted_urls.txt.gz \\
        --negatives-wet-dir /shared-data/english-wet-data \\
        --num-positives 20000 --num-negatives 20000 \\
        --output shared-data/classifiers/quality_classifier.bin

本机验证管线(没有 wiki URL 时, 用 Paloma 验证集里的文本当正例。作业允许把
Paloma 用于构造过滤器/分类器, 但不允许把它拷进训练数据 —— 这里只是拿来训分类器):

    uv run python scripts/train_quality_classifier.py \\
        --paloma-bin shared-data/tokenized_paloma_c4_100_domains_validation.bin \\
        --negatives-wet-dir shared-data/CC \\
        --num-positives 1000 --num-negatives 450 --negative-sample-rate 1.0 \\
        --output shared-data/classifiers/quality_classifier.bin
"""
from __future__ import annotations

import argparse
import gzip
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import fasttext
import numpy as np
from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_data.filters import (  # noqa: E402
    classify_nsfw,
    classify_toxic_speech,
    extract_text_from_html_bytes,
    gopher_quality_filter,
    identify_language,
)
from lm_data.pipeline import iter_wet_documents  # noqa: E402

LABEL_POSITIVE = "__label__wiki"
LABEL_NEGATIVE = "__label__cc"


def sample_lines_reservoir(path: Path, num_lines: int, seed: int) -> list[str]:
    """蓄水池抽样, 不需要把整个 URL 文件读进内存。"""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(reservoir) < num_lines:
                reservoir.append(line)
            else:
                j = rng.randint(0, seen - 1)
                if j < num_lines:
                    reservoir[j] = line
    return reservoir


def fetch_warcs(urls: list[str], work_dir: Path, workers: int) -> list[Path]:
    """按 PDF 建议用 wget 抓 URL 列表; 分片并行, 每片一个 WARC。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_idx in range(workers):
        shard = urls[shard_idx::workers]
        if not shard:
            continue
        shard_file = work_dir / f"urls_{shard_idx:02d}.txt"
        shard_file.write_text("\n".join(shard) + "\n", encoding="utf-8")
        warc_prefix = work_dir / f"pages_{shard_idx:02d}"
        cmd = [
            "wget", "--timeout=5", "--tries=2", "--no-verbose",
            "-i", str(shard_file), f"--warc-file={warc_prefix}", "-O", "/dev/null",
        ]
        print(f"[fetch] shard {shard_idx}: {len(shard)} urls -> {warc_prefix}.warc.gz", flush=True)
        processes.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for process in processes:
        process.wait()
    return sorted(work_dir.glob("pages_*.warc.gz"))


def iter_warc_texts(warc_paths: list[Path]) -> Iterator[str]:
    for path in warc_paths:
        with gzip.open(path, "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type != "response":
                    continue
                content_type = record.http_headers.get_header("Content-Type") or ""
                if "html" not in content_type.lower():
                    continue
                try:
                    yield extract_text_from_html_bytes(record.content_stream().read())
                except Exception:
                    continue


def load_paloma_texts(bin_path: Path, num_docs: int, seed: int, min_chars: int = 200) -> list[str]:
    """把 Paloma 的 uint16 token 流按 <|endoftext|> 切回文档文本(本地代理正例用)。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = np.fromfile(bin_path, dtype=np.uint16)
    boundaries = np.flatnonzero(data == tokenizer.eos_token_id)
    starts = np.concatenate([[0], boundaries[:-1] + 1])
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(starts)), min(num_docs, len(starts))))

    docs = []
    for i in picked:
        text = tokenizer.decode(data[starts[i] : boundaries[i]].astype(int), skip_special_tokens=True)
        if len(text.strip()) >= min_chars:
            docs.append(text)
    return docs


def sample_wet_texts(
    wet_dir: Path, num_docs: int, seed: int, sample_rate: float, langid_threshold: float = 0.7
) -> list[str]:
    """从 WET 文件里随机抽英文文档(负例: 未做质量过滤的真实 CC 页面)。"""
    wet_files = sorted(Path(wet_dir).glob("*.warc.wet.gz"))
    if not wet_files:
        raise FileNotFoundError(f"no *.warc.wet.gz under {wet_dir}")
    rng = random.Random(seed)
    rng.shuffle(wet_files)

    docs: list[str] = []
    for path in wet_files:
        for _, text in iter_wet_documents(path):
            if len(docs) >= num_docs:
                return docs
            if rng.random() > sample_rate:
                continue
            language, score = identify_language(text)
            if language == "en" and score >= langid_threshold:
                docs.append(text)
    return docs


def clean_positives(texts: list[str], langid_threshold: float = 0.7) -> tuple[list[str], Counter]:
    """用已实现的过滤原语清洗正例(语言/规则/有害内容), 返回清洗后文本与统计。"""
    dropped: Counter = Counter()
    kept: list[str] = []
    for text in texts:
        language, score = identify_language(text)
        if language != "en" or score < langid_threshold:
            dropped["langid"] += 1
            continue
        if not gopher_quality_filter(text):
            dropped["gopher"] += 1
            continue
        label, score = classify_nsfw(text)
        if label == "nsfw" and score >= 0.9:
            dropped["nsfw"] += 1
            continue
        label, score = classify_toxic_speech(text)
        if label == "toxic" and score >= 0.9:
            dropped["toxic"] += 1
            continue
        kept.append(text)
    return kept, dropped


def write_fasttext_file(rows: list[tuple[str, str]], path: Path, max_chars: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for label, text in rows:
            f.write(f"{label} {' '.join(text.split())[:max_chars]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wiki-urls", type=Path, help="wiki 外链 URL 文件(.txt.gz), 服务器模式")
    source.add_argument("--paloma-bin", type=Path, help="Paloma token bin, 本地代理模式")
    parser.add_argument("--negatives-wet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--num-positives", type=int, default=20_000)
    parser.add_argument("--num-negatives", type=int, default=20_000)
    parser.add_argument("--negative-sample-rate", type=float, default=0.2)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--no-clean-positives", action="store_true")
    parser.add_argument("--max-chars", type=int, default=10_000)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    # fastText 默认 bucket=2,000,000, 光哈希桶就是 200万 x dim x 4B ≈ 800MB ——
    # Dolma 那两个 ~1GB 的分类器就是这么来的。这个作业的数据规模用 20 万桶足够。
    parser.add_argument("--bucket", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    work_dir = args.work_dir or args.output.parent / "quality_classifier_build"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.wiki_urls:
        urls = sample_lines_reservoir(args.wiki_urls, args.num_positives, args.seed)
        print(f"[positives] sampled {len(urls)} urls from {args.wiki_urls}")
        try:
            warcs = fetch_warcs(urls, work_dir, args.fetch_workers)
        except FileNotFoundError:
            raise SystemExit("wget not found; install it or use --paloma-bin locally")
        print(f"[positives] {len(warcs)} warc files fetched")
        positives = list(iter_warc_texts(warcs))
    else:
        positives = load_paloma_texts(args.paloma_bin, args.num_positives, args.seed)
    print(f"[positives] extracted {len(positives)} texts")

    negatives = sample_wet_texts(
        args.negatives_wet_dir, args.num_negatives, args.seed, args.negative_sample_rate
    )
    print(f"[negatives] sampled {len(negatives)} texts from {args.negatives_wet_dir}")

    if not args.no_clean_positives:
        positives, dropped = clean_positives(positives)
        print(f"[positives] after cleaning: {len(positives)} (dropped: {dict(dropped)})")

    rows = [(LABEL_POSITIVE, text) for text in positives] + [(LABEL_NEGATIVE, text) for text in negatives]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split = int(len(rows) * 0.95)
    train_path, valid_path = work_dir / "train.txt", work_dir / "valid.txt"
    write_fasttext_file(rows[:split], train_path, args.max_chars)
    write_fasttext_file(rows[split:], valid_path, args.max_chars)
    print(f"[data] train={split} valid={len(rows) - split} rows")

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        dim=args.dim,
        bucket=args.bucket,
        minCount=2,
        thread=os.cpu_count(),
        verbose=2,
    )
    n, precision, recall = model.test(str(valid_path))
    print(f"[eval] held-out: n={n} precision={precision:.3f} recall={recall:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    print(f"[done] saved model to {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
