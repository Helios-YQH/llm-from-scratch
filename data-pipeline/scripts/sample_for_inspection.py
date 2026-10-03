"""§4.2 抽样: 从产物里抽"保留"样例, 从原始 WET 里抽"被丢弃/被修改"样例。

    .venv-a4/bin/python scripts/sample_for_inspection.py \\
        --wet-dir shared-data/raw-wet --filtered-dir shared-data/deduped/2_minhash_dedup \\
        --output-dir shared-data/inspection --wet-files 4

输出三个 JSONL 到 --output-dir:
  kept.jsonl      随机保留的文档(url + text)
  discarded.jsonl 被过滤器丢弃的文档(带丢弃原因 + 命中的那一步)
  modified.jsonl  被 PII 掩码改写的文档(原文 + 改写后 + 各类掩码计数)
每行都是完整文档; 下游人工/模型标注时按行看即可。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_data.pipeline import FilterConfig, filter_document, iter_wet_documents, read_jsonl  # noqa: E402


def sample_kept(filtered_dir: Path, count: int, seed: int) -> list[dict]:
    files = sorted(filtered_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl under {filtered_dir}")
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    for path in files:
        for doc in read_jsonl(path):
            seen += 1
            if len(reservoir) < count:
                reservoir.append(doc)
            else:
                j = rng.randint(0, seen - 1)
                if j < count:
                    reservoir[j] = doc
    return reservoir


def sample_dropped_and_modified(
    wet_dir: Path, count: int, seed: int, wet_files: int, min_chars: int = 400
) -> tuple[list[dict], list[dict]]:
    """重跑过滤链, 收集被丢弃的文档(带原因)与被 PII 掩码改写的文档。"""
    rng = random.Random(seed + 1)
    files = sorted(wet_dir.glob("*.warc.wet.gz"))
    rng.shuffle(files)
    cfg = FilterConfig()

    discarded: dict[str, list[dict]] = {}
    modified: list[dict] = []
    for path in files[:wet_files]:
        for url, text in iter_wet_documents(path):
            if len(text) < min_chars:
                continue
            kept_text, reason, masks = filter_document(text, cfg)
            if reason is not None:
                bucket = discarded.setdefault(reason, [])
                if len(bucket) < count:
                    bucket.append({"url": url, "reason": reason, "text": text})
            elif masks:
                if len(modified) < count:
                    modified.append(
                        {"url": url, "masks": dict(masks), "original": text, "masked": kept_text}
                    )

    # 每个原因最多留 count 篇, 再均匀挑一些, 保证样例覆盖不同过滤器
    picked: list[dict] = []
    per_reason = max(1, count // max(len(discarded), 1))
    for reason, docs in sorted(discarded.items()):
        picked.extend(docs[:per_reason])
    return picked, modified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wet-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-kept", type=int, default=8)
    parser.add_argument("--num-discarded", type=int, default=20)
    parser.add_argument("--num-modified", type=int, default=6)
    parser.add_argument("--wet-files", type=int, default=4, help="重跑过滤链时扫描几个原始 WET 文件")
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    kept = sample_kept(args.filtered_dir, args.num_kept, args.seed)
    discarded, modified = sample_dropped_and_modified(
        args.wet_dir, args.num_discarded, args.seed, args.wet_files
    )

    for name, rows in [("kept", kept), ("discarded", discarded), ("modified", modified)]:
        out = args.output_dir / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
