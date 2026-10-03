"""§4.1 并行过滤 English WET 文件, 产出 JSONL(每行一篇文档)并统计各过滤器贡献。

    uv run python scripts/filter_data.py \\
        --wet-dir /shared-data/english-wet-data \\
        --output-dir /root/data/filtered \\
        --workers 8

冒烟测试(只处理前几个文件):

    uv run python scripts/filter_data.py --wet-dir shared-data/CC \\
        --output-dir shared-data/analysis/filtered_smoke --limit 1 --workers 2

每个 worker 会各自加载一遍 fastText 模型(lid 131MB + 两个 dolma ~1GB + 质量
分类器), 常驻内存约 2.2GB/进程 —— workers 数要按机器内存来定。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_data.pipeline import (  # noqa: E402
    FilterConfig,
    FilterStats,
    filter_document,
    iter_wet_documents,
    write_jsonl,
)


def process_wet_file(wet_path_str: str, out_path_str: str, cfg: FilterConfig) -> dict:
    wet_path = Path(wet_path_str)
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = FilterStats()
    with open(out_path, "w", encoding="utf-8") as fout:
        for url, text in iter_wet_documents(wet_path):
            stats.total += 1
            kept_text, reason, masks = filter_document(text, cfg)
            if kept_text is None:
                stats.dropped[reason] += 1
            else:
                stats.kept += 1
                stats.pii_masks.update(masks)
                write_jsonl(fout, {"url": url, "text": kept_text})

    return {"file": wet_path.name, **stats.as_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wet-dir", type=Path, required=True, help="含 *.warc.wet.gz 的目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件(冒烟)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--langid-threshold", type=float, default=0.7)
    parser.add_argument("--nsfw-threshold", type=float, default=0.9)
    parser.add_argument("--toxic-threshold", type=float, default=0.9)
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument("--no-pii-mask", action="store_true")
    parser.add_argument(
        "--only-langid",
        action="store_true",
        help="只做语言过滤(跳过 gopher/nsfw/toxic/quality/PII) —— 用作数据消融的对照组",
    )
    parser.add_argument("--report", type=Path, default=None, help="统计 JSON 输出路径")
    args = parser.parse_args()

    wet_files = sorted(args.wet_dir.glob("*.warc.wet.gz"))
    if args.limit:
        wet_files = wet_files[: args.limit]
    if not wet_files:
        raise SystemExit(f"no *.warc.wet.gz under {args.wet_dir}")

    cfg = FilterConfig(
        langid_threshold=args.langid_threshold,
        nsfw_threshold=args.nsfw_threshold,
        toxic_threshold=args.toxic_threshold,
        quality_threshold=args.quality_threshold,
        mask_pii=not args.no_pii_mask,
        skip=frozenset({"gopher", "nsfw", "toxic", "quality", "pii"}) if args.only_langid else frozenset(),
    )
    print(f"filtering {len(wet_files)} WET files with {args.workers} workers -> {args.output_dir}")

    total = FilterStats()
    per_file = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_wet_file, str(path), str(args.output_dir / f"{path.name.removesuffix('.warc.wet.gz')}.jsonl"), cfg
            ): path
            for path in wet_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="WET files"):
            result = future.result()
            per_file.append(result)
            total.merge(FilterStats.from_dict(result))

    print(f"\n=== 过滤统计: {total.total} 篇 -> 保留 {total.kept} ({total.kept / max(total.total, 1):.1%}) ===")
    for reason, count in total.dropped.most_common():
        print(f"  {reason:<8} 丢弃 {count:>10,} ({count / max(total.total, 1):.1%})")
    print(f"  PII 掩码: {dict(total.pii_masks)}")

    report_path = args.report or args.output_dir / "filter_stats.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                # skip 是 frozenset, 不能直接 JSON 序列化
                "config": {**asdict(cfg), "skip": sorted(cfg.skip)},
                "total": total.as_dict(),
                "per_file": per_file,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
