"""§4 去重: JSONL 目录 → 行级去重 → 文档级 MinHash/LSH 去重。

    uv run python scripts/deduplicate_data.py \\
        --input-dir /root/data/filtered \\
        --output-dir /root/data/deduped \\
        --num-hashes 128 --num-bands 16 --ngrams 5 --jaccard-threshold 0.8

中间产物: <output-dir>/1_line_dedup/ 与 <output-dir>/2_minhash_dedup/(最终结果)。
注意 MinHash 一步把全部文档文本 + n-gram 集合常驻内存, 是最吃内存的阶段。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_data.pipeline import exact_line_dedup_jsonl, minhash_dedup_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件(冒烟)")
    parser.add_argument("--no-line-dedup", action="store_true")
    parser.add_argument("--no-minhash", action="store_true")
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--ngrams", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="MinHash 签名并行进程数")
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.jsonl"))
    if args.limit:
        input_files = input_files[: args.limit]
    if not input_files:
        raise SystemExit(f"no *.jsonl under {args.input_dir}")

    report: dict = {"input_files": len(input_files)}

    if args.no_line_dedup:
        line_out_dir = args.input_dir
    else:
        line_out_dir = args.output_dir / "1_line_dedup"
        print(f"[line dedup] {len(input_files)} files -> {line_out_dir}")
        report["line_dedup"] = exact_line_dedup_jsonl(input_files, line_out_dir)
        print(f"[line dedup] {report['line_dedup']}")

    line_files = sorted(line_out_dir.glob("*.jsonl"))

    if args.no_minhash:
        return
    minhash_out_dir = args.output_dir / "2_minhash_dedup"
    print(f"[minhash] {len(line_files)} files -> {minhash_out_dir}")
    report["minhash"] = minhash_dedup_jsonl(
        line_files,
        minhash_out_dir,
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        ngrams=args.ngrams,
        jaccard_threshold=args.jaccard_threshold,
        workers=args.workers,
    )
    print(f"[minhash] {report['minhash']}")

    report_path = args.output_dir / "dedup_stats.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
