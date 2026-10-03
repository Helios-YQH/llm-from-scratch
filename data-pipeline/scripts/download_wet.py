"""下载 Common Crawl 的 WET 分片(原始未过滤), 供 §4 数据管线使用。

    # 先下 100 个分片试试(约 6GB)
    python scripts/download_wet.py --count 100 --output-dir shared-data/raw-wet --workers 4

    # 全量(2500 个, 约 157GB)
    python scripts/download_wet.py --count 2500 --output-dir shared-data/raw-wet --workers 8

说明: 作业的官方流程是先用 fastText 只留英文 WET 再交给管线; 这里直接下原始 WET,
让管线的第一步(langid >= 0.7)承担英文过滤 —— 报告里因此会多出"语言过滤"这一级统计。
已存在的文件会跳过, 可反复运行/中断续传。
"""
from __future__ import annotations

import argparse
import gzip
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://data.commoncrawl.org/"


def sample_wet_paths(crawl_id: str, count: int, seed: int) -> list[str]:
    paths_url = f"{BASE_URL}crawl-data/{crawl_id}/wet.paths.gz"
    with urllib.request.urlopen(paths_url) as response:
        lines = gzip.decompress(response.read()).decode().splitlines()
    rng = random.Random(seed)
    return sorted(rng.sample(lines, min(count, len(lines))))


def download_one(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100, help="下载多少个 WET 分片")
    parser.add_argument("--crawl-id", default="CC-MAIN-2026-17")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, help="并行下载连接数")
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wet_paths = sample_wet_paths(args.crawl_id, args.count, args.seed)
    print(f"{len(wet_paths)} WET files from {args.crawl_id} -> {args.output_dir}", flush=True)

    done = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, BASE_URL + path, args.output_dir / Path(path).name): path
            for path in wet_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="WET files"):
            try:
                dest = future.result()
                done += 1
                total_bytes += dest.stat().st_size
            except Exception as exc:  # 单个文件失败不影响整体
                print(f"[fail] {futures[future]}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"downloaded {done}/{len(wet_paths)} files, {total_bytes / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
