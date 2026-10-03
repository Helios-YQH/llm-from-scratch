"""CPU-side measurements for the tech report's tokenizer gaps (run on the server).

Covers the two items from 实验记录.md that need the training host:

  * §2.5  BPE training: wall-clock, peak memory of the whole process tree,
          and a stage profile (which phase dominates).
  * §2.7c tokenizer encoding throughput, and the extrapolation to the Pile.

Writes everything to stdout; run_report_gaps.sh redirects it into logs/report_gaps/.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import subprocess
import sys
import time
from pathlib import Path

import psutil

CORPUS = Path("data/TinyStories-train.txt")
SUBSET = Path("/tmp/ts_200mb.txt")
VOCAB_SIZE = 10_000
SPECIAL = ["<|endoftext|>"]
PILE_BYTES = 825 * 1024**3

_TRAIN_CHILD = """
import time
from lm_basics.tokenizer import train_bpe, save_bpe
t0 = time.time()
vocab, merges = train_bpe({corpus!r}, {vocab_size}, {special!r})
t1 = time.time()
save_bpe(vocab, merges, "data/report_gaps_vocab.json", "data/report_gaps_merges.txt")
print(f"TRAIN_BPE_SECONDS {{t1 - t0:.1f}}")
print(f"TOTAL_SECONDS {{time.time() - t0:.1f}}")
print(f"VOCAB {{len(vocab)}} MERGES {{len(merges)}}")
"""


def hdr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}", flush=True)


def bpe_timing() -> None:
    """Full-corpus BPE run: wall clock + peak RSS over the whole process tree."""
    hdr("BPE training on the full 1.92 GB TinyStories corpus")
    print(f"corpus: {CORPUS} ({CORPUS.stat().st_size / 1e9:.2f} GB), "
          f"vocab_size={VOCAB_SIZE}, special={SPECIAL}")

    code = _TRAIN_CHILD.format(corpus=str(CORPUS), vocab_size=VOCAB_SIZE, special=SPECIAL)
    proc = subprocess.Popen([sys.executable, "-c", code])
    child = psutil.Process(proc.pid)
    peak_rss = 0
    t0 = time.time()
    while proc.poll() is None:
        try:
            rss = child.memory_info().rss
            rss += sum(c.memory_info().rss for c in child.children(recursive=True))
        except psutil.Error:
            rss = 0
        peak_rss = max(peak_rss, rss)
        time.sleep(0.2)

    print(f"wall clock (incl. save)    : {time.time() - t0:.1f} s")
    print(f"PEAK_RSS_GIB               : {peak_rss / 1024**3:.2f} GiB "
          f"(main process + children)")


def bpe_profile() -> None:
    """Which stage dominates, on a 200 MB subset (children are not profiled)."""
    hdr("BPE training profile (200 MB subset)")
    with open(CORPUS, "rb") as src, open(SUBSET, "wb") as dst:
        dst.write(src.read(200_000_000))
    print(f"copied the first {SUBSET.stat().st_size / 1e6:.0f} MB of the corpus")

    from lm_basics.tokenizer import train_bpe

    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    train_bpe(str(SUBSET), VOCAB_SIZE, SPECIAL)
    pr.disable()
    print(f"subset wall clock: {time.time() - t0:.1f} s")

    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("cumulative").print_stats(15)
    print("\ntop functions by cumulative time (main process only):")
    print(stream.getvalue())
    SUBSET.unlink(missing_ok=True)


def encode_throughput() -> None:
    hdr("Tokenizer encoding throughput on this host")
    from lm_basics.tokenizer import Tokenizer

    tok = Tokenizer.from_files("data/vocab.json", "data/merges.txt", SPECIAL)
    lines: list[str] = []
    n_bytes = 0
    with open(CORPUS, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            n_bytes += len(line.encode("utf-8"))
            if n_bytes >= 20 * 1024 * 1024:
                break

    t0 = time.perf_counter()
    n_tok = sum(1 for _ in tok.encode_iterable(lines))
    elapsed = time.perf_counter() - t0

    rate = n_bytes / elapsed
    print(f"encoded {n_bytes / 1e6:.1f} MB -> {n_tok} tokens in {elapsed:.1f} s "
          f"(single process)")
    print(f"THROUGHPUT                 : {rate / 1e6:.3f} MB/s, "
          f"{n_tok / elapsed / 1e3:.1f} K tokens/s")
    print(f"Pile (825 GB) extrapolation: {PILE_BYTES / rate / 3600:.1f} h "
          f"= {PILE_BYTES / rate / 86400:.1f} days")


def main() -> None:
    bpe_timing()
    bpe_profile()
    encode_throughput()


if __name__ == "__main__":
    main()
