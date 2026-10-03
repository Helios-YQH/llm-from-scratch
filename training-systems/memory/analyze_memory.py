"""Analyze PyTorch memory snapshot pickle files.

Reconstructs the active-memory timeline, reports peak memory, and groups
allocations by their Python call-site to identify what dominates memory.
"""
import pickle
import sys
import re
from collections import defaultdict


def parse_snapshot(path):
    with open(path, "rb") as f:
        snap = pickle.load(f)
    return snap["device_traces"][0]


def python_frames(frames):
    """Return frames that have a real Python file (nn_utils.py, model.py, etc.)."""
    out = []
    for fr in frames:
        fn = fr.get("filename", "")
        if fn.endswith(".py"):
            out.append(f"{fr.get('name', '?')} @ {fn}:{fr.get('line', 0)}")
    return out


def categorize(frames):
    """Classify an allocation by the deepest Python call-site."""
    py = python_frames(frames)
    if not py:
        return "(no python frame)"
    # Find the first frame that is in lm_basics (our model code) or the
    # generic op name, to group sensibly.
    deepest = py[0]
    # pull out the file basename
    m = re.search(r"([\w.]+\.py):(\d+)", deepest)
    if m:
        file = m.group(1)
        line = int(m.group(2))
        # model.py line -> identify attention vs ffn
        if "nn_utils" in file and "softmax" in deepest:
            return "softmax (exp) — nn_utils.py"
        if "model.py" in file and line in range(425, 435):
            return "scaled_dot_product_attention — model.py:432"
        return f"{deepest}"
    return deepest


def main():
    for path in sys.argv[1:]:
        events = parse_snapshot(path)
        active = 0
        peak = 0
        allocs = {}  # addr -> (size, category)
        by_cat = defaultdict(int)  # category -> total bytes currently/live

        for e in events:
            if e["action"] == "alloc":
                addr, size = e["addr"], e["size"]
                cat = categorize(e.get("frames", []))
                allocs[addr] = (size, cat)
                active += size
                peak = max(peak, active)
            elif e["action"] == "free":
                addr = e["addr"]
                if addr in allocs:
                    size, cat = allocs.pop(addr)
                    active -= size

        print(f"\n===== {path} =====")
        print(f"peak active memory: {peak/1024**2:.1f} MiB")

        # Live allocations at end grouped by category
        live_by_cat = defaultdict(int)
        for addr, (size, cat) in allocs.items():
            live_by_cat[cat] += size
        print("live allocations by category (end of profile):")
        for cat, total in sorted(live_by_cat.items(), key=lambda kv: -kv[1]):
            print(f"  {total/1024**2:8.1f} MiB  {cat}")


if __name__ == "__main__":
    main()
