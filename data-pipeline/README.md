# data-pipeline

Turns raw Common Crawl WET shards into language-model pretraining data, and
measures what each stage contributes: HTML-to-text extraction with encoding
detection, fastText language identification, regular-expression PII masking,
Dolma's Jigsaw harmful-content classifiers, the Gopher quality heuristics, a
fastText quality classifier, exact line deduplication, and MinHash+LSH document
deduplication — followed by a paired training ablation and a sample-level audit
of the pipeline's decisions.

**Report: [report/tech_report.pdf](report/tech_report.pdf)**

## Layout

- `lm_data/` — the pipeline package (filters, deduplication, pipeline
  components, WET handling) with its unit tests
- `scripts/` — `download_data.py` → `filter_data.py` → `deduplicate_data.py`
  → `tokenize_data.py`, plus `train_ablation.py` (the paired experiment) and
  the quality-classifier trainer
- `lm_basics/` — the baseline trainer (vendored, MIT-licensed) used by the
  ablation
- `tests/` — 21 unit tests across the primitives
- `report/` — the report, `make_figures.py`, and the committed measurement CSVs
  the figures read

## Quickstart

```bash
uv sync
uv run pytest                                  # 21 unit tests

# Full corpus build (see scripts/*.py --help for flags):
uv run python scripts/download_data.py --count 100 --output-dir shared-data/raw-wet
uv run python scripts/filter_data.py   --wet-dir shared-data/raw-wet \
        --output-dir shared-data/filtered --workers 12
```

Environment notes: fastText requires NumPy < 2 (its prediction path calls
`np.array(..., copy=False)`), and on Windows the `fasttext-wheel` package
provides the same module without needing a C++ toolchain.
