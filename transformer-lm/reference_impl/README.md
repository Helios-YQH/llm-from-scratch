# reference_impl — library-based cross-checks

The main implementation is deliberately from-scratch; knowing how the same job
is done with real libraries is equally important. This folder implements each
component **with the libraries** (tiktoken / HuggingFace `tokenizers` /
`torch` / `transformers`) as an independent oracle for the from-scratch code.

## BPE tokenizer

| Script | Library | What it does |
|---|---|---|
| `seed_tiktoken_cache.py` | tiktoken | Seeds the GPT-2 encoding cache offline (the fix when the download is blocked) |
| `tokenize_with_tiktoken.py` | tiktoken | Byte-level BPE encode/decode with tiktoken, compared byte-for-byte with the from-scratch `Tokenizer` |
| `train_with_tokenizers.py` | tokenizers | Trains a byte-level BPE with HuggingFace `tokenizers`; artifacts land in `output/` |

## Running

```bash
cd transformer-lm
uv run python reference_impl/seed_tiktoken_cache.py
uv run python reference_impl/tokenize_with_tiktoken.py
uv run python reference_impl/train_with_tokenizers.py
```

## Correspondence (from-scratch → off-the-shelf)

| From-scratch component | Off-the-shelf equivalent |
|---|---|
| `train_bpe` (incremental merges) | `tokenizers.BPE` + `BpeTrainer` (Rust backend, much faster) |
| GPT-2 pretokenization PAT | `tokenizers.pre_tokenizers.ByteLevel()` (or ByteLevel + a custom regex) |
| `Tokenizer.encode` | `tiktoken.get_encoding("gpt2").encode` |
| `save_bpe` / `from_files` | `tokenizer.save("tokenizer.json")` / `Tokenizer.from_file` |
| byte ↔ printable-char mapping | the `bytes_to_unicode` built into tiktoken / transformers |

## What the cross-checks found

On the TinyStories validation set, `tokenize_with_tiktoken.py` shows the
from-scratch `Tokenizer` and tiktoken produce **identical token IDs**;
compression is about 4 bytes/token.

## Transformer resource accounting

`transformer_accounting.py` computes parameter counts, the FLOPs breakdown, and
training-time estimates on A100 / A6000 / B200:

```bash
uv run python reference_impl/transformer_accounting.py
```

## Full training reference

`train_lm.py` runs one end-to-end training with off-the-shelf components:
`tokenizers.BPE` for the tokenizer, `transformers.GPT2LMHeadModel` for the
model, `torch.optim.AdamW` + `SequentialLR` for the optimizer; tokenizer and
weights are saved to disk.

```bash
uv run python reference_impl/train_lm.py --train_txt data/TinyStories-valid.txt \
    --val_txt data/TinyStories-valid.txt --vocab_size 1000 --total_iters 10
```

| From-scratch component | Off-the-shelf equivalent |
|---|---|
| `train_bpe` (incremental merges) | `tokenizers.BPE` + `BpeTrainer` (Rust backend) |
| `TransformerLM` | `transformers.GPT2LMHeadModel` (or `LlamaForCausalLM`) |
| `AdamW` | `torch.optim.AdamW` (built-in) |
| `get_lr_cosine_schedule` | `LinearLR` + `CosineAnnealingLR` → `SequentialLR` |
| `cross_entropy` | `F.cross_entropy` (the same numerical trick) |
| `clip_gradient_norm` | `torch.nn.utils.clip_grad_norm_` (built-in) |
| `run_save_checkpoint` | `model.save_pretrained()` / `torch.save` |
| `run_get_batch` | `IterableDataset` + `DataLoader` |
