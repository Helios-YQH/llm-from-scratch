#!/bin/bash
# Generate the six prediction files S9's 70B judge scores: base / SFT / DPO, each
# on AlpacaEval and SimpleSafetyTests.
#
# None of this needs the rented card -- it is all 8B, one card, ~20 minutes. Only
# the judging has to run where the 70B lives, so this can run on the shared box
# while the rented one is still downloading weights.
#
# Which wrapper goes around the instruction differs by model and is not a detail:
# the supplement prescribes the zero-shot transcript for the base model (3.3/3.4)
# and the Alpaca template for the instruction-tuned ones (5.3/5.4) -- the same
# split `run_pending_evals.sh` uses for MMLU and GSM8K.
#
# util 0.45 because 8B needs ~16GB plus a few GB of KV cache; holding the card at
# the default 0.85 would take 40GB from everyone else for no gain in throughput.
#
#   GPU=0 nohup bash scripts/run_s9_generation.sh > /tmp/s9_gen.log 2>&1 &

set -u
cd /mnt/14T/houyi/slm/post-training
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
GPU="${GPU:-0}"
UTIL="${UTIL:-0.45}"

echo "=== $(date -u +%FT%TZ) start  GPU=$GPU util=$UTIL ==="

run() {
    local name="$1" model="$2" style="$3" port="$4"
    echo "--- $name  ($model, $style)"
    $PY scripts/eval_alpaca_sst.py --model "$model" --prompt-style "$style" \
        --generator-name "llama-3.1-8b-$name" --output-dir "runs/_s9_$name" \
        --gpu "$GPU" --port "$port" --gpu-memory-utilization "$UTIL"
}

# DPO is the val-best checkpoint, not the last one, as the supplement asks (4.2);
# `best` is a bare LoRA adapter, so it is the merged copy that vLLM can load.
run base /mnt/14T/houyi/models/Meta-Llama-3.1-8B  zero-shot-system 8500
run sft  runs/sft/checkpoints/final               alpaca-sft       8501
run dpo  runs/dpo/checkpoints/best_merged         alpaca-sft       8502

echo "=== $(date -u +%FT%TZ) all done ==="
