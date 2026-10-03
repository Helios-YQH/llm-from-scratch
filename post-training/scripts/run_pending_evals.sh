#!/bin/bash
# Fill the evaluation gaps the hand-run experiments left open. One card, sequential, ~1.5h.
#
#   1-2  V2's exact arm and noisy0.1 seed 1, re-evaluated with E1's protocol -- same
#        script, same 1319 test questions, temperature 1.0. The other three V2
#        checkpoints were re-evaluated this way when the validation-grader bug was
#        fixed; without these two the table has no upper endpoint and one arm's
#        second seed is scored by the wrong grader.
#   3    DPO's best-validation checkpoint merged: it is saved as a LoRA adapter, and
#        vLLM needs a plain model directory.
#   4-6  The alignment-tax evaluations (supplement 5.1/5.2/5(d)): SFT, DPO, and the
#        base model as a control, all wrapped in the Alpaca template the supplement
#        prescribes for instruction-tuned models.
#   7    SFT's held-out loss, split into prompt and response halves (the training
#        loss averages over both, and ~34% of the packed tokens are the prompt).
#
# Pick a card with >=20GB free but <38GB, so the training queue (min_free_mib=38000)
# does not try to co-locate a 45GB GRPO job on it.
#
#   GPU=3 bash scripts/run_pending_evals.sh

set -u
cd /mnt/14T/houyi/slm/post-training
export PATH="$PWD/.venv/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
GPU="${GPU:-3}"
UTIL="${UTIL:-0.35}"

echo "=== $(date -u +%FT%TZ) start  GPU=$GPU util=$UTIL ==="

echo "--- 1/7  V2 exact arm re-eval (E4 checkpoint, E1 protocol)"
$PY scripts/eval_prompting.py --model runs/e4_lr1e-05_seed0/checkpoints/final \
    --prompts r1_zero_three_shot_gsm8k --output-dir runs/_reeval_exact_seed0 \
    --gpu "$GPU" --port 8471 --gpu-memory-utilization "$UTIL"

echo "--- 2/7  V2 noisy0.1 seed1 re-eval"
$PY scripts/eval_prompting.py --model runs/track2_veritas_noisy0.1_seed1/checkpoints/final \
    --prompts r1_zero_three_shot_gsm8k --output-dir runs/_reeval_noisy0.1_seed1 \
    --gpu "$GPU" --port 8472 --gpu-memory-utilization "$UTIL"

echo "--- 3/7  merge DPO best adapter"
CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/merge_lora.py \
    --adapter runs/dpo/checkpoints/best --output runs/dpo/checkpoints/best_merged --device cuda:0

echo "--- 4/7  SFT alignment tax (Alpaca protocol)"
$PY scripts/eval_zero_shot.py --model runs/sft/checkpoints/final --prompt-style alpaca-sft \
    --output-dir runs/_tax_sft --gpu "$GPU" --port 8473 \
    --gpu-memory-utilization "$UTIL" --max-model-len 4096

echo "--- 5/7  DPO alignment tax (Alpaca protocol)"
$PY scripts/eval_zero_shot.py --model runs/dpo/checkpoints/best_merged --prompt-style alpaca-sft \
    --output-dir runs/_tax_dpo --gpu "$GPU" --port 8474 \
    --gpu-memory-utilization "$UTIL" --max-model-len 4096

echo "--- 6/7  base model, Alpaca protocol (control)"
$PY scripts/eval_zero_shot.py --model /mnt/14T/houyi/models/Meta-Llama-3.1-8B \
    --prompt-style alpaca-sft --output-dir runs/_tax_base_alpaca --gpu "$GPU" --port 8475 \
    --gpu-memory-utilization "$UTIL" --max-model-len 4096

echo "--- 7/7  SFT held-out loss"
CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/eval_sft_loss.py \
    --adapter runs/sft/checkpoints/step004000 --n-examples 500 --device cuda:0

echo "=== $(date -u +%FT%TZ) all done ==="
