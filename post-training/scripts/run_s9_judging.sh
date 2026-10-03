#!/bin/bash
# Judge the six S9 prediction files with an OpenAI-compatible chat API.
#
# Runs on the box that holds the predictions, needs no GPU, and costs a few yuan
# instead of a rented 80GB card -- see scripts/judge_api.py for what "judge" means
# here and what it does not reproduce.
#
# Each (run, task, judge) pair is cached on disk, so re-running after an
# interruption costs nothing and judging the same files with a second model is
# just a second invocation.
#
#   JUDGE_API_KEY=sk-... bash scripts/run_s9_judging.sh gpt-5.6-luna
#   RUNS="base sft" WORKERS=12 JUDGE_API_KEY=... bash scripts/run_s9_judging.sh claude-haiku-4-5
#
# Only 'base' and 'sft' by default: 'dpo' is still being generated when the first
# judge goes out, and its two files simply are not there yet.

set -u
cd /mnt/14T/houyi/slm/post-training
PY=.venv/bin/python
JUDGE="${1:?usage: run_s9_judging.sh <judge-model>}"
RUNS="${RUNS:-base sft dpo}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
# Defaults to the relay. For a judge served locally by vLLM -- which needs no key --
# point BASE_URL at it and leave JUDGE_API_KEY unset; the placeholder is never checked.
BASE_URL="${BASE_URL:-https://tokeness.ai/v1}"
API_KEY="${JUDGE_API_KEY:-dummy}"

echo "=== $(date -u +%FT%TZ) judge=$JUDGE runs='$RUNS' workers=$WORKERS limit=$LIMIT base=$BASE_URL ==="

for name in $RUNS; do
    for task in alpaca_eval sst; do
        case "$task" in
            alpaca_eval) file="runs/_s9_$name/alpaca_eval.json" ;;
            sst)         file="runs/_s9_$name/sst.jsonl" ;;
        esac
        if [ ! -f "$file" ]; then
            echo "--- skip $name/$task: $file not there yet"
            continue
        fi
        echo "--- $name/$task  →  $JUDGE"
        $PY scripts/judge_api.py --task "$task" --predictions "$file" \
            --judge-model "$JUDGE" --output-dir "runs/_s9_$name/judged" \
            --base-url "$BASE_URL" --api-key "$API_KEY" \
            --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --limit "$LIMIT" \
            || echo "!!! $name/$task exited $? -- the next task is not a substitute for it"
    done
done

echo "=== $(date -u +%FT%TZ) all done ==="
