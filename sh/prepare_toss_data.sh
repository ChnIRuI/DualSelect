#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
TOSS_DIR="${TOSS_DIR:-$ROOT_DIR/baseline/TOSS-main}"

REDORCA_IN=${REDORCA_IN:-".../RedOrca/train.jsonl"}
REDORCA_OUT=${REDORCA_OUT:-"$TOSS_DIR/data/toss/redorca_train_processed.jsonl"}
UTILITY_IN=${UTILITY_IN:-""}
UTILITY_OUT=${UTILITY_OUT:-"$TOSS_DIR/data/toss/utility_ref_messages.jsonl"}
HARMFUL_IN=${HARMFUL_IN:-""}
HARMFUL_OUT=${HARMFUL_OUT:-"$TOSS_DIR/data/toss/harmful_ref_messages.jsonl"}
MAX_UTILITY_SAMPLES=${MAX_UTILITY_SAMPLES:-22400}
MAX_HARMFUL_SAMPLES=${MAX_HARMFUL_SAMPLES:-22400}
UTILITY_RESPONSE_KEY=${UTILITY_RESPONSE_KEY:-auto}
HARMFUL_RESPONSE_KEY=${HARMFUL_RESPONSE_KEY:-auto}

args=(
  --redorca-in "$REDORCA_IN"
  --redorca-out "$REDORCA_OUT"
  --max-utility-samples "$MAX_UTILITY_SAMPLES"
  --max-harmful-samples "$MAX_HARMFUL_SAMPLES"
  --utility-response-key "$UTILITY_RESPONSE_KEY"
  --harmful-response-key "$HARMFUL_RESPONSE_KEY"
)

if [[ -n "$UTILITY_IN" ]]; then
  args+=(--utility-in "$UTILITY_IN" --utility-out "$UTILITY_OUT")
fi

if [[ -n "$HARMFUL_IN" ]]; then
  args+=(--harmful-in "$HARMFUL_IN" --harmful-out "$HARMFUL_OUT")
fi

python "$TOSS_DIR/train/examples/prepare_toss_data.py" "${args[@]}"
