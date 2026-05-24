#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python experiments/reference_diagnostic/prepare_motivating_resources.py \
  --redorca "${REDORCA:-.../RedOrca/train.jsonl}" \
  --output-dir outputs/motivating_diagnostic/resources \
  --num-task-samples "${NUM_TASK_SAMPLES:-1024}" \
  --task-batch-size "${TASK_BATCH_SIZE:-16}" \
  --num-reference-samples "${NUM_REFERENCE_SAMPLES:-4096}" \
  --num-probe-samples "${NUM_PROBE_SAMPLES:-512}" \
  --probe-min-risk-score "${PROBE_MIN_RISK_SCORE:-2}" \
  --seed "${SEED:-42}"
