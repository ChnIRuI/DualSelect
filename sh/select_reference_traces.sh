#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python experiments/reference_diagnostic/select_reference_traces.py \
  --task-embeddings outputs/motivating_diagnostic/redorca_task_proxy.npy \
  --reference-embeddings outputs/motivating_diagnostic/hh_reference_proxy.npy \
  --task-metadata outputs/motivating_diagnostic/redorca_task_meta.jsonl \
  --reference-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --top-k "${TOP_K:-32}" \
  --score-mode "${SCORE_MODE:-conflict}" \
  --risk-weight "${RISK_WEIGHT:-0.2}" \
  --batch-field batch_id \
  --strategies dualselect static_mean_conflict periodic_random uniform \
  --output outputs/motivating_diagnostic/reference_trace.jsonl \
  --write-task-indices
