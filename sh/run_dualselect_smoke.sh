#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TASK_DATA="${TASK_DATA:-outputs/motivating_diagnostic/resources/redorca_task_batches.jsonl}"
REFERENCE_DATA="${REFERENCE_DATA:-outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl}"
IFS=' ' read -r -a TARGET_MODULES_ARRAY <<< "${TARGET_MODULES:-q_proj v_proj}"
EXTRA_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi

python experiments/dualselect/train_dualselect.py \
  --model-name-or-path "$MODEL" \
  --task-data "$TASK_DATA" \
  --reference-data "$REFERENCE_DATA" \
  --output-dir outputs/dualselect/smoke \
  --selection-log outputs/dualselect/smoke/selection_log.jsonl \
  --max-task-samples 32 \
  --max-reference-samples 64 \
  --reference-score-limit 64 \
  --num-train-epochs 1 \
  --max-train-steps 2 \
  --warmup-steps 1 \
  --train-batch-size 4 \
  --task-top-k 2 \
  --reference-top-k 8 \
  --selection-mode hard \
  --correction-eta 0.2 \
  --learning-rate 2e-5 \
  --max-length 1024 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --target-modules "${TARGET_MODULES_ARRAY[@]}" \
  --torch-dtype bfloat16 \
  --logging-steps 1 \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}"
