#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TASK_DATA="${TASK_DATA:-.../RedOrca/train.jsonl}"
REFERENCE_DATA="${REFERENCE_DATA:-outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dualselect/llama3_redorca}"
IFS=' ' read -r -a TARGET_MODULES_ARRAY <<< "${TARGET_MODULES:-q_proj v_proj}"

EXTRA_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi
if [[ "${SAVE_TASK_SELECTION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-task-selection)
fi
if [[ -n "${MAX_TASK_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-task-samples "$MAX_TASK_SAMPLES")
fi
if [[ -n "${MAX_REFERENCE_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-reference-samples "$MAX_REFERENCE_SAMPLES")
fi
if [[ -n "${REFERENCE_SCORE_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--reference-score-limit "$REFERENCE_SCORE_LIMIT")
fi
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi

python experiments/dualselect/train_dualselect.py \
  --model-name-or-path "$MODEL" \
  --task-data "$TASK_DATA" \
  --reference-data "$REFERENCE_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --selection-log "$OUTPUT_DIR/selection_log.jsonl" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --warmup-steps "${WARMUP_STEPS:-10}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-8}" \
  --task-top-k "${TASK_TOP_K:-4}" \
  --reference-top-k "${REFERENCE_TOP_K:-32}" \
  --selection-mode "${SELECTION_MODE:-hard}" \
  --mu-task-loss "${MU_TASK_LOSS:-1.0}" \
  --mu-reference-loss "${MU_REFERENCE_LOSS:-1.0}" \
  --nu-conflict "${NU_CONFLICT:-1.0}" \
  --task-temperature "${TASK_TEMPERATURE:-0.1}" \
  --reference-temperature "${REFERENCE_TEMPERATURE:-0.1}" \
  --correction-eta "${CORRECTION_ETA:-0.2}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --target-modules "${TARGET_MODULES_ARRAY[@]}" \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  --logging-steps "${LOGGING_STEPS:-10}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}"
