#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

python experiments/reference_diagnostic/run_probe_rollout.py \
  --model "$MODEL" \
  --trace outputs/motivating_diagnostic/reference_trace.jsonl \
  --redorca-task-batches outputs/motivating_diagnostic/resources/redorca_task_batches.jsonl \
  --hh-reference-pool outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
  --safety-probe outputs/motivating_diagnostic/resources/safety_probe.jsonl \
  --output outputs/motivating_diagnostic/probe_results.jsonl \
  --strategies dualselect static_mean_conflict periodic_random uniform \
  --max-batches "${MAX_BATCHES:-16}" \
  --probe-max-samples "${PROBE_MAX_SAMPLES:-128}" \
  --rollout-steps "${ROLLOUT_STEPS:-3}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --reference-batch-size "${REFERENCE_BATCH_SIZE:-4}" \
  --reference-loss-weight "${REFERENCE_LOSS_WEIGHT:-5.0}" \
  --correction-mode "${CORRECTION_MODE:-preserve_kl}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --lora-rank "${LORA_RANK:-8}" \
  --target-modules q_proj v_proj \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  --trust-remote-code
