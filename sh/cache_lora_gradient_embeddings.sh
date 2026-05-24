#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PROJECTION_DIM="${PROJECTION_DIM:-512}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LORA_RANK="${LORA_RANK:-8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

python experiments/reference_diagnostic/cache_lora_gradient_embeddings.py \
  --model "$MODEL" \
  --kind task \
  --data outputs/motivating_diagnostic/resources/redorca_task_batches.jsonl \
  --output-embeddings outputs/motivating_diagnostic/redorca_task_lora_grad.npy \
  --output-metadata outputs/motivating_diagnostic/redorca_task_meta.jsonl \
  --projection-dim "$PROJECTION_DIM" \
  --max-length "$MAX_LENGTH" \
  --lora-rank "$LORA_RANK" \
  --target-modules q_proj v_proj \
  --torch-dtype "$TORCH_DTYPE" \
  --trust-remote-code

python experiments/reference_diagnostic/cache_lora_gradient_embeddings.py \
  --model "$MODEL" \
  --kind reference \
  --data outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
  --output-embeddings outputs/motivating_diagnostic/hh_reference_lora_grad.npy \
  --output-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --projection-dim "$PROJECTION_DIM" \
  --max-length "$MAX_LENGTH" \
  --lora-rank "$LORA_RANK" \
  --target-modules q_proj v_proj \
  --torch-dtype "$TORCH_DTYPE" \
  --trust-remote-code

