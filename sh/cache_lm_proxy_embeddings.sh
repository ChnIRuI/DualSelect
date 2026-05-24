#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

python experiments/reference_diagnostic/cache_lm_proxy_embeddings.py \
  --model "$MODEL" \
  --kind task \
  --data outputs/motivating_diagnostic/resources/redorca_task_batches.jsonl \
  --output-embeddings outputs/motivating_diagnostic/redorca_task_proxy.npy \
  --output-metadata outputs/motivating_diagnostic/redorca_task_meta.jsonl \
  --batch-size 4 \
  --mode loss_weighted_hidden \
  --trust-remote-code

python experiments/reference_diagnostic/cache_lm_proxy_embeddings.py \
  --model "$MODEL" \
  --kind reference \
  --data outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
  --output-embeddings outputs/motivating_diagnostic/hh_reference_proxy.npy \
  --output-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --batch-size 4 \
  --mode loss_weighted_hidden \
  --trust-remote-code
