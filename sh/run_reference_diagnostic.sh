#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python experiments/reference_diagnostic/run_reference_diagnostic.py \
  --trace outputs/motivating_diagnostic/reference_trace.jsonl \
  --reference-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --probe-results outputs/motivating_diagnostic/probe_results.jsonl \
  --heatmap-strategy dualselect \
  --output-dir outputs/motivating_diagnostic/tables \
  --figure-path figs/reference_diagnostic.pdf
