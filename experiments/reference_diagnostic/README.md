# Motivating Diagnostic: Are Safety References Task-Conditioned?

This folder implements the diagnostic described in Section
`Motivating Diagnostic: Are Safety References Task-Conditioned?`.

The intended setting is:

- model: `meta-llama/Meta-Llama-3-8B-Instruct`
- task data: fixed REDORCA mini-batches
- safe-response reference pool: chosen responses from the Anthropic HH-RLHF training split
- safety probe: a separate held-out safety-specific probe set, filtered from
  Anthropic HH-RLHF test split by risk/refusal keywords

The REDORCA batches are fixed and are not filtered by the task selector. The experiment therefore isolates whether the reference side changes with the current task-update direction.

## 0. Data Status

The checked-in REDORCA file may be a Git LFS pointer. Verify before running:

```bash
head .../RedOrca/train.jsonl
```

If it prints `version https://git-lfs.github.com/spec/v1`, resolve the real file first:

```bash
git lfs pull
```

or pass `--redorca` to a real REDORCA JSONL path.

## 1. Prepare Disjoint Resources

This creates:

- `redorca_task_batches.jsonl`: fixed REDORCA task mini-batches
- `redorca_task_meta.jsonl`: `batch_id` and coarse `task_group` per REDORCA sample
- `hh_reference_pool.jsonl`: HH-RLHF train-split chosen safe/helpful responses
- `hh_reference_meta.jsonl`: heuristic `safety_category` per HH reference
- `safety_probe.jsonl`: held-out safety probe examples
- `disjointness_report.json`: prompt-hash overlap checks

```bash
python experiments/reference_diagnostic/prepare_motivating_resources.py \
  --redorca .../RedOrca/train.jsonl \
  --output-dir outputs/motivating_diagnostic/resources \
  --num-task-samples 1024 \
  --task-batch-size 16 \
  --num-reference-samples 4096 \
  --num-probe-samples 512 \
  --probe-min-risk-score 2 \
  --seed 42
```

By default, REDORCA task batches are stratified by coarse task group
(`Coding`, `Math`, `Reasoning`, `Writing`, `QA`) and each batch receives a
group-specific `batch_id`. This is important for the middle heatmap: randomly
mixed REDORCA batches are usually majority-`QA`.

The held-out probe is safety-specific rather than a random HH test sample. It
filters for risk/refusal patterns in categories such as `Violence`, `Drugs`,
`Self-harm`, `Privacy`, `Fraud`, and `Hate`, so the rollout panel measures
safety-behavior drift instead of ordinary helpful-QA drift.

If the machine cannot download `Anthropic/hh-rlhf` through Hugging Face, provide local files:

```bash
python experiments/reference_diagnostic/prepare_motivating_resources.py \
  --redorca .../RedOrca/train.jsonl \
  --hh-train-file .../hh_rlhf_train.jsonl \
  --hh-probe-file .../hh_rlhf_test.jsonl \
  --output-dir outputs/motivating_diagnostic/resources
```

You can also pass final evaluation files to check prompt-level disjointness against them:

```bash
--final-eval-files .../eval1.jsonl .../eval2.jsonl
```

## 2. Cache Update-Direction Embeddings

The recommended diagnostic uses LoRA-gradient embeddings: each example is
represented by the response-token SFT-loss gradient with respect to LoRA
parameters, compressed by deterministic feature hashing.

Task side:

```bash
python experiments/reference_diagnostic/cache_lora_gradient_embeddings.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --kind task \
  --data outputs/motivating_diagnostic/resources/redorca_task_batches.jsonl \
  --output-embeddings outputs/motivating_diagnostic/redorca_task_lora_grad.npy \
  --output-metadata outputs/motivating_diagnostic/redorca_task_meta.jsonl \
  --projection-dim 512 \
  --max-length 1024 \
  --lora-rank 8 \
  --target-modules q_proj v_proj \
  --torch-dtype bfloat16 \
  --trust-remote-code
```

Reference side:

```bash
python experiments/reference_diagnostic/cache_lora_gradient_embeddings.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --kind reference \
  --data outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
  --output-embeddings outputs/motivating_diagnostic/hh_reference_lora_grad.npy \
  --output-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --projection-dim 512 \
  --max-length 1024 \
  --lora-rank 8 \
  --target-modules q_proj v_proj \
  --torch-dtype bfloat16 \
  --trust-remote-code
```

If you have exact full gradients from your training stack, replace the two
`.npy` files above with your gradient arrays and keep the metadata files.
For quick debugging only, you can still use `cache_lm_proxy_embeddings.py`.

## 3. Select Task-Conditioned References

```bash
python experiments/reference_diagnostic/select_reference_traces.py \
  --task-embeddings outputs/motivating_diagnostic/redorca_task_lora_grad.npy \
  --reference-embeddings outputs/motivating_diagnostic/hh_reference_lora_grad.npy \
  --task-metadata outputs/motivating_diagnostic/redorca_task_meta.jsonl \
  --reference-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --top-k 32 \
  --score-mode conflict \
  --risk-weight 0.2 \
  --batch-field batch_id \
  --strategies dualselect static_mean_conflict periodic_random uniform \
  --output outputs/motivating_diagnostic/reference_trace.jsonl
```

The default selection score is `conflict = -cos(task_direction, reference_direction)`, with a small z-scored safety-risk bonus. This keeps the reference side task-conditioned while reducing the chance that top-K references are dominated by generic `Other` HH conversations. `static_mean_conflict` is the fixed-set baseline, `periodic_random` is the random selected-set baseline, and `uniform` gives category-balanced references.

## 4. Held-Out Probe Rollout

The aggregation script expects rollout results as JSONL or CSV with:

```json
{"strategy": "dualselect", "kl_before": 0.142, "kl_after": 0.097}
```

Here `kl_before - kl_after` is the safety-probe KL reduction after the K-step rollout. In the default implementation, `kl_before` is the safety-probe KL drift after a task-only REDORCA rollout, and `kl_after` is the drift after a task rollout with selected-reference preservation under the same REDORCA batch and strategy. Keep the rollout probe data separate:

```text
outputs/motivating_diagnostic/resources/safety_probe.jsonl
```

Run:

```bash
python experiments/reference_diagnostic/run_probe_rollout.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --trace outputs/motivating_diagnostic/reference_trace.jsonl \
  --redorca-task-batches outputs/motivating_diagnostic/resources/redorca_task_batches.jsonl \
  --hh-reference-pool outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
  --safety-probe outputs/motivating_diagnostic/resources/safety_probe.jsonl \
  --output outputs/motivating_diagnostic/probe_results.jsonl \
  --strategies dualselect static_mean_conflict periodic_random uniform \
  --max-batches 16 \
  --probe-max-samples 128 \
  --rollout-steps 3 \
  --train-batch-size 4 \
  --reference-batch-size 4 \
  --reference-loss-weight 5.0 \
  --correction-mode preserve_kl \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --target-modules q_proj v_proj \
  --torch-dtype bfloat16 \
  --trust-remote-code
```

This writes one row per rollout batch and strategy to:

```text
outputs/motivating_diagnostic/probe_results.jsonl
```

`--correction-mode preserve_kl` optimizes task SFT loss while penalizing
KL(base || current) on the selected HH references. This is closer to the paper's
motivation: references preserve safe/helpful behavior instead of acting as extra
SFT examples. For ablations, use `project_conflict`, `remove_conflict`,
`align_ref`, or `additive_loss`.

## One-Command Shell Pipeline

The same flow is available through shell wrappers:

```bash
bash sh/prepare_motivating_resources.sh
bash sh/cache_lora_gradient_embeddings.sh
bash sh/select_reference_traces_lora_grad.sh
MAX_BATCHES=64 PROBE_MAX_SAMPLES=512 ROLLOUT_STEPS=3 bash sh/run_probe_rollout.sh
bash sh/plot_reference_diagnostic.sh
```

## 5. Aggregate And Plot

```bash
python experiments/reference_diagnostic/run_reference_diagnostic.py \
  --trace outputs/motivating_diagnostic/reference_trace.jsonl \
  --reference-metadata outputs/motivating_diagnostic/hh_reference_meta.jsonl \
  --probe-results outputs/motivating_diagnostic/probe_results.jsonl \
  --heatmap-strategy dualselect \
  --output-dir outputs/motivating_diagnostic/tables \
  --figure-path figs/reference_diagnostic.pdf
```

Outputs:

- `overlap_by_strategy.csv`
- `category_heatmap.csv`
- `probe_reduction.csv`, if probe values are available
- `metrics.json`
- `reference_diagnostic.pdf`, if Matplotlib is installed

## Trace Schema

`run_reference_diagnostic.py` consumes a JSONL trace with one row per REDORCA batch and strategy:

```json
{
  "strategy": "dualselect",
  "batch_id": "redorca_batch_000123",
  "task_group": "Coding",
  "selected_ref_indices": [12, 91, 103],
  "selected_ref_scores": [0.81, 0.79, 0.77],
  "reference_categories": ["Privacy", "Fraud", "Privacy"],
  "pool_size": 4096,
  "top_k": 32
}
```
