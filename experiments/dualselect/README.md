# DualSelect Training

This folder contains a runnable HF/PEFT implementation of the DualSelect
algorithm from the paper draft.

It implements:

- task-only warm-up to estimate the initial task direction,
- lazy reference refresh once per epoch,
- task-conditioned safe-response reference selection,
- per-batch reference-conditioned task selection,
- corrected update direction `g_task + eta * g_ref`.

## Quick Smoke Test

Run a tiny job first:

```bash
bash sh/run_dualselect_smoke.sh
```

## Main Run

```bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
TASK_DATA=.../RedOrca/train.jsonl \
REFERENCE_DATA=outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
OUTPUT_DIR=outputs/dualselect/llama3_redorca \
bash sh/run_dualselect_train.sh
```

Important knobs:

- `REFERENCE_TOP_K`: number of safe-response references selected per lazy refresh.
- `TASK_TOP_K`: number of task examples selected inside each mini-batch.
- `CORRECTION_ETA`: update-stage safety correction strength.
- `NU_CONFLICT`: strength of task-reference gradient interaction in scores.
- `REFERENCE_SCORE_LIMIT`: optional cap for reference scoring during debugging.
- `LOAD_IN_4BIT=1`: load base model in 4-bit for memory-limited GPUs.

The final LoRA adapter is saved to `OUTPUT_DIR`; selected references and,
optionally, selected task examples are written to `selection_log.jsonl`.
