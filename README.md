# ⚖️ Two to Tango: Coupled Task--Reference Selection for Safe LLM Fine-Tuning

This is the public release for the **DualSelect** codebase.

## 📖 Overview

Large Language Model (LLM) fine-tuning can improve task performance while weakening previously learned safe and helpful behavior. This often happens when task gradients conflict with gradients induced by safety-oriented reference examples. A fixed reference set is not always sufficient, because the most useful references depend on the current task-update direction.

**DualSelect** addresses this problem with a coupled task--reference selection pipeline. It first selects task-conditioned safe-response references with high preservation loss and strong conflict with the current task direction. It then filters whole task samples that are compatible with the induced reference direction and applies reference-gradient correction during fine-tuning.

The released code supports:

1. **Task-conditioned safe-response reference selection** from a disjoint reference pool
2. **Whole-sample task filtering** based on compatibility with the selected reference direction
3. **Reference-gradient correction** using the update rule `g_task + rho * g_ref`
4. **LoRA-gradient scoring utilities** for the reference-selection diagnostic
5. **Probe rollout analysis** for measuring held-out safety preservation


![DualSelect Framework](resource/dualselect.png)

## 🎉 News
* Our paper has been accepted to **EMNLP 2026**!


## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

For 4-bit loading, install a compatible `bitsandbytes` build separately.

### 2. Prepare data

DualSelect expects task data and reference data in JSON, JSONL, or CSV format. Each record should contain a prompt/instruction field and a response/completion field.

Supported task fields include:

- prompt-like fields: `instruction`, `prompt`, `question`, `query`, or `messages`
- response-like fields: `output`, `response`, `completion`, `answer`, or `chosen`

For diagnostic scripts, task records may also include:

- `batch_id`: batch identifier used to group task examples
- `task_group`: coarse task category, such as `Coding`, `Math`, `Reasoning`, `Writing`, or `QA`

Reference records may include:

- `safety_category`: reference category used for analysis
- `risk_score`: optional diagnostic metadata used by the reference-analysis scripts

In the paper setting, the reference pool is a prompt-disjoint Anthropic HH-RLHF safe-response pool. More generally, references can be any retention pool whose behavior should be preserved.

Any path shown as `...` is a placeholder and must be replaced with your local path.

### 3. Run DualSelect fine-tuning

The main training entry point is:

```bash
experiments/dualselect/train_dualselect.py
```

The provided shell wrapper exposes the most common options through environment variables:

```bash
MODEL=.../base-model \
TASK_DATA=.../task_train.jsonl \
REFERENCE_DATA=.../reference_pool.jsonl \
OUTPUT_DIR=outputs/dualselect/run \
bash sh/run_dualselect_train.sh
```

The final LoRA adapter is saved to `OUTPUT_DIR`. Reference and task-selection traces are written to:

```text
OUTPUT_DIR/selection_log.jsonl
```

For a small debug run, limit the number of examples and steps:

```bash
MODEL=.../base-model \
TASK_DATA=.../task_train.jsonl \
REFERENCE_DATA=.../reference_pool.jsonl \
OUTPUT_DIR=outputs/dualselect/debug \
MAX_TASK_SAMPLES=32 \
MAX_REFERENCE_SAMPLES=64 \
REFERENCE_SCORE_LIMIT=64 \
MAX_TRAIN_STEPS=2 \
TRAIN_BATCH_SIZE=4 \
TASK_TOP_K=2 \
REFERENCE_TOP_K=8 \
bash sh/run_dualselect_train.sh
```


## 🔧 Training Options

Common environment variables for `sh/run_dualselect_train.sh`:

- `MODEL`: base Hugging Face model name or local model path
- `TASK_DATA`: task fine-tuning data file
- `REFERENCE_DATA`: candidate reference pool
- `OUTPUT_DIR`: output directory for the LoRA adapter and logs
- `NUM_TRAIN_EPOCHS`: number of training epochs, default `1`
- `WARMUP_STEPS`: task-only warmup steps before selection, default `10`
- `TRAIN_BATCH_SIZE`: mini-batch size, default `8`
- `TASK_TOP_K`: selected task examples inside each mini-batch, default `4`
- `REFERENCE_TOP_K`: selected references per refresh, default `32`
- `SELECTION_MODE`: `hard` or `soft`, default `hard`
- `CORRECTION_ETA`: reference correction strength, default `0.2`
- `LEARNING_RATE`: optimizer learning rate, default `2e-5`
- `MAX_LENGTH`: tokenizer max sequence length, default `2048`
- `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`: LoRA configuration
- `TARGET_MODULES`: space-separated LoRA target modules, default `q_proj v_proj`
- `LOAD_IN_4BIT=1`: enable 4-bit model loading
- `SAVE_TASK_SELECTION=1`: write selected task indices to the selection log

In the paper experiments, selectors are matched by supervised target-token budget, with 90% retained target tokens and `Kref = ceil(0.1 * |Dref|)` for each lazy reference refresh. This public wrapper exposes compact Top-K controls (`TASK_TOP_K` and `REFERENCE_TOP_K`) so small debug runs can be launched without reproducing the full experimental budget.


## 📌 Paper Setting

The paper evaluates DualSelect on safe LLM fine-tuning with:

- **Models**: Gemma-3-1B-It, Qwen3-4B-Instruct-2507, and Llama-3-8B-Instruct
- **Task datasets**: REDORCA and GSM8K
- **Reference pool**: prompt-disjoint Anthropic HH-RLHF safe-response examples
- **Safety metrics**: HH and HEx-PHI win rates against Standard SFT, averaged as Safety Avg.
- **Utility metrics**: SlimOrca win rate for REDORCA and exact-match accuracy for GSM8K
- **Training recipe**: LoRA rank 16, scaling 32, dropout 0.05, AdamW, lazy reference refresh, and validation-tuned `rho`

The main empirical claim is that task-conditioned references preserve safety better than fixed or globally scored safety references while maintaining task utility. The diagnostic in the paper shows structured but non-static reference selection and higher held-out probe KL reduction than static references.


## 🧪 Optional Reference Diagnostic

The diagnostic pipeline studies whether selected references change with the task direction. It is optional and can be skipped if you only need training.

### 1. Build diagnostic resources

If you want to construct task batches, reference pools, and safety probes from REDORCA and HH-RLHF-style data, run:

```bash
python experiments/reference_diagnostic/prepare_motivating_resources.py \
  --redorca .../RedOrca/train.jsonl \
  --output-dir outputs/motivating_diagnostic/resources \
  --num-task-samples 1024 \
  --task-batch-size 64 \
  --num-reference-samples 4096 \
  --num-probe-samples 1024 \
  --seed 42
```

If Hugging Face dataset download is unavailable, pass local HH-RLHF-style files:

```bash
python experiments/reference_diagnostic/prepare_motivating_resources.py \
  --redorca .../RedOrca/train.jsonl \
  --hh-train-file .../hh_train.jsonl \
  --hh-probe-file .../hh_test.jsonl \
  --output-dir outputs/motivating_diagnostic/resources
```

This writes:

- `redorca_task_batches.jsonl`
- `redorca_task_meta.jsonl`
- `hh_reference_pool.jsonl`
- `hh_reference_meta.jsonl`
- `safety_probe.jsonl`
- `disjointness_report.json`

### 2. Cache task/reference embeddings

For the main diagnostic, use LoRA-gradient embeddings:

```bash
MODEL=.../base-model \
bash sh/cache_lora_gradient_embeddings.sh
```

For lightweight debugging only, the repository also includes LM-proxy embeddings:

```bash
MODEL=.../base-model \
bash sh/cache_lm_proxy_embeddings.sh
```

### 3. Select references

LoRA-gradient trace:

```bash
bash sh/select_reference_traces_lora_grad.sh
```

LM-proxy trace:

```bash
bash sh/select_reference_traces.sh
```

Both variants write:

```text
outputs/motivating_diagnostic/reference_trace.jsonl
```

### 4. Run probe rollout

```bash
MODEL=.../base-model \
bash sh/run_probe_rollout.sh
```

This writes rollout metrics to:

```text
outputs/motivating_diagnostic/probe_results.jsonl
```

### 5. Aggregate diagnostic results

```bash
python experiments/reference_diagnostic/run_reference_diagnostic.py \
  --trace outputs/motivating_diagnostic/reference_trace.jsonl \
  --reference-metadata outputs/motivating_diagnostic/resources/hh_reference_meta.jsonl \
  --probe-results outputs/motivating_diagnostic/probe_results.jsonl \
  --output-dir outputs/motivating_diagnostic/tables \
  --figure-path outputs/motivating_diagnostic/reference_diagnostic.pdf
```


## 📂 Repository Structure

- `experiments/dualselect/`: core DualSelect LoRA training implementation
- `experiments/reference_diagnostic/`: optional reference-selection diagnostics
- `sh/`: shell wrappers for training, embedding, selection, and rollout
- `resource/`: figures and static resources
- `requirements.txt`: version-pinned Python dependency list


## 📝 Notes

- Machine-specific paths, usernames, and local training artifacts have been removed from this public release.
- Replace every `...` placeholder with your local dataset or model path before running commands.
- Raw datasets, model checkpoints, generated outputs, logs, and caches are intentionally not bundled.
- The code assumes a Hugging Face-compatible causal language model and PEFT/LoRA support.
- The diagnostic scripts are optional; the training entry point can be used directly with your own task and reference data.
