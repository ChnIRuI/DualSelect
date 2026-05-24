# ⚖️ Two to Tango: Coupled Task--Reference Selection for Safe LLM Fine-Tuning

This is the public release for the **DualSelect** codebase.

## 📖 Overview

Large Language Model (LLM) fine-tuning can degrade safety-aligned behavior when task updates conflict with helpful or safe response patterns. A fixed reference set may be inefficient because the most useful safety references can vary with the current task-update direction.

**DualSelect** addresses this problem with a task-conditioned selection pipeline that chooses reference examples according to the current task direction and uses them to guide fine-tuning.

The current public release implements a practical workflow:

1. **Resource preparation** for task batches, safety references, and held-out probes
2. **Embedding or gradient caching** for task and reference examples
3. **Task-conditioned reference selection** with static and random baselines
4. **Diagnostic rollout and plotting** to measure reference behavior
5. **LoRA fine-tuning** with DualSelect-based reference and task selection


![DualSelect Framework](resource/dualselect.png)


## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Cache update-direction embeddings

Use LoRA-gradient embeddings for the main diagnostic:

```bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
bash sh/cache_lora_gradient_embeddings.sh
```

For lightweight debugging, the LM proxy embedding script is also provided:

```bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
bash sh/cache_lm_proxy_embeddings.sh
```

### 3. Select references

```bash
bash sh/select_reference_traces_lora_grad.sh
```

### 4. Run DualSelect fine-tuning

Before training, replace `MODEL` and `TASK_DATA` with your local model and dataset paths.

```bash
MODEL=.../base-model \
TASK_DATA=.../RedOrca/train.jsonl \
REFERENCE_DATA=outputs/motivating_diagnostic/resources/hh_reference_pool.jsonl \
OUTPUT_DIR=outputs/dualselect/run \
bash sh/run_dualselect_train.sh
```

For a small smoke test:

```bash
MODEL=.../base-model \
bash sh/run_dualselect_smoke.sh
```


## 📂 Repository Structure

- `experiments/dualselect/`: DualSelect LoRA training implementation
- `experiments/reference_diagnostic/`: resource preparation, embedding, selection, rollout, and plotting utilities
- `sh/`: shell templates for common experiment stages
- `resource/`: project figures and static resources
- `README.md`: public-facing project instructions
- `requirements.txt`: minimal Python dependency list


## 📝 Notes

- Machine-specific paths, usernames, and local training artifacts have been removed from this public release.
- Any remaining `...` placeholder must be replaced with your local path before running the corresponding script.
- Third-party datasets, checkpoints, generated outputs, logs, and caches are intentionally not bundled.
- Scripts assume a Hugging Face-compatible causal language model and JSON/JSONL/CSV training data.
- `LOAD_IN_4BIT=1` requires a working `bitsandbytes` installation.
