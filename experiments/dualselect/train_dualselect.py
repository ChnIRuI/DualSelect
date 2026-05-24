from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


IGNORE_INDEX = -100


def read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        for key in ("train", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported data file: {path}")


def first_existing(record: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def example_prompt_response(record: Dict[str, Any]) -> Tuple[str, str]:
    if "messages" in record and isinstance(record["messages"], list):
        prompt_parts = []
        response = ""
        for message in record["messages"]:
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if role == "assistant":
                response = content
            else:
                prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts), response

    prompt = first_existing(record, ["instruction", "prompt", "question", "query"])
    user_input = str(record.get("input", "")).strip()
    if user_input and user_input != prompt:
        prompt = prompt + "\n" + user_input if prompt else user_input
    response = first_existing(record, ["output", "response", "completion", "answer", "chosen"])
    return prompt, response


def encode_record(record: Dict[str, Any], tokenizer: Any, max_length: int, with_prompt_token: bool) -> Dict[str, Any]:
    import torch

    prompt, response = example_prompt_response(record)
    if not prompt or not response:
        raise ValueError("Record has empty prompt or response")
    if prompt.lstrip().startswith(("Human:", "User:", "<|user|>", "<|begin_of_text|>")):
        prompt_text = prompt.rstrip() + "\nAssistant: "
    else:
        prompt_text = f"Human: {prompt}\nAssistant: "
    full_text = prompt_text + response + " " + tokenizer.eos_token

    tokenized = tokenizer(full_text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = tokenized.input_ids[0]
    labels = input_ids.clone()
    if not with_prompt_token:
        prompt_len = tokenizer(prompt_text, return_tensors="pt", max_length=max_length, truncation=True).input_ids.shape[1]
        labels[:prompt_len] = IGNORE_INDEX
    if labels.ne(IGNORE_INDEX).sum().item() == 0:
        raise ValueError("Encoded record has no supervised response tokens")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }


def collate_encoded(examples: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    import torch

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in examples], batch_first=True, padding_value=pad_token_id
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in examples], batch_first=True, padding_value=IGNORE_INDEX
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in examples], batch_first=True, padding_value=0
        ),
    }


def batch_indices(num_items: int, batch_size: int, rng: random.Random, shuffle: bool) -> List[List[int]]:
    indices = list(range(num_items))
    if shuffle:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, num_items, batch_size)]


def select_trainable_params(model: Any) -> List[Tuple[str, Any]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def zero_like_trainable(model: Any) -> List[Any]:
    import torch

    return [torch.zeros_like(param, device=param.device) for _, param in select_trainable_params(model)]


def clone_grads(model: Any) -> List[Any]:
    return [
        param.grad.detach().clone() if param.grad is not None else None
        for _, param in select_trainable_params(model)
    ]


def add_scaled_(target: List[Any], source: Sequence[Any], scale: float) -> None:
    for idx, grad in enumerate(source):
        if grad is not None:
            target[idx].add_(grad, alpha=float(scale))


def grad_dot(left: Sequence[Any], right: Sequence[Any]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        total += float((a.float() * b.float()).sum().detach().cpu().item())
    return total


def grad_norm(grads: Sequence[Any], eps: float = 1e-12) -> float:
    return math.sqrt(max(grad_dot(grads, grads), eps))


def grad_cosine(left: Sequence[Any], right: Sequence[Any], eps: float = 1e-12) -> float:
    return grad_dot(left, right) / ((grad_norm(left, eps) + eps) * (grad_norm(right, eps) + eps))


def assign_grads(model: Any, grads: Sequence[Any]) -> None:
    for (_, param), grad in zip(select_trainable_params(model), grads):
        param.grad = None if grad is None else grad.detach().clone()


def weighted_average_grads(grads: Sequence[Sequence[Any]], weights: Sequence[float]) -> List[Any]:
    import torch

    if not grads:
        raise ValueError("Cannot average empty gradient list")
    out = [torch.zeros_like(g) if g is not None else None for g in grads[0]]  # noqa: F821
    total = max(float(sum(weights)), 1e-12)
    for grad_list, weight in zip(grads, weights):
        for idx, grad in enumerate(grad_list):
            if grad is not None and out[idx] is None:
                out[idx] = torch.zeros_like(grad)
            if grad is not None and out[idx] is not None:
                out[idx].add_(grad, alpha=float(weight) / total)
    return out


def softmax_np(scores: Sequence[float], temperature: float) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    temp = max(float(temperature), 1e-8)
    arr = arr / temp
    arr = arr - arr.max()
    weights = np.exp(arr)
    return weights / max(float(weights.sum()), 1e-12)


def softmin_np(scores: Sequence[float], temperature: float) -> np.ndarray:
    return softmax_np([-float(score) for score in scores], temperature)


@dataclass
class ExampleGrad:
    index: int
    loss: float
    grads: List[Any]
    score: float = 0.0
    cosine: float = 0.0


class DualSelectTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        np.random.seed(args.seed)
        self._setup_model()
        self._setup_data()

    def _setup_model(self) -> None:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.args.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.args.device

        dtype = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.args.torch_dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=not self.args.use_slow_tokenizer,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token

        model_kwargs: Dict[str, Any] = {"trust_remote_code": self.args.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype is None else dtype,
            )
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, **model_kwargs)
        self.model.config.use_cache = False
        if self.args.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        if not self.args.load_in_4bit:
            self.model.to(self.device)
        self.model.print_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            [param for _, param in select_trainable_params(self.model)],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _setup_data(self) -> None:
        task_records = read_records(self.args.task_data)
        ref_records = read_records(self.args.reference_data)
        if self.args.max_task_samples is not None:
            task_records = task_records[: self.args.max_task_samples]
        if self.args.max_reference_samples is not None:
            ref_records = ref_records[: self.args.max_reference_samples]

        self.task_examples = self._encode_many(task_records, "task")
        self.reference_examples = self._encode_many(ref_records, "reference")

    def _encode_many(self, records: Sequence[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
        encoded = []
        skipped = 0
        for record in records:
            try:
                encoded.append(encode_record(record, self.tokenizer, self.args.max_length, self.args.with_prompt_token))
            except ValueError:
                skipped += 1
        print(f"Loaded {len(encoded)} {name} examples; skipped {skipped}")
        if not encoded:
            raise ValueError(f"No usable {name} examples")
        return encoded

    def loss_for_encoded(self, encoded: Dict[str, Any]) -> Any:
        batch = collate_encoded([encoded], self.tokenizer.pad_token_id)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        ).loss

    def loss_and_grad(self, encoded: Dict[str, Any]) -> Tuple[float, List[Any]]:
        self.model.zero_grad(set_to_none=True)
        loss = self.loss_for_encoded(encoded)
        loss.backward()
        grads = clone_grads(self.model)
        return float(loss.detach().cpu().item()), grads

    def optimizer_step_with_grads(self, grads: Sequence[Any]) -> None:
        import torch

        assign_grads(self.model, grads)
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for _, param in select_trainable_params(self.model)], self.args.max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def warmup(self) -> List[Any]:
        print(f"Running {self.args.warmup_steps} task-only warm-up updates")
        epoch_direction = zero_like_trainable(self.model)
        if self.args.warmup_steps <= 0:
            return epoch_direction

        batches = batch_indices(len(self.task_examples), self.args.train_batch_size, self.rng, shuffle=True)
        step = 0
        while step < self.args.warmup_steps:
            for batch in batches:
                grads = []
                losses = []
                for idx in batch:
                    loss, grad = self.loss_and_grad(self.task_examples[idx])
                    grads.append(grad)
                    losses.append(loss)
                avg_grad = weighted_average_grads(grads, [1.0] * len(grads))
                self.optimizer_step_with_grads(avg_grad)
                add_scaled_(epoch_direction, avg_grad, 1.0)
                step += 1
                if self.args.logging_steps and step % self.args.logging_steps == 0:
                    print(f"warmup_step={step} loss={float(np.mean(losses)):.6f}")
                if step >= self.args.warmup_steps:
                    break
        return epoch_direction

    def refresh_references(self, task_direction: Sequence[Any], epoch: int) -> Tuple[List[Any], Dict[str, Any]]:
        print(f"Refreshing references for epoch {epoch}")
        candidates = []
        for idx, encoded in enumerate(self.reference_examples):
            loss, grads = self.loss_and_grad(encoded)
            conflict = -grad_cosine(grads, task_direction)
            score = self.args.mu_reference_loss * loss + self.args.nu_conflict * conflict
            candidates.append(ExampleGrad(index=idx, loss=loss, grads=grads, score=score, cosine=-conflict))
            if self.args.reference_score_limit and len(candidates) >= self.args.reference_score_limit:
                break

        if self.args.selection_mode == "hard":
            selected = sorted(candidates, key=lambda item: item.score, reverse=True)[: self.args.reference_top_k]
            weights = [1.0] * len(selected)
        else:
            weights_all = softmax_np([item.score for item in candidates], self.args.reference_temperature)
            top_indices = np.argsort(-weights_all)[: self.args.reference_top_k]
            selected = [candidates[int(i)] for i in top_indices]
            weights = [float(weights_all[int(i)]) for i in top_indices]

        ref_direction = weighted_average_grads([item.grads for item in selected], weights)
        stats = {
            "selected_ref_mean_loss": float(np.mean([item.loss for item in selected])),
            "selected_ref_mean_score": float(np.mean([item.score for item in selected])),
            "selected_ref_mean_cosine": float(np.mean([item.cosine for item in selected])),
            "num_scored_refs": len(candidates),
            "num_selected_refs": len(selected),
        }
        self.save_selection(epoch, "reference", selected, weights, stats)
        print("reference_refresh", stats)
        return ref_direction, stats

    def select_task_batch(
        self,
        batch: Sequence[int],
        ref_direction: Sequence[Any],
        epoch: int,
        global_step: int,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        candidates = []
        for idx in batch:
            loss, grads = self.loss_and_grad(self.task_examples[idx])
            compatibility = grad_cosine(grads, ref_direction)
            score = self.args.mu_task_loss * loss - self.args.nu_conflict * compatibility
            candidates.append(ExampleGrad(index=idx, loss=loss, grads=grads, score=score, cosine=compatibility))

        if self.args.selection_mode == "hard":
            k = min(max(1, self.args.task_top_k), len(candidates))
            selected = sorted(candidates, key=lambda item: item.score)[:k]
            weights = [1.0] * len(selected)
        else:
            weights_all = softmin_np([item.score for item in candidates], self.args.task_temperature)
            if self.args.task_top_k > 0:
                top_indices = np.argsort(-weights_all)[: min(self.args.task_top_k, len(candidates))]
                selected = [candidates[int(i)] for i in top_indices]
                weights = [float(weights_all[int(i)]) for i in top_indices]
            else:
                selected = candidates
                weights = [float(w) for w in weights_all]

        task_direction = weighted_average_grads([item.grads for item in selected], weights)
        stats = {
            "task_loss": float(np.mean([item.loss for item in selected])),
            "task_score": float(np.mean([item.score for item in selected])),
            "task_ref_cosine": grad_cosine(task_direction, ref_direction),
            "num_selected_task": len(selected),
        }
        if self.args.save_task_selection:
            self.save_selection(epoch, f"task_step_{global_step}", selected, weights, stats)
        return task_direction, stats

    def corrected_direction(self, task_direction: Sequence[Any], ref_direction: Sequence[Any]) -> List[Any]:
        final = []
        for task_grad, ref_grad in zip(task_direction, ref_direction):
            if task_grad is None:
                final.append(None)
            elif ref_grad is None:
                final.append(task_grad.clone())
            else:
                final.append(task_grad.clone().add(ref_grad, alpha=self.args.correction_eta))
        return final

    def train(self) -> None:
        self.model.train()
        task_direction = self.warmup()
        global_step = 0
        for epoch in range(1, self.args.num_train_epochs + 1):
            ref_direction, ref_stats = self.refresh_references(task_direction, epoch)
            epoch_direction = zero_like_trainable(self.model)
            num_epoch_updates = 0
            batches = batch_indices(len(self.task_examples), self.args.train_batch_size, self.rng, shuffle=True)

            for batch in batches:
                task_grad, task_stats = self.select_task_batch(batch, ref_direction, epoch, global_step)
                final_grad = self.corrected_direction(task_grad, ref_direction)
                self.optimizer_step_with_grads(final_grad)
                add_scaled_(epoch_direction, task_grad, 1.0)
                num_epoch_updates += 1
                global_step += 1

                if self.args.logging_steps and global_step % self.args.logging_steps == 0:
                    logs = {**ref_stats, **task_stats, "epoch": epoch, "global_step": global_step}
                    print(json.dumps(logs, sort_keys=True))
                if self.args.max_train_steps and global_step >= self.args.max_train_steps:
                    break

            scale = 1.0 / max(num_epoch_updates, 1)
            for grad in epoch_direction:
                if grad is not None:
                    grad.mul_(scale)
            task_direction = epoch_direction
            self.save_checkpoint(epoch, global_step)
            if self.args.max_train_steps and global_step >= self.args.max_train_steps:
                break

        self.save_final()

    def save_selection(
        self,
        epoch: int,
        kind: str,
        selected: Sequence[ExampleGrad],
        weights: Sequence[float],
        stats: Dict[str, Any],
    ) -> None:
        if not self.args.selection_log:
            return
        path = Path(self.args.selection_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "epoch": epoch,
            "kind": kind,
            "indices": [item.index for item in selected],
            "weights": [float(w) for w in weights],
            "scores": [float(item.score) for item in selected],
            "losses": [float(item.loss) for item in selected],
            "cosines": [float(item.cosine) for item in selected],
            **stats,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_checkpoint(self, epoch: int, global_step: int) -> None:
        if not self.args.save_each_epoch:
            return
        path = Path(self.args.output_dir) / f"epoch_{epoch}_step_{global_step}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved checkpoint to {path}")

    def save_final(self) -> None:
        path = Path(self.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved final adapter to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--task-data", required=True, type=Path)
    parser.add_argument("--reference-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-log", default=None)
    parser.add_argument("--max-task-samples", type=int, default=None)
    parser.add_argument("--max-reference-samples", type=int, default=None)
    parser.add_argument("--reference-score-limit", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--task-top-k", type=int, default=4)
    parser.add_argument("--reference-top-k", type=int, default=32)
    parser.add_argument("--selection-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--mu-task-loss", type=float, default=1.0)
    parser.add_argument("--mu-reference-loss", type=float, default=1.0)
    parser.add_argument("--nu-conflict", type=float, default=1.0)
    parser.add_argument("--task-temperature", type=float, default=0.1)
    parser.add_argument("--reference-temperature", type=float, default=0.1)
    parser.add_argument("--correction-eta", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--with-prompt-token", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-slow-tokenizer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--save-task-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = DualSelectTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
