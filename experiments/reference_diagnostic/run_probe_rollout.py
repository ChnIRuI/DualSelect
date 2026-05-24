from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from cache_lm_proxy_embeddings import collate, encode_record, read_records


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_by_field(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return grouped


def select_trace_rows(
    trace_rows: Sequence[Dict[str, Any]],
    strategies: Sequence[str],
    max_batches: int | None,
) -> List[Dict[str, Any]]:
    allowed = set(strategies)
    by_batch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if str(row.get("strategy")) in allowed:
            by_batch[str(row["batch_id"])].append(row)
    selected_batches = sorted(by_batch)
    if max_batches is not None:
        selected_batches = selected_batches[:max_batches]
    return [row for batch_id in selected_batches for row in by_batch[batch_id]]


def cycle_batch(
    encoded: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    start: int,
    size: int,
) -> List[Dict[str, Any]]:
    if not indices:
        return []
    return [encoded[indices[(start + offset) % len(indices)]] for offset in range(size)]


def reset_trainable_params(model: Any, initial_state: Dict[str, Any]) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(initial_state[name].to(param.device, dtype=param.dtype))


def snapshot_trainable_params(model: Any) -> Dict[str, Any]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def sft_loss(model: Any, batch: Dict[str, Any]) -> Any:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    ).loss


def kl_to_base_loss(model: Any, batch: Dict[str, Any]) -> Any:
    """KL(base || current) on response tokens for reference preservation."""
    import torch
    import torch.nn.functional as F

    labels = batch["labels"][:, 1:]
    mask = labels != -100
    if not mask.any():
        return torch.zeros((), device=batch["input_ids"].device, requires_grad=True)

    with torch.no_grad():
        with model.disable_adapter():
            base_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits[:, :-1, :].float()
            base_logp = F.log_softmax(base_logits, dim=-1)
            base_p = base_logp.exp()

    current_logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits[:, :-1, :].float()
    current_logp = F.log_softmax(current_logits, dim=-1)
    token_kl = (base_p * (base_logp - current_logp)).sum(dim=-1)
    return token_kl[mask].mean()


def trainable_params(model: Any) -> List[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def collect_grads(model: Any) -> List[Any]:
    return [
        param.grad.detach().clone() if param.grad is not None else None
        for param in trainable_params(model)
    ]


def grad_dot(left: Sequence[Any], right: Sequence[Any]) -> float:
    total = 0.0
    for grad_left, grad_right in zip(left, right):
        if grad_left is None or grad_right is None:
            continue
        total += float((grad_left.float() * grad_right.float()).sum().detach().cpu().item())
    return total


def grad_norm_sq(grads: Sequence[Any]) -> float:
    return max(grad_dot(grads, grads), 1e-12)


def assign_corrected_grads(
    model: Any,
    task_grads: Sequence[Any],
    ref_grads: Sequence[Any],
    correction_mode: str,
    correction_strength: float,
) -> Dict[str, float]:
    dot = grad_dot(task_grads, ref_grads)
    ref_norm_sq = grad_norm_sq(ref_grads)
    task_norm_sq = grad_norm_sq(task_grads)
    conflict_projection = min(0.0, dot) / ref_norm_sq

    for param, task_grad, ref_grad in zip(trainable_params(model), task_grads, ref_grads):
        if task_grad is None:
            param.grad = None
            continue
        corrected = task_grad.clone()
        if ref_grad is not None:
            if correction_mode == "align_ref":
                corrected = corrected + correction_strength * ref_grad
            elif correction_mode == "project_conflict":
                if dot < 0:
                    corrected = corrected - conflict_projection * ref_grad
                corrected = corrected + correction_strength * ref_grad
            elif correction_mode == "remove_conflict":
                if dot < 0:
                    corrected = corrected - conflict_projection * ref_grad
            else:
                raise ValueError(f"Unknown correction mode: {correction_mode}")
        param.grad = corrected

    denom = max(task_norm_sq * ref_norm_sq, 1e-12) ** 0.5
    return {
        "task_ref_dot": dot,
        "task_ref_cosine": dot / denom,
        "task_grad_norm": task_norm_sq**0.5,
        "ref_grad_norm": ref_norm_sq**0.5,
    }


def rollout(
    model: Any,
    tokenizer: Any,
    optimizer_cls: Any,
    initial_state: Dict[str, Any],
    task_examples: Sequence[Dict[str, Any]],
    ref_examples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    device: str,
) -> Dict[str, float]:
    import torch

    reset_trainable_params(model, initial_state)
    model.train()
    optimizer = optimizer_cls([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    task_order = list(range(len(task_examples)))
    ref_order = list(range(len(ref_examples)))
    rng.shuffle(task_order)
    rng.shuffle(ref_order)

    diagnostics: Dict[str, float] = {}
    for step in range(args.rollout_steps):
        task_items = cycle_batch(task_examples, task_order, step * args.train_batch_size, args.train_batch_size)
        task_batch = collate(task_items, tokenizer.pad_token_id)
        task_batch = {key: value.to(device) for key, value in task_batch.items()}

        if args.correction_mode == "preserve_kl":
            optimizer.zero_grad(set_to_none=True)
            task_loss = sft_loss(model, task_batch)
            loss = task_loss
            ref_kl = None
            if ref_examples and args.reference_loss_weight > 0:
                ref_items = cycle_batch(ref_examples, ref_order, step * args.reference_batch_size, args.reference_batch_size)
                ref_batch = collate(ref_items, tokenizer.pad_token_id)
                ref_batch = {key: value.to(device) for key, value in ref_batch.items()}
                ref_kl = kl_to_base_loss(model, ref_batch)
                loss = loss + args.reference_loss_weight * ref_kl

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            diagnostics["last_task_loss"] = float(task_loss.detach().cpu().item())
            diagnostics["last_ref_preservation_kl"] = float(ref_kl.detach().cpu().item()) if ref_kl is not None else 0.0
            diagnostics["last_loss"] = float(loss.detach().cpu().item())
            continue

        if not ref_examples or args.correction_mode == "additive_loss":
            loss = sft_loss(model, task_batch)
            if ref_examples and args.reference_loss_weight > 0:
                ref_items = cycle_batch(ref_examples, ref_order, step * args.reference_batch_size, args.reference_batch_size)
                ref_batch = collate(ref_items, tokenizer.pad_token_id)
                ref_batch = {key: value.to(device) for key, value in ref_batch.items()}
                loss = loss + args.reference_loss_weight * sft_loss(model, ref_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            diagnostics["last_loss"] = float(loss.detach().cpu().item())
            continue

        optimizer.zero_grad(set_to_none=True)
        task_loss = sft_loss(model, task_batch)
        task_loss.backward()
        task_grads = collect_grads(model)

        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            ref_items = cycle_batch(ref_examples, ref_order, step * args.reference_batch_size, args.reference_batch_size)
            ref_batch = collate(ref_items, tokenizer.pad_token_id)
            ref_batch = {key: value.to(device) for key, value in ref_batch.items()}
            ref_loss = sft_loss(model, ref_batch)
            ref_loss.backward()
        ref_grads = collect_grads(model)

        diagnostics.update(
            assign_corrected_grads(
                model,
                task_grads,
                ref_grads,
                args.correction_mode,
                args.reference_loss_weight,
            )
        )
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        diagnostics["last_task_loss"] = float(task_loss.detach().cpu().item())
        diagnostics["last_ref_loss"] = float(ref_loss.detach().cpu().item())
    return diagnostics


def probe_kl(model: Any, tokenizer: Any, probe_examples: Sequence[Dict[str, Any]], args: argparse.Namespace, device: str) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(probe_examples), args.probe_batch_size):
            batch = collate(probe_examples[start : start + args.probe_batch_size], tokenizer.pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"][:, 1:]
            mask = labels != -100
            if not mask.any():
                continue

            with model.disable_adapter():
                base_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits[:, :-1, :].float()
            current_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits[:, :-1, :].float()

            base_logp = F.log_softmax(base_logits, dim=-1)
            current_logp = F.log_softmax(current_logits, dim=-1)
            base_p = base_logp.exp()
            token_kl = (base_p * (base_logp - current_logp)).sum(dim=-1)
            values.append(token_kl[mask].detach().cpu())

    if not values:
        return 0.0
    return float(torch.cat(values).mean().item())


def encode_records(records: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int, with_prompt_token: bool) -> List[Dict[str, Any]]:
    return [encode_record(record, tokenizer, max_length, with_prompt_token) for record in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--redorca-task-batches", required=True, type=Path)
    parser.add_argument("--hh-reference-pool", required=True, type=Path)
    parser.add_argument("--safety-probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strategies", nargs="+", default=["dualselect", "static_mean_conflict", "periodic_random", "uniform"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--probe-max-samples", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--reference-batch-size", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--reference-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--correction-mode",
        choices=["preserve_kl", "project_conflict", "remove_conflict", "align_ref", "additive_loss"],
        default="preserve_kl",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--with-prompt-token", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    initial_state = snapshot_trainable_params(model)

    trace_rows = select_trace_rows(read_jsonl(args.trace), args.strategies, args.max_batches)
    task_records = read_jsonl(args.redorca_task_batches)
    reference_records = read_jsonl(args.hh_reference_pool)
    probe_records = read_records(args.safety_probe)
    if args.probe_max_samples is not None:
        probe_records = probe_records[: args.probe_max_samples]

    task_by_batch = group_by_field(task_records, "batch_id")
    task_encoded_by_batch = {
        batch_id: encode_records(records, tokenizer, args.max_length, args.with_prompt_token)
        for batch_id, records in task_by_batch.items()
    }
    reference_encoded = encode_records(reference_records, tokenizer, args.max_length, args.with_prompt_token)
    probe_encoded = encode_records(probe_records, tokenizer, args.max_length, args.with_prompt_token)

    optimizer_cls = torch.optim.AdamW
    task_only_kl_cache: Dict[str, float] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row_id, row in enumerate(trace_rows):
            strategy = str(row["strategy"])
            batch_id = str(row["batch_id"])
            task_examples = task_encoded_by_batch[batch_id]

            if batch_id not in task_only_kl_cache:
                rollout(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer_cls=optimizer_cls,
                    initial_state=initial_state,
                    task_examples=task_examples,
                    ref_examples=[],
                    args=args,
                    rng=rng,
                    device=device,
                )
                task_only_kl_cache[batch_id] = probe_kl(model, tokenizer, probe_encoded, args, device)

            selected_refs = [reference_encoded[int(idx)] for idx in row.get("selected_ref_indices", [])]
            diagnostics = rollout(
                model=model,
                tokenizer=tokenizer,
                optimizer_cls=optimizer_cls,
                initial_state=initial_state,
                task_examples=task_examples,
                ref_examples=selected_refs,
                args=args,
                rng=rng,
                device=device,
            )
            strategy_kl = probe_kl(model, tokenizer, probe_encoded, args, device)
            out = {
                "strategy": strategy,
                "batch_id": batch_id,
                "task_group": row.get("task_group"),
                "kl_before": task_only_kl_cache[batch_id],
                "kl_after": strategy_kl,
                "kl_reduction": task_only_kl_cache[batch_id] - strategy_kl,
                "rollout_steps": args.rollout_steps,
                "num_probe_samples": len(probe_encoded),
                "num_selected_refs": len(selected_refs),
                "correction_mode": args.correction_mode,
                "reference_objective": "kl_to_base" if args.correction_mode == "preserve_kl" else "sft_gradient",
            }
            out.update(diagnostics)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[{row_id + 1}/{len(trace_rows)}] {strategy} {batch_id}: "
                f"before={out['kl_before']:.6f} after={out['kl_after']:.6f} "
                f"reduction={out['kl_reduction']:.6f}"
            )

    print(f"Wrote probe rollout results to {args.output}")


if __name__ == "__main__":
    main()
