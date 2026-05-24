#!/usr/bin/env python3
"""Cache LoRA-gradient embeddings for task/reference diagnostic selection.

Unlike `cache_lm_proxy_embeddings.py`, this script represents each example by
the gradient of its response-token SFT loss with respect to LoRA parameters.
The high-dimensional gradient is compressed with deterministic feature hashing,
which keeps the diagnostic lightweight while preserving update-direction
information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from cache_lm_proxy_embeddings import encode_record, metadata_for, read_records


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def projection_cache_for_param(
    name: str,
    numel: int,
    projection_dim: int,
    seed: int,
    cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    key = (name, numel)
    if key not in cache:
        rng = np.random.default_rng(stable_seed(name, seed))
        buckets = rng.integers(0, projection_dim, size=numel, dtype=np.int64)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=numel)
        cache[key] = (buckets, signs)
    return cache[key]


def hashed_trainable_gradient(model: Any, projection_dim: int, seed: int, cache: Dict[Any, Any]) -> np.ndarray:
    projected = np.zeros(projection_dim, dtype=np.float32)
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float().cpu().reshape(-1).numpy()
        if grad.size == 0:
            continue
        buckets, signs = projection_cache_for_param(name, grad.size, projection_dim, seed, cache)
        projected += np.bincount(buckets, weights=grad * signs, minlength=projection_dim).astype(np.float32)
    norm = np.linalg.norm(projected)
    if norm > 0:
        projected /= norm
    return projected


def collate_single(example: Dict[str, Any], pad_id: int) -> Dict[str, Any]:
    import torch

    return {
        "input_ids": example["input_ids"].unsqueeze(0),
        "labels": example["labels"].unsqueeze(0),
        "attention_mask": example["attention_mask"].unsqueeze(0),
    }


def loss_for_example(model: Any, batch: Dict[str, Any]) -> Any:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    ).loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-embeddings", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--kind", choices=["task", "reference"], default="task")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
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

    records = read_records(args.data)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    end_index = args.end_index if args.end_index is not None else len(records)
    records = records[args.start_index : end_index]

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
    model.train()
    model.print_trainable_parameters()

    projection_cache: Dict[Any, Any] = {}
    embeddings: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    for local_idx, record in enumerate(records):
        global_idx = args.start_index + local_idx
        encoded = encode_record(record, tokenizer, args.max_length, args.with_prompt_token)
        batch = collate_single(encoded, tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}

        model.zero_grad(set_to_none=True)
        loss = loss_for_example(model, batch)
        loss.backward()
        embeddings.append(hashed_trainable_gradient(model, args.projection_dim, args.seed, projection_cache))
        row = metadata_for(record, global_idx, args.kind)
        row["loss"] = float(loss.detach().cpu().item())
        metadata.append(row)

        if (local_idx + 1) % 25 == 0 or local_idx + 1 == len(records):
            print(f"Encoded {local_idx + 1}/{len(records)} gradient embeddings")

    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_embeddings, np.stack(embeddings, axis=0))
    with args.output_metadata.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(embeddings)} gradient embeddings to {args.output_embeddings}")
    print(f"Wrote metadata to {args.output_metadata}")


if __name__ == "__main__":
    main()

