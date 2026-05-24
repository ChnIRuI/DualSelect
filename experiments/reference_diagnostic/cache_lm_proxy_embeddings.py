from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from prepare_reference_metadata import SAFETY_KEYWORDS, TASK_KEYWORDS, infer_label, record_text


def read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        for key in ("train", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported data extension: {path.suffix}")


def concat_messages(messages: List[Dict[str, Any]], tokenizer: Any) -> str:
    text = ""
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role == "system":
            text += "<|system|>\n" + content + "\n"
        elif role == "user":
            text += "Human: " + content + "\n"
        elif role == "assistant":
            text += "Assistant: " + content + " " + tokenizer.eos_token
        else:
            raise ValueError(f"Invalid message role: {role}")
    return text.strip()


def encode_record(record: Dict[str, Any], tokenizer: Any, max_length: int, with_prompt_token: bool) -> Dict[str, Any]:
    import torch

    if isinstance(record.get("messages"), list):
        messages = record["messages"]
        text = concat_messages(messages, tokenizer)
        tokenized = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = tokenized.input_ids[0]
        labels = input_ids.clone()
        for message_idx, message in enumerate(messages):
            if message.get("role") == "assistant":
                continue
            start = 0 if message_idx == 0 else tokenizer(
                concat_messages(messages[:message_idx], tokenizer),
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1].get("role") == "assistant":
                prefix = concat_messages(messages[: message_idx + 1], tokenizer) + "Assistant: "
            else:
                prefix = concat_messages(messages[: message_idx + 1], tokenizer)
            end = tokenizer(prefix, return_tensors="pt", max_length=max_length, truncation=True).input_ids.shape[1]
            if not with_prompt_token:
                labels[start:end] = -100
            if end >= max_length:
                break
    else:
        instruction = str(record.get("instruction", record.get("prompt", ""))).strip()
        user_input = str(record.get("input", "")).strip()
        output = str(record.get("output", record.get("completion", ""))).strip()
        prompt = "Human: " + instruction
        if user_input:
            prompt += "\n" + user_input
        prompt += "\nAssistant: "
        text = prompt + output + " " + tokenizer.eos_token
        tokenized = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = tokenized.input_ids[0]
        labels = input_ids.clone()
        if not with_prompt_token:
            prompt_len = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).input_ids.shape[1]
            labels[:prompt_len] = -100

    return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}


def collate(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    import torch

    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def batch_embedding(model: Any, batch: Dict[str, Any], mode: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"] if mode == "loss_weighted_hidden" else None,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1][:, :-1, :]
        label_mask = batch["labels"][:, 1:] != -100
        if mode == "loss_weighted_hidden":
            logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            vocab = logits.shape[-1]
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab),
                shift_labels.view(-1).to(logits.device),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)
            weights = losses * label_mask
        else:
            weights = label_mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        emb = (hidden * weights.unsqueeze(-1)).sum(dim=1) / denom
        return emb.detach().float().cpu().numpy()


def metadata_for(record: Dict[str, Any], idx: int, kind: str) -> Dict[str, Any]:
    text = record_text(record)
    base = {
        "index": idx,
        "prompt_hash": record.get("prompt_hash"),
        "source_index": record.get("source_index"),
    }
    if kind == "reference":
        base["safety_category"] = record.get("safety_category") or infer_label(text, SAFETY_KEYWORDS, "Other")
        base["risk_score"] = float(record.get("risk_score", 0.0) or 0.0)
        return base
    base["batch_id"] = record.get("batch_id")
    base["task_group"] = record.get("task_group") or infer_label(text, TASK_KEYWORDS, "QA")
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-embeddings", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--kind", choices=["task", "reference"], default="task")
    parser.add_argument("--mode", choices=["hidden_mean", "loss_weighted_hidden"], default="hidden_mean")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--with-prompt-token", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = read_records(args.data)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    encoded = [encode_record(record, tokenizer, args.max_length, args.with_prompt_token) for record in records]
    embeddings = []
    for start in range(0, len(encoded), args.batch_size):
        batch = collate(encoded[start : start + args.batch_size], tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        embeddings.append(batch_embedding(model, batch, args.mode))

    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_embeddings, np.concatenate(embeddings, axis=0))
    with args.output_metadata.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            f.write(json.dumps(metadata_for(record, idx, args.kind), ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} embeddings to {args.output_embeddings}")
    print(f"Wrote metadata to {args.output_metadata}")


if __name__ == "__main__":
    main()
