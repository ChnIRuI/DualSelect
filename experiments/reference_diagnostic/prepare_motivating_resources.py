#!/usr/bin/env python3
"""Prepare REDORCA/HH-RLHF resources for the motivating diagnostic.

The diagnostic in the paper uses three disjoint resources:

1. Fixed REDORCA mini-batches to induce task-update directions.
2. Anthropic HH-RLHF train-split chosen responses as safe-response references.
3. A separate safety probe set for rollout evaluation.

This script materializes those resources as local JSONL/metadata files. It does
not train a model or compute embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from prepare_reference_metadata import SAFETY_KEYWORDS, TASK_KEYWORDS, infer_label, record_text


SAFETY_PROBE_KEYWORDS = {
    "Violence": [
        "weapon",
        "kill",
        "attack",
        "bomb",
        "assault",
        "shoot",
        "stab",
        "poison",
        "harm someone",
        "hurt someone",
    ],
    "Drugs": [
        "cocaine",
        "heroin",
        "meth",
        "opioid",
        "illegal drug",
        "drug dealer",
        "overdose",
        "make drugs",
    ],
    "Self-harm": [
        "suicide",
        "self-harm",
        "self harm",
        "kill myself",
        "hurt myself",
        "end my life",
        "cut myself",
    ],
    "Privacy": [
        "password",
        "credential",
        "dox",
        "address",
        "ssn",
        "social security",
        "credit card",
        "private information",
    ],
    "Fraud": [
        "fraud",
        "scam",
        "phishing",
        "steal",
        "fake id",
        "counterfeit",
        "forgery",
        "hack",
        "malware",
        "bypass",
    ],
    "Hate": [
        "racist",
        "sexist",
        "slur",
        "hate",
        "genocide",
        "discriminate",
        "inferior",
    ],
}

REFUSAL_AND_SAFETY_LANGUAGE = [
    "i can't",
    "i cannot",
    "i won't",
    "i'm sorry",
    "i'm sorry",
    "not able to help",
    "can't help",
    "cannot help",
    "illegal",
    "dangerous",
    "harmful",
    "unsafe",
    "seek help",
    "emergency",
    "professional",
]


def read_local_records(path: Path) -> List[Dict[str, Any]]:
    if is_git_lfs_pointer(path):
        raise ValueError(
            f"{path} is a Git LFS pointer, not the real dataset. Run `git lfs pull` "
            "or pass a path to the resolved REDORCA JSONL file."
        )
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ("train", "test", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"Unsupported JSON shape in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    head = path.read_text(encoding="utf-8", errors="ignore")[:200]
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def load_hh_split(dataset_name: str, split: str, local_file: Path | None) -> List[Dict[str, Any]]:
    if local_file is not None:
        return read_local_records(local_file)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading Anthropic HH-RLHF from Hugging Face requires `datasets`. "
            "Install it or provide --hh-train-file/--hh-probe-file."
        ) from exc
    return list(load_dataset(dataset_name, split=split))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def prompt_hash(text: str) -> str:
    return hashlib.sha1(normalize_space(text).lower().encode("utf-8")).hexdigest()


def first_existing(record: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def messages_to_io(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    prompt_parts = []
    output = ""
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role == "assistant":
            output = content
        else:
            prompt_parts.append(f"{role or 'user'}: {content}")
    return "\n".join(prompt_parts), output


def redorca_to_example(record: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    if isinstance(record.get("messages"), list):
        instruction, output = messages_to_io(record["messages"])
        user_input = ""
    else:
        instruction = first_existing(record, ["instruction", "prompt", "question", "query", "input"])
        user_input = "" if "input" in record and instruction == str(record.get("input", "")).strip() else str(record.get("input", "")).strip()
        output = first_existing(record, ["output", "completion", "response", "answer", "chosen"])
    example = {
        "source": "redorca",
        "source_index": source_index,
        "instruction": instruction,
        "input": user_input,
        "output": output,
        "prompt_hash": prompt_hash(instruction + "\n" + user_input),
    }
    example["task_group"] = infer_redorca_task_group(example)
    return example


def infer_redorca_task_group(record: Dict[str, Any]) -> str:
    text = normalize_space(record_text(record)).lower()
    instruction = normalize_space(str(record.get("instruction", ""))).lower()

    coding_patterns = [
        r"\bpython\b",
        r"\bjava(script)?\b",
        r"\bc\+\+\b",
        r"\bsql\b",
        r"\bcode\b",
        r"\bprogram\b",
        r"\bfunction\b",
        r"\bdebug\b",
        r"\balgorithm\b",
        r"\bregex\b",
    ]
    math_patterns = [
        r"\bcalculate\b",
        r"\bsolve\b",
        r"\bequation\b",
        r"\bprobability\b",
        r"\bgeometry\b",
        r"\balgebra\b",
        r"\binteger\b",
        r"\b\d+\s*[\+\-\*/]\s*\d+\b",
    ]
    writing_patterns = [
        r"\bwrite\b",
        r"\brewrite\b",
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\bemail\b",
        r"\bessay\b",
        r"\bstory\b",
        r"\bpoem\b",
        r"\bparagraph\b",
        r"\btranslate\b",
    ]
    reasoning_patterns = [
        r"\bwhich of the following\b",
        r"\banswer options\b",
        r"\bstep by step\b",
        r"\binfer\b",
        r"\bdeduce\b",
        r"\bentailment\b",
        r"\blogic\b",
        r"\breasonable\b",
        r"\bexplain why\b",
    ]
    ordered = [
        ("Coding", coding_patterns),
        ("Math", math_patterns),
        ("Writing", writing_patterns),
        ("Reasoning", reasoning_patterns),
    ]
    for label, patterns in ordered:
        if any(re.search(pattern, text) for pattern in patterns):
            return label
    if instruction.startswith(("what", "who", "when", "where", "why", "how", "is ", "are ", "does ", "do ")):
        return "QA"
    return infer_label(text, TASK_KEYWORDS, "QA")


def split_hh_chosen(chosen: str) -> Tuple[str, str]:
    marker = "\n\nAssistant:"
    idx = chosen.rfind(marker)
    if idx < 0:
        idx = chosen.rfind("Assistant:")
        marker = "Assistant:"
    if idx < 0:
        return chosen.strip(), ""
    prompt = chosen[:idx].strip()
    response = chosen[idx + len(marker) :].strip()
    return prompt, response


def safety_category_and_score(prompt: str, response: str) -> Tuple[str, int]:
    text = normalize_space(prompt + "\n" + response).lower()
    prompt_text = normalize_space(prompt).lower()
    scores = {
        label: sum(1 for keyword in keywords if keyword in text)
        for label, keywords in SAFETY_PROBE_KEYWORDS.items()
    }
    category, keyword_score = max(scores.items(), key=lambda item: item[1])
    refusal_score = sum(1 for phrase in REFUSAL_AND_SAFETY_LANGUAGE if phrase in response.lower())
    # Prompt-side risk matters more for probe construction; refusal language helps
    # distinguish safe/helpful HH answers from generic helpful answers.
    prompt_risk = sum(
        1
        for keywords in SAFETY_PROBE_KEYWORDS.values()
        for keyword in keywords
        if keyword in prompt_text
    )
    risk_score = keyword_score + refusal_score + prompt_risk
    return (category if keyword_score > 0 else "Other"), int(risk_score)


def hh_to_example(record: Dict[str, Any], split: str, source_index: int) -> Dict[str, Any]:
    chosen = str(record.get("chosen", "")).strip()
    prompt, response = split_hh_chosen(chosen)
    safety_category, risk_score = safety_category_and_score(prompt, response)
    return {
        "source": "anthropic_hh_rlhf",
        "split": split,
        "source_index": source_index,
        "instruction": prompt,
        "input": "",
        "output": response,
        "prompt_hash": prompt_hash(prompt),
        "safety_category": safety_category,
        "risk_score": risk_score,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sample_disjoint(
    rows: Sequence[Dict[str, Any]],
    n: int,
    forbidden_hashes: set[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if row.get("prompt_hash") not in forbidden_hashes and row.get("instruction")]
    rng.shuffle(candidates)
    sampled = []
    seen_hashes = set(forbidden_hashes)
    for row in candidates:
        row_hash = row.get("prompt_hash")
        if not row_hash or row_hash in seen_hashes:
            continue
        sampled.append(row)
        seen_hashes.add(str(row_hash))
        if len(sampled) == n:
            break
    return sampled


def sample_safety_probe(
    rows: Sequence[Dict[str, Any]],
    n: int,
    forbidden_hashes: set[str],
    rng: random.Random,
    min_risk_score: int,
    allowed_categories: set[str] | None,
) -> List[Dict[str, Any]]:
    candidates = []
    for row in rows:
        row_hash = row.get("prompt_hash")
        category = str(row.get("safety_category", "Other"))
        risk_score = int(row.get("risk_score", 0) or 0)
        if not row_hash or row_hash in forbidden_hashes or not row.get("instruction"):
            continue
        if allowed_categories and category not in allowed_categories:
            continue
        if category == "Other" or risk_score < min_risk_score:
            continue
        candidates.append(row)
    rng.shuffle(candidates)
    sampled = []
    seen_hashes = set(forbidden_hashes)
    for row in sorted(candidates, key=lambda item: int(item.get("risk_score", 0) or 0), reverse=True):
        row_hash = str(row.get("prompt_hash"))
        if row_hash in seen_hashes:
            continue
        sampled.append(row)
        seen_hashes.add(row_hash)
        if len(sampled) == n:
            break
    return sampled


def attach_task_batches(rows: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied["diagnostic_index"] = idx
        copied["batch_id"] = f"redorca_batch_{idx // batch_size:06d}"
        copied["task_group"] = copied.get("task_group") or infer_redorca_task_group(copied)
        out.append(copied)
    return out


def build_stratified_task_batches(
    rows: Sequence[Dict[str, Any]],
    total_samples: int,
    batch_size: int,
    task_groups: Sequence[str],
    forbidden_hashes: set[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = {group: [] for group in task_groups}
    for row in rows:
        group = str(row.get("task_group") or infer_redorca_task_group(row))
        if group in by_group and row.get("prompt_hash") not in forbidden_hashes and row.get("instruction"):
            by_group[group].append(row)

    target_batches = max(1, total_samples // batch_size)
    base_batches = max(1, target_batches // max(1, len(task_groups)))
    remainder = target_batches % max(1, len(task_groups))
    out: List[Dict[str, Any]] = []
    diagnostic_index = 0

    for group_index, group in enumerate(task_groups):
        candidates = by_group[group]
        rng.shuffle(candidates)
        group_batches = base_batches + (1 if group_index < remainder else 0)
        target = group_batches * batch_size
        selected = candidates[: min(target, len(candidates))]
        for local_idx, row in enumerate(selected):
            copied = dict(row)
            copied["task_group"] = group
            copied["diagnostic_index"] = diagnostic_index
            copied["batch_id"] = f"redorca_{group.lower()}_batch_{local_idx // batch_size:06d}"
            out.append(copied)
            diagnostic_index += 1

    if len(out) < total_samples:
        seen_hashes = hashes(out) | forbidden_hashes
        leftovers = [
            row
            for group_rows in by_group.values()
            for row in group_rows
            if row.get("prompt_hash") not in seen_hashes
        ]
        rng.shuffle(leftovers)
        for row in leftovers[: total_samples - len(out)]:
            group = str(row.get("task_group") or infer_redorca_task_group(row))
            copied = dict(row)
            copied["task_group"] = group
            copied["diagnostic_index"] = diagnostic_index
            copied["batch_id"] = f"redorca_{group.lower()}_extra_batch_{diagnostic_index // batch_size:06d}"
            out.append(copied)
            diagnostic_index += 1
    return out[:total_samples]


def metadata_rows(rows: Sequence[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        text = record_text(row)
        if kind == "reference":
            out.append(
                {
                    "index": idx,
                    "prompt_hash": row.get("prompt_hash"),
                    "safety_category": row.get("safety_category") or infer_label(text, SAFETY_KEYWORDS, "Other"),
                    "risk_score": row.get("risk_score", 0),
                    "source_index": row.get("source_index"),
                }
            )
        else:
            out.append(
                {
                    "index": idx,
                    "prompt_hash": row.get("prompt_hash"),
                    "batch_id": row.get("batch_id"),
                    "task_group": row.get("task_group"),
                    "source_index": row.get("source_index"),
                }
            )
    return out


def hashes(rows: Sequence[Dict[str, Any]]) -> set[str]:
    return {str(row["prompt_hash"]) for row in rows if row.get("prompt_hash")}


def disjoint_report(resources: Dict[str, Sequence[Dict[str, Any]]], final_eval_hashes: set[str]) -> Dict[str, Any]:
    hash_sets = {name: hashes(rows) for name, rows in resources.items()}
    report: Dict[str, Any] = {
        "counts": {name: len(rows) for name, rows in resources.items()},
        "unique_prompt_hashes": {name: len(values) for name, values in hash_sets.items()},
        "pairwise_prompt_overlaps": {},
        "final_eval_overlaps": {},
    }
    names = sorted(hash_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            report["pairwise_prompt_overlaps"][f"{left}__{right}"] = len(hash_sets[left] & hash_sets[right])
    if final_eval_hashes:
        for name, values in hash_sets.items():
            report["final_eval_overlaps"][name] = len(values & final_eval_hashes)
    return report


def load_final_eval_hashes(paths: Sequence[Path]) -> set[str]:
    all_hashes: set[str] = set()
    for path in paths:
        for idx, record in enumerate(read_local_records(path)):
            example = redorca_to_example(record, idx)
            all_hashes.add(str(example["prompt_hash"]))
    return all_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redorca", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hh-dataset-name", default="Anthropic/hh-rlhf")
    parser.add_argument("--hh-train-file", type=Path, default=None)
    parser.add_argument("--hh-probe-file", type=Path, default=None)
    parser.add_argument("--num-task-samples", type=int, default=1024)
    parser.add_argument("--task-batch-size", type=int, default=16)
    parser.add_argument("--task-groups", nargs="+", default=["Coding", "Math", "Reasoning", "Writing", "QA"])
    parser.add_argument("--no-stratified-task-batches", action="store_true")
    parser.add_argument("--num-reference-samples", type=int, default=4096)
    parser.add_argument("--num-probe-samples", type=int, default=512)
    parser.add_argument("--probe-min-risk-score", type=int, default=2)
    parser.add_argument(
        "--probe-categories",
        nargs="*",
        default=["Violence", "Drugs", "Self-harm", "Privacy", "Fraud", "Hate"],
    )
    parser.add_argument("--final-eval-files", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    redorca_raw = read_local_records(args.redorca)
    redorca_examples = [redorca_to_example(record, idx) for idx, record in enumerate(redorca_raw)]

    final_eval_hashes = load_final_eval_hashes(args.final_eval_files)
    if args.no_stratified_task_batches:
        redorca_batches = attach_task_batches(
            sample_disjoint(redorca_examples, args.num_task_samples, final_eval_hashes, rng),
            args.task_batch_size,
        )
    else:
        redorca_batches = build_stratified_task_batches(
            redorca_examples,
            args.num_task_samples,
            args.task_batch_size,
            args.task_groups,
            final_eval_hashes,
            rng,
        )
    redorca_hashes = hashes(redorca_batches)

    hh_train_raw = load_hh_split(args.hh_dataset_name, "train", args.hh_train_file)
    hh_train = [hh_to_example(record, "train", idx) for idx, record in enumerate(hh_train_raw)]
    reference_pool = sample_disjoint(
        hh_train,
        args.num_reference_samples,
        redorca_hashes | final_eval_hashes,
        rng,
    )
    reference_hashes = hashes(reference_pool)

    if args.hh_probe_file is not None:
        hh_probe_raw = read_local_records(args.hh_probe_file)
        probe_split = "local_probe"
    else:
        hh_probe_raw = load_hh_split(args.hh_dataset_name, "test", None)
        probe_split = "test"
    hh_probe = [hh_to_example(record, probe_split, idx) for idx, record in enumerate(hh_probe_raw)]
    safety_probe = sample_safety_probe(
        hh_probe,
        args.num_probe_samples,
        redorca_hashes | reference_hashes | final_eval_hashes,
        rng,
        args.probe_min_risk_score,
        set(args.probe_categories) if args.probe_categories else None,
    )
    if len(safety_probe) < args.num_probe_samples:
        print(
            f"Warning: requested {args.num_probe_samples} safety probes but only found "
            f"{len(safety_probe)} with min risk score {args.probe_min_risk_score}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "redorca_task_batches.jsonl", redorca_batches)
    write_jsonl(args.output_dir / "redorca_task_meta.jsonl", metadata_rows(redorca_batches, "task"))
    write_jsonl(args.output_dir / "hh_reference_pool.jsonl", reference_pool)
    write_jsonl(args.output_dir / "hh_reference_meta.jsonl", metadata_rows(reference_pool, "reference"))
    write_jsonl(args.output_dir / "safety_probe.jsonl", safety_probe)

    report = disjoint_report(
        {
            "redorca_task_batches": redorca_batches,
            "hh_reference_pool": reference_pool,
            "safety_probe": safety_probe,
        },
        final_eval_hashes,
    )
    report["config"] = {
        "redorca": str(args.redorca),
        "hh_dataset_name": args.hh_dataset_name,
        "seed": args.seed,
        "task_batch_size": args.task_batch_size,
        "task_groups": args.task_groups,
        "stratified_task_batches": not args.no_stratified_task_batches,
        "probe_min_risk_score": args.probe_min_risk_score,
        "probe_categories": args.probe_categories,
    }
    (args.output_dir / "disjointness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
