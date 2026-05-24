#!/usr/bin/env python3
"""Create task-conditioned safety-reference selection traces from cached embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def read_metadata(path: Path | None, length: int) -> List[Dict[str, Any]]:
    if path is None:
        return [{"index": i} for i in range(length)]
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", [])
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported metadata extension: {path.suffix}")
    if len(rows) != length:
        raise ValueError(f"{path} has {len(rows)} rows but embeddings have {length} rows")
    return rows


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def zscore(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    std = values.std()
    if std < eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / std


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, scores.shape[0]))
    unsorted = np.argpartition(-scores, k - 1)[:k]
    return unsorted[np.argsort(-scores[unsorted])]


def build_batches(
    task_meta: Sequence[Dict[str, Any]],
    batch_size: int,
    batch_field: str,
) -> List[List[int]]:
    if batch_field and any(batch_field in row for row in task_meta):
        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(task_meta):
            grouped[str(row.get(batch_field, f"batch_{idx // batch_size:06d}"))].append(idx)
        return [grouped[key] for key in sorted(grouped)]
    return [list(range(start, min(start + batch_size, len(task_meta)))) for start in range(0, len(task_meta), batch_size)]


def task_group_for(indices: Sequence[int], task_meta: Sequence[Dict[str, Any]], group_field: str) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for idx in indices:
        counts[str(task_meta[idx].get(group_field, "Unknown"))] += 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "Unknown"


def reference_categories(indices: Iterable[int], ref_meta: Sequence[Dict[str, Any]], category_field: str) -> List[str]:
    return [str(ref_meta[int(idx)].get(category_field, "Other")) for idx in indices]


def score_references(
    task_direction: np.ndarray,
    ref_embeddings: np.ndarray,
    risk_bonus: np.ndarray,
    score_mode: str,
    risk_weight: float,
) -> np.ndarray:
    cosine = ref_embeddings @ task_direction
    if score_mode == "conflict":
        base_score = -cosine
    elif score_mode == "alignment":
        base_score = cosine
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")
    return base_score + risk_weight * risk_bonus


def fixed_uniform_indices(
    ref_meta: Sequence[Dict[str, Any]],
    category_field: str,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_category: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(ref_meta):
        by_category[str(row.get(category_field, "Other"))].append(idx)
    for indices in by_category.values():
        rng.shuffle(indices)
    selected: List[int] = []
    while len(selected) < k and by_category:
        for category in sorted(list(by_category)):
            if by_category[category]:
                selected.append(by_category[category].pop())
                if len(selected) == k:
                    break
            else:
                by_category.pop(category, None)
    return np.array(selected, dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-embeddings", required=True, type=Path)
    parser.add_argument("--reference-embeddings", required=True, type=Path)
    parser.add_argument("--task-metadata", type=Path, default=None)
    parser.add_argument("--reference-metadata", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["dualselect", "static_mean_conflict", "periodic_random", "uniform"],
        choices=["dualselect", "static_mean_conflict", "periodic_random", "uniform"],
    )
    parser.add_argument("--score-mode", choices=["conflict", "alignment"], default="conflict")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-field", default="batch_id")
    parser.add_argument("--task-group-field", default="task_group")
    parser.add_argument("--reference-category-field", default="safety_category")
    parser.add_argument("--risk-field", default="risk_score")
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-full-scores", action="store_true")
    parser.add_argument("--write-task-indices", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_embeddings = normalize(np.load(args.task_embeddings).astype(np.float64))
    ref_embeddings = normalize(np.load(args.reference_embeddings).astype(np.float64))
    if task_embeddings.ndim != 2 or ref_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays")
    if task_embeddings.shape[1] != ref_embeddings.shape[1]:
        raise ValueError("Task and reference embeddings must have the same dimension")

    task_meta = read_metadata(args.task_metadata, task_embeddings.shape[0])
    ref_meta = read_metadata(args.reference_metadata, ref_embeddings.shape[0])
    batches = build_batches(task_meta, args.batch_size, args.batch_field)

    if args.top_k is None and args.top_ratio is None:
        raise ValueError("Provide --top-k or --top-ratio")
    top_k_value = args.top_k or max(1, int(round(ref_embeddings.shape[0] * args.top_ratio)))

    risk = np.array([float(row.get(args.risk_field, 0.0) or 0.0) for row in ref_meta], dtype=np.float64)
    risk_bonus = zscore(risk)
    rng = np.random.default_rng(args.seed)

    global_direction = normalize(task_embeddings.mean(axis=0, keepdims=True))[0]
    static_scores = score_references(global_direction, ref_embeddings, risk_bonus, args.score_mode, args.risk_weight)
    static_selected = topk(static_scores, top_k_value)
    uniform_selected = fixed_uniform_indices(ref_meta, args.reference_category_field, top_k_value, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for batch_num, task_indices in enumerate(batches):
            batch_id = str(task_meta[task_indices[0]].get(args.batch_field, f"batch_{batch_num:06d}"))
            group = task_group_for(task_indices, task_meta, args.task_group_field)
            batch_direction = normalize(task_embeddings[task_indices].mean(axis=0, keepdims=True))[0]
            dual_scores = score_references(batch_direction, ref_embeddings, risk_bonus, args.score_mode, args.risk_weight)

            strategy_to_indices = {
                "dualselect": topk(dual_scores, top_k_value),
                "static_mean_conflict": static_selected,
                "periodic_random": rng.choice(ref_embeddings.shape[0], size=top_k_value, replace=False),
                "uniform": uniform_selected,
            }
            strategy_to_scores = {
                "dualselect": dual_scores,
                "static_mean_conflict": static_scores,
                "periodic_random": np.zeros(ref_embeddings.shape[0], dtype=np.float64),
                "uniform": np.zeros(ref_embeddings.shape[0], dtype=np.float64),
            }

            for strategy in args.strategies:
                selected = np.asarray(strategy_to_indices[strategy], dtype=int)
                scores = strategy_to_scores[strategy]
                row: Dict[str, Any] = {
                    "strategy": strategy,
                    "batch_id": batch_id,
                    "task_group": group,
                    "selected_ref_indices": selected.tolist(),
                    "selected_ref_scores": [float(scores[idx]) for idx in selected],
                    "reference_categories": reference_categories(selected, ref_meta, args.reference_category_field),
                    "pool_size": int(ref_embeddings.shape[0]),
                    "top_k": int(top_k_value),
                }
                if args.write_task_indices:
                    row["task_indices"] = list(map(int, task_indices))
                if args.write_full_scores:
                    row["reference_scores"] = scores.astype(float).tolist()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(batches) * len(args.strategies)} trace rows to {args.output}")


if __name__ == "__main__":
    main()

