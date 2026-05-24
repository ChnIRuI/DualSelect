from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def read_table(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data", [])
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def mean_pairwise_jaccard(sets: Sequence[Sequence[int]]) -> Tuple[float, int]:
    values = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return (float(mean(values)), len(values)) if values else (0.0, 0)


def random_null_jaccard(
    sets: Sequence[Sequence[int]],
    pool_size: int,
    repeats: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    if len(sets) < 2:
        return 0.0, 0.0
    ks = [len(s) for s in sets]
    values = []
    for _ in range(repeats):
        sampled = [rng.choice(pool_size, size=k, replace=False).tolist() for k in ks]
        value, _ = mean_pairwise_jaccard(sampled)
        values.append(value)
    return float(np.mean(values)), float(np.std(values))


def compute_overlap(rows: Sequence[Dict[str, Any]], repeats: int, seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("strategy", "unknown")), str(row.get("task_group", "Unknown")))].append(row)

    rng = np.random.default_rng(seed)
    out = []
    for (strategy, task_group), group_rows in sorted(grouped.items()):
        selected_sets = [list(map(int, row.get("selected_ref_indices", []))) for row in group_rows]
        pool_size = int(group_rows[0].get("pool_size", max((max(s) if s else 0 for s in selected_sets), default=0) + 1))
        overlap, num_pairs = mean_pairwise_jaccard(selected_sets)
        null_mean, null_std = random_null_jaccard(selected_sets, pool_size, repeats, rng)
        out.append(
            {
                "strategy": strategy,
                "task_group": task_group,
                "mean_jaccard": overlap,
                "num_pairs": num_pairs,
                "random_null_mean": null_mean,
                "random_null_std": null_std,
                "num_batches": len(group_rows),
                "pool_size": pool_size,
            }
        )
    return out


def compute_category_heatmap(rows: Sequence[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if str(row.get("strategy")) != strategy:
            continue
        task_group = str(row.get("task_group", "Unknown"))
        categories = row.get("reference_categories")
        if categories is None:
            categories = ["Other"] * len(row.get("selected_ref_indices", []))
        counts[task_group].update(map(str, categories))

    all_categories = sorted({category for counter in counts.values() for category in counter})
    out = []
    for task_group in sorted(counts):
        total = sum(counts[task_group].values())
        for category in all_categories:
            out.append(
                {
                    "strategy": strategy,
                    "task_group": task_group,
                    "safety_category": category,
                    "count": counts[task_group][category],
                    "selection_ratio": counts[task_group][category] / total if total else 0.0,
                }
            )
    return out


def enrich_reference_categories(
    rows: Sequence[Dict[str, Any]],
    reference_metadata: Sequence[Dict[str, Any]] | None,
    category_field: str,
) -> List[Dict[str, Any]]:
    if reference_metadata is None:
        return list(rows)
    enriched = []
    for row in rows:
        copied = dict(row)
        if "reference_categories" not in copied:
            categories = []
            for idx in copied.get("selected_ref_indices", []):
                try:
                    categories.append(str(reference_metadata[int(idx)].get(category_field, "Other")))
                except (IndexError, TypeError, ValueError):
                    categories.append("Other")
            copied["reference_categories"] = categories
        enriched.append(copied)
    return enriched


def probe_rows_from_trace(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if "probe_kl_before" in row and "probe_kl_after" in row:
            out.append(
                {
                    "strategy": row.get("strategy", "unknown"),
                    "kl_before": row["probe_kl_before"],
                    "kl_after": row["probe_kl_after"],
                }
            )
    return out


def compute_probe_reduction(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            before = float(row.get("kl_before", row.get("probe_kl_before")))
            after = float(row.get("kl_after", row.get("probe_kl_after")))
        except (TypeError, ValueError):
            continue
        grouped[str(row.get("strategy", "unknown"))].append(before - after)

    out = []
    for strategy, reductions in sorted(grouped.items()):
        arr = np.array(reductions, dtype=np.float64)
        out.append(
            {
                "strategy": strategy,
                "median_reduction": float(np.median(arr)),
                "p90_reduction": float(np.percentile(arr, 90)),
                "mean_reduction": float(np.mean(arr)),
                "num_probes": int(arr.shape[0]),
            }
        )
    return out


def plot_figure(
    overlap_rows: Sequence[Dict[str, Any]],
    heatmap_rows: Sequence[Dict[str, Any]],
    probe_rows: Sequence[Dict[str, Any]],
    heatmap_strategy: str,
    figure_path: Path,
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "dual_selection_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("Matplotlib is not installed; skipping figure generation.")
        return

    deep_purple = "#7D73D1"
    mid_purple = "#A9A0E8"
    pale_blue = "#D6EFF6"
    mid_blue = "#B8D9F0"
    dark_text = "#2E2E2E"
    grid_color = "#D9D9D9"
    cmap = LinearSegmentedColormap.from_list("paper_pastel", [pale_blue, mid_blue, mid_purple, deep_purple])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    def beautify(ax: Any) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", colors=dark_text, length=3, width=0.8)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.6, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.35, 1.3], wspace=0.34)

    ax1 = fig.add_subplot(gs[0, 0])
    beautify(ax1)
    selected_overlap = [r for r in overlap_rows if r["strategy"] == heatmap_strategy]
    if selected_overlap:
        mean_overlap = float(np.mean([float(r["mean_jaccard"]) for r in selected_overlap]))
        null_mean = float(np.mean([float(r["random_null_mean"]) for r in selected_overlap]))
        null_std = float(np.mean([float(r["random_null_std"]) for r in selected_overlap]))
    else:
        mean_overlap, null_mean, null_std = 0.0, 0.0, 0.0
    x = np.arange(2)
    ax1.bar(
        x,
        [null_mean, mean_overlap],
        yerr=[null_std, 0.0],
        capsize=4,
        width=0.56,
        color=[mid_blue, deep_purple],
        edgecolor="#5B5B5B",
        linewidth=0.8,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Random\nnull", heatmap_strategy])
    ax1.set_ylabel("Mean Jaccard overlap")
    ax1.set_title("(a) Cross-batch top-$K$ overlap", pad=8)
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, color=grid_color)
    ax1.set_ylim(0, max(0.05, max(null_mean + null_std, mean_overlap) * 1.35))
    for i, value in enumerate([null_mean, mean_overlap]):
        ax1.text(i, value + ax1.get_ylim()[1] * 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9.5)

    ax2 = fig.add_subplot(gs[0, 1])
    beautify(ax2)
    task_groups = sorted({str(r["task_group"]) for r in heatmap_rows})
    categories = sorted({str(r["safety_category"]) for r in heatmap_rows})
    matrix = np.zeros((len(task_groups), len(categories)), dtype=np.float64)
    index = {(r["task_group"], r["safety_category"]): float(r["selection_ratio"]) for r in heatmap_rows}
    for i, task_group in enumerate(task_groups):
        for j, category in enumerate(categories):
            matrix[i, j] = index.get((task_group, category), 0.0)
    im = ax2.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(0.01, float(matrix.max()) if matrix.size else 0.01))
    ax2.set_xticks(np.arange(len(categories)))
    ax2.set_xticklabels(categories, rotation=35, ha="right")
    ax2.set_yticks(np.arange(len(task_groups)))
    ax2.set_yticklabels(task_groups)
    ax2.set_xlabel("Safety category")
    ax2.set_ylabel("Task group")
    ax2.set_title("(b) Selected reference categories", pad=8)
    ax2.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(task_groups), 1), minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax2.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax2.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value >= matrix.max() * 0.6 else dark_text)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Selection ratio", fontsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    beautify(ax3)
    if probe_rows:
        strategies = [str(r["strategy"]) for r in probe_rows]
        medians = [float(r["median_reduction"]) for r in probe_rows]
        p90s = [float(r["p90_reduction"]) for r in probe_rows]
        x = np.arange(len(strategies))
        width = 0.34
        ax3.bar(x - width / 2, medians, width=width, color=mid_blue, edgecolor="#5B5B5B", linewidth=0.8, label="Median")
        ax3.bar(x + width / 2, p90s, width=width, color=deep_purple, edgecolor="#5B5B5B", linewidth=0.8, label="90th percentile")
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace("_", "-") for s in strategies], rotation=0)
        ax3.legend(loc="upper left", frameon=True)
        all_values = medians + p90s
        value_min = min(all_values)
        value_max = max(all_values)
        span = max(value_max - value_min, 1e-6)
        lower = min(0.0, value_min - 0.15 * span)
        upper = max(0.0, value_max + 0.15 * span)
        if upper - lower < 0.002:
            center = 0.5 * (upper + lower)
            lower, upper = center - 0.001, center + 0.001
        ax3.axhline(0.0, color="#4F4F4F", linewidth=0.8)
        ax3.set_ylim(lower, upper)
    else:
        ax3.text(0.5, 0.5, "Probe KL not provided", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_ylim(0, 1)
    ax3.set_ylabel("Probe KL reduction")
    ax3.set_title("(c) Held-out safety probe", pad=8)
    ax3.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, color=grid_color)

    plt.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote figure to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--probe-results", type=Path, default=None)
    parser.add_argument("--reference-metadata", type=Path, default=None)
    parser.add_argument("--reference-category-field", default="safety_category")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-path", type=Path, default=None)
    parser.add_argument("--heatmap-strategy", default="dualselect")
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_table(args.trace)
    reference_metadata = read_table(args.reference_metadata) if args.reference_metadata else None
    rows = enrich_reference_categories(rows, reference_metadata, args.reference_category_field)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows = compute_overlap(rows, args.random_repeats, args.seed)
    heatmap_rows = compute_category_heatmap(rows, args.heatmap_strategy)
    probe_input = read_table(args.probe_results) if args.probe_results else probe_rows_from_trace(rows)
    probe_rows = compute_probe_reduction(probe_input)

    write_csv(
        args.output_dir / "overlap_by_strategy.csv",
        overlap_rows,
        ["strategy", "task_group", "mean_jaccard", "num_pairs", "random_null_mean", "random_null_std", "num_batches", "pool_size"],
    )
    write_csv(
        args.output_dir / "category_heatmap.csv",
        heatmap_rows,
        ["strategy", "task_group", "safety_category", "count", "selection_ratio"],
    )
    if probe_rows:
        write_csv(
            args.output_dir / "probe_reduction.csv",
            probe_rows,
            ["strategy", "median_reduction", "p90_reduction", "mean_reduction", "num_probes"],
        )

    metrics = {
        "trace": str(args.trace),
        "heatmap_strategy": args.heatmap_strategy,
        "num_trace_rows": len(rows),
        "num_overlap_rows": len(overlap_rows),
        "num_heatmap_rows": len(heatmap_rows),
        "num_probe_rows": len(probe_rows),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.figure_path:
        plot_figure(overlap_rows, heatmap_rows, probe_rows, args.heatmap_strategy, args.figure_path)

    print(f"Wrote diagnostic tables to {args.output_dir}")


if __name__ == "__main__":
    main()
