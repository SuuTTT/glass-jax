"""Benchmark cluster_idea_lib on exact-labeled and planted graph datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import json
import math
import time
from pathlib import Path

import numpy as np

from glass.cluster_idea_lib import (
    build_structural_entropy_dataset,
    cem_node_move_search,
    cluster_graph,
    multistart_se_heuristic,
    ring_of_triangles,
    run_official_sep_coding_tree,
    seeded_sbm,
    structural_entropy,
    weighted_bridge_graph,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    adjacency: np.ndarray
    reference_labels: np.ndarray
    reference_entropy: float
    reference_kind: str


def comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("ARI inputs must have the same shape")
    n = y_true.size
    if n < 2:
        return 1.0

    contingency = Counter(zip(y_true.tolist(), y_pred.tolist()))
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    sum_comb = sum(comb2(count) for count in contingency.values())
    true_comb = sum(comb2(count) for count in true_counts.values())
    pred_comb = sum(comb2(count) for count in pred_counts.values())
    total_comb = comb2(n)
    expected = true_comb * pred_comb / total_comb if total_comb else 0.0
    maximum = 0.5 * (true_comb + pred_comb)
    denom = maximum - expected
    if abs(denom) < 1e-12:
        return 1.0
    return float((sum_comb - expected) / denom)


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("NMI inputs must have the same shape")
    n = float(y_true.size)
    if n == 0:
        return 1.0

    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    joint_counts = Counter(zip(y_true.tolist(), y_pred.tolist()))

    mi = 0.0
    for (true_label, pred_label), count in joint_counts.items():
        pxy = count / n
        px = true_counts[true_label] / n
        py = pred_counts[pred_label] / n
        mi += pxy * math.log(pxy / (px * py))

    h_true = -sum((count / n) * math.log(count / n) for count in true_counts.values())
    h_pred = -sum((count / n) * math.log(count / n) for count in pred_counts.values())
    denom = 0.5 * (h_true + h_pred)
    if denom < 1e-12:
        return 1.0
    return float(mi / denom)


def planted_bridge_case(left: int, right: int, bridge_weight: float) -> BenchmarkCase:
    adj = weighted_bridge_graph(left, right, bridge_weight=bridge_weight)
    labels = np.concatenate([np.zeros(left, dtype=np.int32), np.ones(right, dtype=np.int32)])
    return BenchmarkCase(
        name=f"Bridge({left}+{right}, w={bridge_weight:g})",
        adjacency=adj,
        reference_labels=labels,
        reference_entropy=structural_entropy(adj, labels),
        reference_kind="planted",
    )


def planted_ring_case(count: int) -> BenchmarkCase:
    adj = ring_of_triangles(count=count, bridge_weight=1.0)
    labels = np.repeat(np.arange(count, dtype=np.int32), 3)
    return BenchmarkCase(
        name=f"RingTriangles({count})",
        adjacency=adj,
        reference_labels=labels,
        reference_entropy=structural_entropy(adj, labels),
        reference_kind="planted",
    )


def planted_sbm_case(name: str, sizes: tuple[int, ...], seed: int) -> BenchmarkCase:
    adj = seeded_sbm(sizes=sizes, p_in=0.78, p_out=0.06, seed=seed)
    labels = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
    return BenchmarkCase(
        name=name,
        adjacency=adj,
        reference_labels=labels,
        reference_entropy=structural_entropy(adj, labels),
        reference_kind="planted",
    )


def get_cases() -> list[BenchmarkCase]:
    exact_cases = [
        BenchmarkCase(
            name=f"Exact/{item.name}",
            adjacency=item.adjacency,
            reference_labels=item.best_labels,
            reference_entropy=item.best_entropy,
            reference_kind="global optimum",
        )
        for item in build_structural_entropy_dataset(max_nodes=9)
    ]
    planted_cases = [
        planted_bridge_case(10, 10, bridge_weight=1.0),
        planted_ring_case(6),
        planted_sbm_case("SBM(24, 3x8)", (8, 8, 8), seed=11),
    ]
    return exact_cases + planted_cases


def run_algorithm(name: str, adj: np.ndarray, reference_k: int, seed: int):
    start = time.time()
    if name == "ClusterIdea-Auto":
        labels = cluster_graph(adj, mode="auto", exact_max_nodes=9, heuristic_starts=6, seed=seed).labels
    elif name == "ClusterIdea-Heuristic":
        labels = multistart_se_heuristic(adj, starts=4, max_passes=10, seed=seed).labels
    elif name == "ClusterIdea-CEM":
        labels = cem_node_move_search(adj, episodes=4, horizon=min(48, 2 * adj.shape[0]), seed=seed).labels
    elif name == "Official-SEP":
        labels = run_official_sep_coding_tree(adj, k=max(2, reference_k)).labels
    else:
        raise ValueError(f"unknown algorithm {name}")
    return labels, time.time() - start


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = ["Dataset", "Reference", "Algorithm", "K", "SE", "Gap", "ARI", "NMI", "Time(s)", "Status"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["reference_kind"]),
                    str(row["algorithm"]),
                    str(row["k"]),
                    format_float(row.get("se")),
                    format_float(row.get("gap")),
                    format_float(row.get("ari")),
                    format_float(row.get("nmi")),
                    format_float(row.get("seconds"), digits=3),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def benchmark(seed: int = 42) -> list[dict[str, object]]:
    algorithms = ["ClusterIdea-Auto", "ClusterIdea-Heuristic", "ClusterIdea-CEM", "Official-SEP"]
    rows: list[dict[str, object]] = []
    for case in get_cases():
        reference_k = int(len(np.unique(case.reference_labels)))
        print(f"Running {case.name} (N={case.adjacency.shape[0]}, reference={case.reference_kind})...", flush=True)
        for algorithm in algorithms:
            try:
                labels, seconds = run_algorithm(algorithm, case.adjacency, reference_k, seed=seed)
                se = structural_entropy(case.adjacency, labels)
                rows.append(
                    {
                        "dataset": case.name,
                        "reference_kind": case.reference_kind,
                        "algorithm": algorithm,
                        "n": int(case.adjacency.shape[0]),
                        "k": int(len(np.unique(labels))),
                        "reference_entropy": case.reference_entropy,
                        "se": se,
                        "gap": se - case.reference_entropy,
                        "ari": adjusted_rand_index(case.reference_labels, labels),
                        "nmi": normalized_mutual_info(case.reference_labels, labels),
                        "seconds": seconds,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": case.name,
                        "reference_kind": case.reference_kind,
                        "algorithm": algorithm,
                        "n": int(case.adjacency.shape[0]),
                        "k": "NA",
                        "reference_entropy": case.reference_entropy,
                        "se": None,
                        "gap": None,
                        "ari": None,
                        "nmi": None,
                        "seconds": None,
                        "status": repr(exc),
                    }
                )
    return rows


def write_artifacts(rows: list[dict[str, object]]) -> Path:
    out_dir = Path("docs/experimental_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cluster_idea_lib_benchmark_20260507.json"
    report_path = out_dir / "cluster_idea_lib_benchmark_20260507.md"

    serializable = []
    for row in rows:
        serializable.append(
            {
                key: (float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
                for key, value in row.items()
            }
        )
    json_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")

    exact_rows = [row for row in rows if row["reference_kind"] == "global optimum"]
    planted_rows = [row for row in rows if row["reference_kind"] == "planted"]
    auto_rows = [row for row in rows if row["algorithm"] == "ClusterIdea-Auto" and row["status"] == "ok"]
    sep_rows = [row for row in rows if row["algorithm"] == "Official-SEP" and row["status"] == "ok"]
    exact_auto_hits = sum(abs(float(row["gap"])) < 1e-10 for row in auto_rows if row["reference_kind"] == "global optimum")
    sep_beats = 0
    sep_total = 0
    by_dataset_algo = {(row["dataset"], row["algorithm"]): row for row in rows}
    for row in sep_rows:
        ours = by_dataset_algo.get((row["dataset"], "ClusterIdea-Auto"))
        if ours is not None and ours["status"] == "ok":
            sep_total += 1
            sep_beats += float(ours["se"]) <= float(row["se"]) + 1e-12

    report = f"""# Cluster Idea Lib Benchmark: Non-Differentiable Structural Entropy Minimization

**Date:** {date.today().strftime("%B %-d, %Y")}  
**Project:** glass-jax / `glass.cluster_idea_lib`

## 1. Abstract
This experiment evaluates `cluster_idea_lib`, a NumPy-based, non-differentiable structural entropy clustering library. The benchmark mirrors the script-style workflow in `tests/benchmark_full.py`: each dataset is loaded, each algorithm is run, clustering metrics are computed, a markdown summary is printed, and result artifacts are written under `docs/experimental_reports/`.

The core question is whether the high-level `cluster_graph()` API can recover globally minimum 2D structural entropy partitions on exact-labeled small graphs, and whether its heuristic path remains competitive with the official SEP coding tree baseline on larger planted graphs.

## 2. Experimental Setup
### 2.1 Datasets
- **Exact-labeled graphs:** 5 graphs generated by `build_structural_entropy_dataset(max_nodes=9)`. Their reference labels are the exhaustive global minimum structural entropy partitions.
- **Planted graphs:** 3 larger synthetic graphs with known planted structure: two-clique bridge, ring of triangle modules, and a 3-block SBM.

### 2.2 Algorithms
- **ClusterIdea-Auto:** `cluster_graph(mode="auto")`; exact exhaustive search for N <= 9 and multistart local-search heuristic otherwise.
- **ClusterIdea-Heuristic:** direct multistart local node-move search.
- **ClusterIdea-CEM:** lightweight cross-entropy-method node-move policy search, included as the ML/RL exploration hook.
- **Official-SEP:** wrapper around `official_baselines/SEP/SEPN/codingTree.py`.

### 2.3 Metrics
- **SE:** hard 2D structural entropy; lower is better.
- **Gap:** `SE - reference_entropy`; lower is better, with 0 meaning exact match to the reference partition objective.
- **ARI/NMI:** agreement with exact optimum labels or planted labels.
- **Time:** wall-clock seconds on this workspace.

## 3. Results
### 3.1 Exact-Labeled Graphs
{markdown_table(exact_rows)}

### 3.2 Planted Larger Graphs
{markdown_table(planted_rows)}

## 4. Summary
- `ClusterIdea-Auto` matched the exact global optimum on **{exact_auto_hits}/{len([row for row in auto_rows if row["reference_kind"] == "global optimum"])}** exact-labeled graphs.
- `ClusterIdea-Auto` was no worse than Official-SEP by structural entropy on **{sep_beats}/{sep_total}** comparable runs.
- The exact path is intentionally exponential; it is appropriate for labeling small graphs and producing ground-truth supervision.
- The heuristic path scales to larger graphs, but the first implementation recomputes full SE after each proposal. This is correct and simple, but the next performance step is an incremental delta scorer.
- The CEM/RL scaffold is wired and measurable, but it is not yet competitive with deterministic SE local search. It is best treated as an experiment harness for learned action policies.

Raw JSON results are saved at `{json_path}`.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    results = benchmark()
    print("\n--- Summary Table ---")
    print(markdown_table(results))
    path = write_artifacts(results)
    print(f"\nBenchmark report saved to {path}")
