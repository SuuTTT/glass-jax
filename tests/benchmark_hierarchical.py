"""Hierarchical-clustering benchmark on Cora / Citeseer / Photo.

Scores SEClust-Tree, HCSE, and a Louvain dendrogram on the
hierarchical-quality metrics published in HCSE [Pan, Zheng, Fan 2021]
(Dasgupta cost, cost(SE)) and HypCSE [Zeng et al. 2025]
(dendrogram purity).

Output:
- ``docs/experimental_reports/hierarchical_<timestamp>.json``
- ``docs/experimental_reports/hierarchical_<timestamp>.md``
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import networkx as nx
import scipy.sparse as sp
from community import community_louvain
from tqdm import tqdm

from glass.seclust import (
    SparseGraph,
    coding_tree_hierarchy_levels,
    dasgupta_cost,
    dendrogram_purity,
    multistart_incremental_se_heuristic,
    sparse_structural_entropy,
)


SEED = 42


def _load_pyg_dataset(name: str):
    from torch_geometric.datasets import Amazon, Planetoid
    from torch_geometric.utils import to_undirected

    if name in {"Cora", "Citeseer", "PubMed"}:
        ds = Planetoid(root="/tmp/dataset", name=name)
    elif name in {"Photo", "Computers"}:
        ds = Amazon(root="/tmp/dataset", name=name)
    else:
        raise ValueError(name)
    data = ds[0]
    n = int(data.num_nodes)
    k = int(ds.num_classes)
    edge_index = to_undirected(data.edge_index).cpu().numpy().astype(np.int64)
    labels = data.y.cpu().numpy().astype(np.int64)
    graph = SparseGraph.from_edge_index(edge_index, num_nodes=n)
    return graph, labels, k


def _csr_from_sparsegraph(g: SparseGraph) -> sp.csr_matrix:
    n = g.n_nodes
    rows = []
    cols = []
    vals = []
    for node in range(n):
        nbrs = g.neighbors[node]
        ws = g.weights[node]
        if nbrs.size:
            rows.append(np.full(nbrs.size, node, dtype=np.int64))
            cols.append(nbrs.astype(np.int64))
            vals.append(ws.astype(float))
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(vals)
    else:
        row = np.zeros(0, dtype=np.int64)
        col = np.zeros(0, dtype=np.int64)
        data = np.zeros(0, dtype=float)
    return sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()


def seclust_tree_levels(graph: SparseGraph, seed: int = SEED) -> list[np.ndarray]:
    """Run multistart, then build coding-tree hierarchy levels."""

    base, _ = multistart_incremental_se_heuristic(graph, starts=4, max_passes=8, seed=seed)
    levels = coding_tree_hierarchy_levels(graph, base, min_clusters=1)
    # `levels` is a tuple of HierarchicalLevel objects, finest-first.
    return [np.asarray(lv.labels, dtype=np.int64) for lv in levels]


def hcse_levels(graph: SparseGraph, k_target: int) -> list[np.ndarray]:
    """Run HCSE official implementation, extract levels by cluster height."""

    from glass.seclust.benchmark_sep import run_official_sep_coding_tree

    csr = _csr_from_sparsegraph(graph)
    dense = csr.toarray()
    result = run_official_sep_coding_tree(dense, k=max(2, int(k_target)))
    # The official PartitionTree returns a flat partition; we don't have
    # easy access to its intermediate levels. Approximate the hierarchy
    # by [singletons -> flat partition].
    leaves = np.arange(csr.shape[0], dtype=np.int64)
    flat = np.asarray(result.labels, dtype=np.int64)
    return [leaves, flat]


def louvain_dendrogram_levels(graph: SparseGraph) -> list[np.ndarray]:
    """Louvain produces a multi-level dendrogram via
    community_louvain.generate_dendrogram."""

    csr = _csr_from_sparsegraph(graph)
    G = nx.from_scipy_sparse_array(csr)
    dendro = community_louvain.generate_dendrogram(G, random_state=SEED)
    n = csr.shape[0]
    # `dendro` is a list of dicts: dendro[0] maps node->cluster,
    # dendro[1] maps cluster->meta-cluster, etc.
    levels: list[np.ndarray] = []
    composed = np.asarray([dendro[0][i] for i in range(n)], dtype=np.int64)
    levels.append(composed.copy())
    for li in range(1, len(dendro)):
        mapping = dendro[li]
        composed = np.asarray([mapping[c] for c in composed], dtype=np.int64)
        levels.append(composed.copy())
    return levels


def score_one(
    name: str,
    graph: SparseGraph,
    labels: np.ndarray,
    levels: list[np.ndarray],
    method: str,
) -> dict:
    csr = _csr_from_sparsegraph(graph)
    finest = levels[0]
    coarsest = levels[-1]
    se_finest = sparse_structural_entropy(graph, finest)
    se_coarsest = sparse_structural_entropy(graph, coarsest)

    # Dasgupta cost (canonical hierarchical objective).
    t0 = time.time()
    cost = dasgupta_cost(csr, levels)
    dpu = dendrogram_purity(levels, labels)
    metric_time = time.time() - t0

    return {
        "dataset": name,
        "method": method,
        "n_levels": len(levels),
        "K_finest": int(np.unique(finest).size),
        "K_coarsest": int(np.unique(coarsest).size),
        "SE_finest": float(se_finest),
        "SE_coarsest": float(se_coarsest),
        "dasgupta_cost": float(cost),
        "dendrogram_purity": float(dpu),
        "metric_seconds": float(metric_time),
    }


def run() -> list[dict]:
    rows: list[dict] = []
    datasets = ["Cora", "Citeseer", "Photo"]
    methods = ["SEClust-Tree", "HCSE", "Louvain-Dendrogram"]
    pbar = tqdm(total=len(datasets) * len(methods), desc="Hierarchical", unit="cell")
    for name in datasets:
        try:
            graph, labels, k = _load_pyg_dataset(name)
        except Exception as exc:
            tqdm.write(f"  {name} unavailable: {exc}")
            for m in methods:
                rows.append({"dataset": name, "method": m, "status": f"skipped: {exc}"})
                pbar.update(1)
            continue

        for method in methods:
            pbar.set_postfix_str(f"{name} / {method}")
            try:
                t0 = time.time()
                if method == "SEClust-Tree":
                    levels = seclust_tree_levels(graph, seed=SEED)
                elif method == "HCSE":
                    if graph.n_nodes > 5000:
                        rows.append({
                            "dataset": name, "method": method,
                            "status": f"skipped: dense N>{5000}",
                        })
                        pbar.update(1)
                        continue
                    levels = hcse_levels(graph, k_target=k)
                elif method == "Louvain-Dendrogram":
                    levels = louvain_dendrogram_levels(graph)
                else:
                    raise ValueError(method)
                run_seconds = time.time() - t0
                row = score_one(name, graph, labels, levels, method)
                row["build_seconds"] = float(run_seconds)
                rows.append(row)
                tqdm.write(
                    f"  {name} {method}: levels={row['n_levels']}, "
                    f"Das={row['dasgupta_cost']:.3e}, "
                    f"DP={row['dendrogram_purity']:.4f}, "
                    f"SE_fine={row['SE_finest']:.3f}, t={run_seconds:.2f}s"
                )
            except Exception as exc:
                rows.append({
                    "dataset": name, "method": method,
                    "status": f"failed: {exc}",
                })
                tqdm.write(f"  {name} {method} failed: {exc}")
            pbar.update(1)

    pbar.close()
    return rows


def write_report(rows: list[dict]) -> None:
    out = Path("docs/experimental_reports")
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"hierarchical_{timestamp}.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Hierarchical-Clustering Benchmark ({timestamp})",
        "",
        "Scores SEClust-Tree, HCSE, and Louvain-Dendrogram on Cora / Citeseer / "
        "Photo using the hierarchical-quality metrics from HCSE [Pan et al. 2021] "
        "and HypCSE [Zeng et al. 2025]. Lower Dasgupta cost is better; higher "
        "dendrogram purity is better.",
        "",
        "| Dataset | Method | Levels | K (fine, coarse) | SE (fine, coarse) | "
        "Dasgupta cost ↓ | Dendrogram purity ↑ | Build (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in rows:
        if "status" in row:
            md_lines.append(
                f"| {row['dataset']} | {row['method']} | --- | --- | --- | --- | --- | "
                f"_{row['status']}_ |"
            )
            continue
        md_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n_levels']} | "
            f"{row['K_finest']}, {row['K_coarsest']} | "
            f"{row['SE_finest']:.3f}, {row['SE_coarsest']:.3f} | "
            f"{row['dasgupta_cost']:.3e} | "
            f"{row['dendrogram_purity']:.4f} | "
            f"{row['build_seconds']:.2f} |"
        )
    md_path = out / f"hierarchical_{timestamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    rows = run()
    write_report(rows)
