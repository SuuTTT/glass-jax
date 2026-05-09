"""Scalability sweep: SEClust vs Louvain vs Leiden on synthetic SBMs.

Generates a sparse SBM at increasing N, runs each algorithm, measures wall
time and structural entropy / modularity, and writes a JSON report plus
log-log plots into ``docs/experimental_reports``.

Graphs are built directly into a SciPy CSR adjacency without any dense
``(N, N)`` materialization, so the sweep can scale beyond N=1e5.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time

import igraph
import infomap
import leidenalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from community import community_louvain
from tqdm import tqdm

from glass.seclust import (
    IncrementalSEState,
    SparseGraph,
    cluster_graph,
    sparse_structural_entropy,
)


SEED = 42
SECLUST_STARTS = 4
SECLUST_MAX_PASSES = 8
TIME_LIMIT_SECONDS = 600.0


@dataclass
class SweepRow:
    n_nodes: int
    n_edges: int
    n_communities: int
    algorithm: str
    runtime_seconds: float | None
    structural_entropy: float | None
    modularity: float | None
    n_clusters: int | None
    status: str

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_communities": self.n_communities,
            "algorithm": self.algorithm,
            "runtime_seconds": self.runtime_seconds,
            "structural_entropy": self.structural_entropy,
            "modularity": self.modularity,
            "n_clusters": self.n_clusters,
            "status": self.status,
        }


def sparse_sbm(n_nodes: int, n_communities: int, p_in: float, p_out: float, seed: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build an SBM as a CSR adjacency without dense materialization.

    For each pair of community blocks, draw the number of edges from a binomial
    on the number of slots, then sample positions without replacement. This is
    O(num_edges) instead of O(N^2).
    """

    rng = np.random.default_rng(seed)
    base = n_nodes // n_communities
    sizes = [base] * (n_communities - 1) + [n_nodes - base * (n_communities - 1)]
    starts = np.cumsum([0] + sizes)
    labels = np.repeat(np.arange(n_communities, dtype=np.int32), sizes)

    rows = []
    cols = []
    for a in range(n_communities):
        for b in range(a, n_communities):
            sa, ea = int(starts[a]), int(starts[a + 1])
            sb, eb = int(starts[b]), int(starts[b + 1])
            if a == b:
                # Pairs (i<j) within block a
                size_a = ea - sa
                slots = size_a * (size_a - 1) // 2
                if slots == 0:
                    continue
                m = int(rng.binomial(slots, p_in))
                if m == 0:
                    continue
                picks = rng.choice(slots, size=m, replace=False)
                # Decode to (i,j) pairs in upper triangle inside block.
                # Using inverse triangular indexing is unstable for large N;
                # use numpy.triu_indices on a single block.
                triu_i, triu_j = np.triu_indices(size_a, k=1)
                pi = sa + triu_i[picks]
                pj = sa + triu_j[picks]
                rows.extend([pi, pj])
                cols.extend([pj, pi])
            else:
                size_a = ea - sa
                size_b = eb - sb
                slots = size_a * size_b
                if slots == 0:
                    continue
                m = int(rng.binomial(slots, p_out))
                if m == 0:
                    continue
                picks = rng.choice(slots, size=m, replace=False)
                pi = sa + (picks // size_b)
                pj = sb + (picks % size_b)
                rows.extend([pi, pj])
                cols.extend([pj, pi])

    if rows:
        row = np.concatenate(rows).astype(np.int64)
        col = np.concatenate(cols).astype(np.int64)
    else:
        row = np.zeros(0, dtype=np.int64)
        col = np.zeros(0, dtype=np.int64)
    data = np.ones(row.size, dtype=float)
    matrix = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    matrix.sum_duplicates()
    return matrix, labels


def hard_modularity_csr(csr: sp.csr_matrix, labels: np.ndarray) -> float:
    degrees = np.asarray(csr.sum(axis=1)).flatten()
    volume = float(degrees.sum())
    if volume <= 1e-12:
        return 0.0
    score = 0.0
    for cluster in np.unique(labels):
        mask = labels == cluster
        cluster_degree = float(degrees[mask].sum())
        sub = csr[mask][:, mask]
        internal = float(sub.sum())
        score += internal - (cluster_degree * cluster_degree / volume)
    return float(score / volume)


def run_louvain(csr: sp.csr_matrix, seed: int) -> tuple[np.ndarray, float]:
    start = time.time()
    graph = nx.from_scipy_sparse_array(csr)
    partition = community_louvain.best_partition(graph, random_state=seed)
    n = csr.shape[0]
    labels = np.array([partition[i] for i in range(n)], dtype=np.int32)
    return labels, time.time() - start


def run_leiden(csr: sp.csr_matrix, seed: int) -> tuple[np.ndarray, float]:
    start = time.time()
    coo = sp.triu(csr, k=1).tocoo()
    edges = list(zip(coo.row.astype(np.int64).tolist(), coo.col.astype(np.int64).tolist()))
    g = igraph.Graph(n=csr.shape[0], edges=edges, directed=False)
    weights = coo.data.astype(float).tolist()
    g.es["weight"] = weights
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=weights, seed=seed)
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for i, cluster in enumerate(partition):
        for node in cluster:
            labels[node] = i
    return labels, time.time() - start


def run_infomap(csr: sp.csr_matrix, seed: int) -> tuple[np.ndarray, float]:
    start = time.time()
    coo = csr.tocoo()
    model = infomap.Infomap(f"--two-level --silent --seed {seed}")
    for r, c, v in zip(coo.row, coo.col, coo.data):
        model.add_link(int(r), int(c), float(v))
    model.run()
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for node in model.tree:
        if node.is_leaf:
            labels[node.node_id] = node.module_id - 1
    return labels, time.time() - start


def run_seclust(graph: SparseGraph, seed: int, k_hint: int | None) -> tuple[np.ndarray, float, float]:
    start = time.time()
    result = cluster_graph(
        graph,
        mode="heuristic",
        heuristic_starts=SECLUST_STARTS,
        max_passes=SECLUST_MAX_PASSES,
        seed=seed,
    )
    elapsed = time.time() - start
    return result.labels, elapsed, float(result.entropy)


def estimate_seclust_seconds(graph: SparseGraph) -> float:
    n = graph.n_nodes
    rng = np.random.default_rng(SEED)
    init_labels = rng.integers(0, max(2, min(16, int(math.sqrt(max(n, 2))) + 2)), size=n, dtype=np.int32)
    state = IncrementalSEState(graph, init_labels)
    sample = rng.choice(n, size=min(n, 64), replace=False)
    evaluations = 0
    candidate_counts: list[int] = []
    start = time.time()
    for node in sample:
        candidates = state.candidate_clusters(int(node), allow_new_cluster=True)
        candidate_counts.append(len(candidates))
        for target in candidates:
            state.move_delta(int(node), int(target))
            evaluations += 1
    elapsed = max(time.time() - start, 1e-9)
    per_eval = elapsed / max(evaluations, 1)
    avg_candidates = float(np.mean(candidate_counts)) if candidate_counts else 1.0
    estimated = SECLUST_STARTS * SECLUST_MAX_PASSES * n * avg_candidates
    return float(per_eval * estimated * 1.25)


def evaluate_se(graph: SparseGraph, labels: np.ndarray) -> float:
    return float(sparse_structural_entropy(graph, labels))


def run_sweep(sizes: list[int]) -> list[SweepRow]:
    rows: list[SweepRow] = []
    algorithms = ["Louvain", "Leiden", "Infomap", "SEClust"]
    pbar = tqdm(total=len(sizes) * len(algorithms), desc="Scalability", unit="cell", dynamic_ncols=True)
    for n_nodes in sizes:
        # Keep average degree roughly stable as N grows; communities grow as sqrt(N).
        n_communities = max(2, int(round(math.sqrt(n_nodes / 10))))
        avg_in_degree = 12.0
        avg_out_degree = 1.5
        block_size = n_nodes / n_communities
        p_in = min(0.9, avg_in_degree / max(block_size - 1, 1.0))
        p_out = min(0.5, avg_out_degree / max(n_nodes - block_size, 1.0))
        print(f"\n=== N={n_nodes}, K={n_communities}, p_in={p_in:.4f}, p_out={p_out:.4f} ===", flush=True)

        t0 = time.time()
        csr, true_labels = sparse_sbm(n_nodes, n_communities, p_in, p_out, SEED)
        print(f"  built graph: nnz={csr.nnz} ({time.time() - t0:.2f}s)", flush=True)
        graph = SparseGraph.from_csr(csr)
        n_edges = int(csr.nnz / 2)

        for algo, runner in [
            ("Louvain", lambda: run_louvain(csr, SEED)),
            ("Leiden", lambda: run_leiden(csr, SEED)),
            ("Infomap", lambda: run_infomap(csr, SEED)),
        ]:
            pbar.set_postfix_str(f"N={n_nodes} / {algo}")
            try:
                labels, elapsed = runner()
                se_val = evaluate_se(graph, labels)
                mod = hard_modularity_csr(csr, labels)
                rows.append(SweepRow(n_nodes, n_edges, n_communities, algo, elapsed, se_val, mod, int(len(np.unique(labels))), "ok"))
                tqdm.write(f"  N={n_nodes} {algo}: {elapsed:.2f}s, SE={se_val:.4f}, Q={mod:.4f}, K={len(np.unique(labels))}")
            except Exception as exc:
                rows.append(SweepRow(n_nodes, n_edges, n_communities, algo, None, None, None, None, f"failed: {exc}"))
                tqdm.write(f"  N={n_nodes} {algo} failed: {exc}")
            pbar.update(1)

        pbar.set_postfix_str(f"N={n_nodes} / SEClust")
        try:
            estimate = estimate_seclust_seconds(graph)
            tqdm.write(f"  N={n_nodes} SEClust estimate: {estimate:.1f}s")
            if estimate > TIME_LIMIT_SECONDS:
                rows.append(SweepRow(n_nodes, n_edges, n_communities, "SEClust", None, None, None, None, f"skipped (estimate {estimate:.0f}s)"))
            else:
                labels, elapsed, se_val = run_seclust(graph, SEED, k_hint=n_communities)
                mod = hard_modularity_csr(csr, labels)
                rows.append(SweepRow(n_nodes, n_edges, n_communities, "SEClust", elapsed, se_val, mod, int(len(np.unique(labels))), "ok"))
                tqdm.write(f"  N={n_nodes} SEClust: {elapsed:.2f}s, SE={se_val:.4f}, Q={mod:.4f}, K={len(np.unique(labels))}")
        except Exception as exc:
            rows.append(SweepRow(n_nodes, n_edges, n_communities, "SEClust", None, None, None, None, f"failed: {exc}"))
            tqdm.write(f"  N={n_nodes} SEClust failed: {exc}")
        pbar.update(1)

    pbar.close()
    return rows


def make_plots(rows: list[SweepRow], out_dir: Path, timestamp: str) -> tuple[Path, Path]:
    by_algo: dict[str, list[SweepRow]] = {}
    for row in rows:
        if row.runtime_seconds is None:
            continue
        by_algo.setdefault(row.algorithm, []).append(row)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"Louvain": "o", "Leiden": "s", "Infomap": "^", "SEClust": "D"}
    colors = {"Louvain": "#1f77b4", "Leiden": "#ff7f0e", "Infomap": "#2ca02c", "SEClust": "#d62728"}
    for algo, group in sorted(by_algo.items()):
        group_sorted = sorted(group, key=lambda r: r.n_nodes)
        ns = [r.n_nodes for r in group_sorted]
        ts = [r.runtime_seconds for r in group_sorted]
        ax.loglog(ns, ts, marker=markers.get(algo, "o"), color=colors.get(algo, None), label=algo, linewidth=1.6, markersize=7)
    ax.set_xlabel("Number of nodes (N)")
    ax.set_ylabel("Runtime (seconds, log-log)")
    ax.set_title("Runtime scalability on synthetic SBMs")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    runtime_path = out_dir / f"scalability_runtime_{timestamp}.png"
    fig.savefig(runtime_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for algo, group in sorted(by_algo.items()):
        group_sorted = sorted(group, key=lambda r: r.n_nodes)
        ns = [r.n_nodes for r in group_sorted if r.structural_entropy is not None]
        ses = [r.structural_entropy for r in group_sorted if r.structural_entropy is not None]
        if ns:
            ax.semilogx(ns, ses, marker=markers.get(algo, "o"), color=colors.get(algo, None), label=algo, linewidth=1.6, markersize=7)
    ax.set_xlabel("Number of nodes (N, log)")
    ax.set_ylabel("Structural entropy (bits, lower is better)")
    ax.set_title("Partition quality (SE) vs N on synthetic SBMs")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    quality_path = out_dir / f"scalability_quality_{timestamp}.png"
    fig.savefig(quality_path, dpi=140)
    plt.close(fig)
    return runtime_path, quality_path


def main() -> None:
    sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000]
    rows = run_sweep(sizes)
    out_dir = Path("docs/experimental_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"scalability_seclust_{timestamp}.json"
    json_path.write_text(json.dumps([r.to_dict() for r in rows], indent=2) + "\n", encoding="utf-8")
    runtime_path, quality_path = make_plots(rows, out_dir, timestamp)
    print(f"\nResults: {json_path}")
    print(f"Runtime plot: {runtime_path}")
    print(f"Quality plot: {quality_path}")


if __name__ == "__main__":
    main()
