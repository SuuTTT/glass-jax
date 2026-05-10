"""Idea 015: ogbn-arxiv benchmark for SEClust at $N=169{,}343$.

Loads ogbn-arxiv via OGB, runs Louvain/Leiden/Infomap baselines and
the SEClust-Auto / SEClust-ConstrainedK variants, and reports
cluster-quality metrics (ACC against the 40 paper-topic classes,
NMI, ARI) plus runtime. The largest empirical cell in the paper.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OGB_DOWNLOAD_DIR", "/tmp/ogb")

import numpy as np
import scipy.sparse as sp


def load_ogbn_arxiv():
    from ogb.nodeproppred import NodePropPredDataset
    ds = NodePropPredDataset(name="ogbn-arxiv", root="/tmp/ogb")
    graph, labels = ds[0]
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    n = int(graph["num_nodes"])
    y = np.asarray(labels, dtype=np.int64).flatten()
    return edge_index, n, y


def best_perm_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(int(y_true.max()), int(y_pred.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    if D > 200:
        # Hungarian on huge K is slow; greedy fallback.
        used_rows, used_cols, total = set(), set(), 0
        entries = sorted(((w[r, c], r, c) for r in range(D) for c in range(D) if w[r, c] > 0), reverse=True)
        for v, r, c in entries:
            if r not in used_rows and c not in used_cols:
                total += v
                used_rows.add(r); used_cols.add(c)
        return float(total / y_pred.size)
    ind = linear_sum_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in zip(*ind)) / y_pred.size)


def run_louvain(csr: sp.csr_matrix, seed: int):
    import networkx as nx
    from community import community_louvain
    t0 = time.time()
    g = nx.from_scipy_sparse_array(csr)
    part = community_louvain.best_partition(g, random_state=seed)
    n = csr.shape[0]
    labels = np.array([part[i] for i in range(n)], dtype=np.int32)
    return labels, time.time() - t0


def run_leiden(csr: sp.csr_matrix, seed: int):
    import igraph
    import leidenalg
    t0 = time.time()
    coo = sp.triu(csr, k=1).tocoo()
    edges = list(zip(coo.row.astype(np.int64).tolist(), coo.col.astype(np.int64).tolist()))
    g = igraph.Graph(n=csr.shape[0], edges=edges, directed=False)
    weights = coo.data.astype(float).tolist()
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights=weights, seed=seed,
    )
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for i, cluster in enumerate(part):
        for node in cluster:
            labels[node] = i
    return labels, time.time() - t0


def run_seclust_auto(graph, seed: int, starts: int = 2, max_passes: int = 6):
    from glass.seclust import cluster_graph
    t0 = time.time()
    res = cluster_graph(
        graph, mode="heuristic", heuristic_starts=starts, max_passes=max_passes, seed=seed,
    )
    return np.asarray(res.labels, dtype=np.int32), float(res.entropy), time.time() - t0


def run_seclust_constrained(graph, target_k: int, seed: int, starts: int = 2, max_passes: int = 6):
    from glass.seclust import constrained_k_multistart
    t0 = time.time()
    labels, ent = constrained_k_multistart(
        graph, target_clusters=target_k,
        starts=starts, max_passes=max_passes, seed=seed,
    )
    return np.asarray(labels, dtype=np.int32), float(ent), time.time() - t0


def main():
    print("== Loading ogbn-arxiv ==", flush=True)
    edge_index, n, y_true = load_ogbn_arxiv()
    print(f"  N={n}, edges={edge_index.shape[1]}, K_gt={len(np.unique(y_true))}", flush=True)

    print("== Building SparseGraph + CSR ==", flush=True)
    from glass.seclust import SparseGraph
    t0 = time.time()
    graph = SparseGraph.from_edge_index(edge_index, num_nodes=n)
    rows = np.concatenate([edge_index[0], edge_index[1]])
    cols = np.concatenate([edge_index[1], edge_index[0]])
    data = np.ones(rows.size, dtype=float) * 0.5
    csr = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    csr.sum_duplicates()
    print(f"  built in {time.time()-t0:.1f}s, volume={graph.volume:.0f}", flush=True)

    K_gt = int(len(np.unique(y_true)))
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    rows_out: list[dict] = []
    SEED = 42

    def add(name: str, labels: np.ndarray, t: float, se: float | None = None):
        K = int(np.unique(labels).size)
        acc = best_perm_acc(y_true, labels)
        nmi = float(normalized_mutual_info_score(y_true, labels))
        ari = float(adjusted_rand_score(y_true, labels))
        row = {
            "method": name, "K": K, "ACC": acc, "NMI": nmi, "ARI": ari,
            "time_s": float(t), "structural_entropy": se,
        }
        rows_out.append(row)
        se_str = f"  SE={se:.3f}" if se is not None else ""
        print(f"  {name:<22} K={K:5d}  ACC={acc:.3f}  NMI={nmi:.3f}  ARI={ari:.3f}  t={t:.1f}s{se_str}", flush=True)

    print("== Running Louvain ==", flush=True)
    try:
        labels, t = run_louvain(csr, SEED)
        add("Louvain", labels, t)
    except Exception as exc:
        print(f"  Louvain failed: {exc}", flush=True)
        rows_out.append({"method": "Louvain", "status": f"failed: {exc}"})

    print("== Running Leiden ==", flush=True)
    try:
        labels, t = run_leiden(csr, SEED)
        add("Leiden", labels, t)
    except Exception as exc:
        print(f"  Leiden failed: {exc}", flush=True)
        rows_out.append({"method": "Leiden", "status": f"failed: {exc}"})

    print("== Running SEClust-Auto (free K, multistart) ==", flush=True)
    try:
        labels, se, t = run_seclust_auto(graph, SEED, starts=2, max_passes=6)
        add("SEClust-Auto", labels, t, se)
    except Exception as exc:
        print(f"  SEClust-Auto failed: {exc}", flush=True)
        rows_out.append({"method": "SEClust-Auto", "status": f"failed: {exc}"})

    print(f"== Running SEClust-ConstrainedK (K={K_gt}) ==", flush=True)
    try:
        labels, se, t = run_seclust_constrained(graph, K_gt, SEED, starts=2, max_passes=6)
        add("SEClust-ConstrainedK", labels, t, se)
    except Exception as exc:
        print(f"  SEClust-ConstrainedK failed: {exc}", flush=True)
        rows_out.append({"method": "SEClust-ConstrainedK", "status": f"failed: {exc}"})

    # Save report.
    out = Path("docs/experimental_reports")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"ogbn_arxiv_{ts}.json"
    json_path.write_text(json.dumps({
        "dataset": "ogbn-arxiv",
        "n_nodes": int(n),
        "n_edges_directed": int(edge_index.shape[1]),
        "K_gt": K_gt,
        "seed": SEED,
        "results": rows_out,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
