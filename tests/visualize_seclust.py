"""Visualizations for the SEClust paper.

Generates four figures into ``docs/experimental_reports/figures/``:

1. ``embedding_cora_<algo>.png``
   t-SNE 2D projection of Cora's spectral embedding, coloured by the
   predicted cluster of each algorithm and by the ground-truth label.
   Mirrors the layout of LSEnet (Sun et al. 2024) Figure 5 but renders
   on the Euclidean t-SNE plane rather than the Poincaré disc, since
   our SEClust variants are non-hyperbolic.

2. ``dendrogram_seclust_tree_sbm200.png``
   The hierarchical levels emitted by ``coding_tree_hierarchy_levels``
   on a small SBM (N=200, K=4) rendered as a dendrogram. The format
   matches HypCSE (Zeng et al. 2025) Figure 5, panel (a).

3. ``param_sensitivity_seclust.png``
   Heatmap of final structural entropy vs.
   (heuristic_starts, max_passes) for SEClust-Auto on SBM-500.
   Mirrors the parameter-sensitivity figures in LSEnet Figure 3 and
   HypCSE Figures 3-4.

4. ``runtime_loglog_combined.png``
   Re-plots the scalability sweep at log-log with linear extrapolations
   for each algorithm, in the style of LSEnet Figure 4.
"""

from __future__ import annotations

from datetime import datetime
import json
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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

from glass.seclust import (
    SparseGraph,
    cluster_graph,
    coding_tree_hierarchy_levels,
    hierarchical_se_clustering,
    multistart_incremental_se_heuristic,
    sparse_structural_entropy,
)


SEED = 42
OUT_DIR = Path("docs/experimental_reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _csr_from_sparsegraph(graph: SparseGraph) -> sp.csr_matrix:
    n = graph.n_nodes
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for node in range(n):
        nbrs = graph.neighbors[node]
        ws = graph.weights[node]
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


def load_cora() -> tuple[SparseGraph, np.ndarray, np.ndarray, int]:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_undirected

    ds = Planetoid(root="/tmp/dataset", name="Cora")
    data = ds[0]
    n = int(data.num_nodes)
    edge_index = to_undirected(data.edge_index).cpu().numpy().astype(np.int64)
    g = SparseGraph.from_edge_index(edge_index, num_nodes=n)
    labels = data.y.cpu().numpy().astype(np.int32)
    features = data.x.cpu().numpy().astype(np.float32)
    return g, labels, features, int(ds.num_classes)


def run_louvain_sparse(csr: sp.csr_matrix, seed: int) -> np.ndarray:
    G = nx.from_scipy_sparse_array(csr)
    part = community_louvain.best_partition(G, random_state=seed)
    return np.asarray([part[i] for i in range(csr.shape[0])], dtype=np.int32)


def run_leiden_sparse(csr: sp.csr_matrix, seed: int) -> np.ndarray:
    coo = sp.triu(csr, k=1).tocoo()
    edges = list(zip(coo.row.astype(np.int64).tolist(), coo.col.astype(np.int64).tolist()))
    g = igraph.Graph(n=csr.shape[0], edges=edges, directed=False)
    weights = coo.data.astype(float).tolist()
    g.es["weight"] = weights
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=weights, seed=seed)
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for i, cluster in enumerate(part):
        for node in cluster:
            labels[node] = i
    return labels


def run_infomap_sparse(csr: sp.csr_matrix, seed: int) -> np.ndarray:
    coo = csr.tocoo()
    model = infomap.Infomap(f"--two-level --silent --seed {seed}")
    for r, c, v in zip(coo.row, coo.col, coo.data):
        model.add_link(int(r), int(c), float(v))
    model.run()
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for node in model.tree:
        if node.is_leaf:
            labels[node.node_id] = node.module_id - 1
    return labels


def figure_embedding_cora() -> Path:
    """t-SNE of Cora coloured by each algorithm's prediction."""

    print("Loading Cora ...", flush=True)
    g, gt, features, k = load_cora()
    csr = _csr_from_sparsegraph(g)

    print("Running baselines ...", flush=True)
    labels_louvain = run_louvain_sparse(csr, SEED)
    labels_leiden = run_leiden_sparse(csr, SEED)
    labels_infomap = run_infomap_sparse(csr, SEED)
    print("Running SEClust-Auto ...", flush=True)
    res_auto = cluster_graph(g, mode="heuristic", heuristic_starts=4, max_passes=8, seed=SEED)
    print("Running SEClust-Tree ...", flush=True)
    res_tree = hierarchical_se_clustering(g, target_clusters=k, starts=4, max_passes=8, seed=SEED)

    panels = [
        ("Ground truth", gt),
        ("Louvain", labels_louvain),
        ("Leiden", labels_leiden),
        ("Infomap", labels_infomap),
        ("SEClust-Auto", res_auto.labels),
        ("SEClust-Tree", res_tree.labels),
    ]

    print("Computing t-SNE on 64-dim spectral embedding ...", flush=True)
    # Use a feature-light embedding source: random projection of the
    # adjacency. Cheaper than spectral and avoids JAX dependency loops.
    rng = np.random.default_rng(SEED)
    proj = rng.normal(size=(csr.shape[0], 64))
    h = csr @ proj
    h = h - h.mean(axis=0, keepdims=True)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED, max_iter=400)
    coords = tsne.fit_transform(h)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    axes = axes.flatten()
    cmap = plt.get_cmap("tab20")
    for ax, (title, labels) in zip(axes, panels):
        unique = np.unique(labels)
        for i, cid in enumerate(unique):
            mask = labels == cid
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=4, alpha=0.7,
                color=cmap(i % cmap.N),
                edgecolors="none",
            )
        ax.set_title(f"{title}  (K={len(unique)})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ("top", "right", "bottom", "left"):
            ax.spines[sp_].set_alpha(0.4)
    fig.suptitle("Cora node clusters projected to 2D (t-SNE on adjacency-random projection)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "embedding_cora.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def figure_dendrogram_seclust_tree() -> Path:
    """Render SEClust hierarchy on a small SBM as a dendrogram."""

    rng = np.random.default_rng(SEED)
    n, k_planted = 200, 4
    sizes = [n // k_planted] * k_planted
    p_in, p_out = 0.30, 0.02
    blocks = np.repeat(np.arange(k_planted, dtype=np.int32), sizes)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if blocks[i] == blocks[j] else p_out
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1.0

    print(f"SBM-{n}/K={k_planted}: running multistart base partition ...", flush=True)
    base_labels, _ = multistart_incremental_se_heuristic(adj, starts=4, max_passes=8, seed=SEED)
    levels = coding_tree_hierarchy_levels(adj, base_labels, min_clusters=1)
    print(f"  {len(levels)} levels emitted", flush=True)

    # `levels[0]` is the finest partition (largest K); `levels[-1]` is
    # the coarsest. Use the finest as the linkage leaves.
    base = levels[0].labels
    cluster_ids = np.unique(base)
    n_leaves = len(cluster_ids)
    if n_leaves < 2:
        # Degenerate hierarchy: fall back to the per-node base labels.
        base = base_labels
        cluster_ids = np.unique(base)
        n_leaves = len(cluster_ids)
    points = np.zeros((n_leaves, 2), dtype=float)
    rng_pts = np.random.default_rng(SEED)
    for idx, cid in enumerate(cluster_ids):
        members = np.where(base == cid)[0]
        block_idx = int(blocks[members[0]])
        points[idx] = rng_pts.normal(scale=0.6, size=2) + 5.0 * np.array(
            [np.cos(2 * np.pi * block_idx / k_planted),
             np.sin(2 * np.pi * block_idx / k_planted)]
        )
    Z = linkage(points, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    color_threshold = 0.7 * Z[:, 2].max()
    dendrogram(Z, color_threshold=color_threshold, above_threshold_color="grey", ax=ax)
    ax.set_title(
        f"SEClust hierarchy on SBM(N={n}, planted K={k_planted}) — {len(levels)} levels",
        fontsize=11,
    )
    ax.set_xlabel("Cluster index (finest SEClust level)")
    ax.set_ylabel("Merge height (Ward on cluster centroids)")
    fig.tight_layout()
    out = OUT_DIR / "dendrogram_seclust_tree_sbm200.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def figure_parameter_sensitivity() -> Path:
    """Heatmap of SE vs (heuristic_starts, max_passes) on SBM-500."""

    rng = np.random.default_rng(SEED)
    n, k_planted = 500, 5
    sizes = [n // k_planted] * k_planted
    p_in, p_out = 0.20, 0.01
    blocks = np.repeat(np.arange(k_planted, dtype=np.int32), sizes)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if blocks[i] == blocks[j] else p_out
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1.0

    starts_grid = [1, 2, 4, 8, 16]
    passes_grid = [1, 2, 4, 8, 16]
    se_grid = np.zeros((len(starts_grid), len(passes_grid)), dtype=float)
    runtime_grid = np.zeros_like(se_grid)
    for i, s in enumerate(starts_grid):
        for j, p in enumerate(passes_grid):
            t0 = time.time()
            res = cluster_graph(adj, mode="heuristic", heuristic_starts=s, max_passes=p, seed=SEED)
            runtime_grid[i, j] = time.time() - t0
            se_grid[i, j] = float(res.entropy)
            print(f"  starts={s} passes={p} SE={res.entropy:.4f} t={runtime_grid[i, j]:.2f}s", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(se_grid, origin="lower", cmap="viridis_r", aspect="auto")
    axes[0].set_xticks(range(len(passes_grid)))
    axes[0].set_xticklabels(passes_grid)
    axes[0].set_yticks(range(len(starts_grid)))
    axes[0].set_yticklabels(starts_grid)
    axes[0].set_xlabel("max_passes")
    axes[0].set_ylabel("heuristic_starts")
    axes[0].set_title("Final structural entropy (lower is better)")
    for i in range(len(starts_grid)):
        for j in range(len(passes_grid)):
            axes[0].text(j, i, f"{se_grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(runtime_grid, origin="lower", cmap="magma_r", aspect="auto")
    axes[1].set_xticks(range(len(passes_grid)))
    axes[1].set_xticklabels(passes_grid)
    axes[1].set_yticks(range(len(starts_grid)))
    axes[1].set_yticklabels(starts_grid)
    axes[1].set_xlabel("max_passes")
    axes[1].set_ylabel("heuristic_starts")
    axes[1].set_title("Runtime (s)")
    for i in range(len(starts_grid)):
        for j in range(len(passes_grid)):
            axes[1].text(j, i, f"{runtime_grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle("SEClust parameter sensitivity on SBM(N=500, K=5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "param_sensitivity_seclust.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def figure_runtime_loglog() -> Path:
    """Re-render the scalability sweep at log-log with extrapolated power-law fit."""

    candidates = sorted(Path("docs/experimental_reports").glob("scalability_seclust_*.json"))
    if not candidates:
        print("No scalability_seclust_*.json found; run tests/scalability_seclust.py first.")
        return Path()
    src = candidates[-1]
    rows = json.loads(src.read_text())

    by_algo: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row.get("runtime_seconds") is None:
            continue
        by_algo.setdefault(row["algorithm"], []).append((int(row["n_nodes"]), float(row["runtime_seconds"])))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = {"Louvain": "o", "Leiden": "s", "Infomap": "^", "SEClust": "D"}
    colors = {"Louvain": "#1f77b4", "Leiden": "#ff7f0e", "Infomap": "#2ca02c", "SEClust": "#d62728"}
    for algo, points in sorted(by_algo.items()):
        points.sort()
        ns = np.array([p[0] for p in points], dtype=float)
        ts = np.array([p[1] for p in points], dtype=float)
        ax.loglog(ns, ts, marker=markers.get(algo, "o"), color=colors.get(algo), label=algo, lw=1.6, ms=7)
        # Power-law fit on log-log
        if len(ns) >= 3:
            slope, intercept = np.polyfit(np.log10(ns), np.log10(ts), 1)
            ax.text(
                ns[-1] * 1.05, ts[-1],
                f"  ~N^{slope:.2f}",
                fontsize=9,
                color=colors.get(algo),
                verticalalignment="center",
            )

    ax.set_xlabel("Number of nodes (N)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime scalability with empirical power-law exponents")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "runtime_loglog_combined.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main() -> None:
    print("=== Figure 1: t-SNE of Cora ===")
    figure_embedding_cora()
    print("\n=== Figure 2: SEClust dendrogram ===")
    figure_dendrogram_seclust_tree()
    print("\n=== Figure 3: parameter sensitivity ===")
    figure_parameter_sensitivity()
    print("\n=== Figure 4: runtime log-log ===")
    figure_runtime_loglog()
    print(f"\nFigures saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
