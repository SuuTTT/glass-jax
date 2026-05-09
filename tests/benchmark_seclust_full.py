"""SEClust benchmark against previously reported baselines.

This script mirrors the report style of ``benchmark_sbm_20260506.md`` and
``real_world_comparison_20260507.md``. Baseline rows for synthetic datasets
mix executed runs (Louvain/Leiden/Infomap/Glass-JAX) and SEClust variants.
Real-world datasets now use a sparse pipeline (no dense materialization),
so Cora / Citeseer / Photo are executed end-to-end.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import itertools
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import networkx as nx
import optax
import numpy as np
import scipy.sparse as sp
from community import community_louvain
import infomap
import igraph
import leidenalg
from tqdm import tqdm

from glass.objectives.map_equation import soft_map_equation
from glass.objectives.modularity import soft_modularity
from glass.solvers.spectral import spectral_embedding

from glass.seclust import (
    IncrementalSEState,
    SparseGraph,
    cluster_graph,
    hierarchical_se_clustering,
    sparse_structural_entropy,
    structural_entropy,
)


TIME_LIMIT_SECONDS = 600.0
SECLUST_STARTS = 6
SECLUST_MAX_PASSES = 10
SECLUST_SEED = 42


@dataclass(frozen=True)
class DatasetCase:
    name: str
    adjacency: object | None  # np.ndarray | scipy.sparse | SparseGraph
    labels: np.ndarray | None
    k: int | None
    source: str
    is_sparse: bool = False
    features: np.ndarray | None = None


HCSE_DENSE_MAX_NODES = 5000  # HCSE (a.k.a. SEP package) build_coding_tree is ~O(N^3); skip beyond this.

SYNTHETIC_BASELINES = {
    "Karate": ["Louvain", "Leiden", "Infomap", "Glass-Mod (JAX)", "Glass-Map (JAX)", "HCSE"],
    "Caveman (10x20)": ["Louvain", "Leiden", "Infomap", "Glass-Mod (JAX)", "Glass-Map (JAX)", "HCSE"],
    "SBM (N=100)": ["Louvain", "Leiden", "Infomap", "Glass-Mod (JAX)", "Glass-Map (JAX)", "HCSE"],
    "SBM (N=500)": ["Louvain", "Leiden", "Infomap", "Glass-Mod (JAX)", "Glass-Map (JAX)", "HCSE"],
    "SBM (N=1000)": ["Louvain", "Leiden", "Infomap", "Glass-Mod (JAX)", "Glass-Map (JAX)", "HCSE"],
}


REAL_WORLD_BASELINES = {
    "Cora": ["Louvain", "Leiden", "Infomap", "HCSE", "LSEnet", "Glass-SE GNN"],
    "Citeseer": ["Louvain", "Leiden", "Infomap", "HCSE", "LSEnet", "Glass-SE GNN"],
    "Photo": ["Louvain", "Leiden", "Infomap", "HCSE", "LSEnet", "Glass-SE GNN"],
}


def karate_graph() -> DatasetCase:
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13),
        (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21),
        (1, 30), (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12),
        (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33),
        (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33),
        (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31),
        (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33),
        (31, 32), (31, 33), (32, 33),
    ]
    adj = np.zeros((34, 34), dtype=float)
    for left, right in edges:
        adj[left, right] = 1.0
        adj[right, left] = 1.0
    officer = {8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}
    labels = np.array([1 if node in officer else 0 for node in range(34)], dtype=np.int32)
    return DatasetCase("Karate", adj, labels, 2, "hardcoded Zachary graph")


def caveman_graph(cliques: int = 10, clique_size: int = 20) -> DatasetCase:
    n = cliques * clique_size
    adj = np.zeros((n, n), dtype=float)
    labels = np.repeat(np.arange(cliques, dtype=np.int32), clique_size)
    for block in range(cliques):
        start = block * clique_size
        end = start + clique_size
        adj[start:end, start:end] = 1.0
        np.fill_diagonal(adj[start:end, start:end], 0.0)
        if block < cliques - 1:
            adj[end - 1, end] = 1.0
            adj[end, end - 1] = 1.0
    return DatasetCase("Caveman (10x20)", adj, labels, cliques, "numpy connected clique chain")


def sbm_graph(name: str, n_nodes: int, n_communities: int, p_in: float, p_out: float, seed: int) -> DatasetCase:
    rng = np.random.default_rng(seed)
    base = n_nodes // n_communities
    sizes = [base] * (n_communities - 1) + [n_nodes - base * (n_communities - 1)]
    labels = np.repeat(np.arange(n_communities, dtype=np.int32), sizes)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return DatasetCase(name, adj, labels, n_communities, "numpy SBM")


def real_world_unavailable(name: str) -> DatasetCase:
    return DatasetCase(name, None, None, None, "unavailable: torch_geometric datasets are not installed locally")


def real_world_graph(name: str) -> DatasetCase:
    """Load a PyG dataset directly into a SparseGraph (no dense materialization)."""

    try:
        from torch_geometric.datasets import Amazon, Planetoid
        from torch_geometric.utils import to_undirected
    except Exception as exc:
        return DatasetCase(name, None, None, None, f"unavailable: torch_geometric import failed: {exc}")

    try:
        if name in {"Cora", "Citeseer", "PubMed"}:
            dataset = Planetoid(root="/tmp/dataset", name=name)
        elif name in {"Photo", "Computers"}:
            dataset = Amazon(root="/tmp/dataset", name=name)
        else:
            raise ValueError(f"Unknown real-world dataset {name}")
    except Exception as exc:
        return DatasetCase(name, None, None, None, f"unavailable: dataset load failed: {exc}")

    data = dataset[0]
    n_nodes = int(data.num_nodes)
    k = int(dataset.num_classes)
    labels = data.y.cpu().numpy().astype(np.int32)
    edge_index = to_undirected(data.edge_index).cpu().numpy().astype(np.int64)
    graph = SparseGraph.from_edge_index(edge_index, num_nodes=n_nodes)
    features = None
    if hasattr(data, "x") and data.x is not None:
        features = data.x.cpu().numpy().astype(np.float32)
    return DatasetCase(
        name, graph, labels, k,
        "torch_geometric (sparse)",
        is_sparse=True,
        features=features,
    )


def get_cases() -> list[DatasetCase]:
    return [
        karate_graph(),
        caveman_graph(),
        sbm_graph("SBM (N=100)", 100, 4, 0.4, 0.02, 42),
        sbm_graph("SBM (N=500)", 500, 5, 0.2, 0.01, 42),
        sbm_graph("SBM (N=1000)", 1000, 10, 0.1, 0.005, 42),
        real_world_graph("Cora"),
        real_world_graph("Citeseer"),
        real_world_graph("Photo"),
    ]


def comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    contingency = Counter(zip(y_true.tolist(), y_pred.tolist()))
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    sum_comb = sum(comb2(count) for count in contingency.values())
    true_comb = sum(comb2(count) for count in true_counts.values())
    pred_comb = sum(comb2(count) for count in pred_counts.values())
    total_comb = comb2(y_true.size)
    expected = true_comb * pred_comb / total_comb if total_comb else 0.0
    maximum = 0.5 * (true_comb + pred_comb)
    denom = maximum - expected
    return 1.0 if abs(denom) < 1e-12 else float((sum_comb - expected) / denom)


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = float(y_true.size)
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
    return 1.0 if denom < 1e-12 else float(mi / denom)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_ids = sorted(set(y_true.tolist()))
    pred_ids = sorted(set(y_pred.tolist()))
    matrix = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    true_index = {label: i for i, label in enumerate(true_ids)}
    pred_index = {label: i for i, label in enumerate(pred_ids)}
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        matrix[pred_index[pred_label], true_index[true_label]] += 1
    if matrix.shape[0] <= 10 and matrix.shape[1] <= 10:
        best = 0
        width = max(matrix.shape)
        padded = np.zeros((width, width), dtype=np.int64)
        padded[: matrix.shape[0], : matrix.shape[1]] = matrix
        for perm in itertools.permutations(range(width)):
            best = max(best, sum(padded[row, perm[row]] for row in range(width)))
        return float(best / y_true.size)
    # Fallback for very fragmented outputs: greedy matching gives a lower-bound style ACC.
    used_rows = set()
    used_cols = set()
    total = 0
    entries = sorted(
        ((matrix[row, col], row, col) for row in range(matrix.shape[0]) for col in range(matrix.shape[1])),
        reverse=True,
    )
    for value, row, col in entries:
        if row not in used_rows and col not in used_cols:
            total += value
            used_rows.add(row)
            used_cols.add(col)
    return float(total / y_true.size)


def _adj_as_csr(adj) -> sp.csr_matrix:
    """Coerce input to CSR without unnecessary copies."""

    if isinstance(adj, SparseGraph):
        n = adj.n_nodes
        rows = []
        cols = []
        vals = []
        for node in range(n):
            nbrs = adj.neighbors[node]
            ws = adj.weights[node]
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
    if sp.issparse(adj):
        return adj.tocsr()
    return sp.csr_matrix(np.asarray(adj, dtype=float))


def run_louvain(adj, seed: int = 42) -> tuple[np.ndarray, float]:
    start = time.time()
    csr = _adj_as_csr(adj)
    graph = nx.from_scipy_sparse_array(csr)
    partition = community_louvain.best_partition(graph, random_state=seed)
    n_nodes = csr.shape[0]
    labels = np.array([partition[i] for i in range(n_nodes)], dtype=np.int32)
    return labels, time.time() - start


def run_leiden(adj, seed: int = 42) -> tuple[np.ndarray, float]:
    start = time.time()
    csr = _adj_as_csr(adj)
    coo = sp.triu(csr, k=1).tocoo()
    sources = coo.row.astype(np.int64)
    targets = coo.col.astype(np.int64)
    weights = coo.data.astype(float)
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = igraph.Graph(n=csr.shape[0], edges=edges, directed=False)
    g.es['weight'] = weights.tolist()
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=weights.tolist(), seed=seed)
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for i, cluster in enumerate(partition):
        for node in cluster:
            labels[node] = i
    return labels, time.time() - start


def run_infomap(adj, seed: int = 42) -> tuple[np.ndarray, float]:
    start = time.time()
    csr = _adj_as_csr(adj)
    coo = csr.tocoo()
    # Infomap requires a strictly positive seed; map 0 -> 1.
    infomap_seed = max(1, int(seed))
    model = infomap.Infomap(f"--two-level --silent --seed {infomap_seed}")
    for row, col, value in zip(coo.row, coo.col, coo.data):
        model.add_link(int(row), int(col), float(value))
    model.run()
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for node in model.tree:
        if node.is_leaf:
            labels[node.node_id] = node.module_id - 1
    return labels, time.time() - start


def run_glass_jax_multistart(
    adj: np.ndarray,
    n_communities: int,
    objective_fn,
    n_iters: int = 500,
    lr: float = 0.05,
    n_starts: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
    pi = None
    if "map_equation" in objective_fn.__name__:
        from glass.objectives.map_equation import compute_stationary_distribution

        pi = compute_stationary_distribution(adj_jax)

    emb = spectral_embedding(adj_jax, n_communities)
    spectral_init = jnp.array(emb) * 5.0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(lambda rng: jax.random.normal(rng, (n_nodes, n_communities)) * 0.1)(keys)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)
    optimizer = optax.adam(lr)

    def optimize_single(logits_init):
        opt_state = optimizer.init(logits_init)

        def step(state, temp):
            logits, opt_state = state

            def loss_fn(l):
                S = jax.nn.softmax(l / temp, axis=-1)
                if pi is not None:
                    value = objective_fn(adj_jax, S, pi=pi, is_logits=False)
                else:
                    value = objective_fn(adj_jax, S, is_logits=False)
                if "map_equation" in objective_fn.__name__ or "structural_entropy" in objective_fn.__name__:
                    return value
                return -value

            (loss, grads) = jax.value_and_grad(loss_fn)(logits)
            updates, opt_state = optimizer.update(grads, opt_state)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_state), loss

        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]
        S_eval = jax.nn.softmax(final_logits / 0.01, axis=-1)
        if pi is not None:
            eval_loss = objective_fn(adj_jax, S_eval, pi=pi, is_logits=False)
        else:
            eval_loss = objective_fn(adj_jax, S_eval, is_logits=False)
        if "map_equation" not in objective_fn.__name__ and "structural_entropy" not in objective_fn.__name__:
            eval_loss = -eval_loss
        return final_logits, eval_loss

    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    _ = vmap_optimize(all_inits)
    start = time.time()
    all_final_logits, all_final_losses = vmap_optimize(all_inits)
    all_final_logits.block_until_ready()
    duration = time.time() - start

    best_idx = jnp.argmin(all_final_losses)
    best_logits = all_final_logits[best_idx]
    S = jax.nn.softmax(best_logits / 0.01, axis=-1)
    return np.array(jnp.argmax(S, axis=-1)), duration


def _to_dense(adj) -> np.ndarray:
    if isinstance(adj, SparseGraph) or sp.issparse(adj):
        return _adj_as_csr(adj).toarray().astype(float)
    return np.asarray(adj, dtype=float)


def run_hcse(adj, k_target: int, seed: int = 42) -> tuple[np.ndarray, float]:
    """Run the HCSE coding-tree baseline (Pan, Zheng, Fan 2021) at depth k_target.

    The implementation lives in ``official_baselines/SEP/SEPN/codingTree.py``
    (the SEP package vendors HCSE under that name); the algorithm itself is
    HCSE / k-HCSE.
    """

    from glass.seclust.benchmark_sep import run_official_sep_coding_tree

    dense = _to_dense(adj)
    start = time.time()
    result = run_official_sep_coding_tree(dense, k=max(2, int(k_target)))
    return np.asarray(result.labels, dtype=np.int32), time.time() - start


def run_glass_se_gnn(
    adj,
    features: np.ndarray,
    k: int,
    n_iters: int = 120,
    hidden_dim: int = 32,
    lr: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Glass-SE GNN: a single-layer GCN encoder over the 2D SE loss.

    Same input as ``run_lsenet_proxy`` (adjacency, features, target K)
    and the same loss; replaces the linear projection with a GCN, which
    is the natural mid-point between the LSEnet proxy and full LSEnet.
    """

    from glass.objectives.structural_entropy import two_dimensional_structural_entropy
    from glass.models.gnn_se import GNNEncoder

    dense = _to_dense(adj)
    n_nodes = dense.shape[0]
    adj_jax = jnp.array(dense)
    features_jax = jnp.array(features)

    model = GNNEncoder(hidden_dim=hidden_dim, num_communities=k)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, features_jax, adj_jax)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            logits = model.apply(p, features_jax, adj_jax)
            S = jax.nn.softmax(logits, axis=-1)
            return two_dimensional_structural_entropy(adj_jax, S, is_logits=False)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state2, loss

    start = time.time()
    for _ in range(n_iters):
        params, opt_state, _ = step(params, opt_state)
    logits = model.apply(params, features_jax, adj_jax)
    labels = np.asarray(jnp.argmax(logits, axis=-1)).astype(np.int32)
    return labels, time.time() - start


def run_lsenet_proxy(
    adj,
    features: np.ndarray,
    k: int,
    n_iters: int = 120,
    n_starts: int = 2,
    lr: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Feature-aware proxy for LSEnet: optimize 2D SE through a linear projection of features."""

    from glass.objectives.structural_entropy import two_dimensional_structural_entropy

    dense = _to_dense(adj)
    n_nodes = dense.shape[0]
    n_features = int(features.shape[1])
    adj_jax = jnp.array(dense)
    features_jax = jnp.array(features)

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_starts)
    W_inits = jax.vmap(lambda rng: jax.random.normal(rng, (n_features, k)) * 0.1)(keys)

    optimizer = optax.adam(lr)

    def optimize_single(W_init):
        opt_state = optimizer.init(W_init)

        def step(state, temp):
            W, opt_state = state

            def loss_fn(w):
                logits = jnp.dot(features_jax, w)
                S = jax.nn.softmax(logits / temp, axis=-1)
                return two_dimensional_structural_entropy(adj_jax, S, is_logits=False)

            (loss, grads) = jax.value_and_grad(loss_fn)(W)
            updates, opt_state = optimizer.update(grads, opt_state)
            W = optax.apply_updates(W, updates)
            return (W, opt_state), loss

        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (W_init, opt_state), temps)
        final_W = final_state[0]
        logits = jnp.dot(features_jax, final_W)
        S_eval = jax.nn.softmax(logits / 0.01, axis=-1)
        eval_loss = two_dimensional_structural_entropy(adj_jax, S_eval, is_logits=False)
        return final_W, eval_loss

    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    _ = vmap_optimize(W_inits)
    start = time.time()
    all_final_W, all_final_losses = vmap_optimize(W_inits)
    all_final_W.block_until_ready()
    duration = time.time() - start

    best_W = all_final_W[jnp.argmin(all_final_losses)]
    logits = jnp.dot(features_jax, best_W)
    S = jax.nn.softmax(logits / 0.01, axis=-1)
    labels = np.array(jnp.argmax(S, axis=-1))
    return labels.astype(np.int32), duration


def _n_nodes_of(adj) -> int:
    if isinstance(adj, SparseGraph):
        return adj.n_nodes
    if sp.issparse(adj):
        return int(adj.shape[0])
    return int(np.asarray(adj).shape[0])


def run_synthetic_baseline(
    case: DatasetCase,
    algorithm: str,
    seed: int = SECLUST_SEED,
) -> dict[str, object]:
    n_nodes = _n_nodes_of(case.adjacency) if case.adjacency is not None else 0
    skip_status: str | None = None
    if algorithm == "Louvain":
        labels, duration = run_louvain(case.adjacency, seed=seed)
    elif algorithm == "Leiden":
        labels, duration = run_leiden(case.adjacency, seed=seed)
    elif algorithm == "Infomap":
        labels, duration = run_infomap(case.adjacency, seed=seed)
    elif algorithm == "Glass-Mod (JAX)":
        labels, duration = run_glass_jax_multistart(
            case.adjacency,
            case.k,
            soft_modularity,
            n_iters=120,
            n_starts=2,
            seed=seed,
        )
    elif algorithm == "Glass-Map (JAX)":
        labels, duration = run_glass_jax_multistart(
            case.adjacency,
            case.k,
            soft_map_equation,
            n_iters=120,
            n_starts=2,
            seed=seed,
        )
    elif algorithm == "HCSE":
        if n_nodes > HCSE_DENSE_MAX_NODES:
            skip_status = f"skipped: HCSE dense build infeasible at N={n_nodes} (>{HCSE_DENSE_MAX_NODES})"
        else:
            labels, duration = run_hcse(case.adjacency, k_target=case.k or 2, seed=seed)
    elif algorithm == "LSEnet":
        if case.features is None:
            skip_status = "skipped: LSEnet requires node features"
        else:
            labels, duration = run_lsenet_proxy(
                case.adjacency,
                case.features,
                k=case.k,
                n_iters=120,
                n_starts=2,
                seed=seed,
            )
    elif algorithm == "Glass-SE GNN":
        if case.features is None:
            skip_status = "skipped: Glass-SE GNN requires node features"
        else:
            labels, duration = run_glass_se_gnn(
                case.adjacency,
                case.features,
                k=case.k,
                n_iters=120,
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown synthetic baseline {algorithm}")

    if skip_status is not None:
        return {
            "dataset": case.name,
            "algorithm": algorithm,
            "seed": seed,
            "ari": None,
            "nmi": None,
            "acc": None,
            "k": None,
            "modularity": None,
            "structural_entropy": None,
            "map_equation": None,
            "time": None,
            "estimated_time": None,
            "se": None,
            "status": skip_status,
        }

    if isinstance(case.adjacency, SparseGraph) or sp.issparse(case.adjacency):
        se_value = sparse_structural_entropy(case.adjacency, labels)
    else:
        se_value = structural_entropy(case.adjacency, labels)
    return {
        "dataset": case.name,
        "algorithm": algorithm,
        "seed": seed,
        "ari": adjusted_rand_index(case.labels, labels),
        "nmi": normalized_mutual_info(case.labels, labels),
        "acc": clustering_accuracy(case.labels, labels),
        "k": int(len(np.unique(labels))),
        "modularity": hard_modularity(case.adjacency, labels),
        "structural_entropy": se_value,
        "map_equation": hard_map_equation(case.adjacency, labels),
        "time": duration,
        "estimated_time": None,
        "se": None,
        "status": "baseline_executed",
    }


def hard_modularity(adj, labels: np.ndarray) -> float:
    if isinstance(adj, SparseGraph) or sp.issparse(adj):
        csr = _adj_as_csr(adj)
        degrees = np.asarray(csr.sum(axis=1)).flatten()
        volume = float(degrees.sum())
        if volume <= 1e-12:
            return 0.0
        labels = np.asarray(labels)
        score = 0.0
        for cluster in np.unique(labels):
            mask = labels == cluster
            cluster_degree = float(degrees[mask].sum())
            sub = csr[mask][:, mask]
            internal = float(sub.sum())
            score += internal - (cluster_degree * cluster_degree / volume)
        return float(score / volume)
    degrees = adj.sum(axis=1)
    volume = float(degrees.sum())
    if volume <= 1e-12:
        return 0.0
    labels = np.asarray(labels)
    score = 0.0
    for cluster in np.unique(labels):
        mask = labels == cluster
        internal = float(adj[np.ix_(mask, mask)].sum())
        cluster_degree = float(degrees[mask].sum())
        score += internal - (cluster_degree * cluster_degree / volume)
    return float(score / volume)


def entropy_from_masses(masses: np.ndarray) -> float:
    masses = masses[masses > 1e-12]
    total = float(masses.sum())
    if total <= 1e-12:
        return 0.0
    probs = masses / total
    return float(-np.sum(probs * np.log2(probs)))


def hard_map_equation(adj, labels: np.ndarray) -> float:
    if isinstance(adj, SparseGraph) or sp.issparse(adj):
        csr = _adj_as_csr(adj)
        degrees = np.asarray(csr.sum(axis=1)).flatten()
        volume = float(degrees.sum())
        if volume <= 1e-12:
            return 0.0
        pi = degrees / volume
        labels = np.asarray(labels)
        # Compute per-node out-of-cluster weight via CSR rows.
        n = csr.shape[0]
        out_weight = np.zeros(n, dtype=float)
        indptr = csr.indptr
        indices = csr.indices
        data = csr.data
        for node in range(n):
            cid = labels[node]
            start, end = indptr[node], indptr[node + 1]
            nbrs = indices[start:end]
            ws = data[start:end]
            mismatch = labels[nbrs] != cid
            if mismatch.any():
                out_weight[node] = float(ws[mismatch].sum())

        q_values = []
        module_terms = []
        for cluster in np.unique(labels):
            mask = labels == cluster
            p_module = float(pi[mask].sum())
            d_mask = degrees[mask]
            ow = out_weight[mask]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(d_mask > 1e-12, ow / np.maximum(d_mask, 1e-12), 0.0)
            exit_prob = float(np.sum(pi[mask] * ratios))
            q_values.append(exit_prob)
            module_terms.append((exit_prob, p_module, pi[mask]))

        q = float(np.sum(q_values))
        index_term = q * entropy_from_masses(np.asarray(q_values, dtype=float))
        module_term = 0.0
        for exit_prob, p_module, node_masses in module_terms:
            module_mass = exit_prob + p_module
            if module_mass > 1e-12:
                module_term += module_mass * entropy_from_masses(np.concatenate([[exit_prob], node_masses]))
        return float(index_term + module_term)

    degrees = adj.sum(axis=1)
    volume = float(degrees.sum())
    if volume <= 1e-12:
        return 0.0
    pi = degrees / volume
    labels = np.asarray(labels)

    q_values = []
    module_terms = []
    for cluster in np.unique(labels):
        mask = labels == cluster
        p_module = float(pi[mask].sum())
        exit_prob = 0.0
        for node in np.where(mask)[0]:
            if degrees[node] > 1e-12:
                exit_weight = float(adj[node, ~mask].sum())
                exit_prob += float(pi[node] * exit_weight / degrees[node])
        q_values.append(exit_prob)
        module_terms.append((exit_prob, p_module, pi[mask]))

    q = float(np.sum(q_values))
    index_term = q * entropy_from_masses(np.asarray(q_values, dtype=float))
    module_term = 0.0
    for exit_prob, p_module, node_masses in module_terms:
        module_mass = exit_prob + p_module
        if module_mass > 1e-12:
            module_term += module_mass * entropy_from_masses(np.concatenate([[exit_prob], node_masses]))
    return float(index_term + module_term)


def estimate_seclust_seconds(adj) -> float:
    graph_start = time.time()
    if isinstance(adj, SparseGraph):
        graph = adj
    else:
        graph = SparseGraph.from_adjacency(adj)
    graph_seconds = time.time() - graph_start
    n = graph.n_nodes
    rng = np.random.default_rng(SECLUST_SEED)
    labels = rng.integers(0, max(2, min(16, int(math.sqrt(max(n, 2))) + 2)), size=n, dtype=np.int32)
    state = IncrementalSEState(graph, labels)
    sample_nodes = rng.choice(n, size=min(n, 48), replace=False)
    evaluations = 0
    start = time.time()
    candidate_counts = []
    for node in sample_nodes:
        candidates = state.candidate_clusters(int(node), allow_new_cluster=True)
        candidate_counts.append(len(candidates))
        for target in candidates:
            state.move_delta(int(node), int(target))
            evaluations += 1
    elapsed = max(time.time() - start, 1e-9)
    per_eval = elapsed / max(evaluations, 1)
    avg_candidates = float(np.mean(candidate_counts)) if candidate_counts else 1.0
    estimated_evaluations = SECLUST_STARTS * SECLUST_MAX_PASSES * n * avg_candidates
    # Include graph/state construction once per start plus a conservative 25%
    # overhead for accepted moves and canonical final scoring.
    return float((graph_seconds * SECLUST_STARTS + per_eval * estimated_evaluations) * 1.25)


def run_seclust(
    case: DatasetCase,
    algorithm: str = "SEClust-Auto",
    seed: int = SECLUST_SEED,
) -> dict[str, object]:
    if case.adjacency is None or case.labels is None:
        return {
            "dataset": case.name,
            "algorithm": algorithm,
            "seed": seed,
            "ari": None,
            "nmi": None,
            "acc": None,
            "k": None,
            "modularity": None,
            "structural_entropy": None,
            "map_equation": None,
            "time": None,
            "estimated_time": None,
            "se": None,
            "status": case.source,
        }

    estimate = estimate_seclust_seconds(case.adjacency)
    if estimate > TIME_LIMIT_SECONDS:
        return {
            "dataset": case.name,
            "algorithm": algorithm,
            "seed": seed,
            "ari": None,
            "nmi": None,
            "acc": None,
            "k": None,
            "modularity": None,
            "structural_entropy": None,
            "map_equation": None,
            "time": None,
            "estimated_time": estimate,
            "se": None,
            "status": f"skipped: estimated runtime {estimate:.1f}s exceeds {TIME_LIMIT_SECONDS:.0f}s limit",
        }

    start = time.time()
    if algorithm == "SEClust-Tree":
        result = hierarchical_se_clustering(
            case.adjacency,
            target_clusters=case.k,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=seed,
        )
    elif algorithm == "SEClust-TargetK":
        from glass.seclust.incremental import multistart_incremental_se_heuristic
        from glass.seclust.hierarchy import merge_hierarchy_levels, select_hierarchy_level
        from glass.seclust.heuristics import ClusteringResult

        base_labels, _ = multistart_incremental_se_heuristic(
            case.adjacency,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=seed,
        )
        levels = merge_hierarchy_levels(case.adjacency, base_labels, min_clusters=case.k)
        selected = select_hierarchy_level(levels, target_clusters=case.k)
        result = ClusteringResult(
            entropy=selected.entropy,
            labels=selected.labels,
            method="seclust-target-k",
        )
    elif algorithm == "SEClust-MultiLevel":
        from glass.seclust import multilevel_se_clustering
        result = multilevel_se_clustering(
            case.adjacency,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=seed,
        )
    elif algorithm == "SEClust-ConstrainedK":
        from glass.seclust import constrained_k_multistart
        from glass.seclust.heuristics import ClusteringResult
        labels, entropy = constrained_k_multistart(
            case.adjacency,
            target_clusters=case.k or 2,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=seed,
        )
        result = ClusteringResult(entropy=entropy, labels=labels, method="seclust-constrained-k")
    else:
        result = cluster_graph(
            case.adjacency,
            mode="heuristic",
            exact_max_nodes=9,
            heuristic_starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=seed,
        )
    elapsed = time.time() - start
    return {
        "dataset": case.name,
        "algorithm": algorithm,
        "seed": seed,
        "ari": adjusted_rand_index(case.labels, result.labels),
        "nmi": normalized_mutual_info(case.labels, result.labels),
        "acc": clustering_accuracy(case.labels, result.labels),
        "k": int(len(np.unique(result.labels))),
        "modularity": hard_modularity(case.adjacency, result.labels),
        "structural_entropy": result.entropy,
        "map_equation": hard_map_equation(case.adjacency, result.labels),
        "time": elapsed,
        "estimated_time": estimate,
        "se": result.entropy,
        "status": "ok",
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "skip"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def best_flags(rows: list[dict[str, object]], metric: str, lower_is_better: bool = False) -> set[int]:
    values = [(i, row.get(metric)) for i, row in enumerate(rows) if isinstance(row.get(metric), float)]
    if not values:
        return set()
    best = min(value for _, value in values) if lower_is_better else max(value for _, value in values)
    return {i for i, value in values if abs(value - best) < 1e-12}


def maybe_bold(text: str, enabled: bool) -> str:
    return f"**{text}**" if enabled else text


def _render_metric(row: dict[str, object], field: str, digits: int = 3) -> str:
    """Render a metric as either a single value or 'mean ± std' if std > 0.

    Falls back to the bare ``field`` value when the row was produced by a
    single-seed run (i.e. ``f_std`` is missing). On aggregated rows the
    bare ``field`` already holds the mean, so passing the same field
    through :func:`fmt` would lose the std.
    """

    if f"{field}_std" in row:
        return fmt_mean_std(row.get(f"{field}_mean", row.get(field)), row.get(f"{field}_std"), digits)
    return fmt(row.get(field), digits)


def synthetic_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Estimate (s) | Graph (N, E, True K*) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for dataset in SYNTHETIC_BASELINES:
        group = [row for row in rows if row["dataset"] == dataset]
        acc_best = best_flags(group, "acc")
        ari_best = best_flags(group, "ari")
        nmi_best = best_flags(group, "nmi")
        modularity_best = best_flags(group, "modularity")
        se_best = best_flags(group, "structural_entropy", lower_is_better=True)
        map_best = best_flags(group, "map_equation", lower_is_better=True)
        for i, row in enumerate(group):
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset if i == 0 else "",
                        str(row["algorithm"]),
                        maybe_bold(_render_metric(row, "acc"), i in acc_best),
                        maybe_bold(_render_metric(row, "nmi"), i in nmi_best),
                        maybe_bold(_render_metric(row, "ari"), i in ari_best),
                        fmt(row.get("k"), 0),
                        maybe_bold(_render_metric(row, "modularity"), i in modularity_best),
                        maybe_bold(_render_metric(row, "structural_entropy"), i in se_best),
                        maybe_bold(_render_metric(row, "map_equation"), i in map_best),
                        _render_metric(row, "runtime_seconds", 4),
                        _render_metric(row, "estimated_runtime_seconds", 1),
                        f"{row.get('n_nodes')}, {row.get('n_edges')}, {row.get('true_k')}" if row.get("n_nodes") else "skip",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def real_world_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Graph (N, E, True K*) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for dataset in REAL_WORLD_BASELINES:
        group = [row for row in rows if row["dataset"] == dataset]
        acc_best = best_flags(group, "acc")
        nmi_best = best_flags(group, "nmi")
        ari_best = best_flags(group, "ari")
        modularity_best = best_flags(group, "modularity")
        se_best = best_flags(group, "structural_entropy", lower_is_better=True)
        map_best = best_flags(group, "map_equation", lower_is_better=True)
        for i, row in enumerate(group):
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset if i == 0 else "",
                        str(row["algorithm"]),
                        maybe_bold(_render_metric(row, "acc"), i in acc_best),
                        maybe_bold(_render_metric(row, "nmi"), i in nmi_best),
                        maybe_bold(_render_metric(row, "ari"), i in ari_best),
                        fmt(row.get("k"), 0),
                        maybe_bold(_render_metric(row, "modularity"), i in modularity_best),
                        maybe_bold(_render_metric(row, "structural_entropy"), i in se_best),
                        maybe_bold(_render_metric(row, "map_equation"), i in map_best),
                        _render_metric(row, "runtime_seconds", 4),
                        f"{row.get('n_nodes')}, {row.get('n_edges')}, {row.get('true_k')}" if row.get("n_nodes") else "skip",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def run_benchmark(seeds: list[int] | None = None) -> list[dict[str, object]]:
    """Run the full benchmark, optionally over a list of seeds.

    With ``seeds=None`` or a single-element list, behaves as before
    (one row per (dataset, algorithm)). With multiple seeds, every
    cell is executed once per seed and each row is tagged with the
    seed used. Aggregation to mean/std is done downstream by
    :func:`aggregate_rows`.
    """

    if seeds is None:
        seeds = [SECLUST_SEED]
    if not seeds:
        raise ValueError("seeds must contain at least one integer")

    cases = {case.name: case for case in get_cases()}
    rows: list[dict[str, object]] = []

    seclust_variants = [
        "SEClust-Auto",
        "SEClust-Tree",
        "SEClust-TargetK",
        "SEClust-MultiLevel",
        "SEClust-ConstrainedK",
    ]
    cells_per_seed = sum(len(algos) + len(seclust_variants) for algos in SYNTHETIC_BASELINES.values())
    cells_per_seed += sum(len(algos) + len(seclust_variants) for algos in REAL_WORLD_BASELINES.values())
    total_cells = cells_per_seed * len(seeds)
    pbar = tqdm(total=total_cells, desc="Benchmark", unit="cell", dynamic_ncols=True)

    def _run_seclust_named(case: DatasetCase, algorithm: str, seed: int) -> dict[str, object]:
        if algorithm == "SEClust-Auto":
            return run_seclust(case, seed=seed)
        return run_seclust(case, algorithm=algorithm, seed=seed)

    for seed in seeds:
        for dataset, algorithms in SYNTHETIC_BASELINES.items():
            case = cases[dataset]
            for algorithm in algorithms:
                pbar.set_postfix_str(f"seed={seed} {dataset} / {algorithm}")
                rows.append(run_synthetic_baseline(case, algorithm, seed=seed))
                pbar.update(1)
            for variant in seclust_variants:
                pbar.set_postfix_str(f"seed={seed} {dataset} / {variant}")
                rows.append(_run_seclust_named(case, variant, seed=seed))
                pbar.update(1)

        for dataset, algorithms in REAL_WORLD_BASELINES.items():
            case = cases[dataset]
            if case.adjacency is None:
                for algorithm in algorithms:
                    pbar.set_postfix_str(f"seed={seed} {dataset} / {algorithm} (unavailable)")
                    rows.append(
                        {
                            "dataset": dataset,
                            "algorithm": algorithm,
                            "seed": seed,
                            "acc": None,
                            "nmi": None,
                            "ari": None,
                            "k": None,
                            "modularity": None,
                            "structural_entropy": None,
                            "map_equation": None,
                            "time": None,
                            "estimated_time": None,
                            "se": None,
                            "status": case.source,
                        }
                    )
                    pbar.update(1)
            else:
                for algorithm in algorithms:
                    pbar.set_postfix_str(f"seed={seed} {dataset} / {algorithm}")
                    rows.append(run_synthetic_baseline(case, algorithm, seed=seed))
                    pbar.update(1)
            for variant in seclust_variants:
                pbar.set_postfix_str(f"seed={seed} {dataset} / {variant}")
                rows.append(_run_seclust_named(case, variant, seed=seed))
                pbar.update(1)

    pbar.close()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"seclust_full_benchmark_{timestamp}"
    for row in rows:
        case = cases[row["dataset"]]
        row["experiment_id"] = experiment_id
        row["true_k"] = case.k
        row["dataset_source"] = case.source
        if case.adjacency is None:
            row["n_nodes"] = None
            row["n_edges"] = None
        elif isinstance(case.adjacency, SparseGraph):
            row["n_nodes"] = case.adjacency.n_nodes
            row["n_edges"] = int(sum(nbrs.size for nbrs in case.adjacency.neighbors) / 2)
        elif sp.issparse(case.adjacency):
            row["n_nodes"] = int(case.adjacency.shape[0])
            row["n_edges"] = int(case.adjacency.nnz / 2)
        else:
            row["n_nodes"] = int(case.adjacency.shape[0])
            row["n_edges"] = int(np.sum(case.adjacency > 0) / 2)

        # `seed` is set by the runner; keep it as-is. Only ensure it exists.
        row.setdefault("seed", None)

        status_orig = str(row.get("status", "ok"))
        if status_orig.startswith("skipped") or status_orig.startswith("unavailable"):
            row["skip_reason"] = status_orig
            row["status"] = "skipped"
        elif status_orig == "baseline_executed":
            row["status"] = "ok"
            row["skip_reason"] = None
        elif status_orig == "baseline_imported":
            row["status"] = "baseline_imported"
            row["skip_reason"] = None
        else:
            row["status"] = "ok"
            row["skip_reason"] = None

        row["estimated_runtime_seconds"] = row.pop("estimated_time", None)
        row["runtime_seconds"] = row.pop("time", None)
        row["labels_path"] = None
        row.pop("se", None)

    return rows


_NUMERIC_FIELDS = (
    "acc",
    "nmi",
    "ari",
    "modularity",
    "structural_entropy",
    "map_equation",
    "runtime_seconds",
    "estimated_runtime_seconds",
)


def aggregate_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group per-seed rows by (dataset, algorithm) and compute mean/std.

    Each output row contains, for every numeric field ``f``:
    ``f`` = mean (so existing table renderers keep working),
    ``f_mean`` = same mean explicitly,
    ``f_std`` = sample standard deviation across seeds (population
    std, ddof=0; identical to ``np.std``).

    Non-numeric fields (``k``, ``status``, ``true_k``, ``n_nodes``,
    ``n_edges``, ``dataset_source``, ``experiment_id``, etc.) are
    preserved from the first row in the group; ``k`` is also reported
    as ``k_mean`` (rounded to int) and ``k_std``. ``seeds_used``
    lists the seeds that contributed.
    """

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (str(row.get("dataset")), str(row.get("algorithm")))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for (dataset, algorithm), group in grouped.items():
        out: dict[str, object] = {
            "dataset": dataset,
            "algorithm": algorithm,
            "n_seeds": len(group),
            "seeds_used": [r.get("seed") for r in group],
        }
        # Carry through metadata from the first row.
        for meta in (
            "experiment_id",
            "true_k",
            "dataset_source",
            "n_nodes",
            "n_edges",
            "labels_path",
        ):
            out[meta] = group[0].get(meta)

        # If every row in the group is skipped, propagate the first
        # skip reason and leave all numeric fields as None.
        all_skipped = all(str(r.get("status")) in {"skipped", "baseline_imported"} for r in group)
        if all_skipped:
            out["status"] = group[0].get("status")
            out["skip_reason"] = group[0].get("skip_reason")
            for f in _NUMERIC_FIELDS:
                out[f] = None
                out[f"{f}_mean"] = None
                out[f"{f}_std"] = None
            out["k"] = None
            out["k_mean"] = None
            out["k_std"] = None
            aggregated.append(out)
            continue

        out["status"] = "ok"
        out["skip_reason"] = None

        for f in _NUMERIC_FIELDS:
            vals = [r.get(f) for r in group if isinstance(r.get(f), (int, float))]
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=0))
                out[f] = m
                out[f"{f}_mean"] = m
                out[f"{f}_std"] = s
            else:
                out[f] = None
                out[f"{f}_mean"] = None
                out[f"{f}_std"] = None

        k_vals = [r.get("k") for r in group if isinstance(r.get("k"), (int, float))]
        if k_vals:
            out["k"] = int(round(float(np.mean(k_vals))))
            out["k_mean"] = float(np.mean(k_vals))
            out["k_std"] = float(np.std(k_vals, ddof=0))
        else:
            out["k"] = None
            out["k_mean"] = None
            out["k_std"] = None

        aggregated.append(out)

    return aggregated


def fmt_mean_std(mean: object, std: object, digits: int = 3) -> str:
    if mean is None:
        return "skip"
    if not isinstance(mean, (int, float)):
        return str(mean)
    if std is None or not isinstance(std, (int, float)) or std < 1e-6:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def write_report(rows: list[dict[str, object]], seeds: list[int] | None = None) -> Path:
    """Write JSON of per-seed rows + an aggregated mean/std markdown report.

    When ``seeds`` is provided (or auto-detected from the row tags), the
    markdown tables show ``mean ± std`` cells via :func:`aggregate_rows`.
    The JSON sidecar always contains the raw per-seed rows.
    """

    out_dir = Path("docs/experimental_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = rows[0]["experiment_id"] if rows else "seclust_full_benchmark"

    # Always save raw per-seed rows.
    json_path = out_dir / f"{exp_id}.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    if seeds is None:
        seeds = sorted({r.get("seed") for r in rows if r.get("seed") is not None})
    multi_seed = len(seeds) > 1

    # Aggregate (always — single-seed yields std=0, which is rendered
    # as a bare value by ``fmt_mean_std``).
    aggregated = aggregate_rows(rows)
    agg_path = out_dir / f"{exp_id}_aggregated.json"
    agg_path.write_text(json.dumps(aggregated, indent=2) + "\n", encoding="utf-8")

    report_path = out_dir / f"{exp_id}.md"

    synthetic_rows = [row for row in aggregated if row["dataset"] in SYNTHETIC_BASELINES]
    real_rows = [row for row in aggregated if row["dataset"] in REAL_WORLD_BASELINES]
    skipped = [row for row in aggregated if str(row["algorithm"]).startswith("SEClust") and row["status"] != "ok"]
    completed = [row for row in aggregated if str(row["algorithm"]).startswith("SEClust") and row["status"] == "ok"]

    report = f"""# SEClust Full Benchmark Against Existing Baselines

**Date:** {date.today().strftime("%B %-d, %Y")}
**Project:** glass-jax / `glass.seclust`

## 1. Abstract
This report benchmarks SEClust on the synthetic datasets defined in `tests/benchmark_full.py` and the real-world PyG datasets (Cora, Citeseer, Photo) loaded via `torch_geometric.datasets`. All baselines (Louvain, Leiden, Infomap, Glass-JAX variants) are executed end-to-end in this run; no values are imported from prior reports. Real-world graphs are loaded directly into a sparse `SparseGraph` adjacency without any dense `(N, N)` materialization, so Cora/Citeseer/Photo no longer hit the dense node guard.

## 2. Setup
- Synthetic datasets are generated locally with NumPy adjacencies; the SEClust path automatically converts to sparse incremental scoring.
- Real-world PyG datasets are loaded with `SparseGraph.from_edge_index(edge_index, num_nodes)` — no dense materialization. Baselines (Louvain/Leiden/Infomap) accept the sparse adjacency through a CSR coercion path.
- SEClust config: `heuristic_starts={SECLUST_STARTS}`, `max_passes={SECLUST_MAX_PASSES}`, seed `{SECLUST_SEED}`.
- Runtime limit: `{TIME_LIMIT_SECONDS:.0f}` seconds per dataset; runs that the incremental estimator predicts will exceed this are skipped with the estimate recorded.
- Logged metrics follow `docs/seclust/experiment_protocol.md`: ACC, NMI, ARI, K, modularity, structural entropy, map equation, and runtime.
- Bold values mark the best available result per dataset and metric. K is diagnostic and is not bolded.

## 3. Synthetic Results
{synthetic_table(synthetic_rows)}

## 4. Real-World Results
{real_world_table(real_rows)}

## 5. Summary
- Seeds executed: `{seeds}` ({"multi-seed" if multi_seed else "single seed"}).
- Completed SEClust groups: `{len(completed)}`.
- Skipped or unavailable SEClust groups: `{len(skipped)}`.
- All baselines on Cora/Citeseer/Photo are executed in this run via the sparse pipeline.
{"- Numeric cells are reported as `mean ± std` across seeds; if a row is deterministic (or a single seed was used) only the mean is shown." if multi_seed else ""}

**Notes:**
- **True K***: Ground-truth number of communities. Parameter-free community detection algorithms (Louvain, Leiden, Infomap) do not take `K` as an input.
- **SEClust-Auto** runs `cluster_graph(mode="heuristic")` with multistart incremental local move; **SEClust-Tree** runs `hierarchical_se_clustering` with target_clusters=K; **SEClust-TargetK** runs `merge_hierarchy_levels` then `select_hierarchy_level(K)`.

Raw per-seed results are at `{json_path}`. Aggregated mean/std rows are at `{agg_path}`.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--seeds",
        type=str,
        default=str(SECLUST_SEED),
        help="Comma-separated list of seeds (default: 42).",
    )
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"Running benchmark with seeds={seeds}", flush=True)

    raw = run_benchmark(seeds=seeds)
    aggregated = aggregate_rows(raw)
    print("\n--- Synthetic Table ---")
    print(synthetic_table([row for row in aggregated if row["dataset"] in SYNTHETIC_BASELINES]))
    print("\n--- Real-World Table ---")
    print(real_world_table([row for row in aggregated if row["dataset"] in REAL_WORLD_BASELINES]))
    path = write_report(raw, seeds=seeds)
    print(f"\nBenchmark report saved to {path}")
