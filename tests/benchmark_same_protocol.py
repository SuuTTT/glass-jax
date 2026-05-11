"""Same-protocol benchmark harness for SEClust journal experiments.

This script is the executable version of the TPAMI/TKDE/TNNLS empirical
protocol: every method sees the same graph, node features, target-K when
required, seed list, and metric suite. It writes seed-level raw results,
aggregated JSON, and a Markdown report under ``docs/experimental_reports``.

Example smoke run:

    python tests/benchmark_same_protocol.py \
        --block topology \
        --datasets Karate \
        --methods Louvain,Spectral,LabelPropagation,SEClust-ConstrainedK \
        --seeds 0

Full runs are intentionally expensive and may download PyG/OGB datasets.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import signal
import time
from typing import Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp

from benchmark_seclust_full import (
    DatasetCase,
    adjusted_rand_index,
    clustering_accuracy,
    hard_map_equation,
    hard_modularity,
    karate_graph,
    normalized_mutual_info,
    run_glass_se_gnn,
    run_hcse,
    run_infomap,
    run_leiden,
    run_louvain,
    run_lsenet_proxy,
    sparse_structural_entropy,
    structural_entropy,
    _adj_as_csr,
)

from glass.seclust import SparseGraph


TOPOLOGY_DATASETS = [
    "Karate",
    "SBM-Easy",
    "SBM-Noisy",
    "DCSBM",
    "LFR",
    "Cora",
    "Citeseer",
    "PubMed",
    "Photo",
    "Computers",
    "Coauthor-CS",
    "Coauthor-Physics",
    "ogbn-arxiv",
]

ATTRIBUTED_DATASETS = [
    "Cora",
    "Citeseer",
    "PubMed",
    "Photo",
    "Computers",
    "Coauthor-CS",
    "Coauthor-Physics",
    "ogbn-arxiv",
]

HIERARCHY_DATASETS = ["Cora", "Citeseer", "Photo", "Computers", "Coauthor-CS"]
SCALE_DATASETS = ["SBM-10k", "SBM-50k", "ogbn-arxiv"]

TOPOLOGY_METHODS = [
    "Louvain",
    "Leiden",
    "Infomap",
    "Spectral",
    "LabelPropagation",
    "HCSE",
    "SEClust-Auto",
    "SEClust-Tree",
    "SEClust-ConstrainedK",
    "SEClust-MultiLevel",
]

ATTRIBUTED_METHODS = [
    "RawKMeans",
    "PCAKMeans",
    "AdjSVDKMeans",
    "DeepWalkKMeans",
    "Node2VecKMeans",
    "GAE",
    "VGAE",
    "DMoN",
    "MinCutPool",
    "LSEnet-Proxy",
    "Glass-SE-GNN",
    "SEClust-ConstrainedK",
]

HIERARCHY_METHODS = ["SEClust-Tree", "HCSE", "Louvain-Dendrogram", "Paris", "Agglomerative"]
SCALE_METHODS = ["Louvain", "Leiden", "SEClust-Auto", "SEClust-ConstrainedK"]

HCSE_DENSE_MAX_NODES = 5_000
DENSE_GNN_MAX_NODES = 8_000
SPECTRAL_DENSE_MAX_NODES = 10_000
NEURAL_DEFAULT_EPOCHS = 200
LFR_ATTEMPT_TIMEOUT_SECONDS = 5


class SkipMethod(RuntimeError):
    """Raised for expected protocol skips such as optional dependencies or size caps."""


@contextmanager
def bounded_lfr_attempt(seconds: int):
    """Bound NetworkX LFR retries so bad seeds cannot hang full benchmark runs."""

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"LFR generation exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class RunConfig:
    block: str
    seeds: list[int]
    datasets: list[str]
    methods: list[str]
    output_dir: Path
    quick: bool
    max_nodes: int | None
    time_limit: float | None
    neural_epochs: int
    seclust_starts: int
    seclust_passes: int
    dry_run: bool


def canonical_dataset_name(name: str) -> str:
    aliases = {
        "CiteSeer": "Citeseer",
        "CoauthorCS": "Coauthor-CS",
        "CoauthorPhysics": "Coauthor-Physics",
        "OGBN-Arxiv": "ogbn-arxiv",
        "Ogbn-Arxiv": "ogbn-arxiv",
    }
    return aliases.get(name, name)


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip().lower() in {"", "default"}:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def n_nodes(case: DatasetCase) -> int:
    return int(_adj_as_csr(case.adjacency).shape[0]) if case.adjacency is not None else 0


def n_edges(case: DatasetCase) -> int:
    return int(_adj_as_csr(case.adjacency).nnz // 2) if case.adjacency is not None else 0


def csr_to_edge_index(csr: sp.csr_matrix) -> np.ndarray:
    coo = csr.tocoo()
    return np.vstack([coo.row.astype(np.int64), coo.col.astype(np.int64)])


def make_sbm(
    name: str,
    n: int,
    k: int,
    p_in: float,
    p_out: float,
    seed: int,
) -> DatasetCase:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(k, dtype=np.int32), math.ceil(n / k))[:n]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for block_i in range(k):
        idx_i = np.flatnonzero(labels == block_i)
        for block_j in range(block_i, k):
            idx_j = np.flatnonzero(labels == block_j)
            p = p_in if block_i == block_j else p_out
            if block_i == block_j:
                tri = rng.random((idx_i.size, idx_i.size)) < p
                tri = np.triu(tri, k=1)
                rr, cc = np.where(tri)
                src = idx_i[rr]
                dst = idx_i[cc]
            else:
                mat = rng.random((idx_i.size, idx_j.size)) < p
                rr, cc = np.where(mat)
                src = idx_i[rr]
                dst = idx_j[cc]
            if src.size:
                rows.extend([src, dst])
                cols.extend([dst, src])
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
    else:
        row = np.zeros(0, dtype=np.int64)
        col = np.zeros(0, dtype=np.int64)
    csr = sp.coo_matrix((np.ones(row.size), (row, col)), shape=(n, n)).tocsr()
    return DatasetCase(name, SparseGraph.from_csr(csr), labels, k, "synthetic SBM", is_sparse=True)


def make_dcsbm(seed: int) -> DatasetCase:
    rng = np.random.default_rng(seed)
    n, k = 1_000, 5
    labels = np.repeat(np.arange(k, dtype=np.int32), n // k)
    theta = rng.lognormal(mean=0.0, sigma=0.9, size=n)
    theta /= theta.mean()
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        same = labels[i] == labels[i + 1 :]
        base = np.where(same, 0.025, 0.0025)
        prob = np.minimum(base * theta[i] * theta[i + 1 :], 0.35)
        hits = np.flatnonzero(rng.random(n - i - 1) < prob) + i + 1
        if hits.size:
            rows.extend([i] * hits.size)
            cols.extend(hits.tolist())
    row = np.asarray(rows + cols, dtype=np.int64)
    col = np.asarray(cols + rows, dtype=np.int64)
    csr = sp.coo_matrix((np.ones(row.size), (row, col)), shape=(n, n)).tocsr()
    return DatasetCase("DCSBM", SparseGraph.from_csr(csr), labels, k, "synthetic DCSBM", is_sparse=True)


def make_lfr_fallback(seed: int, n: int) -> DatasetCase:
    """Generate an LFR-style graph when NetworkX cannot realize a seed.

    This is not used when NetworkX succeeds. It preserves the journal protocol's
    intent for the problematic seeds: power-law-like community sizes, degree
    heterogeneity, and a low external mixing rate.
    """

    rng = np.random.default_rng(seed + 91_337)
    sizes: list[int] = []
    remaining = n
    while remaining > 0:
        max_size = min(120, remaining)
        min_size = min(20, max_size)
        if remaining <= 40:
            size = remaining
        else:
            raw = int(np.round((rng.pareto(1.5) + 1.0) * min_size))
            size = int(np.clip(raw, min_size, max_size))
        sizes.append(size)
        remaining -= size
    labels = np.concatenate([np.full(size, idx, dtype=np.int32) for idx, size in enumerate(sizes)])
    rng.shuffle(labels)

    theta = rng.lognormal(mean=0.0, sigma=0.75, size=n)
    theta /= theta.mean()
    rows: list[int] = []
    cols: list[int] = []
    p_in = 0.055
    p_out = 0.003
    for i in range(n - 1):
        same = labels[i] == labels[i + 1 :]
        base = np.where(same, p_in, p_out)
        prob = np.minimum(base * theta[i] * theta[i + 1 :], 0.45)
        hits = np.flatnonzero(rng.random(n - i - 1) < prob) + i + 1
        if hits.size:
            rows.extend([i] * hits.size)
            cols.extend(hits.tolist())
    row = np.asarray(rows + cols, dtype=np.int64)
    col = np.asarray(cols + rows, dtype=np.int64)
    csr = sp.coo_matrix((np.ones(row.size), (row, col)), shape=(n, n)).tocsr()
    return DatasetCase(
        "LFR",
        SparseGraph.from_csr(csr),
        labels,
        int(np.unique(labels).size),
        "LFR-style fallback: power-law communities, degree heterogeneity, mu~0.15",
        is_sparse=True,
    )


def make_lfr(seed: int, quick: bool = False) -> DatasetCase:
    n = 400 if quick else 1_000
    graph = None
    last_exc: Exception | None = None
    # NetworkX LFR generation is stochastic and can fail for otherwise valid
    # settings. Retry a bounded grid while keeping the benchmark family fixed.
    # The looser tail settings are used only to avoid seed-specific generation
    # skips; the source field records the actual parameterization.
    lfr_source = "networkx LFR"
    for offset, tau1, tau2, mu, average_degree, min_community, max_iters in [
        (0, 2.5, 1.5, 0.15, 12, 25, 2_000),
        (1, 2.5, 1.5, 0.15, 10, 20, 2_000),
        (2, 2.5, 1.5, 0.15, 8, 20, 2_000),
        (3, 2.5, 1.5, 0.15, 12, 15, 2_000),
        (4, 2.5, 1.5, 0.15, 15, 15, 2_000),
        (5, 2.5, 1.5, 0.15, 10, 10, 2_000),
        (6, 2.6, 1.7, 0.12, 14, 20, 5_000),
        (7, 2.4, 1.6, 0.18, 16, 15, 5_000),
        (8, 2.8, 1.8, 0.12, 18, 10, 5_000),
        (9, 2.2, 1.4, 0.20, 20, 10, 5_000),
    ]:
        try:
            with bounded_lfr_attempt(LFR_ATTEMPT_TIMEOUT_SECONDS):
                graph = nx.LFR_benchmark_graph(
                    n=n,
                    tau1=tau1,
                    tau2=tau2,
                    mu=mu,
                    average_degree=average_degree,
                    min_community=min_community,
                    max_iters=max_iters,
                    seed=seed + offset * 10_000,
                )
            lfr_source = (
                "networkx LFR"
                f" tau1={tau1} tau2={tau2} mu={mu}"
                f" avg_degree={average_degree} min_community={min_community}"
            )
            break
        except Exception as exc:
            last_exc = exc
    if graph is None:
        return make_lfr_fallback(seed, n)

    communities: dict[frozenset[int], int] = {}
    labels = np.zeros(n, dtype=np.int32)
    for node in range(n):
        comm = frozenset(graph.nodes[node]["community"])
        communities.setdefault(comm, len(communities))
        labels[node] = communities[comm]
    csr = nx.to_scipy_sparse_array(graph, nodelist=list(range(n)), format="csr", dtype=float)
    csr.setdiag(0.0)
    csr.eliminate_zeros()
    return DatasetCase("LFR", SparseGraph.from_csr(csr), labels, len(communities), lfr_source, is_sparse=True)


def make_scale_sbm(name: str, seed: int) -> DatasetCase:
    if name == "SBM-10k":
        return make_sbm(name, n=10_000, k=20, p_in=0.01, p_out=0.00025, seed=seed)
    if name == "SBM-50k":
        return make_sbm(name, n=50_000, k=50, p_in=0.0025, p_out=0.00004, seed=seed)
    raise ValueError(name)


def load_pyg_case(name: str) -> DatasetCase:
    try:
        import torch
        from torch_geometric.datasets import Amazon, Coauthor, Planetoid
        from torch_geometric.utils import to_undirected
    except Exception as exc:
        return DatasetCase(name, None, None, None, f"unavailable: PyG import failed: {exc}")

    try:
        if name in {"Cora", "Citeseer", "PubMed"}:
            dataset = Planetoid(root="/tmp/dataset", name=name)
        elif name in {"Photo", "Computers"}:
            dataset = Amazon(root="/tmp/dataset", name=name)
        elif name == "Coauthor-CS":
            dataset = Coauthor(root="/tmp/dataset", name="CS")
        elif name == "Coauthor-Physics":
            dataset = Coauthor(root="/tmp/dataset", name="Physics")
        else:
            raise ValueError(name)
    except Exception as exc:
        return DatasetCase(name, None, None, None, f"unavailable: dataset load failed: {exc}")

    data = dataset[0]
    edge_index = to_undirected(data.edge_index).cpu().numpy().astype(np.int64)
    graph = SparseGraph.from_edge_index(edge_index, num_nodes=int(data.num_nodes))
    labels = data.y.cpu().numpy().astype(np.int32)
    features = None
    if getattr(data, "x", None) is not None:
        x = data.x
        if hasattr(torch, "is_tensor") and torch.is_tensor(x):
            features = x.detach().cpu().numpy().astype(np.float32)
        else:
            features = np.asarray(x, dtype=np.float32)
    return DatasetCase(
        name=name,
        adjacency=graph,
        labels=labels,
        k=int(dataset.num_classes),
        source="torch_geometric",
        is_sparse=True,
        features=features,
    )


def load_ogbn_arxiv() -> DatasetCase:
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except Exception as exc:
        return DatasetCase("ogbn-arxiv", None, None, None, f"unavailable: OGB import failed: {exc}")

    os.environ.setdefault("OGB_DOWNLOAD_DIR", "/tmp/ogb")
    try:
        dataset = NodePropPredDataset(name="ogbn-arxiv", root="/tmp/ogb")
        graph_dict, labels = dataset[0]
    except Exception as exc:
        return DatasetCase("ogbn-arxiv", None, None, None, f"unavailable: OGB load failed: {exc}")

    edge_index = np.asarray(graph_dict["edge_index"], dtype=np.int64)
    graph = SparseGraph.from_edge_index(edge_index, num_nodes=int(graph_dict["num_nodes"]))
    features = np.asarray(graph_dict.get("node_feat"), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1).astype(np.int32)
    return DatasetCase(
        name="ogbn-arxiv",
        adjacency=graph,
        labels=y,
        k=int(np.unique(y).size),
        source="ogb.nodeproppred",
        is_sparse=True,
        features=features,
    )


def load_case(name: str, seed: int, quick: bool = False) -> DatasetCase:
    name = canonical_dataset_name(name)
    if name == "Karate":
        return karate_graph()
    if name == "SBM-Easy":
        return make_sbm(name, n=500 if quick else 1_000, k=5, p_in=0.08, p_out=0.004, seed=seed)
    if name == "SBM-Noisy":
        return make_sbm(name, n=500 if quick else 1_000, k=5, p_in=0.045, p_out=0.018, seed=seed)
    if name == "DCSBM":
        return make_dcsbm(seed)
    if name == "LFR":
        return make_lfr(seed, quick=quick)
    if name in {"SBM-10k", "SBM-50k"}:
        return make_scale_sbm(name, seed)
    if name == "ogbn-arxiv":
        return load_ogbn_arxiv()
    if name in {"Cora", "Citeseer", "PubMed", "Photo", "Computers", "Coauthor-CS", "Coauthor-Physics"}:
        return load_pyg_case(name)
    return DatasetCase(name, None, None, None, f"unavailable: unknown dataset {name}")


def rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int32)


def mean_conductance(adj, labels: np.ndarray) -> float | None:
    csr = _adj_as_csr(adj)
    degree = np.asarray(csr.sum(axis=1)).reshape(-1)
    total_volume = float(degree.sum())
    if total_volume <= 1e-12:
        return None
    values: list[float] = []
    labels = np.asarray(labels)
    for cluster in np.unique(labels):
        mask = labels == cluster
        vol = float(degree[mask].sum())
        denom = min(vol, total_volume - vol)
        if denom <= 1e-12:
            continue
        internal = float(csr[mask][:, mask].sum())
        cut = max(vol - internal, 0.0)
        values.append(cut / denom)
    return float(np.mean(values)) if values else None


def evaluate_labels(
    case: DatasetCase,
    labels: np.ndarray,
    block: str,
    method: str,
    seed: int,
    runtime_seconds: float,
    status: str = "ok",
    extra: dict | None = None,
) -> dict:
    labels = canonicalize_labels(labels)
    row = {
        "block": block,
        "dataset": case.name,
        "method": method,
        "seed": int(seed),
        "status": status,
        "source": case.source,
        "n_nodes": n_nodes(case),
        "n_edges": n_edges(case),
        "true_k": int(case.k) if case.k is not None else None,
        "pred_k": int(np.unique(labels).size),
        "runtime_seconds": float(runtime_seconds),
        "peak_rss_mb": rss_mb(),
    }
    if case.labels is not None:
        row.update(
            {
                "acc": clustering_accuracy(case.labels, labels),
                "nmi": normalized_mutual_info(case.labels, labels),
                "ari": adjusted_rand_index(case.labels, labels),
            }
        )
    try:
        row["modularity"] = hard_modularity(case.adjacency, labels)
    except Exception:
        row["modularity"] = None
    try:
        row["mean_conductance"] = mean_conductance(case.adjacency, labels)
    except Exception:
        row["mean_conductance"] = None
    try:
        if isinstance(case.adjacency, SparseGraph) or sp.issparse(case.adjacency):
            row["structural_entropy"] = float(sparse_structural_entropy(case.adjacency, labels))
        else:
            row["structural_entropy"] = float(structural_entropy(case.adjacency, labels))
    except Exception:
        row["structural_entropy"] = None
    try:
        row["map_equation"] = hard_map_equation(case.adjacency, labels)
    except Exception:
        row["map_equation"] = None
    if extra:
        row.update(extra)
    return row


def skip_row(case: DatasetCase, block: str, method: str, seed: int, reason: str) -> dict:
    return {
        "block": block,
        "dataset": case.name,
        "method": method,
        "seed": int(seed),
        "status": f"skipped: {reason}",
        "source": case.source,
        "n_nodes": n_nodes(case) if case.adjacency is not None else None,
        "n_edges": n_edges(case) if case.adjacency is not None else None,
        "true_k": int(case.k) if case.k is not None else None,
        "pred_k": None,
        "runtime_seconds": None,
        "peak_rss_mb": rss_mb(),
    }


def failed_row(case: DatasetCase, block: str, method: str, seed: int, exc: Exception) -> dict:
    row = skip_row(case, block, method, seed, f"failed: {type(exc).__name__}: {exc}")
    row["status"] = f"failed: {type(exc).__name__}: {exc}"
    return row


def require_case(case: DatasetCase) -> str | None:
    if case.adjacency is None:
        return case.source
    if case.labels is None:
        return "missing labels"
    if case.k is None:
        return "missing target K"
    return None


def run_spectral(case: DatasetCase, seed: int) -> tuple[np.ndarray, float, dict]:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.decomposition import TruncatedSVD

    csr = _adj_as_csr(case.adjacency)
    k = int(case.k or 2)
    t0 = time.time()
    if csr.shape[0] <= SPECTRAL_DENSE_MAX_NODES:
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
            n_init=20,
        )
        labels = model.fit_predict(csr)
        mode = "normalized spectral clustering"
    else:
        dims = min(max(k, 32), 128, csr.shape[0] - 1)
        emb = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(csr)
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(emb)
        mode = "adjacency SVD + k-means fallback"
    return labels.astype(np.int32), time.time() - t0, {"variant": mode}


def run_label_propagation(case: DatasetCase, seed: int) -> tuple[np.ndarray, float, dict]:
    csr = _adj_as_csr(case.adjacency)
    graph = nx.from_scipy_sparse_array(csr)
    t0 = time.time()
    communities = list(nx.algorithms.community.asyn_lpa_communities(graph, seed=seed, weight="weight"))
    labels = np.zeros(csr.shape[0], dtype=np.int32)
    for idx, community in enumerate(communities):
        labels[list(community)] = idx
    return labels, time.time() - t0, {}


def run_kmeans(features, k: int, seed: int) -> tuple[np.ndarray, float]:
    from sklearn.cluster import KMeans, MiniBatchKMeans

    t0 = time.time()
    x = np.asarray(features, dtype=np.float32)
    if x.shape[0] > 10_000:
        model = MiniBatchKMeans(
            n_clusters=k,
            n_init=3,
            random_state=seed,
            batch_size=8192,
            max_iter=100,
            reassignment_ratio=0.01,
        )
    else:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = model.fit_predict(x)
    return labels.astype(np.int32), time.time() - t0


def run_pca_kmeans(features, k: int, seed: int) -> tuple[np.ndarray, float, dict]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    x = np.asarray(features, dtype=np.float32)
    dims = min(128, x.shape[1] - 1, x.shape[0] - 1)
    if dims < 2:
        emb = StandardScaler(with_mean=False).fit_transform(x)
    else:
        emb = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(x)
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(emb)
    return labels.astype(np.int32), time.time() - t0, {"embedding_dim": int(emb.shape[1])}


def run_adj_svd_kmeans(case: DatasetCase, seed: int) -> tuple[np.ndarray, float, dict]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    csr = _adj_as_csr(case.adjacency)
    k = int(case.k or 2)
    t0 = time.time()
    dims = min(max(k, 32), 128, csr.shape[0] - 1)
    emb = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(csr)
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(emb)
    return labels.astype(np.int32), time.time() - t0, {"embedding_dim": int(dims)}


def run_node2vec(case: DatasetCase, seed: int, method: str, quick: bool) -> tuple[np.ndarray, float, dict]:
    import torch
    from sklearn.cluster import KMeans
    from torch_geometric.nn import Node2Vec

    csr = _adj_as_csr(case.adjacency)
    if csr.shape[0] > 50_000:
        raise SkipMethod("Node2Vec baseline is capped at N<=50000 in this harness")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    edge_index = torch.as_tensor(csr_to_edge_index(csr), dtype=torch.long)
    p, q = (1.0, 1.0) if method == "DeepWalkKMeans" else (1.0, 0.5)
    epochs = 3 if quick else 20
    t0 = time.time()
    try:
        model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=128,
            walk_length=20,
            context_size=10,
            walks_per_node=4 if quick else 10,
            p=p,
            q=q,
            num_negative_samples=1,
            num_nodes=csr.shape[0],
            sparse=True,
        )
    except ImportError as exc:
        raise SkipMethod(str(exc)) from exc
    loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
    model.eval()
    emb = model.embedding.weight.detach().cpu().numpy()
    labels = KMeans(n_clusters=int(case.k or 2), n_init=20, random_state=seed).fit_predict(emb)
    return labels.astype(np.int32), time.time() - t0, {"epochs": epochs, "embedding_dim": 128, "p": p, "q": q}


def run_gae(case: DatasetCase, seed: int, method: str, epochs: int, quick: bool) -> tuple[np.ndarray, float, dict]:
    import torch
    import torch.nn.functional as F
    from sklearn.cluster import KMeans
    from torch_geometric.nn import GAE, VGAE, GCNConv

    if case.features is None:
        raise SkipMethod("GAE/VGAE requires node features")

    csr = _adj_as_csr(case.adjacency)
    if csr.shape[0] > 50_000:
        raise SkipMethod("GAE/VGAE baseline is capped at N<=50000 in this harness")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    edge_index = torch.as_tensor(csr_to_edge_index(csr), dtype=torch.long)
    x = torch.as_tensor(np.asarray(case.features, dtype=np.float32), dtype=torch.float32)
    hidden = 64 if quick else 128
    z_dim = 32 if quick else 64
    train_epochs = min(epochs, 30) if quick else epochs

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, variational: bool):
            super().__init__()
            self.variational = variational
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv_mu = GCNConv(hidden_channels, out_channels)
            self.conv_logstd = GCNConv(hidden_channels, out_channels) if variational else None

        def forward(self, features, edges):
            h = F.relu(self.conv1(features, edges))
            if self.variational:
                return self.conv_mu(h, edges), self.conv_logstd(h, edges)
            return self.conv_mu(h, edges)

    variational = method == "VGAE"
    model = (VGAE if variational else GAE)(Encoder(x.shape[1], hidden, z_dim, variational))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    t0 = time.time()
    model.train()
    for _ in range(train_epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        if variational:
            loss = loss + (1.0 / x.shape[0]) * model.kl_loss()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    emb = z.detach().cpu().numpy()
    labels = KMeans(n_clusters=int(case.k or 2), n_init=20, random_state=seed).fit_predict(emb)
    return labels.astype(np.int32), time.time() - t0, {"epochs": train_epochs, "embedding_dim": z_dim}


def run_dmon(case: DatasetCase, seed: int, epochs: int, quick: bool) -> tuple[np.ndarray, float, dict]:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import DMoNPooling, GCNConv

    if case.features is None:
        raise SkipMethod("DMoN requires node features")
    csr = _adj_as_csr(case.adjacency)
    if csr.shape[0] > DENSE_GNN_MAX_NODES:
        raise SkipMethod(f"DMoN dense pooling is capped at N<={DENSE_GNN_MAX_NODES}")

    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(case.features, dtype=np.float32), dtype=torch.float32)
    edge_index = torch.as_tensor(csr_to_edge_index(csr), dtype=torch.long)
    dense_adj = torch.as_tensor(csr.toarray(), dtype=torch.float32).unsqueeze(0)
    k = int(case.k or 2)
    hidden = 64 if quick else 128
    train_epochs = min(epochs, 30) if quick else epochs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(x.shape[1], hidden)
            self.pool = DMoNPooling([hidden], k, dropout=0.25)

        def forward(self):
            h = F.relu(self.conv(x, edge_index)).unsqueeze(0)
            s, _, _, spectral_loss, orthogonality_loss, cluster_loss = self.pool(h, dense_adj)
            return spectral_loss + orthogonality_loss + cluster_loss, s.squeeze(0)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    t0 = time.time()
    model.train()
    for _ in range(train_epochs):
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, assignment = model()
    labels = assignment.argmax(dim=-1).cpu().numpy().astype(np.int32)
    return labels, time.time() - t0, {"epochs": train_epochs, "dense_pooling": True}


def run_mincut(case: DatasetCase, seed: int, epochs: int, quick: bool) -> tuple[np.ndarray, float, dict]:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, dense_mincut_pool

    if case.features is None:
        raise SkipMethod("MinCutPool requires node features")
    csr = _adj_as_csr(case.adjacency)
    if csr.shape[0] > DENSE_GNN_MAX_NODES:
        raise SkipMethod(f"MinCutPool dense pooling is capped at N<={DENSE_GNN_MAX_NODES}")

    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(case.features, dtype=np.float32), dtype=torch.float32)
    edge_index = torch.as_tensor(csr_to_edge_index(csr), dtype=torch.long)
    dense_x = x.unsqueeze(0)
    dense_adj = torch.as_tensor(csr.toarray(), dtype=torch.float32).unsqueeze(0)
    k = int(case.k or 2)
    hidden = 64 if quick else 128
    train_epochs = min(epochs, 30) if quick else epochs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(x.shape[1], hidden)
            self.assign = torch.nn.Linear(hidden, k)

        def forward(self):
            h = F.relu(self.conv(x, edge_index))
            logits = self.assign(h).unsqueeze(0)
            _, _, mincut_loss, ortho_loss = dense_mincut_pool(dense_x, dense_adj, logits)
            return mincut_loss + ortho_loss, logits.squeeze(0)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    t0 = time.time()
    model.train()
    for _ in range(train_epochs):
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, logits = model()
    labels = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
    return labels, time.time() - t0, {"epochs": train_epochs, "dense_pooling": True}


def run_louvain_dendrogram(case: DatasetCase, seed: int) -> tuple[list[np.ndarray], float]:
    from community import community_louvain

    csr = _adj_as_csr(case.adjacency)
    graph = nx.from_scipy_sparse_array(csr)
    t0 = time.time()
    dendrogram = community_louvain.generate_dendrogram(graph, random_state=seed)
    n = csr.shape[0]
    levels: list[np.ndarray] = []
    composed = np.asarray([dendrogram[0][i] for i in range(n)], dtype=np.int32)
    levels.append(canonicalize_labels(composed))
    for level in dendrogram[1:]:
        composed = np.asarray([level[int(cluster)] for cluster in composed], dtype=np.int32)
        levels.append(canonicalize_labels(composed))
    return levels, time.time() - t0


def run_seclust_same_protocol(case: DatasetCase, method: str, seed: int, cfg: RunConfig) -> dict:
    from glass.seclust import (
        cluster_graph,
        constrained_k_multistart,
        hierarchical_se_clustering,
        multilevel_se_clustering,
    )
    from glass.seclust.heuristics import ClusteringResult
    from glass.seclust.hierarchy import merge_hierarchy_levels, select_hierarchy_level
    from glass.seclust.incremental import multistart_incremental_se_heuristic

    t0 = time.time()
    if method == "SEClust-Tree":
        result = hierarchical_se_clustering(
            case.adjacency,
            target_clusters=case.k,
            starts=cfg.seclust_starts,
            max_passes=cfg.seclust_passes,
            seed=seed,
        )
    elif method == "SEClust-TargetK":
        base_labels, _ = multistart_incremental_se_heuristic(
            case.adjacency,
            starts=cfg.seclust_starts,
            max_passes=cfg.seclust_passes,
            seed=seed,
        )
        levels = merge_hierarchy_levels(case.adjacency, base_labels, min_clusters=case.k)
        selected = select_hierarchy_level(levels, target_clusters=case.k)
        result = ClusteringResult(entropy=selected.entropy, labels=selected.labels, method="seclust-target-k")
    elif method == "SEClust-MultiLevel":
        result = multilevel_se_clustering(
            case.adjacency,
            starts=cfg.seclust_starts,
            max_passes=cfg.seclust_passes,
            seed=seed,
        )
    elif method == "SEClust-ConstrainedK":
        labels, entropy = constrained_k_multistart(
            case.adjacency,
            target_clusters=case.k or 2,
            starts=cfg.seclust_starts,
            max_passes=cfg.seclust_passes,
            seed=seed,
        )
        result = ClusteringResult(entropy=entropy, labels=labels, method="seclust-constrained-k")
    else:
        result = cluster_graph(
            case.adjacency,
            mode="heuristic",
            exact_max_nodes=9,
            heuristic_starts=cfg.seclust_starts,
            max_passes=cfg.seclust_passes,
            seed=seed,
        )
    runtime = time.time() - t0
    return evaluate_labels(
        case,
        np.asarray(result.labels, dtype=np.int32),
        cfg.block,
        method,
        seed,
        runtime,
        extra={"seclust_entropy": float(result.entropy), "seclust_starts": cfg.seclust_starts, "seclust_passes": cfg.seclust_passes},
    )


def run_paris_or_agglomerative(case: DatasetCase, method: str) -> tuple[list[np.ndarray], float, dict]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    csr = _adj_as_csr(case.adjacency)
    if csr.shape[0] > 5_000:
        raise SkipMethod(f"{method} dense hierarchy is capped at N<=5000")
    if method == "Paris":
        raise SkipMethod(
            "Paris requires the scikit-network Paris implementation; install scikit-network "
            "and add its dendrogram adapter before using this row as a journal baseline"
        )
    t0 = time.time()
    dense = csr.toarray()
    similarity = dense / max(float(dense.max()), 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_method = "average"
    tree = linkage(condensed, method=linkage_method)
    ks = sorted({max(2, int(case.k or 2)), max(2, int(math.sqrt(csr.shape[0]))), 1}, reverse=True)
    levels = [canonicalize_labels(fcluster(tree, t=k, criterion="maxclust") - 1) for k in ks if k > 1]
    levels.append(np.zeros(csr.shape[0], dtype=np.int32))
    return levels, time.time() - t0, {"linkage": linkage_method}


def evaluate_hierarchy(case: DatasetCase, levels: list[np.ndarray], block: str, method: str, seed: int, runtime: float, extra=None) -> dict:
    from glass.seclust import dasgupta_cost, dendrogram_purity

    finest = canonicalize_labels(levels[0])
    row = evaluate_labels(case, finest, block, method, seed, runtime, extra=extra)
    row["n_levels"] = int(len(levels))
    row["coarsest_k"] = int(np.unique(levels[-1]).size)
    try:
        row["dasgupta_cost"] = float(dasgupta_cost(_adj_as_csr(case.adjacency), levels))
    except Exception:
        row["dasgupta_cost"] = None
    try:
        row["dendrogram_purity"] = float(dendrogram_purity(levels, case.labels))
    except Exception:
        row["dendrogram_purity"] = None
    return row


def run_topology_method(case: DatasetCase, method: str, seed: int, cfg: RunConfig) -> dict:
    if method == "Louvain":
        labels, runtime = run_louvain(case.adjacency, seed=seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method == "Leiden":
        labels, runtime = run_leiden(case.adjacency, seed=seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method == "Infomap":
        labels, runtime = run_infomap(case.adjacency, seed=seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method == "Spectral":
        labels, runtime, extra = run_spectral(case, seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method == "LabelPropagation":
        labels, runtime, extra = run_label_propagation(case, seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method == "HCSE":
        if n_nodes(case) > HCSE_DENSE_MAX_NODES:
            return skip_row(case, cfg.block, method, seed, f"HCSE dense implementation capped at N<={HCSE_DENSE_MAX_NODES}")
        labels, runtime = run_hcse(case.adjacency, case.k or 2, seed=seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method.startswith("SEClust-"):
        return run_seclust_same_protocol(case, method, seed, cfg)
    raise ValueError(f"unknown topology method {method}")


def run_attributed_method(case: DatasetCase, method: str, seed: int, cfg: RunConfig) -> dict:
    if method in {"RawKMeans", "PCAKMeans"} and case.features is None:
        return skip_row(case, cfg.block, method, seed, "method requires node features")
    if method == "RawKMeans":
        labels, runtime = run_kmeans(case.features, int(case.k or 2), seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method == "PCAKMeans":
        labels, runtime, extra = run_pca_kmeans(case.features, int(case.k or 2), seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method == "AdjSVDKMeans":
        labels, runtime, extra = run_adj_svd_kmeans(case, seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method in {"DeepWalkKMeans", "Node2VecKMeans"}:
        labels, runtime, extra = run_node2vec(case, seed, method, cfg.quick)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method in {"GAE", "VGAE"}:
        labels, runtime, extra = run_gae(case, seed, method, cfg.neural_epochs, cfg.quick)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method == "DMoN":
        labels, runtime, extra = run_dmon(case, seed, cfg.neural_epochs, cfg.quick)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method == "MinCutPool":
        labels, runtime, extra = run_mincut(case, seed, cfg.neural_epochs, cfg.quick)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime, extra=extra)
    if method == "LSEnet-Proxy":
        if case.features is None:
            return skip_row(case, cfg.block, method, seed, "method requires node features")
        if n_nodes(case) > DENSE_GNN_MAX_NODES:
            return skip_row(case, cfg.block, method, seed, f"LSEnet proxy is dense and capped at N<={DENSE_GNN_MAX_NODES}")
        labels, runtime = run_lsenet_proxy(case.adjacency, case.features, int(case.k or 2), n_iters=cfg.neural_epochs, seed=seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method == "Glass-SE-GNN":
        if case.features is None:
            return skip_row(case, cfg.block, method, seed, "method requires node features")
        if n_nodes(case) > DENSE_GNN_MAX_NODES:
            return skip_row(case, cfg.block, method, seed, f"Glass-SE-GNN is dense and capped at N<={DENSE_GNN_MAX_NODES}")
        labels, runtime = run_glass_se_gnn(case.adjacency, case.features, int(case.k or 2), n_iters=cfg.neural_epochs, seed=seed)
        return evaluate_labels(case, labels, cfg.block, method, seed, runtime)
    if method == "SEClust-ConstrainedK":
        return run_topology_method(case, method, seed, cfg)
    if method == "LSEnet-Official":
        return skip_row(case, cfg.block, method, seed, "official LSEnet repo only ships FootBall config; add dataset configs before using this row as a journal baseline")
    raise ValueError(f"unknown attributed method {method}")


def run_hierarchy_method(case: DatasetCase, method: str, seed: int, cfg: RunConfig) -> dict:
    if method == "SEClust-Tree":
        from glass.seclust import coding_tree_hierarchy_levels, multistart_incremental_se_heuristic

        t0 = time.time()
        base, _ = multistart_incremental_se_heuristic(
            case.adjacency,
            starts=cfg.seclust_starts,
            max_passes=cfg.seclust_passes,
            seed=seed,
        )
        level_objs = coding_tree_hierarchy_levels(case.adjacency, base, min_clusters=1)
        levels = [np.asarray(level.labels, dtype=np.int32) for level in level_objs]
        return evaluate_hierarchy(case, levels, cfg.block, method, seed, time.time() - t0)
    if method == "HCSE":
        if n_nodes(case) > HCSE_DENSE_MAX_NODES:
            return skip_row(case, cfg.block, method, seed, f"HCSE dense implementation capped at N<={HCSE_DENSE_MAX_NODES}")
        labels, runtime = run_hcse(case.adjacency, case.k or 2, seed=seed)
        levels = [np.arange(labels.size, dtype=np.int32), canonicalize_labels(labels), np.zeros(labels.size, dtype=np.int32)]
        return evaluate_hierarchy(case, levels, cfg.block, method, seed, runtime, extra={"hierarchy_note": "flat HCSE partition bracketed by singleton/root levels"})
    if method == "Louvain-Dendrogram":
        levels, runtime = run_louvain_dendrogram(case, seed)
        return evaluate_hierarchy(case, levels, cfg.block, method, seed, runtime)
    if method in {"Paris", "Agglomerative"}:
        levels, runtime, extra = run_paris_or_agglomerative(case, method)
        return evaluate_hierarchy(case, levels, cfg.block, method, seed, runtime, extra=extra)
    raise ValueError(f"unknown hierarchy method {method}")


def run_method(case: DatasetCase, method: str, seed: int, cfg: RunConfig) -> dict:
    reason = require_case(case)
    if reason is not None:
        return skip_row(case, cfg.block, method, seed, reason)
    if cfg.max_nodes is not None and n_nodes(case) > cfg.max_nodes:
        return skip_row(case, cfg.block, method, seed, f"N={n_nodes(case)} exceeds --max-nodes={cfg.max_nodes}")
    if cfg.dry_run:
        return skip_row(case, cfg.block, method, seed, "dry run")

    t0 = time.time()
    try:
        if cfg.block in {"topology", "scale", "synthetic"}:
            row = run_topology_method(case, method, seed, cfg)
        elif cfg.block == "attributed":
            row = run_attributed_method(case, method, seed, cfg)
        elif cfg.block == "hierarchy":
            row = run_hierarchy_method(case, method, seed, cfg)
        else:
            raise ValueError(f"unknown block {cfg.block}")
    except SkipMethod as exc:
        return skip_row(case, cfg.block, method, seed, str(exc))
    except Exception as exc:
        return failed_row(case, cfg.block, method, seed, exc)

    if cfg.time_limit is not None:
        elapsed = row.get("runtime_seconds")
        if isinstance(elapsed, (int, float)) and elapsed > cfg.time_limit:
            row["status"] = f"completed_over_time_limit: {elapsed:.2f}s > {cfg.time_limit:.2f}s"
    row["wall_seconds_including_scoring"] = float(time.time() - t0)
    return row


def aggregate_rows(rows: list[dict]) -> list[dict]:
    numeric_keys = [
        "acc",
        "nmi",
        "ari",
        "modularity",
        "mean_conductance",
        "structural_entropy",
        "map_equation",
        "runtime_seconds",
        "wall_seconds_including_scoring",
        "peak_rss_mb",
        "pred_k",
        "dasgupta_cost",
        "dendrogram_purity",
        "n_levels",
    ]
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["block"], row["dataset"], row["method"]), []).append(row)

    out: list[dict] = []
    for (block, dataset, method), group in sorted(groups.items()):
        item = {
            "block": block,
            "dataset": dataset,
            "method": method,
            "n_runs": len(group),
            "n_ok": sum(1 for row in group if str(row.get("status")) in {"ok", "baseline_executed"}),
            "statuses": sorted(set(str(row.get("status")) for row in group)),
            "n_nodes": group[0].get("n_nodes"),
            "n_edges": group[0].get("n_edges"),
            "true_k": group[0].get("true_k"),
        }
        for key in numeric_keys:
            values = [row.get(key) for row in group if isinstance(row.get(key), (int, float))]
            if values:
                arr = np.asarray(values, dtype=float)
                item[f"{key}_mean"] = float(arr.mean())
                item[f"{key}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(item)
    return out


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "---"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_mean(row: dict, key: str, digits: int = 3) -> str:
    mean = row.get(f"{key}_mean")
    if mean is None:
        return "---"
    std = row.get(f"{key}_std", 0.0)
    if std and abs(float(std)) > 1e-12:
        return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"
    return f"{float(mean):.{digits}f}"


def best_methods(aggregated: list[dict], metric: str, higher: bool = True) -> dict[str, str]:
    result: dict[str, str] = {}
    for dataset in sorted(set(row["dataset"] for row in aggregated)):
        candidates = [row for row in aggregated if row["dataset"] == dataset and row.get(f"{metric}_mean") is not None]
        if not candidates:
            continue
        key: Callable[[dict], float] = lambda row: float(row[f"{metric}_mean"])
        best = max(candidates, key=key) if higher else min(candidates, key=key)
        result[dataset] = str(best["method"])
    return result


def write_outputs(rows: list[dict], cfg: RunConfig) -> tuple[Path, Path, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"same_protocol_{cfg.block}_{timestamp}"
    raw_path = cfg.output_dir / f"{prefix}.json"
    agg_path = cfg.output_dir / f"{prefix}_aggregated.json"
    md_path = cfg.output_dir / f"{prefix}.md"
    aggregated = aggregate_rows(rows)

    payload = {
        "metadata": {
            "timestamp": timestamp,
            "block": cfg.block,
            "datasets": cfg.datasets,
            "methods": cfg.methods,
            "seeds": cfg.seeds,
            "quick": cfg.quick,
            "max_nodes": cfg.max_nodes,
            "neural_epochs": cfg.neural_epochs,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    agg_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    nmi_best = best_methods(aggregated, "nmi", higher=True)
    se_best = best_methods(aggregated, "structural_entropy", higher=False)
    lines = [
        f"# Same-Protocol {cfg.block.title()} Benchmark ({timestamp})",
        "",
        "Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.",
        "",
        f"- Datasets: {', '.join(cfg.datasets)}",
        f"- Methods: {', '.join(cfg.methods)}",
        f"- Seeds: {', '.join(str(seed) for seed in cfg.seeds)}",
        f"- Quick mode: {cfg.quick}",
        "",
        "| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in aggregated:
        method = str(row["method"])
        dataset = str(row["dataset"])
        if nmi_best.get(dataset) == method:
            method = f"**{method}**"
        status = "; ".join(row["statuses"])
        if len(status) > 80:
            status = status[:77] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    method,
                    f"{row['n_ok']} / {row['n_runs']}",
                    fmt_mean(row, "acc"),
                    fmt_mean(row, "nmi"),
                    fmt_mean(row, "ari"),
                    fmt_mean(row, "pred_k", 1),
                    fmt_mean(row, "modularity"),
                    fmt_mean(row, "mean_conductance"),
                    fmt_mean(row, "structural_entropy"),
                    fmt_mean(row, "map_equation"),
                    fmt_mean(row, "runtime_seconds", 2),
                    status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best-By-Dataset Checks",
            "",
            f"- Best mean NMI: {json.dumps(nmi_best, sort_keys=True)}",
            f"- Best mean structural entropy (lower): {json.dumps(se_best, sort_keys=True)}",
            "",
            "## Files",
            "",
            f"- Raw seed-level JSON: `{raw_path}`",
            f"- Aggregated JSON: `{agg_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_path, agg_path, md_path


def write_checkpoint(rows: list[dict], cfg: RunConfig) -> None:
    """Overwrite recoverable checkpoint files during long full-protocol runs."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "checkpoint": True,
            "block": cfg.block,
            "datasets": cfg.datasets,
            "methods": cfg.methods,
            "seeds": cfg.seeds,
            "quick": cfg.quick,
            "rows_completed": len(rows),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    raw_path = cfg.output_dir / f"same_protocol_{cfg.block}_checkpoint.json"
    agg_path = cfg.output_dir / f"same_protocol_{cfg.block}_checkpoint_aggregated.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    agg_path.write_text(json.dumps(aggregate_rows(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_datasets_for_block(block: str) -> list[str]:
    if block == "topology":
        return TOPOLOGY_DATASETS
    if block == "attributed":
        return ATTRIBUTED_DATASETS
    if block == "hierarchy":
        return HIERARCHY_DATASETS
    if block == "scale":
        return SCALE_DATASETS
    if block == "synthetic":
        return ["Karate", "SBM-Easy", "SBM-Noisy", "DCSBM", "LFR"]
    raise ValueError(block)


def default_methods_for_block(block: str) -> list[str]:
    if block in {"topology", "synthetic"}:
        return TOPOLOGY_METHODS
    if block == "attributed":
        return ATTRIBUTED_METHODS
    if block == "hierarchy":
        return HIERARCHY_METHODS
    if block == "scale":
        return SCALE_METHODS
    raise ValueError(block)


def quick_filter(cfg: RunConfig) -> RunConfig:
    if not cfg.quick:
        return cfg
    datasets = [name for name in cfg.datasets if name in {"Karate", "SBM-Easy", "Cora", "Photo"}]
    if not datasets:
        datasets = cfg.datasets[:1]
    methods = cfg.methods[:]
    if cfg.block == "topology":
        methods = [m for m in methods if m in {"Louvain", "Spectral", "LabelPropagation", "SEClust-ConstrainedK"}]
    elif cfg.block == "attributed":
        methods = [m for m in methods if m in {"RawKMeans", "PCAKMeans", "AdjSVDKMeans", "GAE", "SEClust-ConstrainedK"}]
    elif cfg.block == "hierarchy":
        methods = [m for m in methods if m in {"SEClust-Tree", "Louvain-Dendrogram"}]
    return RunConfig(
        block=cfg.block,
        seeds=cfg.seeds,
        datasets=datasets,
        methods=methods or cfg.methods[:1],
        output_dir=cfg.output_dir,
        quick=cfg.quick,
        max_nodes=cfg.max_nodes,
        time_limit=cfg.time_limit,
        neural_epochs=min(cfg.neural_epochs, 30),
        seclust_starts=min(cfg.seclust_starts, 2),
        seclust_passes=min(cfg.seclust_passes, 4),
        dry_run=cfg.dry_run,
    )


def run_block(cfg: RunConfig) -> list[dict]:
    rows: list[dict] = []
    total = len(cfg.datasets) * len(cfg.methods) * len(cfg.seeds)
    completed = 0
    for dataset_name in cfg.datasets:
        for seed in cfg.seeds:
            case = load_case(dataset_name, seed=seed, quick=cfg.quick)
            print(
                f"[dataset] {case.name} seed={seed} source={case.source} "
                f"N={n_nodes(case) if case.adjacency is not None else 'NA'} "
                f"E={n_edges(case) if case.adjacency is not None else 'NA'}",
                flush=True,
            )
            for method in cfg.methods:
                completed += 1
                print(f"[{completed}/{total}] {case.name} / {method} / seed={seed}", flush=True)
                row = run_method(case, method, seed, cfg)
                rows.append(row)
                write_checkpoint(rows, cfg)
                print(f"  -> {row.get('status')} time={fmt(row.get('runtime_seconds'), 2)}", flush=True)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=["topology", "attributed", "hierarchy", "scale", "synthetic"], default="topology")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names. Default depends on --block.")
    parser.add_argument("--methods", default=None, help="Comma-separated method names. Default depends on --block.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated integer seeds.")
    parser.add_argument("--output-dir", default="docs/experimental_reports")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test subset and shorter neural training.")
    parser.add_argument("--max-nodes", type=int, default=None, help="Skip datasets larger than this node count.")
    parser.add_argument("--time-limit", type=float, default=None, help="Mark completed cells that exceed this runtime.")
    parser.add_argument("--neural-epochs", type=int, default=NEURAL_DEFAULT_EPOCHS)
    parser.add_argument("--seclust-starts", type=int, default=6)
    parser.add_argument("--seclust-passes", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Load datasets and emit skipped method rows without executing methods.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = RunConfig(
        block=args.block,
        seeds=parse_seeds(args.seeds),
        datasets=[canonical_dataset_name(x) for x in parse_csv(args.datasets, default_datasets_for_block(args.block))],
        methods=parse_csv(args.methods, default_methods_for_block(args.block)),
        output_dir=Path(args.output_dir),
        quick=bool(args.quick),
        max_nodes=args.max_nodes,
        time_limit=args.time_limit,
        neural_epochs=args.neural_epochs,
        seclust_starts=args.seclust_starts,
        seclust_passes=args.seclust_passes,
        dry_run=bool(args.dry_run),
    )
    cfg = quick_filter(cfg)
    if args.datasets is not None or args.methods is not None:
        cfg = RunConfig(
            block=cfg.block,
            seeds=cfg.seeds,
            datasets=[canonical_dataset_name(x) for x in parse_csv(args.datasets, cfg.datasets)] if args.datasets is not None else cfg.datasets,
            methods=parse_csv(args.methods, cfg.methods) if args.methods is not None else cfg.methods,
            output_dir=cfg.output_dir,
            quick=cfg.quick,
            max_nodes=cfg.max_nodes,
            time_limit=cfg.time_limit,
            neural_epochs=cfg.neural_epochs,
            seclust_starts=cfg.seclust_starts,
            seclust_passes=cfg.seclust_passes,
            dry_run=cfg.dry_run,
        )
    print(f"Same-protocol block={cfg.block} datasets={cfg.datasets} methods={cfg.methods} seeds={cfg.seeds}", flush=True)
    rows = run_block(cfg)
    raw_path, agg_path, md_path = write_outputs(rows, cfg)
    print(f"Saved raw JSON: {raw_path}", flush=True)
    print(f"Saved aggregate JSON: {agg_path}", flush=True)
    print(f"Saved Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
