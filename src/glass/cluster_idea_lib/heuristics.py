"""Discrete heuristic clustering for structural entropy minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .entropy import canonicalize_labels, structural_entropy
from .exact import exact_minimize_structural_entropy


@dataclass(frozen=True)
class ClusteringResult:
    """Output from a cluster-idea-lib optimizer."""

    entropy: float
    labels: np.ndarray
    method: str
    exact: bool = False


def singleton_labels(n_nodes: int) -> np.ndarray:
    return np.arange(n_nodes, dtype=np.int32)


def agglomerative_se_clustering(
    adj: np.ndarray,
    target_clusters: int | None = None,
) -> ClusteringResult:
    """Greedily merge the pair with the largest SE decrease."""

    n_nodes = int(np.asarray(adj).shape[0])
    labels = singleton_labels(n_nodes)
    best_entropy = structural_entropy(adj, labels)

    while len(np.unique(labels)) > 1:
        unique = np.unique(labels)
        if target_clusters is not None and len(unique) <= target_clusters:
            break
        best_pair = None
        best_pair_entropy = best_entropy
        for i, left in enumerate(unique):
            for right in unique[i + 1 :]:
                candidate = labels.copy()
                candidate[candidate == right] = left
                candidate = canonicalize_labels(candidate)
                entropy = structural_entropy(adj, candidate)
                if entropy < best_pair_entropy - 1e-12:
                    best_pair = (left, right)
                    best_pair_entropy = entropy
        if best_pair is None:
            break
        labels[labels == best_pair[1]] = best_pair[0]
        labels = canonicalize_labels(labels)
        best_entropy = best_pair_entropy
    return ClusteringResult(best_entropy, labels, method="agglomerative-se")


def local_move_se_clustering(
    adj: np.ndarray,
    init_labels: np.ndarray | None = None,
    max_passes: int = 50,
    seed: int = 0,
) -> ClusteringResult:
    """Leiden/Louvain-style node moves, scored by hard structural entropy."""

    rng = np.random.default_rng(seed)
    n_nodes = int(np.asarray(adj).shape[0])
    labels = canonicalize_labels(init_labels if init_labels is not None else singleton_labels(n_nodes))
    best_entropy = structural_entropy(adj, labels)

    for _ in range(max_passes):
        changed = False
        for node in rng.permutation(n_nodes):
            current = int(labels[node])
            candidates = set(np.unique(labels).astype(int).tolist())
            candidates.add(int(labels.max()) + 1)
            local_best_label = current
            local_best_entropy = best_entropy
            for candidate_label in candidates:
                if candidate_label == current:
                    continue
                proposal = labels.copy()
                proposal[node] = candidate_label
                proposal = canonicalize_labels(proposal)
                entropy = structural_entropy(adj, proposal)
                if entropy < local_best_entropy - 1e-12:
                    local_best_entropy = entropy
                    local_best_label = candidate_label
            if local_best_label != current:
                labels[node] = local_best_label
                labels = canonicalize_labels(labels)
                best_entropy = local_best_entropy
                changed = True
        if not changed:
            break
    return ClusteringResult(best_entropy, labels, method="local-move-se")


def multistart_se_heuristic(
    adj: np.ndarray,
    starts: int = 16,
    max_passes: int = 50,
    seed: int = 0,
) -> ClusteringResult:
    """Run agglomerative plus random starts and keep the minimum SE partition."""

    n_nodes = int(np.asarray(adj).shape[0])
    rng = np.random.default_rng(seed)
    seeds = [agglomerative_se_clustering(adj).labels, singleton_labels(n_nodes), np.zeros(n_nodes, dtype=np.int32)]
    for _ in range(max(0, starts - len(seeds))):
        k = int(rng.integers(1, n_nodes + 1))
        seeds.append(rng.integers(0, k, size=n_nodes, dtype=np.int32))

    best: ClusteringResult | None = None
    for i, labels in enumerate(seeds):
        result = local_move_se_clustering(adj, labels, max_passes=max_passes, seed=seed + i)
        if best is None or result.entropy < best.entropy:
            best = result
    assert best is not None
    return ClusteringResult(best.entropy, best.labels, method="multistart-local-se")


def cluster_graph(
    adj: np.ndarray,
    mode: Literal["auto", "exact", "heuristic"] = "auto",
    exact_max_nodes: int = 9,
    heuristic_starts: int = 24,
    seed: int = 0,
) -> ClusteringResult:
    """High-level structural entropy clustering entry point."""

    n_nodes = int(np.asarray(adj).shape[0])
    if mode == "exact" or (mode == "auto" and n_nodes <= exact_max_nodes):
        exact = exact_minimize_structural_entropy(adj, max_nodes=max(exact_max_nodes, n_nodes))
        return ClusteringResult(exact.entropy, exact.labels, method="exact-rgs", exact=True)
    if mode not in {"auto", "heuristic"}:
        raise ValueError("mode must be 'auto', 'exact', or 'heuristic'")
    return multistart_se_heuristic(adj, starts=heuristic_starts, seed=seed)
