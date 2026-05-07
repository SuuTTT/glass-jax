"""Sparse incremental structural entropy scoring and local search."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .entropy import as_symmetric_adjacency, canonicalize_labels, structural_entropy


@dataclass(frozen=True)
class SparseGraph:
    """Neighbor-list view of a non-negative undirected graph."""

    neighbors: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    degrees: np.ndarray
    volume: float
    n_nodes: int
    n_edges: int

    @classmethod
    def from_adjacency(cls, adj: np.ndarray) -> "SparseGraph":
        matrix = as_symmetric_adjacency(adj)
        neighbors = []
        weights = []
        degrees = matrix.sum(axis=1)
        edge_count = 0
        for node in range(matrix.shape[0]):
            idx = np.flatnonzero(matrix[node] > 0)
            vals = matrix[node, idx].astype(float, copy=True)
            neighbors.append(idx.astype(np.int32, copy=False))
            weights.append(vals)
            edge_count += int(np.sum(idx > node))
        return cls(
            neighbors=tuple(neighbors),
            weights=tuple(weights),
            degrees=degrees.astype(float, copy=False),
            volume=float(degrees.sum()),
            n_nodes=int(matrix.shape[0]),
            n_edges=edge_count,
        )


class IncrementalSEState:
    """Mutable partition state with O(deg(v)) node-move SE deltas."""

    def __init__(self, graph: SparseGraph, labels: np.ndarray | None = None, eps: float = 1e-12):
        self.graph = graph
        self.eps = eps
        if labels is None:
            labels = np.arange(graph.n_nodes, dtype=np.int32)
        self.labels = canonicalize_labels(labels)
        if self.labels.shape[0] != graph.n_nodes:
            raise ValueError("labels length must match graph size")

        self.capacity = max(2 * graph.n_nodes + 1, int(self.labels.max(initial=0)) + graph.n_nodes + 2)
        self.volume = np.zeros(self.capacity, dtype=float)
        self.cut = np.zeros(self.capacity, dtype=float)
        self.degree_log_degree = np.zeros(self.capacity, dtype=float)
        self.size = np.zeros(self.capacity, dtype=np.int32)
        self.active = np.zeros(self.capacity, dtype=bool)
        self.node_degree_log_degree = np.zeros(graph.n_nodes, dtype=float)

        positive = graph.degrees > eps
        self.node_degree_log_degree[positive] = graph.degrees[positive] * np.log2(graph.degrees[positive])

        for node, cluster in enumerate(self.labels):
            cid = int(cluster)
            self.volume[cid] += graph.degrees[node]
            self.degree_log_degree[cid] += self.node_degree_log_degree[node]
            self.size[cid] += 1
            self.active[cid] = True

        internal_twice = np.zeros(self.capacity, dtype=float)
        for node in range(graph.n_nodes):
            cid = int(self.labels[node])
            for nbr, weight in zip(graph.neighbors[node], graph.weights[node]):
                if self.labels[int(nbr)] == cid:
                    internal_twice[cid] += float(weight)
        self.cut = self.volume - internal_twice
        self.cut[np.abs(self.cut) < 1e-10] = 0.0
        self.entropy = float(sum(self.cluster_entropy(cid) for cid in np.flatnonzero(self.active)))

    def clone(self) -> "IncrementalSEState":
        other = object.__new__(IncrementalSEState)
        other.graph = self.graph
        other.eps = self.eps
        other.labels = self.labels.copy()
        other.capacity = self.capacity
        other.volume = self.volume.copy()
        other.cut = self.cut.copy()
        other.degree_log_degree = self.degree_log_degree.copy()
        other.size = self.size.copy()
        other.active = self.active.copy()
        other.node_degree_log_degree = self.node_degree_log_degree
        other.entropy = self.entropy
        return other

    def cluster_entropy_values(self, volume: float, cut: float, degree_log_degree: float) -> float:
        graph_volume = self.graph.volume
        if graph_volume <= self.eps or volume <= self.eps:
            return 0.0
        cut = max(float(cut), 0.0)
        boundary = -(cut / graph_volume) * math.log2(max(volume / graph_volume, self.eps))
        internal = -((degree_log_degree - volume * math.log2(max(volume, self.eps))) / graph_volume)
        return float(boundary + internal)

    def cluster_entropy(self, cluster: int) -> float:
        return self.cluster_entropy_values(
            self.volume[cluster],
            self.cut[cluster],
            self.degree_log_degree[cluster],
        )

    def active_clusters(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    def first_empty_cluster(self) -> int:
        empty = np.flatnonzero(~self.active)
        if empty.size == 0:
            raise RuntimeError("incremental state ran out of cluster ids")
        return int(empty[0])

    def edge_weight_to_cluster(self, node: int, cluster: int) -> float:
        total = 0.0
        for nbr, weight in zip(self.graph.neighbors[node], self.graph.weights[node]):
            if int(self.labels[int(nbr)]) == int(cluster):
                total += float(weight)
        return total

    def neighboring_clusters(self, node: int) -> set[int]:
        clusters = {int(self.labels[node])}
        for nbr in self.graph.neighbors[node]:
            clusters.add(int(self.labels[int(nbr)]))
        return clusters

    def candidate_clusters(self, node: int, allow_new_cluster: bool = True) -> list[int]:
        candidates = self.neighboring_clusters(node)
        if allow_new_cluster:
            candidates.add(self.first_empty_cluster())
        return sorted(candidates)

    def move_delta(self, node: int, target: int) -> float:
        source = int(self.labels[node])
        target = int(target)
        if source == target:
            return 0.0
        if target >= self.capacity:
            raise ValueError("target cluster id is outside state capacity")

        degree = float(self.graph.degrees[node])
        node_dlogd = float(self.node_degree_log_degree[node])
        source_before = self.cluster_entropy(source)
        target_before = self.cluster_entropy(target) if self.active[target] else 0.0

        w_source = self.edge_weight_to_cluster(node, source)
        w_target = self.edge_weight_to_cluster(node, target) if self.active[target] else 0.0

        source_after = self.cluster_entropy_values(
            self.volume[source] - degree,
            self.cut[source] - degree + 2.0 * w_source,
            self.degree_log_degree[source] - node_dlogd,
        )
        target_after = self.cluster_entropy_values(
            self.volume[target] + degree,
            self.cut[target] + degree - 2.0 * w_target,
            self.degree_log_degree[target] + node_dlogd,
        )
        return float(source_after + target_after - source_before - target_before)

    def apply_move(self, node: int, target: int) -> float:
        delta = self.move_delta(node, target)
        source = int(self.labels[node])
        target = int(target)
        if source == target:
            return 0.0

        degree = float(self.graph.degrees[node])
        node_dlogd = float(self.node_degree_log_degree[node])
        w_source = self.edge_weight_to_cluster(node, source)
        w_target = self.edge_weight_to_cluster(node, target) if self.active[target] else 0.0

        self.volume[source] -= degree
        self.cut[source] = self.cut[source] - degree + 2.0 * w_source
        self.degree_log_degree[source] -= node_dlogd
        self.size[source] -= 1
        if self.size[source] <= 0:
            self.volume[source] = 0.0
            self.cut[source] = 0.0
            self.degree_log_degree[source] = 0.0
            self.size[source] = 0
            self.active[source] = False

        self.active[target] = True
        self.volume[target] += degree
        self.cut[target] = self.cut[target] + degree - 2.0 * w_target
        self.degree_log_degree[target] += node_dlogd
        self.size[target] += 1
        self.labels[node] = target
        self.entropy += delta
        if abs(self.entropy) < 1e-12:
            self.entropy = 0.0
        return delta

    def canonical_labels(self) -> np.ndarray:
        return canonicalize_labels(self.labels)

    def validate_entropy(self, adj: np.ndarray, atol: float = 1e-8) -> None:
        full = structural_entropy(adj, self.canonical_labels())
        if not math.isclose(full, self.entropy, abs_tol=atol, rel_tol=atol):
            raise AssertionError(f"incremental entropy {self.entropy} != full entropy {full}")


def local_move_incremental(
    adj: np.ndarray,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    allow_new_cluster: bool = True,
) -> tuple[np.ndarray, float]:
    """Run sparse incremental node-move SE local search."""

    graph = SparseGraph.from_adjacency(adj)
    state = IncrementalSEState(graph, init_labels)
    rng = np.random.default_rng(seed)

    for _ in range(max_passes):
        changed = False
        for node in rng.permutation(graph.n_nodes):
            current = int(state.labels[node])
            best_target = current
            best_delta = 0.0
            for target in state.candidate_clusters(int(node), allow_new_cluster=allow_new_cluster):
                if target == current:
                    continue
                delta = state.move_delta(int(node), int(target))
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_target = int(target)
            if best_target != current:
                state.apply_move(int(node), best_target)
                changed = True
        if not changed:
            break

    labels = state.canonical_labels()
    # Canonicalization can change ids but not the partition; score exactly once
    # to remove tiny floating drift before returning a public result.
    entropy = structural_entropy(adj, labels)
    return labels, entropy


def multistart_incremental_se_heuristic(
    adj: np.ndarray,
    starts: int = 8,
    max_passes: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Run dependency-light scalable SE local search from several starts."""

    n_nodes = int(np.asarray(adj).shape[0])
    rng = np.random.default_rng(seed)
    seeds = [
        np.arange(n_nodes, dtype=np.int32),
        np.zeros(n_nodes, dtype=np.int32),
    ]
    for _ in range(max(0, starts - len(seeds))):
        k = int(rng.integers(2, max(3, min(n_nodes, int(math.sqrt(max(n_nodes, 2))) + 2))))
        seeds.append(rng.integers(0, k, size=n_nodes, dtype=np.int32))

    best_labels: np.ndarray | None = None
    best_entropy = float("inf")
    for i, labels in enumerate(seeds):
        candidate_labels, entropy = local_move_incremental(
            adj,
            init_labels=labels,
            max_passes=max_passes,
            seed=seed + i,
            allow_new_cluster=True,
        )
        if entropy < best_entropy:
            best_entropy = entropy
            best_labels = candidate_labels
    assert best_labels is not None
    return best_labels, best_entropy
