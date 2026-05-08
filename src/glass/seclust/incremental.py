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
    node_cuts: np.ndarray
    node_degree_log_degree: np.ndarray
    volume: float
    n_nodes: int
    n_edges: int

    @classmethod
    def from_adjacency(cls, adj, node_degree_log_degree=None) -> "SparseGraph":
        import scipy.sparse as sp
        if sp.issparse(adj):
            matrix = adj.tocsr()
            matrix = 0.5 * (matrix + matrix.T)
        else:
            matrix = np.asarray(adj, dtype=float)
            matrix = 0.5 * (matrix + matrix.T)
            matrix = sp.csr_matrix(matrix)
            
        degrees = np.asarray(matrix.sum(axis=1)).flatten()
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()
        node_cuts = np.asarray(matrix.sum(axis=1)).flatten()
        
        if node_degree_log_degree is None:
            node_degree_log_degree = np.zeros_like(degrees)
            positive = degrees > 1e-12
            node_degree_log_degree[positive] = degrees[positive] * np.log2(degrees[positive])
        
        neighbors = []
        weights = []
        edge_count = 0
        for node in range(matrix.shape[0]):
            start = matrix.indptr[node]
            end = matrix.indptr[node+1]
            idx = matrix.indices[start:end]
            vals = matrix.data[start:end]
            
            neighbors.append(idx.astype(np.int32, copy=False))
            weights.append(vals.astype(float, copy=False))
            edge_count += int(np.sum(idx > node))
            
        return cls(
            neighbors=tuple(neighbors),
            weights=tuple(weights),
            degrees=degrees.astype(float, copy=False),
            node_cuts=node_cuts.astype(float, copy=False),
            node_degree_log_degree=np.asarray(node_degree_log_degree, dtype=float),
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
        self.node_degree_log_degree = graph.node_degree_log_degree.copy()

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
        cluster_node_cuts = np.zeros(self.capacity, dtype=float)
        for node in range(graph.n_nodes):
            cid = int(self.labels[node])
            cluster_node_cuts[cid] += graph.node_cuts[node]
        self.cut = cluster_node_cuts - internal_twice
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

    def merge_delta(self, left: int, right: int, weight_between: float) -> float:
        """Calculate flat SE delta for merging two clusters."""
        if left == right:
            return 0.0
            
        vol_L = self.volume[left]
        cut_L = self.cut[left]
        dlogd_L = self.degree_log_degree[left]
        
        vol_R = self.volume[right]
        cut_R = self.cut[right]
        dlogd_R = self.degree_log_degree[right]
        
        new_vol = vol_L + vol_R
        new_cut = cut_L + cut_R - 2.0 * weight_between
        new_dlogd = dlogd_L + dlogd_R
        
        old_entropy_L = self.cluster_entropy_values(vol_L, cut_L, dlogd_L)
        old_entropy_R = self.cluster_entropy_values(vol_R, cut_R, dlogd_R)
        new_entropy = self.cluster_entropy_values(new_vol, new_cut, new_dlogd)
        
        return new_entropy - old_entropy_L - old_entropy_R

    def apply_merge(self, left: int, right: int, weight_between: float) -> float:
        """Apply a merge and return the entropy change."""
        delta = self.merge_delta(left, right, weight_between)
        
        self.volume[left] += self.volume[right]
        self.cut[left] += self.cut[right] - 2.0 * weight_between
        self.degree_log_degree[left] += self.degree_log_degree[right]
        self.size[left] += self.size[right]
        
        self.volume[right] = 0.0
        self.cut[right] = 0.0
        self.degree_log_degree[right] = 0.0
        self.size[right] = 0
        self.active[right] = False
        
        self.labels[self.labels == right] = left
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
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    allow_new_cluster: bool = True,
    node_degree_log_degree=None,
) -> tuple[np.ndarray, float]:
    """Run sparse incremental node-move SE local search."""

    graph = SparseGraph.from_adjacency(adj, node_degree_log_degree)
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
    return labels, state.entropy



import scipy.sparse as sp

def multi_level_local_move(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    return_hierarchy: bool = False,
):
    rng = np.random.default_rng(seed)
    
    current_adj = adj
    current_labels = init_labels
    current_dlogd = None
    
    projections = []
    
    while True:
        labels, _ = local_move_incremental(
            current_adj,
            node_degree_log_degree=current_dlogd,
            init_labels=current_labels,
            max_passes=max_passes,
            seed=int(rng.integers(1<<30)),
            allow_new_cluster=True,
        )
        
        k = int(labels.max()) + 1
        if k == current_adj.shape[0] or k == 1:
            projections.append(labels)
            break
            
        import scipy.sparse.csgraph as csgraph
        new_labels = np.zeros_like(labels)
        next_label = 0
        for i in range(k):
            mask = (labels == i)
            if not np.any(mask): continue
            sub_adj = current_adj[mask][:, mask]
            n_comp, comp_labels = csgraph.connected_components(sub_adj, directed=False)
            new_labels[mask] = comp_labels + next_label
            next_label += n_comp
            
        labels = canonicalize_labels(new_labels)
        k = int(labels.max()) + 1
        
        projections.append(labels)
        
        row = np.arange(len(labels))
        col = labels
        data = np.ones(len(labels), dtype=current_adj.dtype)
        S = sp.csr_matrix((data, (row, col)), shape=(len(labels), k))
        A_sparse = sp.csr_matrix(current_adj)
        current_adj = S.T @ A_sparse @ S
        
        current_labels = np.arange(k, dtype=np.int32)
        
    final_labels = projections[-1]
    for i in range(len(projections) - 2, -1, -1):
        final_labels = final_labels[projections[i]]
        
    final_labels = canonicalize_labels(final_labels)
    entropy = structural_entropy(adj, final_labels)
    if return_hierarchy:
        return final_labels, entropy, projections
    return final_labels, entropy


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
        candidate_labels, entropy = multi_level_local_move(
            adj,
            init_labels=labels,
            max_passes=max_passes,
            seed=seed + i,
        )
        if entropy < best_entropy:
            best_entropy = entropy
            best_labels = candidate_labels
    assert best_labels is not None
    return best_labels, best_entropy
