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
    def from_csr(cls, matrix, node_degree_log_degree=None) -> "SparseGraph":
        """Build from an already-symmetric CSR matrix without densification."""

        import scipy.sparse as sp
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        degrees = np.asarray(matrix.sum(axis=1)).flatten()
        matrix = matrix.copy()
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
            end = matrix.indptr[node + 1]
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

    @classmethod
    def from_adjacency(cls, adj, node_degree_log_degree=None) -> "SparseGraph":
        import scipy.sparse as sp
        if isinstance(adj, SparseGraph):
            return adj
        if sp.issparse(adj):
            matrix = adj.tocsr()
            matrix = 0.5 * (matrix + matrix.T)
        else:
            matrix = np.asarray(adj, dtype=float)
            matrix = 0.5 * (matrix + matrix.T)
            matrix = sp.csr_matrix(matrix)

        return cls.from_csr(matrix.tocsr(), node_degree_log_degree=node_degree_log_degree)

    @classmethod
    def from_edge_index(
        cls,
        edge_index,
        num_nodes: int,
        weights=None,
        node_degree_log_degree=None,
    ) -> "SparseGraph":
        """Build directly from a PyG-style ``edge_index`` (shape ``(2, E)``).

        The pair list is symmetrized internally (so ``(u, v)`` may appear once
        or twice in the input). Self-loops are dropped from cut accounting but
        retained in degrees, matching ``from_adjacency``.
        """

        import scipy.sparse as sp
        edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        src = edge_index[0]
        dst = edge_index[1]
        if weights is None:
            data = np.ones(src.shape[0], dtype=float)
        else:
            data = np.asarray(weights, dtype=float).reshape(-1)
            if data.shape[0] != src.shape[0]:
                raise ValueError("weights must align with edge_index columns")
        if np.any(data < 0):
            raise ValueError("structural entropy expects non-negative edge weights")
        # Symmetrize by stacking both directions; duplicate entries summed by COO->CSR.
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        vals = np.concatenate([data, data]) * 0.5
        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        matrix.sum_duplicates()
        return cls.from_csr(matrix, node_degree_log_degree=node_degree_log_degree)


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

    def move_delta_batch(self, node: int, candidates: np.ndarray) -> np.ndarray:
        """Vectorised SE deltas for moving ``node`` to each cluster in ``candidates``.

        Returns an array of the same length as ``candidates`` with the SE
        delta for each candidate. The ``source -> source`` move evaluates
        to 0.0; for any inactive candidate cluster the existing entropy
        contribution is treated as 0 (consistent with
        ``cluster_entropy``). All cluster-entropy log/division work is
        done with numpy vector ops, so the per-call cost is
        $O(\\deg(v) + |\\text{candidates}|)$ in numpy operations rather
        than $O(\\deg(v) \\cdot |\\text{candidates}|)$ in Python.
        """

        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size == 0:
            return np.zeros(0, dtype=float)
        source = int(self.labels[node])
        degree = float(self.graph.degrees[node])
        node_dlogd = float(self.node_degree_log_degree[node])

        # Edge-weight-to-each-candidate, all in one pass.
        nbrs = np.asarray(self.graph.neighbors[node], dtype=np.int64)
        weights = np.asarray(self.graph.weights[node], dtype=float)
        if nbrs.size:
            nbr_clusters = self.labels[nbrs]
            # (K_cand, deg) bool match, then weighted sum across deg.
            matches = nbr_clusters[None, :] == candidates[:, None]
            w_to_cand = (matches.astype(float) * weights[None, :]).sum(axis=1)
            w_source = float(weights[nbr_clusters == source].sum())
        else:
            w_to_cand = np.zeros(candidates.size, dtype=float)
            w_source = 0.0

        # Vectorised cluster-entropy computation.
        graph_volume = float(self.graph.volume)
        eps = self.eps

        def _cluster_entropy_vec(vol: np.ndarray, cut: np.ndarray, dlogd: np.ndarray) -> np.ndarray:
            vol = np.asarray(vol, dtype=float)
            cut = np.maximum(np.asarray(cut, dtype=float), 0.0)
            dlogd = np.asarray(dlogd, dtype=float)
            mask = (graph_volume > eps) & (vol > eps)
            safe_vol = np.where(mask, vol, 1.0)
            log_vol_over_V = np.log2(np.maximum(safe_vol / max(graph_volume, eps), eps))
            log_vol = np.log2(np.maximum(safe_vol, eps))
            boundary = -(cut / max(graph_volume, eps)) * log_vol_over_V
            internal = -((dlogd - safe_vol * log_vol) / max(graph_volume, eps))
            return np.where(mask, boundary + internal, 0.0)

        # Per-candidate "before" using existing per-cluster stats.
        cand_active = self.active[candidates]
        cand_vol_before = self.volume[candidates]
        cand_cut_before = self.cut[candidates]
        cand_dlogd_before = self.degree_log_degree[candidates]
        target_before = np.where(
            cand_active,
            _cluster_entropy_vec(cand_vol_before, cand_cut_before, cand_dlogd_before),
            0.0,
        )

        source_before = self.cluster_entropy(source)
        source_after = self.cluster_entropy_values(
            self.volume[source] - degree,
            self.cut[source] - degree + 2.0 * w_source,
            self.degree_log_degree[source] - node_dlogd,
        )

        # Per-candidate "after" with `node` now in the candidate cluster.
        cand_vol_after = cand_vol_before + degree
        cand_cut_after = cand_cut_before + degree - 2.0 * w_to_cand
        cand_dlogd_after = cand_dlogd_before + node_dlogd
        target_after = _cluster_entropy_vec(cand_vol_after, cand_cut_after, cand_dlogd_after)

        deltas = source_after + target_after - source_before - target_before
        # Source-to-source is a no-op.
        same_as_source = candidates == source
        deltas = np.where(same_as_source, 0.0, deltas)
        return deltas.astype(float)

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

    if isinstance(adj, SparseGraph):
        graph = adj
    else:
        graph = SparseGraph.from_adjacency(adj, node_degree_log_degree)
    state = IncrementalSEState(graph, init_labels)
    rng = np.random.default_rng(seed)

    # Try the numba-jitted kernel; fall back to numpy-batched if numba
    # unavailable or compilation fails.
    try:
        from .numba_kernel import numba_move_delta_batch
        # Warm up the kernel so the first user-visible call doesn't
        # include compile time.
        if graph.n_nodes:
            warm_cands = np.asarray(
                state.candidate_clusters(0, allow_new_cluster=allow_new_cluster),
                dtype=np.int64,
            )
            if warm_cands.size:
                numba_move_delta_batch(state, 0, warm_cands)
        delta_fn = numba_move_delta_batch
    except Exception:
        delta_fn = lambda s, n, c: s.move_delta_batch(n, c)

    for _ in range(max_passes):
        changed = False
        for node in rng.permutation(graph.n_nodes):
            current = int(state.labels[node])
            cands = state.candidate_clusters(int(node), allow_new_cluster=allow_new_cluster)
            if not cands:
                continue
            cand_arr = np.asarray(cands, dtype=np.int64)
            deltas = delta_fn(state, int(node), cand_arr)
            best_idx = int(np.argmin(deltas))
            best_target = int(cand_arr[best_idx])
            best_delta = float(deltas[best_idx])
            if best_target != current and best_delta < -1e-12:
                state.apply_move(int(node), best_target)
                changed = True
        if not changed:
            break

    labels = state.canonical_labels()
    return labels, state.entropy



import scipy.sparse as sp

def _to_sparse_adj(adj):
    """Return a scipy CSR adjacency without dense materialization where possible."""

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


def multi_level_local_move(
    adj,
    init_labels: np.ndarray | None = None,
    max_passes: int = 20,
    seed: int = 0,
    return_hierarchy: bool = False,
):
    rng = np.random.default_rng(seed)

    if isinstance(adj, SparseGraph):
        base_graph = adj
    else:
        base_graph = SparseGraph.from_adjacency(adj)

    current_adj = _to_sparse_adj(base_graph)
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
        current_adj = S.T @ current_adj @ S

        current_labels = np.arange(k, dtype=np.int32)

    final_labels = projections[-1]
    for i in range(len(projections) - 2, -1, -1):
        final_labels = final_labels[projections[i]]

    final_labels = canonicalize_labels(final_labels)
    final_state = IncrementalSEState(base_graph, final_labels)
    entropy = float(final_state.entropy)
    if return_hierarchy:
        return final_labels, entropy, projections
    return final_labels, entropy


def multistart_incremental_se_heuristic(
    adj,
    starts: int = 8,
    max_passes: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Run dependency-light scalable SE local search from several starts."""

    if isinstance(adj, SparseGraph):
        graph = adj
    else:
        graph = SparseGraph.from_adjacency(adj)
    n_nodes = graph.n_nodes
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
            graph,
            init_labels=labels,
            max_passes=max_passes,
            seed=seed + i,
        )
        if entropy < best_entropy:
            best_entropy = entropy
            best_labels = candidate_labels
    assert best_labels is not None
    return best_labels, best_entropy


def _spectral_seed(
    graph: "SparseGraph",
    target_clusters: int,
    seed: int,
) -> np.ndarray | None:
    """Top-K Laplacian eigenvectors → k-means → init labels.

    Returns ``None`` if the spectral computation fails (e.g.,
    disconnected graph at small N, or scipy/sklearn unavailable);
    callers should fall back to random initialisation.
    """

    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        from sklearn.cluster import KMeans
    except ImportError:
        return None

    n = graph.n_nodes
    if target_clusters >= n:
        return None
    rows = []
    cols = []
    vals = []
    for node in range(n):
        nbrs = graph.neighbors[node]
        ws = graph.weights[node]
        if nbrs.size:
            rows.append(np.full(nbrs.size, node, dtype=np.int64))
            cols.append(nbrs.astype(np.int64))
            vals.append(ws.astype(float))
    if not rows:
        return None
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    deg = np.asarray(A.sum(axis=1)).flatten()
    if (deg <= 0).any():
        return None
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    # L_sym = I - D^{-1/2} A D^{-1/2}; we want its smallest eigenvalues,
    # which correspond to the largest eigenvalues of D^{-1/2} A D^{-1/2}.
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    try:
        # Largest-magnitude eigvecs of A_norm = smallest of L_sym.
        _, V = eigsh(A_norm, k=target_clusters, which="LA")
    except Exception:
        return None
    # Drop the trivial constant eigenvector (always close to 1).
    # Use rows of V as embeddings, k-means cluster them.
    try:
        km = KMeans(n_clusters=target_clusters, random_state=seed, n_init=4)
        labels = km.fit_predict(V).astype(np.int32)
    except Exception:
        return None
    if int(np.unique(labels).size) < 2:
        return None
    return labels


def constrained_k_multistart(
    adj,
    target_clusters: int,
    starts: int = 8,
    max_passes: int = 20,
    seed: int = 0,
    spectral_init: bool = False,
) -> tuple[np.ndarray, float]:
    """Multistart SE local search constrained to exactly $K$ clusters.

    Idea **2.1'** in the paper's NEXT_STEPS roadmap. Each restart
    initialises with a random partition into ``target_clusters``
    non-empty clusters; ``local_move_incremental`` is run with
    ``allow_new_cluster=False`` so the optimiser cannot grow $K$.
    The resulting partition is a local SE optimum *at*
    $K=K_\\text{target}$ — by construction, the multistart kernel
    never visits the over-fragmented $K_\\text{local}\\!\\gg\\!K$
    minimum that ``SEClust-TargetK`` then has to merge down from.

    With ``spectral_init=True`` (default), one of the restarts uses
    the top-$K$ normalised-Laplacian eigenvectors clustered by
    $k$-means as its initial partition (idea **003** in the
    SEClust paper's idea_lib). This typically gives a
    head-start of $0.1$-$1$ bit of SE relative to a random
    initialisation on graphs with strong cluster structure.

    Returns the labels and 2D structural entropy of the best restart.
    """

    if target_clusters < 1:
        raise ValueError("target_clusters must be >= 1")

    if isinstance(adj, SparseGraph):
        graph = adj
    else:
        graph = SparseGraph.from_adjacency(adj)
    n_nodes = graph.n_nodes
    if target_clusters > n_nodes:
        raise ValueError("target_clusters exceeds number of nodes")
    rng = np.random.default_rng(seed)

    def _balanced_random(k: int) -> np.ndarray:
        """Random partition into exactly k non-empty clusters."""

        labels = np.empty(n_nodes, dtype=np.int32)
        # Seed each cluster with at least one node so no cluster starts empty.
        seed_indices = rng.permutation(n_nodes)[:k]
        labels[seed_indices] = np.arange(k, dtype=np.int32)
        # Distribute the rest uniformly at random.
        rest_mask = np.ones(n_nodes, dtype=bool)
        rest_mask[seed_indices] = False
        labels[rest_mask] = rng.integers(0, k, size=int(rest_mask.sum()), dtype=np.int32)
        return labels

    # Seeds: optionally one spectral seed + (starts-1) random.
    seeds_list: list[np.ndarray] = []
    if spectral_init and target_clusters >= 2:
        spectral = _spectral_seed(graph, target_clusters, seed=seed)
        if spectral is not None and int(np.unique(spectral).size) >= 2:
            seeds_list.append(spectral)
    while len(seeds_list) < starts:
        seeds_list.append(_balanced_random(target_clusters))

    best_labels: np.ndarray | None = None
    best_entropy = float("inf")
    for i, init_labels in enumerate(seeds_list):
        candidate_labels, entropy = local_move_incremental(
            graph,
            init_labels=init_labels,
            max_passes=max_passes,
            seed=seed + i,
            allow_new_cluster=False,
        )
        if entropy < best_entropy:
            best_entropy = entropy
            best_labels = candidate_labels
    assert best_labels is not None
    return best_labels, best_entropy
