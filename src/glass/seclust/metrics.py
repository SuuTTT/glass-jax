"""Hierarchical-clustering quality metrics.

Implements the metrics that the HCSE [Pan, Zheng, Fan 2021] and HypCSE
[Zeng et al. 2025] papers report against — Dasgupta cost (canonical
hierarchical-clustering objective; Dasgupta STOC 2016) and dendrogram
purity (Heller & Ghahramani 2005, formalised in HypCSE Eq. 19) — so
that SEClust-Tree can be scored on the metrics the SE-family papers
use rather than only on flat ACC / NMI / ARI.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


def _adj_as_csr(adj) -> sp.csr_matrix:
    if sp.issparse(adj):
        return adj.tocsr()
    if hasattr(adj, "n_nodes") and hasattr(adj, "neighbors"):
        n = adj.n_nodes
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
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
            return sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
        return sp.csr_matrix((n, n))
    return sp.csr_matrix(np.asarray(adj, dtype=float))


def hierarchy_to_lca_size(
    levels: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Given a sequence of cluster-label vectors at successive merge
    levels (finest first), build a dense lookup that gives, for any
    pair of nodes (i, j), the size of the smallest cluster containing
    both.

    Returns (lca_size, leaves_per_level): lca_size is an N-by-N int
    array (N = number of leaves); for trees with up to a few thousand
    leaves this is the simplest correct implementation.
    """

    if not levels:
        raise ValueError("levels must contain at least one labelling")
    n = int(np.asarray(levels[0]).size)
    leaves_per_level = np.zeros(len(levels), dtype=np.int64)
    lca = np.full((n, n), n, dtype=np.int64)  # default: root cluster
    # Walk levels from finest (largest K, smallest clusters) to coarsest.
    for li, lbls in enumerate(levels):
        lbls = np.asarray(lbls, dtype=np.int64)
        K = int(lbls.max()) + 1 if lbls.size else 0
        leaves_per_level[li] = K
        # For each cluster at this level, mark its size as the LCA cluster
        # size for every pair of nodes inside it that haven't already been
        # given a finer LCA size.
        for c in range(K):
            members = np.where(lbls == c)[0]
            if members.size < 2:
                continue
            size = int(members.size)
            block = lca[np.ix_(members, members)]
            # Take the minimum (= smallest cluster containing both).
            block = np.minimum(block, size)
            lca[np.ix_(members, members)] = block
    return lca, leaves_per_level


def dasgupta_cost(
    adj,
    levels: Sequence[np.ndarray],
) -> float:
    """Dasgupta cost of a hierarchy on a similarity graph.

    Definition (Dasgupta, STOC 2016):
        cost(T) = sum over edges (i, j) of  w_{ij} * |LCA(i, j)|
    where |LCA(i, j)| is the number of leaves below the lowest common
    ancestor of i and j in the hierarchy. Smaller is better.

    ``levels`` is a sequence of label vectors over the leaves; element
    [0] is the finest (largest K) labelling, element [-1] is the
    coarsest. We score against the cluster sizes induced by the
    levels.
    """

    csr = _adj_as_csr(adj)
    n = csr.shape[0]
    if n == 0:
        return 0.0
    lca, _ = hierarchy_to_lca_size(levels)
    coo = csr.tocoo()
    cost = 0.0
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i == j:
            continue
        cost += float(w) * float(lca[int(i), int(j)])
    return float(cost)


def dendrogram_purity(
    levels: Sequence[np.ndarray],
    labels: np.ndarray,
) -> float:
    """Dendrogram purity (Heller & Ghahramani 2005).

        DP(T) = (1 / |P|) * sum over (i, j) with labels[i] == labels[j]:
                 |{ k in LCA(i, j) leaves : labels[k] == labels[i] }| / |LCA(i, j)|

    where P is the set of unordered pairs (i, j) sharing a class.

    Higher is better; perfect hierarchy yields DP = 1.0.
    """

    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.size)
    if n < 2:
        return 1.0

    lca, _ = hierarchy_to_lca_size(levels)

    # For each level, we also need the LCA *cluster id* per pair so we
    # can count same-class members inside it. The simplest way is to
    # carry an LCA-cluster-id grid alongside lca_size; but we can be
    # cheaper: at each level, compute for each cluster the per-class
    # counts and use them when that level happens to be the LCA.
    classes = np.unique(labels)

    # Build LCA-level-index per pair: which level's cluster is the LCA.
    # Walk finest-first; pair gets level li the first time both its
    # endpoints share a cluster at that level.
    level_idx = np.full((n, n), len(levels) - 1, dtype=np.int64)
    assigned = np.zeros((n, n), dtype=bool)
    for li in range(len(levels)):
        lbls = np.asarray(levels[li], dtype=np.int64)
        K = int(lbls.max()) + 1 if lbls.size else 0
        for c in range(K):
            members = np.where(lbls == c)[0]
            if members.size < 2:
                continue
            block = assigned[np.ix_(members, members)]
            new_pairs = ~block
            sub_idx = level_idx[np.ix_(members, members)]
            sub_idx = np.where(new_pairs, li, sub_idx)
            level_idx[np.ix_(members, members)] = sub_idx
            new_assigned = np.ones_like(block, dtype=bool)
            assigned[np.ix_(members, members)] = block | new_assigned

    # Per (level, cluster, class) count of same-class members.
    same_class_count: dict[tuple[int, int, int], int] = {}
    for li, lbls in enumerate(levels):
        lbls = np.asarray(lbls, dtype=np.int64)
        K = int(lbls.max()) + 1 if lbls.size else 0
        for c in range(K):
            mask = lbls == c
            members = np.where(mask)[0]
            if members.size == 0:
                continue
            cls_labels = labels[members]
            for cls in classes:
                same_class_count[(li, c, int(cls))] = int(np.sum(cls_labels == cls))

    same_class_pairs = 0
    weighted_sum = 0.0
    for cls in classes:
        idx = np.where(labels == cls)[0]
        if idx.size < 2:
            continue
        for ii in range(idx.size):
            for jj in range(ii + 1, idx.size):
                i = int(idx[ii])
                j = int(idx[jj])
                li = int(level_idx[i, j])
                cluster_id = int(np.asarray(levels[li])[i])
                lca_size = int(lca[i, j])
                same_class = same_class_count.get((li, cluster_id, int(cls)), 0)
                weighted_sum += same_class / max(lca_size, 1)
                same_class_pairs += 1

    return float(weighted_sum / max(same_class_pairs, 1))


def cluster_labels_to_singleton_levels(
    labels_seq: Iterable[np.ndarray],
) -> list[np.ndarray]:
    """Convenience: take a sequence of partitions and prepend a
    singleton level if the finest is not already singletons."""

    levels = list(np.asarray(l, dtype=np.int64) for l in labels_seq)
    if not levels:
        return levels
    finest = levels[0]
    if int(np.unique(finest).size) < int(finest.size):
        levels.insert(0, np.arange(int(finest.size), dtype=np.int64))
    return levels
