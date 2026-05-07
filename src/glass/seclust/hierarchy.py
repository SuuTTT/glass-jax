"""High-level hierarchical structural entropy clustering."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .entropy import canonicalize_labels, structural_entropy
from .incremental import multistart_incremental_se_heuristic


@dataclass(frozen=True)
class HierarchicalLevel:
    """One flat cut through the hierarchical merge process."""

    k: int
    labels: np.ndarray
    entropy: float
    tree_entropy: float | None = None


@dataclass(frozen=True)
class HierarchicalClusteringResult:
    """Result from the high-level SEClust hierarchy."""

    labels: np.ndarray
    entropy: float
    method: str
    levels: tuple[HierarchicalLevel, ...]
    base_labels: np.ndarray
    target_clusters: int | None


def _cluster_pairs_with_edges(adj: np.ndarray, labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    for left_node, right_node in zip(rows, cols):
        left = int(labels[int(left_node)])
        right = int(labels[int(right_node)])
        if left != right:
            pairs.add((min(left, right), max(left, right)))
    return pairs


def _all_cluster_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    clusters = np.unique(labels).astype(int)
    pairs: set[tuple[int, int]] = set()
    for i, left in enumerate(clusters):
        for right in clusters[i + 1 :]:
            pairs.add((int(left), int(right)))
    return pairs


def _merge_pair(labels: np.ndarray, left: int, right: int) -> np.ndarray:
    merged = labels.copy()
    merged[merged == right] = left
    return canonicalize_labels(merged)


@dataclass
class _TreeNode:
    node_id: int
    volume: float
    cut: float
    base_modules: frozenset[int]
    left: int | None = None
    right: int | None = None


def _term(cut: float, volume: float, parent_volume: float, graph_volume: float, eps: float = 1e-12) -> float:
    if graph_volume <= eps or volume <= eps or parent_volume <= eps:
        return 0.0
    return float(-(max(cut, 0.0) / graph_volume) * math.log2(max(volume / parent_volume, eps)))


def _initial_module_stats(adj: np.ndarray, base_labels: np.ndarray):
    labels = canonicalize_labels(base_labels)
    degrees = adj.sum(axis=1)
    graph_volume = float(degrees.sum())
    n_modules = int(labels.max()) + 1 if labels.size else 0
    volumes = np.zeros(n_modules, dtype=float)
    cuts = np.zeros(n_modules, dtype=float)
    fixed_leaf_entropy = 0.0
    for module in range(n_modules):
        mask = labels == module
        volume = float(degrees[mask].sum())
        internal = float(adj[np.ix_(mask, mask)].sum())
        cut = max(volume - internal, 0.0)
        volumes[module] = volume
        cuts[module] = cut
        positive_degrees = degrees[mask]
        positive_degrees = positive_degrees[positive_degrees > 1e-12]
        if graph_volume > 1e-12 and volume > 1e-12 and positive_degrees.size:
            fixed_leaf_entropy += float(
                -np.sum((positive_degrees / graph_volume) * np.log2(positive_degrees / volume))
            )

    between = np.zeros((max(n_modules, 1), max(n_modules, 1)), dtype=float)
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    for row, col in zip(rows, cols):
        left = int(labels[int(row)])
        right = int(labels[int(col)])
        if left != right:
            between[left, right] += float(adj[int(row), int(col)])
            between[right, left] += float(adj[int(row), int(col)])
    return labels, degrees, graph_volume, volumes, cuts, between, fixed_leaf_entropy


def _labels_from_module_assignment(base_labels: np.ndarray, module_assignment: dict[int, int]) -> np.ndarray:
    labels = np.array([module_assignment[int(module)] for module in base_labels], dtype=np.int32)
    return canonicalize_labels(labels)


def high_dimensional_tree_entropy(
    adj: np.ndarray,
    base_labels: np.ndarray,
    parent: dict[int, int],
    nodes: dict[int, _TreeNode],
    root_ids: set[int] | None = None,
) -> float:
    """Compute high-dimensional SE for a coding tree over base modules."""

    _, _, graph_volume, _, _, _, fixed_leaf_entropy = _initial_module_stats(adj, base_labels)
    total = fixed_leaf_entropy
    root_volume = graph_volume
    if root_ids is None:
        root_ids = {node_id for node_id in nodes if node_id not in parent}
    for node_id, node in nodes.items():
        if node_id in parent:
            parent_volume = nodes[parent[node_id]].volume
        elif node_id in root_ids:
            parent_volume = root_volume
        else:
            continue
        total += _term(node.cut, node.volume, parent_volume, graph_volume)
    return float(total)


def coding_tree_hierarchy_levels(
    adj: np.ndarray,
    base_labels: np.ndarray,
    min_clusters: int = 1,
) -> tuple[HierarchicalLevel, ...]:
    """Build a hierarchy using high-dimensional coding-tree merge deltas."""

    base_labels, _, graph_volume, volumes, cuts, between, fixed_leaf_entropy = _initial_module_stats(adj, base_labels)
    if min_clusters < 1:
        raise ValueError("min_clusters must be >= 1")
    n_modules = int(base_labels.max()) + 1 if base_labels.size else 0
    if n_modules == 0:
        return tuple()

    nodes: dict[int, _TreeNode] = {
        module: _TreeNode(module, float(volumes[module]), float(cuts[module]), frozenset({module}))
        for module in range(n_modules)
    }
    parent: dict[int, int] = {}
    active: set[int] = set(range(n_modules))
    next_id = n_modules

    def module_assignment() -> dict[int, int]:
        out: dict[int, int] = {}
        for cluster_idx, node_id in enumerate(sorted(active)):
            for module in nodes[node_id].base_modules:
                out[module] = cluster_idx
        return out

    def current_level() -> HierarchicalLevel:
        labels = _labels_from_module_assignment(base_labels, module_assignment())
        tree_entropy = high_dimensional_tree_entropy(adj, base_labels, parent, nodes, root_ids=set(active))
        return HierarchicalLevel(
            k=int(len(active)),
            labels=labels,
            entropy=structural_entropy(adj, labels),
            tree_entropy=tree_entropy,
        )

    levels = [current_level()]

    while len(active) > min_clusters:
        best_pair = None
        best_delta = float("inf")
        active_list = sorted(active)
        for i, left in enumerate(active_list):
            for right in active_list[i + 1 :]:
                cut_between = 0.0
                for lmod in nodes[left].base_modules:
                    for rmod in nodes[right].base_modules:
                        cut_between += float(between[lmod, rmod])
                new_volume = nodes[left].volume + nodes[right].volume
                new_cut = nodes[left].cut + nodes[right].cut - 2.0 * cut_between
                delta = (
                    _term(new_cut, new_volume, graph_volume, graph_volume)
                    + _term(nodes[left].cut, nodes[left].volume, new_volume, graph_volume)
                    + _term(nodes[right].cut, nodes[right].volume, new_volume, graph_volume)
                    - _term(nodes[left].cut, nodes[left].volume, graph_volume, graph_volume)
                    - _term(nodes[right].cut, nodes[right].volume, graph_volume, graph_volume)
                )
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_pair = (left, right, new_volume, max(new_cut, 0.0))
        if best_pair is None:
            break

        left, right, new_volume, new_cut = best_pair
        nodes[next_id] = _TreeNode(
            next_id,
            float(new_volume),
            float(new_cut),
            nodes[left].base_modules | nodes[right].base_modules,
            left=left,
            right=right,
        )
        parent[left] = next_id
        parent[right] = next_id
        active.remove(left)
        active.remove(right)
        active.add(next_id)
        next_id += 1
        levels.append(current_level())

    # Touch fixed_leaf_entropy so static analysis does not lose the intent: it
    # is included through high_dimensional_tree_entropy at each level.
    _ = fixed_leaf_entropy
    return tuple(levels)


def merge_hierarchy_levels(
    adj: np.ndarray,
    base_labels: np.ndarray,
    min_clusters: int = 1,
) -> tuple[HierarchicalLevel, ...]:
    """Build a merge hierarchy by least flat-SE increase between modules."""

    labels = canonicalize_labels(base_labels)
    if min_clusters < 1:
        raise ValueError("min_clusters must be >= 1")
    levels = [HierarchicalLevel(k=int(len(np.unique(labels))), labels=labels.copy(), entropy=structural_entropy(adj, labels))]

    while len(np.unique(labels)) > min_clusters:
        pairs = _cluster_pairs_with_edges(adj, labels)
        if not pairs:
            pairs = _all_cluster_pairs(labels)
        best_labels = None
        best_entropy = float("inf")
        best_pair = None
        for left, right in pairs:
            candidate = _merge_pair(labels, left, right)
            entropy = structural_entropy(adj, candidate)
            if entropy < best_entropy - 1e-12:
                best_entropy = entropy
                best_labels = candidate
                best_pair = (left, right)
        if best_labels is None or best_pair is None:
            break
        labels = best_labels
        levels.append(HierarchicalLevel(k=int(len(np.unique(labels))), labels=labels.copy(), entropy=best_entropy))

    return tuple(levels)


def select_hierarchy_level(
    levels: tuple[HierarchicalLevel, ...],
    target_clusters: int | None = None,
) -> HierarchicalLevel:
    """Select a flat level from a hierarchy."""

    if not levels:
        raise ValueError("levels must not be empty")
    if target_clusters is not None:
        eligible = [level for level in levels if level.k <= target_clusters]
        if eligible:
            return min(eligible, key=lambda level: (abs(level.k - target_clusters), level.entropy))
        return min(levels, key=lambda level: (abs(level.k - target_clusters), level.entropy))

    if len(levels) <= 2:
        return levels[0]

    # Simple unsupervised fallback: pick the last level before a large entropy
    # jump. This keeps useful coarse structure without forcing one block.
    deltas = np.array([levels[i + 1].entropy - levels[i].entropy for i in range(len(levels) - 1)], dtype=float)
    if np.all(deltas <= 1e-12):
        return min(levels, key=lambda level: level.entropy)
    jump_index = int(np.argmax(deltas))
    return levels[jump_index]


def hierarchical_se_clustering(
    adj: np.ndarray,
    target_clusters: int | None = None,
    base_labels: np.ndarray | None = None,
    starts: int = 6,
    max_passes: int = 10,
    seed: int = 0,
) -> HierarchicalClusteringResult:
    """Build a high-level SE hierarchy and return a selected flat cut.

    The first implementation builds a hierarchy over fast flat SEClust modules.
    It is a practical bridge toward full high-dimensional coding trees: it
    exposes coarser levels immediately and fixes the flat over-partitioning
    behavior when a target cluster count is known.
    """

    if target_clusters is not None and target_clusters < 1:
        raise ValueError("target_clusters must be >= 1")
    if base_labels is None:
        base_labels, _ = multistart_incremental_se_heuristic(
            adj,
            starts=starts,
            max_passes=max_passes,
            seed=seed,
        )
    else:
        base_labels = canonicalize_labels(base_labels)

    min_clusters = target_clusters if target_clusters is not None else 1
    levels = coding_tree_hierarchy_levels(adj, base_labels, min_clusters=min_clusters)
    selected = select_hierarchy_level(levels, target_clusters=target_clusters)
    return HierarchicalClusteringResult(
        labels=selected.labels,
        entropy=selected.entropy,
        method="hierarchical-se",
        levels=levels,
        base_labels=canonicalize_labels(base_labels),
        target_clusters=target_clusters,
    )
