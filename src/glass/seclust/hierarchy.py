"""High-level hierarchical structural entropy clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entropy import canonicalize_labels, structural_entropy
from .incremental import multistart_incremental_se_heuristic


@dataclass(frozen=True)
class HierarchicalLevel:
    """One flat cut through the hierarchical merge process."""

    k: int
    labels: np.ndarray
    entropy: float


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
    levels = merge_hierarchy_levels(adj, base_labels, min_clusters=min_clusters)
    selected = select_hierarchy_level(levels, target_clusters=target_clusters)
    return HierarchicalClusteringResult(
        labels=selected.labels,
        entropy=selected.entropy,
        method="hierarchical-se",
        levels=levels,
        base_labels=canonicalize_labels(base_labels),
        target_clusters=target_clusters,
    )
