"""Discrete structural entropy utilities for hard graph partitions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PartitionScore:
    """Structural entropy score and the canonical partition labels."""

    entropy: float
    labels: np.ndarray


@dataclass(frozen=True)
class StructuralEntropyScorer:
    """Reusable hard structural entropy scorer for one graph."""

    adjacency: np.ndarray
    degrees: np.ndarray
    graph_volume: float
    eps: float = 1e-12

    @classmethod
    def from_adjacency(cls, adj: np.ndarray, eps: float = 1e-12) -> "StructuralEntropyScorer":
        matrix = as_symmetric_adjacency(adj)
        degrees = matrix.sum(axis=1)
        return cls(matrix, degrees, float(degrees.sum()), eps=eps)

    def score(self, labels: Iterable[int]) -> float:
        labels_arr = canonicalize_labels(labels)
        if labels_arr.shape[0] != self.adjacency.shape[0]:
            raise ValueError("labels length must match adjacency size")
        if self.graph_volume <= self.eps:
            return 0.0

        entropy = 0.0
        for cluster_id in range(int(labels_arr.max()) + 1):
            mask = labels_arr == cluster_id
            volume = float(self.degrees[mask].sum())
            if volume <= self.eps:
                continue
            internal_twice = float(self.adjacency[np.ix_(mask, mask)].sum())
            cut = max(volume - internal_twice, 0.0)
            entropy -= (cut / self.graph_volume) * math.log2(max(volume / self.graph_volume, self.eps))
            positive_degrees = self.degrees[mask]
            positive_degrees = positive_degrees[positive_degrees > self.eps]
            if positive_degrees.size:
                entropy -= float(
                    np.sum((positive_degrees / self.graph_volume) * np.log2(positive_degrees / volume))
                )
        return float(entropy)


def as_symmetric_adjacency(adj: np.ndarray) -> np.ndarray:
    """Return a float, square, loop-free, symmetric adjacency matrix."""

    matrix = np.asarray(adj, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if np.any(matrix < 0):
        raise ValueError("structural entropy expects non-negative edge weights")
    matrix = np.array(matrix, dtype=float, copy=True)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def canonicalize_labels(labels: Iterable[int]) -> np.ndarray:
    """Relabel arbitrary cluster ids to contiguous ids in first-seen order."""

    mapping: dict[int, int] = {}
    out = []
    for raw in labels:
        value = int(raw)
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return np.asarray(out, dtype=np.int32)


def labels_to_partition(labels: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Convert labels into a canonical tuple-of-tuples partition."""

    canonical = canonicalize_labels(labels)
    groups: list[list[int]] = [[] for _ in range(int(canonical.max()) + 1 if canonical.size else 0)]
    for node, label in enumerate(canonical):
        groups[int(label)].append(node)
    return tuple(tuple(group) for group in groups)


def partition_to_labels(partition: Iterable[Iterable[int]], n_nodes: int | None = None) -> np.ndarray:
    """Convert an iterable partition into canonical labels."""

    groups = [tuple(group) for group in partition]
    if n_nodes is None:
        n_nodes = max((node for group in groups for node in group), default=-1) + 1
    labels = np.full(n_nodes, -1, dtype=np.int32)
    for group_id, group in enumerate(groups):
        for node in group:
            if node < 0 or node >= n_nodes:
                raise ValueError(f"node {node} is outside 0..{n_nodes - 1}")
            if labels[node] != -1:
                raise ValueError(f"node {node} appears in multiple groups")
            labels[node] = group_id
    if np.any(labels < 0):
        missing = np.where(labels < 0)[0].tolist()
        raise ValueError(f"partition is missing nodes: {missing}")
    return canonicalize_labels(labels)


def structural_entropy(adj: np.ndarray, labels: Iterable[int], eps: float = 1e-12) -> float:
    """Compute two-dimensional structural entropy for a hard partition.

    The objective matches ``glass.objectives.structural_entropy`` when ``S`` is
    one-hot and ``is_logits=False``:

    H = -sum_C g_C / vol(G) log2(vol(C) / vol(G))
        -sum_C sum_{v in C} d_v / vol(G) log2(d_v / vol(C)).
    """

    return StructuralEntropyScorer.from_adjacency(adj, eps=eps).score(labels)


def structural_entropy_many(adj: np.ndarray, labelings: Iterable[Iterable[int]]) -> list[PartitionScore]:
    """Score several hard partitions."""

    return [
        PartitionScore(entropy=structural_entropy(adj, labels), labels=canonicalize_labels(labels))
        for labels in labelings
    ]
