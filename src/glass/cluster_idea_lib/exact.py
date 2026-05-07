"""Exact structural entropy minimization over hard partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .entropy import PartitionScore, StructuralEntropyScorer, canonicalize_labels


@dataclass(frozen=True)
class ExactSearchResult:
    """Result returned by exhaustive partition search."""

    entropy: float
    labels: np.ndarray
    partitions_evaluated: int


def iter_restricted_growth_strings(n_nodes: int, max_clusters: int | None = None) -> Iterator[np.ndarray]:
    """Yield each set partition once as a restricted-growth label vector."""

    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    if n_nodes == 0:
        yield np.zeros(0, dtype=np.int32)
        return

    labels = np.zeros(n_nodes, dtype=np.int32)

    def visit(pos: int, current_max: int) -> Iterator[np.ndarray]:
        if pos == n_nodes:
            yield labels.copy()
            return

        upper = current_max + 1
        if max_clusters is not None:
            upper = min(upper, max_clusters - 1)
        for value in range(upper + 1):
            labels[pos] = value
            yield from visit(pos + 1, max(current_max, value))

    yield from visit(1, 0)


def exact_minimize_structural_entropy(
    adj: np.ndarray,
    max_nodes: int = 10,
    max_clusters: int | None = None,
) -> ExactSearchResult:
    """Find the global minimum-SE flat partition by exhaustive search.

    This enumerates Bell-number many partitions, so ``max_nodes`` defaults to a
    conservative value. For dataset labeling, n <= 9 is usually comfortable.
    """

    n_nodes = int(np.asarray(adj).shape[0])
    if n_nodes > max_nodes:
        raise ValueError(
            f"exact search requested for {n_nodes} nodes, above max_nodes={max_nodes}; "
            "raise max_nodes explicitly if this is intentional"
        )
    best = PartitionScore(entropy=float("inf"), labels=np.zeros(n_nodes, dtype=np.int32))
    scorer = StructuralEntropyScorer.from_adjacency(adj)
    count = 0
    for labels in iter_restricted_growth_strings(n_nodes, max_clusters=max_clusters):
        score = scorer.score(labels)
        count += 1
        if score < best.entropy - 1e-12:
            best = PartitionScore(entropy=score, labels=canonicalize_labels(labels))
    return ExactSearchResult(entropy=best.entropy, labels=best.labels, partitions_evaluated=count)
