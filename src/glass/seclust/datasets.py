"""Small graph datasets labeled by exact structural entropy minimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entropy import structural_entropy
from .exact import exact_minimize_structural_entropy


@dataclass(frozen=True)
class StructuralEntropyGraph:
    """A graph with exact structural entropy labels."""

    name: str
    adjacency: np.ndarray
    best_labels: np.ndarray
    best_entropy: float
    partitions_evaluated: int


def weighted_bridge_graph(left: int, right: int, bridge_weight: float = 1.0, clique_weight: float = 3.0) -> np.ndarray:
    """Two dense modules connected by one tunable bridge."""

    n_nodes = left + right
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    adj[:left, :left] = clique_weight
    adj[left:, left:] = clique_weight
    np.fill_diagonal(adj, 0.0)
    adj[left - 1, left] = bridge_weight
    adj[left, left - 1] = bridge_weight
    return adj


def ring_of_triangles(count: int = 3, bridge_weight: float = 1.0) -> np.ndarray:
    """A compact motif with local cliques and weak cyclic bridges."""

    n_nodes = count * 3
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for block in range(count):
        nodes = np.arange(block * 3, block * 3 + 3)
        for i in nodes:
            for j in nodes:
                if i != j:
                    adj[i, j] = 2.0
        adj[nodes[-1], ((block + 1) % count) * 3] = bridge_weight
        adj[((block + 1) % count) * 3, nodes[-1]] = bridge_weight
    return adj


def seeded_sbm(
    sizes: tuple[int, ...] = (3, 3, 3),
    p_in: float = 0.85,
    p_out: float = 0.12,
    seed: int = 0,
) -> np.ndarray:
    """Generate a small undirected stochastic block model without networkx."""

    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n_nodes = int(labels.size)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                weight = float(rng.integers(1, 4))
                adj[i, j] = adj[j, i] = weight
    return adj


def label_graph(name: str, adjacency: np.ndarray, max_nodes: int = 9) -> StructuralEntropyGraph:
    """Attach the global minimum SE label to one graph."""

    result = exact_minimize_structural_entropy(adjacency, max_nodes=max_nodes)
    return StructuralEntropyGraph(
        name=name,
        adjacency=np.asarray(adjacency, dtype=float),
        best_labels=result.labels,
        best_entropy=result.entropy,
        partitions_evaluated=result.partitions_evaluated,
    )


def build_structural_entropy_dataset(max_nodes: int = 9) -> list[StructuralEntropyGraph]:
    """Build a deterministic exact-labeled benchmark dataset."""

    graphs = [
        ("bridge_4_4_w1", weighted_bridge_graph(4, 4, bridge_weight=1.0)),
        ("bridge_3_5_w2", weighted_bridge_graph(3, 5, bridge_weight=2.0)),
        ("ring_triangles_3", ring_of_triangles(3, bridge_weight=1.0)),
        ("sbm_333_seed1", seeded_sbm(seed=1)),
        ("sbm_234_seed2", seeded_sbm((2, 3, 4), seed=2)),
    ]
    dataset = [label_graph(name, adj, max_nodes=max_nodes) for name, adj in graphs]
    # Keep only graphs where the optimum improves over the trivial one-block partition.
    return [
        item
        for item in dataset
        if item.best_entropy <= structural_entropy(item.adjacency, np.zeros(item.adjacency.shape[0])) + 1e-12
    ]
