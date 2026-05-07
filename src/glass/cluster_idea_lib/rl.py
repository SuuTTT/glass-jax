"""RL-friendly structural entropy search components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entropy import canonicalize_labels, structural_entropy
from .heuristics import ClusteringResult


@dataclass
class StructuralEntropyMoveEnv:
    """Tiny environment where actions move one node to one cluster."""

    adjacency: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.adjacency = np.asarray(self.adjacency, dtype=float)
        self.labels = canonicalize_labels(self.labels)
        self.entropy = structural_entropy(self.adjacency, self.labels)

    def legal_actions(self) -> list[tuple[int, int]]:
        n_nodes = self.labels.shape[0]
        clusters = set(np.unique(self.labels).astype(int).tolist())
        clusters.add(int(self.labels.max()) + 1)
        return [(node, cluster) for node in range(n_nodes) for cluster in clusters if cluster != int(self.labels[node])]

    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool]:
        node, cluster = action
        old_entropy = self.entropy
        proposal = self.labels.copy()
        proposal[int(node)] = int(cluster)
        self.labels = canonicalize_labels(proposal)
        self.entropy = structural_entropy(self.adjacency, self.labels)
        reward = old_entropy - self.entropy
        return self.labels.copy(), float(reward), reward <= 1e-12


def cem_node_move_search(
    adj: np.ndarray,
    episodes: int = 64,
    horizon: int | None = None,
    elite_frac: float = 0.25,
    seed: int = 0,
) -> ClusteringResult:
    """Cross-entropy-method policy search over node-move actions.

    This is intentionally small and dependency-free. It gives a concrete ML/RL
    hook for learning action preferences, while exact labels from
    ``datasets.py`` can serve as supervision or validation.
    """

    rng = np.random.default_rng(seed)
    n_nodes = int(np.asarray(adj).shape[0])
    if horizon is None:
        horizon = 3 * n_nodes
    node_logits = np.zeros(n_nodes, dtype=float)
    cluster_logits = np.zeros(n_nodes + 1, dtype=float)
    best_labels = np.arange(n_nodes, dtype=np.int32)
    best_entropy = structural_entropy(adj, best_labels)

    for _ in range(episodes):
        rollouts = []
        for _ in range(max(4, n_nodes)):
            env = StructuralEntropyMoveEnv(adj, np.arange(n_nodes, dtype=np.int32))
            chosen_nodes = []
            chosen_clusters = []
            for _ in range(horizon):
                node_probs = np.exp(node_logits - node_logits.max())
                node_probs /= node_probs.sum()
                cluster_probs = np.exp(cluster_logits - cluster_logits.max())
                cluster_probs /= cluster_probs.sum()
                node = int(rng.choice(n_nodes, p=node_probs))
                cluster = int(rng.choice(n_nodes + 1, p=cluster_probs))
                if cluster == int(env.labels[node]):
                    continue
                env.step((node, cluster))
                chosen_nodes.append(node)
                chosen_clusters.append(cluster)
            rollouts.append((env.entropy, env.labels.copy(), chosen_nodes, chosen_clusters))
            if env.entropy < best_entropy:
                best_entropy = env.entropy
                best_labels = env.labels.copy()

        rollouts.sort(key=lambda item: item[0])
        elite_count = max(1, int(len(rollouts) * elite_frac))
        node_logits *= 0.8
        cluster_logits *= 0.8
        for _, _, nodes, clusters in rollouts[:elite_count]:
            for node in nodes:
                node_logits[node] += 0.1
            for cluster in clusters:
                cluster_logits[cluster] += 0.1

    return ClusteringResult(best_entropy, canonicalize_labels(best_labels), method="cem-node-move")
