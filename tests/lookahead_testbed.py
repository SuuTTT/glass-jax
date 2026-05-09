"""A tiny testbed for iterating on the SEClust lookahead merge.

Seven small graphs (N <= 200), each with a known target K and a ground
truth labelling, designed so each algorithm cell completes in well
under a second. The graphs are chosen to exercise distinct failure
modes of one-step greedy merge:

- ``karate``        — classic 2-faction graph; trivial test.
- ``caveman``       — 4 cliques in a chain; pure greedy normally fine.
- ``sbm_aligned``   — SBM(N=120, K=4) with K_local = K_target. Sanity.
- ``sbm_mismatch``  — SBM(N=120, K=4) but target_K=2; forces the
                      collapse failure mode at small scale.
- ``hier_sbm``      — 2-level SBM: 2 super-clusters of 3 sub-clusters
                      each. Tests whether lookahead recovers the
                      hierarchical merge order.
- ``bridge``        — 3 dense cliques connected by single bridge edges
                      to a central star. Greedy can pick the wrong
                      first merge; lookahead should rescue.
- ``noisy_planted`` — N=80, planted K=4, p_in=0.6, p_out=0.05 with a
                      few extra cross-cluster edges. Multistart
                      typically over-fragments to K_local ~ 8;
                      target_K=4 forces a non-trivial merge schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TestCase:
    name: str
    adjacency: np.ndarray
    labels: np.ndarray
    target_k: int
    description: str


def _karate() -> TestCase:
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),
        (0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),
        (1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),
        (2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),(6,16),
        (8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),(15,33),
        (18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),(23,25),
        (23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),
        (26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),(30,32),
        (30,33),(31,32),(31,33),(32,33),
    ]
    adj = np.zeros((34, 34), dtype=float)
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1.0
    officer = {8,9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33}
    labels = np.array([1 if i in officer else 0 for i in range(34)], dtype=np.int32)
    return TestCase("karate", adj, labels, target_k=2,
                    description="34-node Zachary graph, 2 factions")


def _caveman(n_cliques: int = 4, clique_size: int = 8) -> TestCase:
    n = n_cliques * clique_size
    adj = np.zeros((n, n), dtype=float)
    labels = np.repeat(np.arange(n_cliques, dtype=np.int32), clique_size)
    for c in range(n_cliques):
        s = c * clique_size; e = s + clique_size
        adj[s:e, s:e] = 1.0
        np.fill_diagonal(adj[s:e, s:e], 0.0)
        if c < n_cliques - 1:
            adj[e - 1, e] = adj[e, e - 1] = 1.0
    return TestCase("caveman", adj, labels, target_k=n_cliques,
                    description=f"{n_cliques} cliques of size {clique_size} in a chain")


def _sbm(name: str, n: int, k: int, p_in: float, p_out: float, target: int, seed: int) -> TestCase:
    rng = np.random.default_rng(seed)
    sizes = [n // k] * (k - 1) + [n - (n // k) * (k - 1)]
    labels = np.repeat(np.arange(k, dtype=np.int32), sizes)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1.0
    return TestCase(name, adj, labels, target_k=target,
                    description=f"SBM N={n}, planted K={k}, target K={target}")


def _hier_sbm(seed: int = 7) -> TestCase:
    """2 super-clusters of 3 sub-clusters each (6 sub-clusters total).

    Sub-clusters within a super-cluster are tightly connected
    (p_intra=0.5). Sub-clusters across super-clusters share a small
    p_super=0.10. Cross-everything baseline noise p_out=0.01. Target
    K=2 (recover super-clusters); ground truth labels are the
    super-cluster ids.
    """

    rng = np.random.default_rng(seed)
    sub_size = 20
    n_sub = 6
    n = sub_size * n_sub  # 120
    sub_labels = np.repeat(np.arange(n_sub, dtype=np.int32), sub_size)
    super_labels = np.array([s // 3 for s in sub_labels], dtype=np.int32)
    adj = np.zeros((n, n), dtype=float)
    p_intra = 0.5
    p_super = 0.08
    p_out = 0.01
    for i in range(n):
        for j in range(i + 1, n):
            if sub_labels[i] == sub_labels[j]:
                p = p_intra
            elif super_labels[i] == super_labels[j]:
                p = p_super
            else:
                p = p_out
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1.0
    return TestCase("hier_sbm", adj, super_labels, target_k=2,
                    description="2-level SBM (2 super-clusters of 3 sub each); target K=2")


def _bridge(seed: int = 11) -> TestCase:
    """3 dense cliques connected to a small central star.

    Greedy may absorb the central star into the wrong clique first.
    """

    rng = np.random.default_rng(seed)
    clique_size = 12
    n_cliques = 3
    star_size = 4
    n = n_cliques * clique_size + star_size
    adj = np.zeros((n, n), dtype=float)
    labels = np.zeros(n, dtype=np.int32)
    for c in range(n_cliques):
        s = c * clique_size; e = s + clique_size
        adj[s:e, s:e] = 1.0
        np.fill_diagonal(adj[s:e, s:e], 0.0)
        labels[s:e] = c
    star_start = n_cliques * clique_size
    # Star ring (bridge nodes weakly connected)
    for i in range(star_start, n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    # Each star node has one weak bridge to one clique
    for k in range(star_size):
        clique = k % n_cliques
        target = clique * clique_size + (k % clique_size)
        adj[star_start + k, target] = adj[target, star_start + k] = 1.0
    labels[star_start:] = -1  # bridge nodes have no ground-truth assignment
    # Recode -1 to 3 so labels are valid; we'll treat the star as its own
    # ill-defined cluster for the ACC metric.
    labels[labels == -1] = n_cliques
    return TestCase("bridge", adj, labels, target_k=3,
                    description="3 cliques + 4-node bridge star; target K=3")


def _noisy_planted(seed: int = 13) -> TestCase:
    """N=80 SBM with K=4, p_in=0.6, p_out=0.05 + bonus cross-cluster edges.

    Multistart will typically over-fragment (K_local ~ 8). target_K=4
    forces a 4-step merge schedule whose order matters.
    """

    rng = np.random.default_rng(seed)
    return _sbm("noisy_planted", n=80, k=4, p_in=0.6, p_out=0.05,
                target=4, seed=seed)


def get_testbed() -> list[TestCase]:
    return [
        _karate(),
        _caveman(),
        _sbm("sbm_aligned", n=120, k=4, p_in=0.5, p_out=0.02, target=4, seed=21),
        _sbm("sbm_mismatch", n=120, k=4, p_in=0.5, p_out=0.02, target=2, seed=22),
        _hier_sbm(),
        _bridge(),
        _noisy_planted(),
    ]


if __name__ == "__main__":
    cases = get_testbed()
    for c in cases:
        n = c.adjacency.shape[0]
        e = int(np.sum(c.adjacency > 0) // 2)
        gt_k = int(np.unique(c.labels).size)
        print(f"  {c.name:14s} N={n:4d}  E={e:5d}  target_K={c.target_k}  "
              f"true_K={gt_k}  -- {c.description}")
