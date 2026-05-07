"""Comparison helpers for the official SEP codingTree baseline."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from .entropy import canonicalize_labels, structural_entropy
from .heuristics import ClusteringResult, cluster_graph


def run_official_sep_coding_tree(adj: np.ndarray, k: int = 2) -> ClusteringResult:
    """Run ``official_baselines/SEP/SEPN/codingTree.py`` and extract root children."""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sep_path = os.path.join(repo_root, "official_baselines", "SEP")
    if sep_path not in sys.path:
        sys.path.append(sep_path)
    from SEPN.codingTree import PartitionTree

    tree = PartitionTree(adj_matrix=np.asarray(adj, dtype=float))
    tree.build_coding_tree(k=k, mode="v1")
    root = tree.tree_node[tree.root_id]
    labels = np.zeros(np.asarray(adj).shape[0], dtype=np.int32)
    for cluster_id, child_id in enumerate(sorted(root.children)):
        for node in tree.tree_node[child_id].partition:
            labels[int(node)] = cluster_id
    labels = canonicalize_labels(labels)
    return ClusteringResult(structural_entropy(adj, labels), labels, method="official-sep-codingtree")


def compare_on_dataset(dataset, exact_max_nodes: int = 9) -> list[dict[str, object]]:
    """Compare SEClust against SEP on exact-labeled small graphs."""

    rows = []
    for item in dataset:
        start = time.time()
        ours = cluster_graph(item.adjacency, mode="auto", exact_max_nodes=exact_max_nodes)
        ours_seconds = time.time() - start
        try:
            sep = run_official_sep_coding_tree(item.adjacency, k=max(2, len(np.unique(item.best_labels))))
            sep_error = None
        except Exception as exc:  # pragma: no cover - depends on external baseline quirks.
            sep = None
            sep_error = repr(exc)
        rows.append(
            {
                "name": item.name,
                "best_entropy": item.best_entropy,
                "ours_entropy": ours.entropy,
                "ours_method": ours.method,
                "ours_seconds": ours_seconds,
                "sep_entropy": None if sep is None else sep.entropy,
                "sep_error": sep_error,
                "beats_sep": sep is None or ours.entropy <= sep.entropy + 1e-12,
            }
        )
    return rows
