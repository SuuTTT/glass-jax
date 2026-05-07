"""Cluster Idea Lib: non-differentiable structural entropy clustering."""

from .benchmark_sep import compare_on_dataset, run_official_sep_coding_tree
from .datasets import (
    StructuralEntropyGraph,
    build_structural_entropy_dataset,
    label_graph,
    ring_of_triangles,
    seeded_sbm,
    weighted_bridge_graph,
)
from .entropy import (
    PartitionScore,
    StructuralEntropyScorer,
    canonicalize_labels,
    labels_to_partition,
    partition_to_labels,
    structural_entropy,
)
from .exact import ExactSearchResult, exact_minimize_structural_entropy, iter_restricted_growth_strings
from .heuristics import (
    ClusteringResult,
    agglomerative_se_clustering,
    cluster_graph,
    local_move_se_clustering,
    multistart_se_heuristic,
)
from .rl import StructuralEntropyMoveEnv, cem_node_move_search

__all__ = [
    "ClusteringResult",
    "ExactSearchResult",
    "PartitionScore",
    "StructuralEntropyGraph",
    "StructuralEntropyScorer",
    "StructuralEntropyMoveEnv",
    "agglomerative_se_clustering",
    "build_structural_entropy_dataset",
    "canonicalize_labels",
    "cem_node_move_search",
    "cluster_graph",
    "compare_on_dataset",
    "exact_minimize_structural_entropy",
    "iter_restricted_growth_strings",
    "label_graph",
    "labels_to_partition",
    "local_move_se_clustering",
    "multistart_se_heuristic",
    "partition_to_labels",
    "ring_of_triangles",
    "run_official_sep_coding_tree",
    "seeded_sbm",
    "structural_entropy",
    "weighted_bridge_graph",
]
