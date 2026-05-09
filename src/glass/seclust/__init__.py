"""SEClust: non-differentiable structural entropy clustering."""

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
    sparse_structural_entropy,
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
from .hierarchy import (
    HierarchicalClusteringResult,
    HierarchicalLevel,
    coding_tree_hierarchy_levels,
    hierarchical_se_clustering,
    high_dimensional_tree_entropy,
    merge_hierarchy_levels,
    select_hierarchy_level,
)
from .incremental import (
    IncrementalSEState,
    SparseGraph,
    local_move_incremental,
    multistart_incremental_se_heuristic,
)
from .lookahead import (
    merge_to_target_with_lookahead,
    merge_to_target_with_mpc,
    merge_to_target_with_td_bootstrap,
    merge_to_target_with_td_lambda,
    seclust_target_k_lookahead,
)
from .metrics import dasgupta_cost, dendrogram_purity, hierarchy_to_lca_size
from .multilevel import multilevel_se_clustering
from .rl import StructuralEntropyMoveEnv, cem_node_move_search

__all__ = [
    "ClusteringResult",
    "ExactSearchResult",
    "PartitionScore",
    "StructuralEntropyGraph",
    "StructuralEntropyScorer",
    "StructuralEntropyMoveEnv",
    "IncrementalSEState",
    "HierarchicalClusteringResult",
    "HierarchicalLevel",
    "SparseGraph",
    "agglomerative_se_clustering",
    "build_structural_entropy_dataset",
    "canonicalize_labels",
    "cem_node_move_search",
    "cluster_graph",
    "coding_tree_hierarchy_levels",
    "compare_on_dataset",
    "dasgupta_cost",
    "dendrogram_purity",
    "hierarchy_to_lca_size",
    "exact_minimize_structural_entropy",
    "iter_restricted_growth_strings",
    "hierarchical_se_clustering",
    "high_dimensional_tree_entropy",
    "label_graph",
    "labels_to_partition",
    "local_move_se_clustering",
    "local_move_incremental",
    "merge_to_target_with_lookahead",
    "merge_to_target_with_mpc",
    "merge_to_target_with_td_bootstrap",
    "merge_to_target_with_td_lambda",
    "multilevel_se_clustering",
    "seclust_target_k_lookahead",
    "multistart_se_heuristic",
    "multistart_incremental_se_heuristic",
    "merge_hierarchy_levels",
    "partition_to_labels",
    "ring_of_triangles",
    "run_official_sep_coding_tree",
    "seeded_sbm",
    "select_hierarchy_level",
    "sparse_structural_entropy",
    "structural_entropy",
    "weighted_bridge_graph",
]
