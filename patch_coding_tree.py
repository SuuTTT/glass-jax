import re

with open("src/glass/seclust/hierarchy.py", "r") as f:
    content = f.read()

old_func = """def hierarchical_se_clustering(
    adj: np.ndarray,
    target_clusters: int | None = None,
    base_labels: np.ndarray | None = None,
    starts: int = 6,
    max_passes: int = 10,
    seed: int = 0,
) -> HierarchicalClusteringResult:
    \"\"\"Build a high-level SE hierarchy and return a selected flat cut.\"\"\"

    if base_labels is None:
        base_labels, _ = multistart_incremental_se_heuristic(
            adj,
            starts=starts,
            max_passes=max_passes,
            seed=seed,
        )

    levels = coding_tree_hierarchy_levels(adj, base_labels)
    selected = select_hierarchy_level(levels, target_clusters=target_clusters)
    return HierarchicalClusteringResult(
        labels=selected.labels.copy(),
        entropy=selected.entropy,
        method="seclust-tree-v1",
        levels=levels,
        base_labels=base_labels.copy(),
        target_clusters=target_clusters,
    )"""

new_func = """def hierarchical_se_clustering(
    adj: np.ndarray,
    target_clusters: int | None = None,
    base_labels: np.ndarray | None = None,
    starts: int = 6,
    max_passes: int = 10,
    seed: int = 0,
) -> HierarchicalClusteringResult:
    \"\"\"Build a full coding tree SE hierarchy with compression and return a flat cut.\"\"\"

    if base_labels is None:
        base_labels, _ = multistart_incremental_se_heuristic(
            adj,
            starts=starts,
            max_passes=max_passes,
            seed=seed,
        )
        
    from .coding_tree import build_coding_tree_from_modules, extract_flat_labels
    root, nodes = build_coding_tree_from_modules(adj, base_labels, target_k=target_clusters)
    labels = extract_flat_labels(nodes, root, adj.shape[0], base_labels)
    k = int(labels.max()) + 1
    entropy = structural_entropy(adj, labels)
    
    selected = HierarchicalLevel(k=k, labels=labels, entropy=entropy, tree_entropy=entropy)

    return HierarchicalClusteringResult(
        labels=selected.labels.copy(),
        entropy=selected.entropy,
        method="seclust-full-coding-tree",
        levels=(selected,),
        base_labels=base_labels.copy(),
        target_clusters=target_clusters,
    )"""

content = content.replace(old_func, new_func)

with open("src/glass/seclust/hierarchy.py", "w") as f:
    f.write(content)
