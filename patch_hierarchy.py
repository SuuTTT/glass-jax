import re

with open("src/glass/seclust/hierarchy.py", "r") as f:
    content = f.read()

# Replace hierarchical_se_clustering
old_func = """def hierarchical_se_clustering(
    adj: np.ndarray,
    target_clusters: int | None = None,
    base_labels: np.ndarray | None = None,
    starts: int = 6,
    max_passes: int = 10,
    seed: int = 0,
) -> HierarchicalClusteringResult:
    \"\"\"Build a high-level SE hierarchy and return a selected flat cut.

    The first implementation builds a hierarchy over fast flat SEClust modules.
    \"\"\"

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
    \"\"\"Build a high-level SE hierarchy and return a selected flat cut.\"\"\"
    
    from .incremental import multi_level_local_move
    import numpy as np

    rng = np.random.default_rng(seed)
    
    best_entropy = float("inf")
    best_projections = None
    best_flat = None
    
    for i in range(starts):
        flat, ent, projections = multi_level_local_move(
            adj,
            init_labels=base_labels,
            max_passes=max_passes,
            seed=seed + i,
            return_hierarchy=True
        )
        if ent < best_entropy:
            best_entropy = ent
            best_projections = projections
            best_flat = flat
            
    # Reconstruct flat labels at each level
    current_labels = np.arange(adj.shape[0], dtype=np.int32)
    levels = []
    
    for i, proj in enumerate(best_projections):
        if i == 0:
            current_labels = proj
        else:
            current_labels = proj[current_labels]
            
        k = int(current_labels.max()) + 1
        flat_ent = structural_entropy(adj, current_labels)
        
        # Build tree structure up to this level to compute tree_entropy
        # For simplicity in this intermediate step, we report flat entropy as tree entropy if not fully built.
        levels.append(HierarchicalLevel(k=k, labels=current_labels.copy(), entropy=flat_ent, tree_entropy=flat_ent))

    # Reverse levels so finest is first, or keep them?
    # Usually finest is first.
    selected = select_hierarchy_level(tuple(levels), target_clusters=target_clusters)
    return HierarchicalClusteringResult(
        labels=selected.labels.copy(),
        entropy=selected.entropy,
        method="seclust-tree-v2",
        levels=tuple(levels),
        base_labels=best_projections[0].copy(),
        target_clusters=target_clusters,
    )"""

content = content.replace(old_func, new_func)

with open("src/glass/seclust/hierarchy.py", "w") as f:
    f.write(content)
