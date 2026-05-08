import re

with open("tests/benchmark_seclust_full.py", "r") as f:
    content = f.read()

# Add SEClust-TargetK to the benchmark loop
old_loop = """        print(f"Running SEClust-Tree on {dataset}...", flush=True)
        rows.append(run_seclust(cases[dataset], algorithm="SEClust-Tree"))"""
        
new_loop = """        print(f"Running SEClust-Tree on {dataset}...", flush=True)
        rows.append(run_seclust(cases[dataset], algorithm="SEClust-Tree"))
        print(f"Running SEClust-TargetK on {dataset}...", flush=True)
        rows.append(run_seclust(cases[dataset], algorithm="SEClust-TargetK"))"""
content = content.replace(old_loop, new_loop)

old_run = """    if algorithm == "SEClust-Tree":
        result = hierarchical_se_clustering(
            case.adjacency,
            target_clusters=case.k,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=SECLUST_SEED,
        )
    else:
        result = cluster_graph("""

new_run = """    if algorithm == "SEClust-Tree":
        result = hierarchical_se_clustering(
            case.adjacency,
            target_clusters=case.k,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=SECLUST_SEED,
        )
    elif algorithm == "SEClust-TargetK":
        from glass.seclust.incremental import multistart_incremental_se_heuristic
        from glass.seclust.hierarchy import merge_hierarchy_levels, select_hierarchy_level
        from glass.seclust.heuristics import ClusteringResult
        
        base_labels, _ = multistart_incremental_se_heuristic(
            case.adjacency,
            starts=SECLUST_STARTS,
            max_passes=SECLUST_MAX_PASSES,
            seed=SECLUST_SEED,
        )
        levels = merge_hierarchy_levels(case.adjacency, base_labels, min_clusters=case.k)
        selected = select_hierarchy_level(levels, target_clusters=case.k)
        result = ClusteringResult(
            entropy=selected.entropy,
            labels=selected.labels,
            method="seclust-target-k",
        )
    else:
        result = cluster_graph("""
content = content.replace(old_run, new_run)

with open("tests/benchmark_seclust_full.py", "w") as f:
    f.write(content)
