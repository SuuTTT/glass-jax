import numpy as np
from glass.seclust.incremental import multistart_incremental_se_heuristic
from glass.seclust.hierarchy import merge_hierarchy_levels, select_hierarchy_level
from tests.benchmark_seclust_full import get_cases
import time

cases = get_cases()
for case in cases:
    if case.name.startswith("SBM") or case.name == "Karate":
        start = time.time()
        base_labels, _ = multistart_incremental_se_heuristic(case.adjacency, starts=1)
        levels = merge_hierarchy_levels(case.adjacency, base_labels, min_clusters=case.k)
        selected = select_hierarchy_level(levels, target_clusters=case.k)
        duration = time.time() - start
        k = int(selected.labels.max()) + 1
        print(f"{case.name:15} | K: {k:3} | Entropy: {selected.entropy:.4f} | Time: {duration:.4f}s")
