import numpy as np
from glass.seclust.hierarchy import hierarchical_se_clustering
from tests.benchmark_seclust_full import get_cases
import time

cases = get_cases()
for case in cases:
    if case.name.startswith("SBM") or case.name == "Karate":
        start = time.time()
        result = hierarchical_se_clustering(case.adjacency, target_clusters=case.k, starts=1)
        duration = time.time() - start
        k = int(result.labels.max()) + 1
        print(f"{case.name:15} | K: {k:3} | Entropy: {result.entropy:.4f} | Time: {duration:.4f}s")
