import numpy as np
from glass.seclust.incremental import multistart_incremental_se_heuristic
from tests.benchmark_seclust_full import get_cases
import time

cases = get_cases()
for case in cases:
    if case.name.startswith("SBM") or case.name == "Karate":
        start = time.time()
        labels, entropy = multistart_incremental_se_heuristic(case.adjacency, starts=1)
        duration = time.time() - start
        k = int(labels.max()) + 1
        print(f"{case.name:15} | K: {k:3} | Entropy: {entropy:.4f} | Time: {duration:.4f}s")
