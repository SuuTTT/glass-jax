import numpy as np
from glass.seclust.incremental import multi_level_local_move
from tests.benchmark_seclust_full import get_cases
from glass.seclust.entropy import structural_entropy

cases = get_cases()
karate = cases[0]
sbm1000 = cases[4]

labels, ent, projections = multi_level_local_move(sbm1000.adjacency, return_hierarchy=True)
print("SBM1000 Projections:")
for i, p in enumerate(projections):
    print(f"Level {i} nodes: {len(p)}, clusters: {p.max()+1}")
