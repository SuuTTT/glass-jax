# SEClust Design Doc

## Name
`SEClust` means Structural Entropy Clustering.

The prototype name was useful as a sketch, but it did not describe the algorithmic contract. `SEClust` is short, importable as `glass.seclust`, and specific enough to distinguish this package from the differentiable `glass.objectives` code.

Public naming:
- Package path: `glass.seclust`
- High-level API: `cluster_graph()`
- Exact algorithm label: `SEClust-Exact`
- Default high-level algorithm label in benchmarks: `SEClust-Auto`
- Heuristic algorithm label: `SEClust-Heuristic`
- ML/RL exploration label: `SEClust-CEM`

## Problem
Given a weighted, undirected graph `G = (V, E)`, find a hard node partition that minimizes two-dimensional structural entropy:

```text
H(P) = -sum_C g_C / vol(G) log2(vol(C) / vol(G))
       -sum_C sum_{v in C} d_v / vol(G) log2(d_v / vol(C))
```

where:
- `C` is a cluster in partition `P`
- `d_v` is node degree
- `vol(C)` is the sum of degrees inside cluster `C`
- `g_C` is the cut volume leaving `C`
- `vol(G)` is total graph volume

The package is intentionally non-differentiable. It provides exact labels for small graphs, discrete heuristics for larger graphs, and an ML-friendly search environment for learned policies.

## Goals
- Provide a trusted hard structural entropy scorer.
- Produce exact global minimum partitions for small graphs using exhaustive search.
- Generate exact-labeled graph datasets for supervised or benchmark use.
- Provide a high-level clustering API that automatically chooses exact or heuristic search.
- Compare directly against `official_baselines/SEP/SEPN/codingTree.py`.
- Keep the implementation dependency-light and usable without JAX.

## Non-Goals
- Replace differentiable objectives in `glass.objectives`.
- Provide a production-scale Leiden implementation in the first version.
- Guarantee global optimality on large graphs.
- Train a full neural policy in-tree before the discrete baseline is stable.

## Package Layout
```text
src/glass/seclust/
  entropy.py        Hard structural entropy scorer and partition utilities.
  exact.py          Restricted-growth-string exhaustive search.
  heuristics.py     Agglomerative, local-move, multistart, and cluster_graph APIs.
  datasets.py       Deterministic synthetic graphs with exact SE labels.
  benchmark_sep.py  Official SEP codingTree comparison wrapper.
  rl.py             Node-move environment and CEM policy-search scaffold.
```

## Core APIs
```python
from glass.seclust import cluster_graph, structural_entropy

result = cluster_graph(adj, mode="auto")
print(result.entropy, result.labels, result.method)
```

`cluster_graph()` modes:
- `auto`: exact for `N <= exact_max_nodes`, heuristic otherwise.
- `exact`: exhaustive search, guarded by `max_nodes`.
- `heuristic`: multistart local search.

Exact dataset generation:
```python
from glass.seclust import build_structural_entropy_dataset

dataset = build_structural_entropy_dataset(max_nodes=9)
```

SEP comparison:
```python
from glass.seclust import compare_on_dataset

rows = compare_on_dataset(dataset)
```

## Exact Search
Exact search enumerates set partitions using restricted-growth strings. This visits each unlabeled partition once, avoiding duplicates from arbitrary cluster id permutations.

For `N` nodes, the search evaluates the Bell number `B_N` partitions:
- `B_8 = 4,140`
- `B_9 = 21,147`
- `B_10 = 115,975`

This is acceptable for small exact-label datasets and test fixtures. It is not intended as the large-graph path.

## Heuristic Search
The heuristic path has three pieces:
- `agglomerative_se_clustering()`: starts from singleton clusters and greedily merges if SE improves.
- `local_move_se_clustering()`: Leiden/Louvain-style node moves scored by hard SE.
- `multistart_se_heuristic()`: runs deterministic and random starts, then keeps the lowest SE result.

The current implementation recomputes full structural entropy for each candidate move. This keeps the first version simple and auditable. The next performance upgrade should add an incremental delta scorer.

## ML/RL Hook
`StructuralEntropyMoveEnv` exposes a small node-move environment:
- state: current hard labels
- action: `(node, target_cluster)`
- reward: previous SE minus new SE

`cem_node_move_search()` is a minimal cross-entropy-method policy search. It is included to make ML optimization experiments concrete, not as the main production optimizer.

## Benchmark Contract
`tests/benchmark_seclust.py` mirrors the style of `tests/benchmark_full.py`:
- builds datasets
- runs multiple algorithms
- computes SE, gap, ARI, NMI, and timing
- prints a markdown table
- writes report artifacts under `docs/experimental_reports/`

The benchmark algorithms are:
- `SEClust-Auto`
- `SEClust-Heuristic`
- `SEClust-CEM`
- `Official-SEP`

## Known Limitations
- Exact search is exponential.
- Heuristic search is currently correct but not optimized.
- CEM search is measurable but not competitive with deterministic local search.
- The package assumes non-negative undirected adjacency matrices and symmetrizes inputs.

## Next Work
- Add incremental SE delta updates for node moves and cluster merges.
- Add connectedness refinement similar to Leiden.
- Add larger benchmark cases after delta scoring lands.
- Use exact-labeled graphs to train or validate learned move policies.
- Add compatibility shims only if downstream code needs the old prototype import path.
