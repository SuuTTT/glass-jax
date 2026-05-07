# SEClust Launch Doc

## Launch Scope
Launch `SEClust` as the non-differentiable structural entropy clustering module inside `glass-jax`.

This launch promotes the prototype to the `SEClust` name:
- import path: `glass.seclust`
- source path: `src/glass/seclust`
- exact-label benchmark script: `tests/benchmark_seclust.py`
- full benchmark script: `tests/benchmark_seclust_full.py`
- benchmark artifacts: `docs/experimental_reports/seclust_benchmark_20260507.*`
- full benchmark artifacts: `docs/experimental_reports/seclust_full_benchmark_20260507.*`

## User-Facing API
Primary API:
```python
from glass.seclust import cluster_graph

result = cluster_graph(adj, mode="auto")
labels = result.labels
entropy = result.entropy
```

Dataset and benchmark APIs:
```python
from glass.seclust import build_structural_entropy_dataset, compare_on_dataset

dataset = build_structural_entropy_dataset(max_nodes=9)
rows = compare_on_dataset(dataset)
```

ML/RL exploration API:
```python
from glass.seclust import StructuralEntropyMoveEnv, cem_node_move_search
```

Scalable incremental APIs:
```python
from glass.seclust import SparseGraph, IncrementalSEState, local_move_incremental
```

## Launch Criteria
- `glass.seclust` imports successfully.
- Exact search matches known Bell-number enumeration tests.
- Hard SE scorer matches the JAX one-hot objective when JAX is installed.
- `SEClust-Auto` recovers the exact global optimum on the exact-labeled benchmark graphs.
- Benchmark report is generated under `docs/experimental_reports/`.
- No source, test, or report code refers to prototype naming.
- Full synthetic benchmark runs through N=1000 under the 3 minute per-dataset guard.

## Validation Commands
```bash
python -m pytest -q tests/test_seclust.py
python tests/benchmark_seclust.py
python tests/benchmark_seclust_full.py
python -m compileall -q src/glass/seclust
```

Expected benchmark summary from the current launch run:
- `SEClust-Auto` matches the exact global optimum on `5/5` exact-labeled graphs.
- `SEClust-Auto` is no worse than Official-SEP by structural entropy on `8/8` comparable runs.
- `SEClust-CEM` runs successfully but is not yet competitive.
- Sparse incremental SEClust completes synthetic full benchmark cases through `SBM (N=1000)` in seconds.

## Migration Notes
Use the launched import:
```python
from glass.seclust import cluster_graph
```

Use benchmark names:
- `SEClust-Auto`
- `SEClust-Tree`
- `SEClust-Heuristic`
- `SEClust-CEM`

## Release Risks
- Existing notebooks or scripts using prototype imports will break unless updated.
- The flat heuristic can over-partition because `H_2` chooses one global resolution.
- The current scalable backend optimizes flat SE only; high-dimensional coding-tree SE is not implemented yet.
- The official SEP wrapper depends on the local baseline path `official_baselines/SEP/SEPN/codingTree.py`.

## Rollback
If downstream imports unexpectedly depend on the prototype path, add a temporary compatibility package:

```text
src/glass/<prototype_path>/__init__.py
```

that re-exports from `glass.seclust` and emits a deprecation warning. Do this only if needed; the clean launch path is `glass.seclust`.

## Post-Launch Tasks
- Upgrade the implemented `SEClust-Tree` merge hierarchy into a full coding-tree optimizer based on high-dimensional SE.
- Implement incremental merge deltas for the hierarchy.
- Add connectedness refinement.
- Re-run benchmark with larger planted graphs.
- Add exact-labeled datasets as fixtures for learned policy experiments.
