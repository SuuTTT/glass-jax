# SEClust Central Experiment Report

**Date:** May 7, 2026  
**Module:** `glass.seclust`  
**Primary artifacts:**  
- `docs/experimental_reports/seclust_benchmark_20260507.md`
- `docs/experimental_reports/seclust_full_benchmark_20260507.md`

## 1. Executive Summary
SEClust now has two validated capabilities:

1. **Exact small-graph labeling:** `SEClust-Auto` recovers exact global minimum `H_2` partitions on the exact-labeled benchmark set.
2. **Scalable flat local search:** sparse incremental node-move deltas allow the full synthetic benchmark to run through `N=1000` in seconds.
3. **Hierarchical coarse extraction:** `SEClust-Tree` now merges fine SEClust modules into target-`K` levels, fixing the flat over-partitioning issue on the synthetic benchmark when target `K` is known.

The main remaining algorithmic issue is now narrower: the first `SEClust-Tree` implementation controls flat over-partitioning when a target level is supplied, but it is not yet a full high-dimensional coding-tree optimizer. The preferred next step is to upgrade the merge hierarchy into a sparse coding tree, following the design in `official_baselines/SEP/SEPN/codingTree.py`.

## 2. Validation Runs
### 2.1 Unit Tests
Command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_seclust.py
```

Result:

```text
9 passed, 1 skipped
```

The skipped test is the JAX compatibility check when JAX is unavailable in the local environment.

### 2.2 Full Benchmark
Command:

```bash
PYTHONPATH=src python tests/benchmark_seclust_full.py
```

The benchmark imports existing baseline values from:
- `docs/experimental_reports/benchmark_sbm_20260506.md`
- `docs/experimental_reports/real_world_comparison_20260507.md`

Only SEClust is executed in this run.

## 3. Current SEClust Results
| Dataset | Time(s) | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Karate / SEClust-Auto | 0.120 | 0.471 | 0.510 | 0.314 | 6 | 0.342 | 3.427 | 4.612 |
| Karate / SEClust-Tree | 0.056 | 0.971 | 0.836 | 0.882 | 2 | 0.357 | 3.849 | 4.464 |
| Caveman / SEClust-Auto | 0.631 | 1.000 | 1.000 | 1.000 | 10 | 0.895 | 4.337 | 4.380 |
| Caveman / SEClust-Tree | 0.293 | 1.000 | 1.000 | 1.000 | 10 | 0.895 | 4.337 | 4.380 |
| SBM N=100 / SEClust-Auto | 0.201 | 0.720 | 0.850 | 0.731 | 7 | 0.513 | 4.849 | 5.815 |
| SBM N=100 / SEClust-Tree | 0.319 | 1.000 | 1.000 | 1.000 | 4 | 0.626 | 4.840 | 5.399 |
| SBM N=500 / SEClust-Auto | 5.351 | 0.936 | 0.945 | 0.931 | 9 | 0.582 | 7.073 | 7.874 |
| SBM N=500 / SEClust-Tree | 6.307 | 1.000 | 1.000 | 1.000 | 5 | 0.629 | 7.016 | 7.717 |
| SBM N=1000 / SEClust-Auto | 11.167 | 0.944 | 0.968 | 0.947 | 14 | 0.561 | 7.676 | 8.762 |
| SBM N=1000 / SEClust-Tree | 13.767 | 1.000 | 1.000 | 1.000 | 10 | 0.589 | 7.635 | 8.671 |

Cora and Citeseer were not run locally because `torch_geometric` is unavailable in this workspace. Their existing baseline rows remain in the full benchmark report.

## 4. What Improved
### 4.1 Runtime
Before incremental scoring, the flat heuristic recomputed structural entropy for every candidate move. This made N=100+ runs approach or exceed the 3 minute guard.

After adding `src/glass/seclust/incremental.py`:
- `SBM (N=100)` runs in less than 1 second.
- `SBM (N=500)` runs in about 5 seconds.
- `SBM (N=1000)` runs in about 12 seconds.

The core change is that a node move `v: A -> B` only changes two cluster terms:

```text
Delta H = H_after(A) + H_after(B) - H_before(A) - H_before(B)
```

The whole graph no longer needs to be rescored for each candidate.

### 4.2 Objective Tracking
The full benchmark now logs:
- ACC
- NMI
- ARI
- K
- modularity
- structural entropy
- map equation
- runtime/status

This follows `docs/seclust/experiment_protocol.md`.

## 5. Current Issue: Flat Over-Partitioning
SEClust optimizes flat `H_2` today. On several datasets it finds more clusters than the planted or semantic labels:

| Dataset | Expected/Planted K | SEClust K |
| :--- | ---: | ---: |
| Karate | 2 | 6 |
| SBM (N=100) | 4 | 7 |
| SBM (N=500) | 5 | 9 |
| SBM (N=1000) | 10 | 14 |

This is not only an implementation bug. It is a known style of issue for flat community objectives:
- Louvain has a resolution parameter/resolution limit problem.
- Flat `H_2` also has a granularity preference that may split a planted block into smaller structurally meaningful submodules.
- A lower structural entropy partition can disagree with semantic or planted labels if those labels are coarser than the graph's local structure.

The new `SEClust-Tree` row fixes this benchmark issue when the target `K` is supplied:

| Dataset | Target K | SEClust-Tree K | ARI |
| :--- | ---: | ---: | ---: |
| Karate | 2 | 2 | 0.882 |
| SBM (N=100) | 4 | 4 | 1.000 |
| SBM (N=500) | 5 | 5 | 1.000 |
| SBM (N=1000) | 10 | 10 | 1.000 |

The current tree implementation is a high-level merge hierarchy, not yet a full coding-tree optimizer. It proves the direction works: keep fine SE modules, then expose a coarser level for the task.

## 6. Next Fix: Full High-Dimensional Hierarchical SE
The right next step is a full hierarchical clustering algorithm based on high-dimensional structural entropy.

The SEP reference implementation in `official_baselines/SEP/SEPN/codingTree.py` already points to the needed design:
- `CombineDelta()` scores merge deltas.
- `CompressDelta()` scores tree compression.
- `build_coding_tree(k)` constructs a height/depth constrained coding tree.
- `leaf_up()` refines lower-level modules.
- `root_down()` refines upper-level module structure.

Instead of returning only one flat partition, SEClust should build a coding tree. A tree can encode both:
- coarse communities, such as the two factions in Karate
- fine subcommunities, such as structurally tight groups inside those factions

This lets downstream users extract a level appropriate for their task, rather than forcing one global flat resolution. `SEClust-Tree` currently does this at a high level by merging flat modules; the next version should optimize the coding-tree objective directly.

## 7. Next Implementation Plan
### Step 1: Coding Tree Data Model
Add a sparse coding-tree module:

```text
src/glass/seclust/hierarchy.py
```

Core objects:
- `CodingTreeNode`
- `CodingTree`
- node subset volume
- node cut volume
- parent/children links
- height/depth metadata

### Step 2: Hierarchical Objective
Implement high-dimensional SE scoring:

```text
H_T(G) = - sum_{alpha != root} (g_alpha / vol(G)) log2(vol(alpha) / vol(parent(alpha)))
```

Validate that a two-level coding tree matches flat `H_2`.

### Step 3: Incremental Merge Delta
Implement sparse merge deltas analogous to SEP `CombineDelta()`.

Needed state:
- module volume
- module cut
- edge weight between adjacent modules
- active module adjacency

### Step 4: Compression Delta
Implement tree compression analogous to SEP `CompressDelta()`.

Goal:
- remove unnecessary intermediate levels
- prevent excessive depth
- preserve entropy improvements

### Step 5: Leaf-Up and Root-Down Refinement
Implement two local tree refinements:
- **leaf-up:** rebuild local subtrees inside coarse modules
- **root-down:** restructure top-level modules

These are the mechanisms that should reduce flat over-partitioning while preserving meaningful substructure.

### Step 6: Level Selection
Add flat extraction policies:
- `level=k_depth`
- `target_clusters=K`
- best validation metric if labels exist
- MDL-style improvement threshold
- entropy elbow over tree levels

### Step 7: Benchmark Again
Rerun:

```bash
PYTHONPATH=src python tests/benchmark_seclust_full.py
```

Required comparison:
- flat `SEClust-Auto`
- hierarchical `SEClust-Tree`
- Official SEP coding tree
- imported baselines

The target is not simply lower flat `H_2`; it is better control over `K`, ACC, NMI, and ARI while preserving structural entropy quality.

## 8. Success Criteria
For the next milestone:

- `SEClust-Tree` should expose a coding tree and flat labels from selected levels.
- On Karate, it should provide a coarse level near `K=2` while retaining finer substructure below it.
- On SBM benchmarks, selected levels should be closer to planted `K`.
- Runtime should remain under 3 minutes through `N=1000`.
- Reports must include ACC, NMI, ARI, K, modularity, structural entropy, map equation, and runtime.

## 9. Summary
SEClust has moved from a correct small-graph prototype to a scalable flat structural optimizer. The current bottleneck is no longer runtime for thousand-node synthetic graphs; it is resolution control. The next serious algorithmic improvement is a sparse high-dimensional SE coding-tree optimizer inspired by SEP, with explicit level selection for downstream flat labels.
