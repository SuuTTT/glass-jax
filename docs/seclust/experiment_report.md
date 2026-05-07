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
3. **Hierarchical coarse extraction:** `SEClust-Tree` now merges fine SEClust modules with high-dimensional coding-tree deltas and extracts target-`K` levels.

The main remaining algorithmic issue is now narrower: the first high-dimensional `SEClust-Tree` implementation controls `K` when a target level is supplied, but its greedy merge-only coding tree is not yet as strong as a full SEP-style optimizer. It does not yet implement compression, `leaf_up`, or `root_down` refinement from `official_baselines/SEP/SEPN/codingTree.py`.

## 2. Validation Runs
### 2.1 Unit Tests
Command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_seclust.py
```

Result:

```text
11 passed, 1 skipped
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
| Karate / SEClust-Auto | 0.168 | 0.471 | 0.510 | 0.314 | 6 | 0.342 | 3.427 | 4.612 |
| Karate / SEClust-Tree | 0.351 | 0.971 | 0.836 | 0.882 | 2 | 0.357 | 3.849 | 4.464 |
| Caveman / SEClust-Auto | 0.641 | 1.000 | 1.000 | 1.000 | 10 | 0.895 | 4.337 | 4.380 |
| Caveman / SEClust-Tree | 0.363 | 1.000 | 1.000 | 1.000 | 10 | 0.895 | 4.337 | 4.380 |
| SBM N=100 / SEClust-Auto | 0.290 | 0.720 | 0.850 | 0.731 | 7 | 0.513 | 4.849 | 5.815 |
| SBM N=100 / SEClust-Tree | 0.339 | 1.000 | 1.000 | 1.000 | 4 | 0.626 | 4.840 | 5.399 |
| SBM N=500 / SEClust-Auto | 5.831 | 0.936 | 0.945 | 0.931 | 9 | 0.582 | 7.073 | 7.874 |
| SBM N=500 / SEClust-Tree | 6.105 | 0.604 | 0.737 | 0.481 | 5 | 0.441 | 7.758 | 8.298 |
| SBM N=1000 / SEClust-Auto | 12.198 | 0.944 | 0.968 | 0.947 | 14 | 0.561 | 7.676 | 8.762 |
| SBM N=1000 / SEClust-Tree | 12.413 | 0.897 | 0.966 | 0.894 | 10 | 0.575 | 7.765 | 8.780 |

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

The new `SEClust-Tree` row fixes the reported `K` when the target is supplied, but the first high-dimensional merge-only scorer does not always preserve planted-block quality:

| Dataset | Target K | SEClust-Tree K | ARI |
| :--- | ---: | ---: | ---: |
| Karate | 2 | 2 | 0.882 |
| SBM (N=100) | 4 | 4 | 1.000 |
| SBM (N=500) | 5 | 5 | 0.481 |
| SBM (N=1000) | 10 | 10 | 0.894 |

This is the current central issue. The tree has the right target resolution, but greedy high-dimensional merges can combine the wrong fine modules on larger SBM graphs. The next version needs coding-tree refinement and better level/merge selection, not just target-`K` extraction.

## 6. Current High-Dimensional SEClust-Tree
`SEClust-Tree` now uses high-dimensional structural entropy for the hierarchy merge objective.

The implemented coding-tree score is:

```text
H_T(G) = - sum_{alpha != root} (g_alpha / vol(G)) log2(vol(alpha) / vol(parent(alpha)))
```

The implementation operates over flat SEClust base modules:
- original vertices remain leaf terms under their base module
- active module roots are greedily merged
- each merge is scored by its exact `Delta H_T(G)` contribution
- the selected flat labels are extracted from a requested tree level, usually `target_clusters`

For a candidate merge of active roots `a` and `b`:

```text
Delta =
  h(a union b | root) + h(a | a union b) + h(b | a union b)
  - h(a | root) - h(b | root)
```

with `g_{a union b} = g_a + g_b - 2 w(a, b)`.

So yes, the current tree is using the high-dimensional SE formulation now. It is not the complete SEP algorithm yet.

## 7. Next Fix: Full Coding-Tree Refinement
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

This lets downstream users extract a level appropriate for their task, rather than forcing one global flat resolution. `SEClust-Tree` now does this with direct high-dimensional merge scoring; the next version should refine the tree after greedy construction.

Current objective status:

| Algorithm | Hierarchical Output | Objective Used Now | Direct High-Dim SE? |
| :--- | :--- | :--- | :--- |
| `SEClust-Auto` | no | flat `H_2` | no |
| `SEClust-Tree` | yes | greedy coding-tree `H_T(G)` merge deltas | yes, merge-only |
| planned refined tree | yes | `H_T(G)` merge/compress/refine | yes |

## 8. Next Implementation Plan
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
High-dimensional SE scoring is implemented for merge trees:

```text
H_T(G) = - sum_{alpha != root} (g_alpha / vol(G)) log2(vol(alpha) / vol(parent(alpha)))
```

Definitions:
- `alpha` is a non-root coding-tree node.
- `vol(alpha)` is the degree volume of the vertices represented by `alpha`.
- `g_alpha` is the cut volume from `alpha` to its sibling region inside `parent(alpha)`.
- `vol(parent(alpha))` is the parent module volume.

The focused test validates that a two-level coding tree matches flat `H_2`.

### Step 3: Incremental Merge Delta
The current merge delta is exact but still simple. Next, make it sparse and update only affected adjacent pairs, analogous to SEP `CombineDelta()`.

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

## 9. Success Criteria
For the next milestone:

- `SEClust-Tree` should expose a coding tree and flat labels from selected levels.
- On Karate, it should provide a coarse level near `K=2` while retaining finer substructure below it.
- On SBM benchmarks, selected levels should be closer to planted `K`.
- Runtime should remain under 3 minutes through `N=1000`.
- Reports must include ACC, NMI, ARI, K, modularity, structural entropy, map equation, and runtime.

## 10. Summary
SEClust has moved from a correct small-graph prototype to a scalable flat structural optimizer plus a first high-dimensional merge tree. The current bottleneck is no longer runtime for thousand-node synthetic graphs; it is hierarchical merge quality. The next serious algorithmic improvement is sparse high-dimensional SE coding-tree refinement inspired by SEP, with explicit level selection for downstream flat labels.
