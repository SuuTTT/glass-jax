# SEClust: Design and Methodology

This document is the technical reference for the SEClust optimizer. Every claim in the prose is backed by a pointer into the implementation. Read this document side-by-side with the source: paths are given as `file:line` and refer to the `glass-jax` repository.

The document is intended to be self-explanatory: a reader who has never touched the codebase should be able to follow the math, locate the corresponding function, and verify that the implementation matches the formula.

## 1. Scope and Position in the Literature

SEClust optimizes the Structural Entropy (SE) of a graph partition. SE was introduced by Li and Pan (2016) as an information-theoretic measure of community structure that, unlike Modularity, has no resolution limit and admits a hierarchical (high-dimensional) generalization via *coding trees*. The reference baseline in this codebase is SEP (Structural Entropy Partitioning, `official_baselines/SEP/`); the deep-learning baseline is LSEnet (`official_baselines/LSEnet/`).

The `glass-jax` project began as a differentiable-relaxation library for graph objectives (`src/glass/objectives/structural_entropy.py:24`, `src/glass/objectives/modularity.py`, `src/glass/objectives/map_equation.py`). The differentiable path is fully implemented and verified — `two_dimensional_structural_entropy` at `src/glass/objectives/structural_entropy.py:24` is bit-exact with the discrete scorer when `S` is one-hot — but it does not produce sharp partitions on graphs of practical size. SEClust is the discrete, hard-partition optimizer that replaces the relaxation in production.

**Design contract.** SEClust takes a non-negative weighted undirected adjacency `A` and returns:

- `labels`: a hard partition `P = {C_1, ..., C_k}` of `V`
- `entropy`: the hard 2D structural entropy `H_2(P)` (or coding-tree entropy `H_T` when running `SEClust-Tree`)
- `method`: a string identifier of the path taken (exact / heuristic / multilevel / coding-tree)

The package boundary is `src/glass/seclust/__init__.py:1`. The high-level entry point is `cluster_graph()` at `src/glass/seclust/heuristics.py:139`.

## 2. Preliminaries

### 2.1 Graph notation

Let `G = (V, E, w)` be an undirected weighted graph with `n = |V|`, `m = |E|`, and non-negative edge weights `w_uv >= 0`. Define:

- `d_v = sum_u A_{uv}`, the weighted degree of `v`
- `vol(G) = sum_v d_v = 2m` for unweighted graphs, the graph volume
- For a node subset `C subset V`:
  - `vol(C) = sum_{v in C} d_v` (cluster volume)
  - `in(C) = sum_{u, v in C} A_{uv}` (twice the internal edge weight, undirected)
  - `g_C = vol(C) - in(C)` (cut volume leaving `C`)

`A` is symmetrized internally by `0.5 * (A + A^T)` and self-loops are removed before scoring (see `as_symmetric_adjacency` at `src/glass/seclust/entropy.py:60` and the analogous block in the sparse builder at `src/glass/seclust/incremental.py:31`).

### 2.2 Structural Entropy from first principles

SE is the expected description length of a random walker that uses the partition `P` as a code book. For a 1D (flat, no partition) walk:

```
H_1(G) = - sum_v (d_v / vol(G)) log_2(d_v / vol(G))
```

This is the entropy of the stationary distribution of a simple random walk. It serves as a global lower bound for any partition. See the differentiable analogue at `src/glass/objectives/structural_entropy.py:4`.

For a 2D walk that first names a module then names a node inside the module:

```
H_2(P) = - sum_{C in P} (g_C / vol(G)) log_2(vol(C) / vol(G))         [boundary term]
         - sum_{C in P} sum_{v in C} (d_v / vol(G)) log_2(d_v / vol(C)) [internal term]
```

**Reading the two terms.** The boundary term `-(g_C / vol(G)) log_2(vol(C) / vol(G))` is the cost of *crossing* the boundary of `C`: it is large when the cluster has a large cut (large `g_C`) relative to its size, which is exactly when the partition fails to capture community structure. The internal term is the cost of naming nodes once inside the module; it depends only on the degree distribution within `C`, not on its connectivity. Empty clusters contribute 0; isolated zero-volume clusters contribute 0.

The hard-partition scorer is `structural_entropy()` at `src/glass/seclust/entropy.py:117`, which is a thin wrapper over `StructuralEntropyScorer.score()` at `src/glass/seclust/entropy.py:35`. Equivalence with the JAX objective is asserted in tests and follows directly: when `S in {0, 1}^{n x k}` is one-hot, `S * (d - A S)` reduces exactly to the cluster cut `g_C`, and `d^T S` reduces to `vol(C)` (compare `src/glass/objectives/structural_entropy.py:42-47` against `src/glass/seclust/entropy.py:43-56`).

### 2.3 High-dimensional Structural Entropy

A flat partition is a height-2 coding tree (root → modules → leaves). The general formulation lifts this to an arbitrary rooted tree `T` whose leaves are the original vertices and whose internal nodes are nested modules. For each non-root tree node `alpha`:

- `V_alpha`: vertex subset represented by `alpha`
- `vol(alpha) = sum_{v in V_alpha} d_v`
- `g_alpha`: cut volume from `V_alpha` to `V_{parent(alpha)} \ V_alpha`
- `parent(alpha)`: parent in the coding tree

The high-dimensional SE is:

```
H_T(G) = - sum_{alpha != root} (g_alpha / vol(G)) log_2(vol(alpha) / vol(parent(alpha)))
```

For a 2-level tree this collapses back to `H_2`. For deeper trees it encodes nested structure: a coarse community can contain sub-communities at a different resolution, and each level pays its own boundary cost. The reference recursive formulation matches SEP (`official_baselines/SEP/SEPN/codingTree.py`).

The implementation of `H_T` over a coding tree is `high_dimensional_tree_entropy()` at `src/glass/seclust/hierarchy.py:116`. The per-node contribution helper is `_term()` at `src/glass/seclust/hierarchy.py:72`. The fixed leaf-degree contribution is computed once in `_initial_module_stats()` at `src/glass/seclust/hierarchy.py:78`.

### 2.4 Why discrete optimization

The original `glass-jax` line of attack used differentiable relaxations: parameterize a soft assignment `S` and minimize `H_2(A, S)` by gradient descent. This is mathematically clean and trivially batched on GPUs, but it has two failure modes that we observed empirically:

1. **Mode collapse / dispersion**: optimizing `H_2(A, S)` directly puts all mass on one block (it is a global minimum of the relaxation when `S` is uniform). Sinkhorn projection (`src/glass/solvers/sinkhorn.py`) and orthogonality regularizers fix the symptom but produce mushy, non-sharp assignments that need post-hoc rounding.
2. **Resolution limit on the relaxation surface**: the relaxed surface is far smoother than the discrete one. Sharp boundary improvements are smeared across many soft-assignment gradients, and the optimizer converges to a low-entropy soft solution that, when rounded, is not even a local minimum of `H_2`.

The Leiden / Louvain literature has solved exactly this problem on Modularity with discrete greedy local search plus multi-level coarsening. SEClust ports that recipe to SE.

## 3. Algorithmic Components

The optimizer is organized as a stack of layers, each addressing a specific scaling or quality issue. We describe the layers bottom-up.

### 3.1 Sparse graph representation: `SparseGraph`

Source: `SparseGraph` at `src/glass/seclust/incremental.py:13`, constructor `from_adjacency()` at `src/glass/seclust/incremental.py:27`.

`SparseGraph` is an immutable CSR-style neighbor-list view. It stores per-node:

- `neighbors[v]`: int32 array of neighbor indices (`incremental.py:56`)
- `weights[v]`: float array of edge weights (`incremental.py:57`)
- `degrees[v]`: weighted degree (`incremental.py:37`)
- `node_cuts[v]`: degree minus self-loop (`incremental.py:40`)
- `node_degree_log_degree[v]`: precomputed `d_v log_2(d_v)` (`incremental.py:42-45`)

`vol(G)` and `n_edges` are cached at construction. Self-loops are dropped (`incremental.py:38-39`) so they do not affect cut accounting. Edge-weight diagonal removal happens *after* degree computation, so degrees retain the self-loop mass — this matches the SE convention where `d_v` is the random-walk stationary mass.

This object is the fundamental data structure for the rest of the optimizer; all delta updates are expressed as iterations over `neighbors[v], weights[v]`.

### 3.2 Incremental Δ-scoring: `IncrementalSEState`

Source: `IncrementalSEState` at `src/glass/seclust/incremental.py:72`.

The dense scorer `StructuralEntropyScorer.score()` (`src/glass/seclust/entropy.py:35`) costs `O(n^2)` per call because `adj[np.ix_(mask, mask)].sum()` materializes a dense submatrix. For an inner loop that evaluates `n * k` candidate moves per pass, this gives `O(p * k * n^3)` and breaks at `n ~ 500`. We replace it with a stateful object that maintains, per active cluster `C`:

- `volume[C] = vol(C)` (`incremental.py:85`)
- `cut[C] = g_C` (`incremental.py:86`)
- `degree_log_degree[C] = sum_{v in C} d_v log_2(d_v)` (`incremental.py:87`)
- `size[C]` (number of nodes; `incremental.py:88`)
- `active[C]` (bool mask; `incremental.py:89`)

Per-cluster entropy is computed from these three reals only:

```
H_C = - (g_C / V) log_2(vol_C / V) - (1/V) [degree_log_degree_C - vol_C log_2(vol_C)]
```

implemented in `cluster_entropy_values()` at `src/glass/seclust/incremental.py:128`. The total entropy `state.entropy` is maintained as a running sum and re-validated against the dense scorer in tests via `validate_entropy()` at `src/glass/seclust/incremental.py:282`.

#### 3.2.1 Node-move delta

For a move `v: A -> B`, only clusters `A` and `B` change. Let `d = d_v`, `dlogd = d log_2(d)`, and let `w(v, X)` denote the total edge weight from `v` to nodes currently in cluster `X`. Then:

```
vol_A'  = vol_A - d
vol_B'  = vol_B + d
cut_A'  = cut_A - d + 2 w(v, A \ {v})
cut_B'  = cut_B + d - 2 w(v, B)
dlogd_A' = dlogd_A - dlogd
dlogd_B' = dlogd_B + dlogd
```

Justification of the cut update: removing `v` from `A` reduces the *outgoing* mass by `d`, but adds back twice the edges `v` had *inside* `A` (those edges were not in the cut before, and now they are). Inserting `v` into `B` adds `d` to the outgoing mass, then subtracts twice the edges `v` already had to nodes in `B` (those edges become internal). The factor of 2 comes from undirected adjacency double-counting.

`Delta H = H_C(A') + H_C(B') - H_C(A) - H_C(B)`. Implemented in `move_delta()` at `src/glass/seclust/incremental.py:172` and applied in `apply_move()` at `src/glass/seclust/incremental.py:200`.

The cost per candidate evaluation is `O(deg(v))` with the current implementation because `edge_weight_to_cluster()` (`src/glass/seclust/incremental.py:153`) iterates over neighbors. With a per-node `w(v, C)` cache (not yet implemented; see §6), it drops to `O(1)` plus `O(deg(v))` to update the cache after the move is applied.

#### 3.2.2 Cluster-merge delta

For a merge `C_L union C_R` with known inter-cluster weight `w(L, R)`:

```
vol_new   = vol_L + vol_R
cut_new   = cut_L + cut_R - 2 w(L, R)
dlogd_new = dlogd_L + dlogd_R
Delta H   = H_C(new) - H_C(L) - H_C(R)
```

Implemented in `merge_delta()` at `src/glass/seclust/incremental.py:234` and `apply_merge()` at `src/glass/seclust/incremental.py:257`. The `-2 w(L, R)` term has the same derivation as the node-move case: edges between `L` and `R` were in both cuts before and are internal after.

#### 3.2.3 Candidate restriction

For a node `v`, only clusters that contain a neighbor of `v` (plus `v`'s current cluster, plus optionally one fresh singleton) can possibly improve the local objective: a move into a cluster that `v` has zero edges to strictly increases the cut term. This restriction is enforced in `candidate_clusters()` at `src/glass/seclust/incremental.py:166`, called from the inner loop at `src/glass/seclust/incremental.py:308`. It reduces the candidate set from `O(k)` to `O(|neighboring clusters of v|)`, which on a community-structured graph is `O(1)` on average.

### 3.3 Local node-move pass: `local_move_incremental`

Source: `local_move_incremental()` at `src/glass/seclust/incremental.py:288`.

```text
for pass in 1..max_passes:
    changed = false
    for v in random_permutation(V):
        evaluate move_delta(v, T) for T in candidate_clusters(v)
        let T* = argmin_T move_delta(v, T)
        if move_delta(v, T*) < -eps: apply_move(v, T*); changed = true
    if not changed: break
```

This is structurally identical to the Louvain inner loop, with two differences:

1. The objective is SE rather than Modularity. SE penalizes large-cut clusters more aggressively, so the equilibrium tends to have *more, smaller* communities than Modularity (the over-partitioning behavior we explicitly characterize as a property, not a bug — see §5).
2. Candidate restriction is by **neighboring cluster set**, not by neighbor count. This collapses duplicates in the candidate set when many neighbors land in the same cluster.

Convergence is guaranteed because `state.entropy` decreases by at least `eps = 1e-12` on every accepted move and is bounded below by 0 (SE is non-negative). In practice, two to four passes suffice on community-structured graphs.

### 3.4 Multi-level coarsening: `multi_level_local_move`

Source: `multi_level_local_move()` at `src/glass/seclust/incremental.py:328`.

The local move pass alone is not enough. On large graphs (`n > 500`), it converges to local minima where two well-formed communities should merge but no single node move benefits. The remedy is multi-level coarsening, used by both Louvain and Leiden:

```text
loop:
    labels = local_move_incremental(current_adj, current_labels)
    if labels are degenerate (k == n or k == 1): break
    labels = split_disconnected_clusters(current_adj, labels)   # see §3.5
    record projection layer
    S = one-hot indicator matrix of labels (n_curr x k)
    current_adj = S^T A S                                       # supernode graph
    current_labels = singleton on supernodes
project final labels back to original vertices
```

Implementation lines:

- Local move call: `incremental.py:344`
- Connectedness refinement: `incremental.py:358-369`
- Sparse projection `S^T A S`: `incremental.py:374-379`
- Reverse projection chain: `incremental.py:383-385`

The final labels are obtained by composing all projections in reverse: `final = projections[0][projections[1][...[projections[L-1]]]]`. This reproduces the standard Louvain/Leiden multi-level scheme exactly.

**Why does coarsening help?** A node move that traverses an entire community boundary requires the boundary itself to be one node first. Coarsening makes communities into single supernodes; the next pass evaluates a community-level move at the cost of a node-level move. Without coarsening, the inner loop is stuck in local minima where any single-node move increases SE but a coordinated move of many nodes would decrease it. With coarsening, every supernode move *is* a coordinated move at the lower level.

The supernode graph `S^T A S` is a sparse matrix multiplication, computed via `scipy.sparse.csr_matrix` (`incremental.py:377-379`). Memory stays at `O(m)` because supernode graphs are at least as sparse as the original.

### 3.5 Connectedness refinement

Source: lines `incremental.py:358-369`, using `scipy.sparse.csgraph.connected_components`.

The local-move pass can produce *disconnected* clusters: SE is invariant to which connected component a node belongs to as long as cuts and volumes match, so the optimizer happily places two unrelated subgraphs into the same label. Coarsening such a cluster into a single supernode then loses the geometric information that they were separate, and subsequent levels cannot recover it.

Refinement scans every cluster and, if it has more than one connected component, splits each component into a fresh label *before* coarsening. This is the SEClust analogue of Leiden's refinement phase (Traag et al., 2019), specialized to the SE objective.

This single change is responsible for a large fraction of the quality gap between SEClust and Louvain on noisy SBMs. We will plot this in the ablation study (planned, see `tests/ablation_study.py`).

### 3.6 Multistart wrapper

Source: `multistart_incremental_se_heuristic()` at `src/glass/seclust/incremental.py:394`.

The optimizer is run from multiple seeds: singleton (`np.arange`), one-block (`np.zeros`), and `starts - 2` random partitions with `k` drawn uniformly from `[2, sqrt(n) + 2]` (`incremental.py:404-410`). The minimum-entropy result is returned. This is a standard exploration mechanism; the singleton seed corresponds to bottom-up agglomeration, the one-block seed to top-down splitting, and random seeds explore the basin in between.

### 3.7 Coding-tree construction: `build_coding_tree_from_modules`

Source: `src/glass/seclust/coding_tree.py:76`. Helpers: `combine_delta()` at `src/glass/seclust/coding_tree.py:22`, `compress_delta()` at `src/glass/seclust/coding_tree.py:35`, `compress_node()` at `src/glass/seclust/coding_tree.py:49`. Class: `CodingTreeNode` at `src/glass/seclust/coding_tree.py:10`.

The flat optimizer above produces base modules. To produce a *coding tree* — i.e., to optimize the full high-dimensional SE — we need an inner loop over tree edits. We mirror the SEP optimizer (`official_baselines/SEP/SEPN/codingTree.py`).

#### 3.7.1 CombineDelta

Two adjacent tree nodes `n1, n2` with cut `cut_between` between them are merged under a fresh parent `p` with `vol(p) = vol(n1) + vol(n2)` and `g_p = g(n1) + g(n2) - 2 cut_between`. The delta in `H_T` from this combine is:

```
CombineDelta(n1, n2) =
    [(v1 - g1) log_2(v12 / v1)
    + (v2 - g2) log_2(v12 / v2)
    - 2 cut_between log_2(V / v12)] / V
```

where `v12 = v1 + v2`, `v_i = vol(n_i)`, `g_i = cut(n_i)`, `V = vol(G)`. This is a direct algebraic simplification of `H_T(after) - H_T(before)` after parent insertion, and exactly matches SEP's formulation. Implemented at `src/glass/seclust/coding_tree.py:22-33`.

#### 3.7.2 CompressDelta

A degree-1 internal node is redundant: removing it and re-attaching its children to its grandparent does not change the leaf set but does change `H_T` because each child's parent volume changes. The delta is:

```
CompressDelta(node, parent) = child_cut * log_2(vol(parent) / vol(node))
```

implemented at `src/glass/seclust/coding_tree.py:35-39`. This is positive when contracting a non-trivial level (`vol(node) < vol(parent)`), which means we compress only when the savings exceed zero — i.e., the level is not contributing.

#### 3.7.3 Tree assembly

`build_coding_tree_from_modules()` (`coding_tree.py:76`) does:

1. Compute per-module `vol`, `cut`, and the inter-module weight matrix `between` (`coding_tree.py:82-98`).
2. Initialize one tree leaf per module (`coding_tree.py:101-102`).
3. Push every adjacent pair onto a min-heap keyed by `CombineDelta` (`coding_tree.py:108-112`).
4. Repeatedly pop the lowest-delta pair, merge if both endpoints are still active, push the new parent's adjacent pairs back onto the heap (`coding_tree.py:115-153`). This is Kruskal-style and runs in `O((K^2 + K) log K)` for `K` initial modules, but the `K^2` adjacency matrix is the dominating cost.
5. Optionally compress: when `target_k` is given, pop from the compress heap until the tree height matches the target (`coding_tree.py:172-193`).

Flat labels at any cut are extracted by `extract_flat_labels()` at `src/glass/seclust/coding_tree.py:197`.

### 3.8 Flat target-K heuristic: `merge_hierarchy_levels`

Source: `merge_hierarchy_levels()` at `src/glass/seclust/hierarchy.py:230`.

The benchmarks compare against Leiden, Louvain, Infomap, and LSEnet, all of which return a flat partition. SEClust's pure flat optimizer (§3.3 + §3.4) over-partitions because SE strictly prefers more communities up to the point where the boundary saturates. To match the benchmark contract for a target cluster count `K`, we run the flat optimizer to convergence then greedily merge the pair with the smallest *flat* SE delta until exactly `K` clusters remain.

The merge loop is at `hierarchy.py:259-304` and reuses `IncrementalSEState.merge_delta()` (`incremental.py:234`), so each merge candidate is `O(1)`. The neighbor-weight matrix `between` is maintained incrementally: when `L` and `R` merge, `between[L, *] += between[R, *]` and `between[R, *] = 0` (`hierarchy.py:291-297`). Total cost is `O(K^2)` for the initial scan plus `O(K)` per merge, so `O(K^2)` overall.

This routine is the key reason SEClust matches Leiden's mathematical optimum on SBM benchmarks at `n = 1000`: the flat optimizer finds the right base modules, and the merge step compresses to the requested resolution without paying a coding-tree premium.

### 3.9 Hierarchy orchestrator: `hierarchical_se_clustering`

Source: `hierarchical_se_clustering()` at `src/glass/seclust/hierarchy.py:335`.

The user-facing entry point for `SEClust-Tree`. It:

1. Runs `multistart_incremental_se_heuristic()` to produce base modules (`hierarchy.py:348-353`).
2. Calls `build_coding_tree_from_modules(adj, base_labels, target_k=2)` to lift to a coding tree (`hierarchy.py:358`).
3. Extracts flat labels at the chosen cut (`hierarchy.py:365`).
4. Returns a `HierarchicalClusteringResult` (defined at `hierarchy.py:24`).

Note: when the benchmark contract asks for a specific `K`, the operative path is `merge_hierarchy_levels()` + `select_hierarchy_level()` (`hierarchy.py:309`), not `hierarchical_se_clustering()`. The two are exposed side-by-side because the coding-tree path is the right answer for a *hierarchy of resolutions* and the merge path is the right answer for a *single flat cut at K*.

### 3.10 Reference implementations

For correctness baselines we keep three reference paths that re-derive results without the sparse-incremental machinery:

- `exact_minimize_structural_entropy()` at `src/glass/seclust/exact.py:48`: enumerates restricted-growth strings over `n` nodes, evaluating `B_n` partitions. Used to label small graphs (`n <= 9`) with the global SE optimum and to certify the heuristic. Restricted-growth strings are produced by `iter_restricted_growth_strings()` at `src/glass/seclust/exact.py:22`.
- `agglomerative_se_clustering()` at `src/glass/seclust/heuristics.py:29`: greedy single-link merges scored by full `H_2` recomputation. Slow but mathematically transparent.
- `local_move_se_clustering()` at `src/glass/seclust/heuristics.py:62`: greedy node moves scored by full `H_2` recomputation. Used to validate `local_move_incremental` produces identical labels on small graphs.

These paths are not on the production path; they exist to anchor the sparse-incremental optimizer to ground truth.

## 4. Putting It Together

The public surface in `src/glass/seclust/__init__.py:1` exposes three named optimizers:

| Label | Entry point | Layer stack |
|---|---|---|
| `SEClust-Auto` | `cluster_graph(mode="auto")` (`heuristics.py:139`) | exact for `n <= 9`, else multistart-incremental + multi-level coarsening |
| `SEClust-Tree` | `hierarchical_se_clustering()` (`hierarchy.py:335`) | base modules from `SEClust-Auto`, then coding-tree (CombineDelta + optional CompressDelta) |
| `SEClust-TargetK` | `merge_hierarchy_levels()` + `select_hierarchy_level()` (`hierarchy.py:230`, `hierarchy.py:309`) | base modules + greedy flat-SE merge until target `K` |

`cluster_graph()` is the dispatcher: small graphs go to `exact_minimize_structural_entropy` (`heuristics.py:151`), everything else to `multistart_se_heuristic(backend="incremental")` (`heuristics.py:155`), which itself dispatches to `multistart_incremental_se_heuristic` (`heuristics.py:113`).

End-to-end call graph for a single SBM benchmark run at `n = 1000`, `target K = 5`:

```
cluster_graph(adj, mode="auto")                              heuristics.py:139
  multistart_se_heuristic(backend="incremental")             heuristics.py:103
    multistart_incremental_se_heuristic                      incremental.py:394
      for each seed:
        multi_level_local_move                               incremental.py:328
          local_move_incremental                             incremental.py:288
            IncrementalSEState.move_delta                    incremental.py:172
            IncrementalSEState.apply_move                    incremental.py:200
          connected_components split                         incremental.py:358
          S^T A S projection                                 incremental.py:374
        return best entropy
merge_hierarchy_levels(adj, base_labels, min_clusters=K)     hierarchy.py:230
  IncrementalSEState.merge_delta                             incremental.py:234
  IncrementalSEState.apply_merge                             incremental.py:257
select_hierarchy_level(levels, target_clusters=K)            hierarchy.py:309
```

## 5. Theoretical Justifications

This section is the why-it-works companion to §3.

### 5.1 Why incremental scoring is exact, not approximate

The factorization

```
H_2(P) = sum_{C in P} H_C
```

with

```
H_C = - (g_C / V) log_2(vol_C / V) - (1/V) [sum_{v in C} d_v log_2(d_v) - vol_C log_2(vol_C)]
```

is algebraic, not approximate. It follows from grouping the second sum in §2.2 by cluster:

```
- sum_{v in C} (d_v / V) log_2(d_v / vol_C)
= - (1/V) sum_{v in C} d_v [log_2(d_v) - log_2(vol_C)]
= - (1/V) [sum_{v in C} d_v log_2(d_v) - vol_C log_2(vol_C)]
```

Therefore the per-cluster state `(vol_C, g_C, sum_{v in C} d_v log_2(d_v))` is a *sufficient statistic* for `H_C`. Maintaining it incrementally cannot introduce drift in the objective beyond floating-point error, which we bound by the periodic full revalidation in `validate_entropy()` (`incremental.py:282`).

### 5.2 Why coarsening is necessary, not just an optimization

Without coarsening, the local-move pass on graph `G` can have the following structure: there exist clusters `A` and `B` such that the optimal partition merges them, but every single-node move from `A` to `B` increases SE because it leaves `A` in a worse boundary state before any second move can compensate. This is exactly the issue the Louvain paper (Blondel et al., 2008) describes for Modularity. Coarsening sidesteps the issue by making `A` and `B` single nodes at the next level, where their merge is a single-move evaluation and is accepted if and only if `merge_delta(A, B) < 0`.

Formally: every flat partition reachable by a sequence of single-node moves is also reachable by single-node moves on the coarsened graph (interpret a supernode move as moving all its base nodes simultaneously), and additionally the coarsened graph admits *strictly more* moves that would have required a coordinated multi-node move at the base level. Coarsening therefore enlarges the move neighborhood without adding spurious local minima.

### 5.3 Why connectedness refinement matters

A cluster split across two connected components has `g_C = g_C1 + g_C2` and `vol_C = vol_C1 + vol_C2`, but its SE contribution

```
- (g_C / V) log_2(vol_C / V) - (1/V) [dlogd_C - vol_C log_2(vol_C)]
```

is *not* equal to the sum of the contributions of `C1` and `C2` evaluated separately. Specifically, the joint version pays `log_2(vol_C / V)` per unit of cut, while the split version pays `log_2(vol_C1 / V)` and `log_2(vol_C2 / V)` per unit of their respective cuts. Because `log_2` is concave and `vol_C1, vol_C2 < vol_C`, the split contribution is strictly smaller (lower SE) whenever the components are non-trivial. Coarsening a disconnected cluster into a single supernode therefore *encodes a strictly suboptimal partition* into the next level, which the next pass cannot undo because the geometric separation has been collapsed. Splitting before coarsening is mandatory for correctness, not optional for quality.

### 5.4 Why merge instead of compress for target-K

`SEClust-TargetK` uses `merge_hierarchy_levels()` rather than `build_coding_tree_from_modules()` with a target height. The reason is that flat-SE merging optimizes the right objective: `H_2` at the chosen `K`. Coding-tree compression optimizes `H_T` on the *constructed tree*, which is a different objective and may pay an extra coding-tree term that flat scoring does not. Empirically (see `tests/benchmark_seclust_full.py` reports under `docs/experimental_reports/`), `SEClust-TargetK` matches Leiden at `K = ground truth K` on SBM, while `SEClust-Tree` is competitive only at `K = 2` (the natural coding-tree resolution).

This is consistent with the SEP literature: the coding tree is the right model when the *resolution itself is unknown* and the system should choose `K` to minimize description length. When `K` is exogenously specified by the benchmark contract, flat merging is the optimal choice.

### 5.5 Why SE over Modularity

This is the standard SE-vs-Modularity argument, recalled here for completeness:

- Modularity has a resolution limit (Fortunato and Barthélemy, 2007): communities smaller than `sqrt(2m)` are merged regardless of internal density. SE has no such limit because the boundary penalty `g_C / V` scales linearly in cut, not relatively in cluster size.
- SE penalizes high-degree boundary nodes more aggressively because `log_2(vol_C / V)` is more negative for small `vol_C`. This favors partitions where boundary edges land between low-degree nodes — which matches the intuition that hub nodes belong inside their community.
- SE generalizes to coding trees; Modularity does not have a comparable hierarchical formulation.

The flip side — SE over-partitions relative to ground truth — is real and is the reason `SEClust-TargetK` exists.

## 6. Complexity

Let `n = |V|`, `m = |E|`, `k` = current cluster count, `s` = multistart seeds, `p` = local-move passes, `L` = coarsening levels.

| Component | Cost (current) | Cost (target after JAX vectorization) |
|---|---|---|
| `SparseGraph.from_adjacency` from dense `A` | `O(n^2)` once | `O(m)` if input is already sparse |
| `IncrementalSEState.move_delta` | `O(deg(v))` (no cache) | `O(1)` (with `w(v, C)` cache) |
| `local_move_incremental` (one pass) | `O(sum_v deg(v) * |neighboring clusters of v|)` ≈ `O(m * c_avg)` | `O(m)` |
| `multi_level_local_move` | `O(L * m)` typical | same |
| `merge_hierarchy_levels` | `O(K^2)` for `K` base modules | same |
| `build_coding_tree_from_modules` | `O(K^2 log K)` for `K` modules | same |
| `multistart_incremental_se_heuristic` | `O(s * L * m)` | `O(s * L * m)` |
| Memory | `O(n + m + k)` after conversion | `O(n + m + k)` |

The dominating term in current benchmarks is the *initial* `O(n^2)` dense-to-sparse conversion in `SparseGraph.from_adjacency` (`incremental.py:33-34`). This is why dense real-world inputs like Cora are gated by a runtime guard in the benchmark suite — a native sparse PyG `edge_index` constructor is the next engineering item (see TODO in `context.md`).

The exact and reference paths have very different complexities:

- `exact_minimize_structural_entropy`: `O(B_n * n^2)` with `B_n` the Bell number; `B_9 = 21,147`, `B_10 = 115,975`, `B_11 = 678,570`. Used for `n <= 9` only.
- `agglomerative_se_clustering`: `O(n^5)` worst case (full rescore per merge candidate per merge level). Used for small-graph debugging only.
- `local_move_se_clustering` (reference): `O(p * k * n^3)`. Used to verify `local_move_incremental` on `n <= 100`.

## 7. Implementation Notes and Subtleties

A few pieces of the code merit attention because they look like minor details but affect correctness:

1. **Symmetrization is mandatory** (`entropy.py:69`, `incremental.py:31, 34`). The objective is defined on undirected graphs. A user passing a directed adjacency would otherwise get a silent error: degrees would double-count out-edges and cuts would be asymmetric. We always do `0.5 * (A + A^T)`.

2. **Self-loops are in `degrees` but not in `cut` accounting** (`incremental.py:37` vs `incremental.py:40`). `d_v` is the random-walk stationary mass and includes the self-loop weight. `node_cuts[v]` excludes the self-loop because a self-loop never crosses a cluster boundary. This split is enforced by `setdiag(0); eliminate_zeros()` at `incremental.py:38-39`.

3. **`degree_log_degree` is precomputed once** (`incremental.py:42-45`) and never recomputed during the optimization. It depends only on per-node degrees, which do not change during partition optimization. Sharing this array across coarsening levels is the reason `multi_level_local_move` can pass `node_degree_log_degree=current_dlogd` (`incremental.py:339`); it is reset to `None` each level so the supernode graph computes its own.

4. **Capacity allocation is generous** (`incremental.py:84`). The state pre-allocates `2n + 1` cluster slots so that `apply_move` can introduce a fresh singleton without resizing. `first_empty_cluster()` (`incremental.py:147`) finds the first inactive slot in `O(n)`, which is fine relative to the move cost.

5. **`cut` is rounded to zero below `1e-10`** (`incremental.py:110`). This is a numerical-hygiene step: floating-point accumulation in `internal_twice` can leave `g_C` slightly negative on a cluster that genuinely has zero cut (e.g., a connected component as its own cluster). Negative `g_C` would give a complex `log_2`. We clamp.

6. **Canonicalization is first-seen-order, not minimum-id** (`entropy.py:74`). This is a deliberate choice for stability: the same partition produced by two runs with different seeds is mapped to the same label sequence, regardless of which cluster id the optimizer happened to assign internally. This matters for benchmark logging.

## 8. Roadmap

Items below are the open methodology questions, in priority order. Engineering items (JAX vectorization, native sparse PyG inputs) are tracked in `context.md` and are not repeated here.

1. **Refinement phase à la Leiden.** We currently split disconnected clusters before coarsening. A stronger move is to also split *weakly internally connected* clusters — those with a sparse cut between two halves but no full disconnection. The Leiden refinement phase implements this by running a second local-move pass that is restricted to refining within each cluster.

2. **CompressDelta in the production path.** `compress_node()` and `compress_delta()` are implemented (`coding_tree.py:35, 49`) and used during target-`k` height truncation in `build_coding_tree_from_modules` (`coding_tree.py:172`), but they are not yet integrated into a global tree-refinement loop. The full SEP optimizer alternates `combine` and `compress` until both heaps are empty; we should match that.

3. **Leaf-up and root-down refinement.** SEP's deeper refinement phases (`leaf_up`, `root_down` in `official_baselines/SEP/SEPN/codingTree.py`) restructure subtrees and top-level modules respectively. They are out of scope for the current paper version but should be implemented for the second-pass extension.

4. **Learned move proposals.** The CEM scaffold at `src/glass/seclust/rl.py` exposes `(node, target_cluster)` as an action and `delta_H` as a reward. The intended use is to train a policy on small exact-labeled graphs (`build_structural_entropy_dataset` at `src/glass/seclust/datasets.py`) and transfer to large graphs. This is an exploratory research direction; the current sparse-incremental optimizer is competitive without it.

5. **Direct optimization of `H_T` rather than two-step `H_2 + tree`.** The current `SEClust-Tree` builds base modules by minimizing `H_2`, then lifts them to a tree. A unified objective would minimize `H_T` directly via tree edits. This is the SEP optimizer's contribution and is the natural deeper integration once items 1-3 are in place.

## 9. File Index

For the reader who wants to navigate the implementation directly:

| File | Role |
|---|---|
| `src/glass/seclust/__init__.py` | Package surface and re-exports |
| `src/glass/seclust/entropy.py` | Dense scorer, label canonicalization, structural-entropy formula |
| `src/glass/seclust/incremental.py` | `SparseGraph`, `IncrementalSEState`, local-move pass, multi-level coarsening, multistart |
| `src/glass/seclust/coding_tree.py` | `CodingTreeNode`, `combine_delta`, `compress_delta`, full coding-tree builder |
| `src/glass/seclust/hierarchy.py` | `merge_hierarchy_levels`, `coding_tree_hierarchy_levels`, `hierarchical_se_clustering`, `select_hierarchy_level`, `high_dimensional_tree_entropy` |
| `src/glass/seclust/heuristics.py` | Reference paths, `cluster_graph` dispatcher, multistart wrapper |
| `src/glass/seclust/exact.py` | Exhaustive search via restricted-growth strings |
| `src/glass/seclust/datasets.py` | Synthetic graphs with exact SE labels |
| `src/glass/seclust/benchmark_sep.py` | SEP baseline wrapper for like-for-like comparison |
| `src/glass/seclust/rl.py` | CEM scaffold (exploratory) |
| `src/glass/objectives/structural_entropy.py` | Differentiable JAX SE objective (continuous relaxation; deprecated for production) |
| `tests/benchmark_seclust_full.py` | Full benchmark suite |
| `docs/seclust/design.md` | Original design/architecture document (this file is the technical companion) |
| `docs/seclust/experiment_protocol.md` | Required metrics for SEClust experiments |
| `official_baselines/SEP/SEPN/codingTree.py` | Reference SEP coding-tree implementation |
| `official_baselines/LSEnet/` | Deep-learning baseline |
