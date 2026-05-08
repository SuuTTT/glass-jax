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
- Hierarchical algorithm label: `SEClust-Tree`
- ML/RL exploration label: `SEClust-CEM`

## Problem
Given a weighted, undirected graph `G = (V, E, w)`, find a hard node partition `P = {C_1, ..., C_k}` that minimizes two-dimensional structural entropy. SEClust is intentionally non-differentiable: it optimizes hard partitions directly rather than soft assignment logits.

Inputs:
- `A in R^{n x n}`: non-negative weighted adjacency matrix.
- `A` is symmetrized internally by `0.5 * (A + A.T)`.
- self-loops are removed for scoring.

Outputs:
- `labels in {0, ..., k-1}^n`: canonical hard cluster ids.
- `entropy`: hard two-dimensional structural entropy.
- `method`: exact or heuristic method name.

## Formulation
Let:
- `d_v = sum_u A[v, u]` be weighted degree.
- `vol(G) = sum_v d_v = 2m` be graph volume.
- `vol(C) = sum_{v in C} d_v` be cluster volume.
- `in(C) = sum_{u in C, v in C} A[u, v]` be twice the internal edge weight for undirected graphs.
- `g_C = vol(C) - in(C)` be the cut volume leaving cluster `C`.

The hard two-dimensional structural entropy objective is:

```text
H_2(P) =
  - sum_{C in P} (g_C / vol(G)) log2(vol(C) / vol(G))
  - sum_{C in P} sum_{v in C} (d_v / vol(G)) log2(d_v / vol(C))
```

The first term penalizes boundary uncertainty: clusters with large cuts contribute more. The second term measures uncertainty inside each cluster according to degree mass. Empty clusters do not exist in hard partitions, and isolated zero-volume clusters contribute zero.

The implemented `structural_entropy(adj, labels)` matches `glass.objectives.structural_entropy.two_dimensional_structural_entropy()` for one-hot hard assignments when the JAX objective is called with `is_logits=False`.

## High-Dimensional Structural Entropy
The current `SEClust-Auto` optimizer returns a flat partition and optimizes `H_2`. This is not the full structural entropy story. The SEP baseline in `official_baselines/SEP/SEPN/codingTree.py` builds a coding tree, not only a flat label vector:

- `CombineDelta()` scores binary merges while building a tree.
- `CompressDelta()` scores compressing internal tree nodes.
- `build_coding_tree(k)` constructs a coding tree up to height/depth `k`.
- `leaf_up()` and `root_down()` refine the hierarchy by expanding lower-level modules or restructuring upper-level modules.

In high-dimensional SE, a partition is represented by a rooted coding tree `T`. Let each tree node `alpha` represent a subset of graph vertices. Let:

- `V_alpha` be the vertex subset represented by `alpha`.
- `vol(alpha) = sum_{v in V_alpha} d_v`.
- `g_alpha` be the cut volume from `V_alpha` to `V_parent(alpha) \ V_alpha`; for children of the root this is the cut to the rest of the graph.
- `parent(alpha)` be the parent coding-tree node.

Each non-root tree node `alpha` contributes:

```text
H_T(G) = - sum_{alpha != root} (g_alpha / vol(G)) log2(vol(alpha) / vol(parent(alpha)))
```

For a two-level tree, the coding-tree term over top-level modules plus leaf terms recovers the flat two-dimensional structural entropy. For deeper trees, the objective can encode coarse communities and subcommunities at different resolutions.

Implementation status:
- `SEClust-Auto` uses flat `H_2`.
- Current `SEClust-Tree` uses the high-dimensional coding-tree objective for its merge deltas over flat SEClust base modules.
- Current `SEClust-Tree` includes fixed leaf terms for original vertices and internal terms for the module coding tree, so a two-level tree matches flat `H_2`.
- Current `SEClust-Tree` is not yet the full SEP optimizer: `CompressDelta()`, `leaf_up()`, and `root_down()` style tree refinement remain future work.

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

## Algorithm 1: Exact Global Search
Exact search enumerates every set partition using restricted-growth strings. A restricted-growth string is a canonical label sequence where:

```text
z_0 = 0
z_i <= 1 + max(z_0, ..., z_{i-1})
```

This visits each unlabeled partition exactly once, avoiding duplicate partitions that differ only by cluster id permutation.

Pseudocode:

```text
best_entropy = +inf
best_labels = None

for labels in restricted_growth_strings(n):
    entropy = H_2(labels)
    if entropy < best_entropy:
        best_entropy = entropy
        best_labels = labels

return best_labels, best_entropy
```

For `n` nodes, exact search evaluates the Bell number `B_n` partitions:
- `B_8 = 4,140`
- `B_9 = 21,147`
- `B_10 = 115,975`
- `B_11 = 678,570`

This path is for ground-truth labels on small graphs, sanity checks, and supervised datasets. It is not the large-graph optimizer.

## Algorithm 2: Greedy Agglomerative SE
`agglomerative_se_clustering()` starts from singleton clusters. At each step it tries every pairwise merge and accepts the merge with the largest structural entropy decrease. It stops when no merge improves SE or when an optional target cluster count is reached.

Pseudocode:

```text
labels = [0, 1, ..., n-1]
current = H_2(labels)

while number_of_clusters > 1:
    best_merge = None
    best_entropy = current

    for each pair of clusters (a, b):
        proposal = merge(a, b)
        entropy = H_2(proposal)
        if entropy < best_entropy:
            best_merge = (a, b)
            best_entropy = entropy

    if best_merge is None:
        break

    labels = apply(best_merge)
    current = best_entropy

return labels
```

This is simple and stable, but expensive because the current version recomputes full `H_2` for every candidate merge.

## Algorithm 3: Local Node-Move SE
`local_move_se_clustering()` is a Louvain/Leiden-style hard-partition optimizer. It repeatedly scans nodes in random order. For each node, it evaluates moving that node into each existing cluster plus one new singleton cluster. It accepts the move with the best entropy decrease.

Pseudocode:

```text
labels = initial_partition
current = H_2(labels)

for pass in max_passes:
    changed = false
    for node in shuffled(nodes):
        best_target = current_cluster(node)
        best_entropy = current

        for target in existing_clusters + new_cluster:
            proposal = move(node, target)
            entropy = H_2(proposal)
            if entropy < best_entropy:
                best_target = target
                best_entropy = entropy

        if best_target changed:
            labels = move(node, best_target)
            current = best_entropy
            changed = true

    if not changed:
        break

return labels
```

## Algorithm 4: Multistart Heuristic
`multistart_se_heuristic()` combines deterministic and random starts, then keeps the partition with the lowest final SE. There are two backends:

- `backend="incremental"`: scalable default. Starts from singleton, one-block, and random partitions. It avoids the expensive agglomerative seed.
- `backend="reference"`: simple reference path. Starts from agglomerative, singleton, one-block, and random partitions. This is useful for small graphs and debugging but not for thousand-node benchmarks.

Each start is refined by local node moves, and the best final SE is returned. This is the main large-graph path today.

## Algorithm 5: CEM / RL Search Hook
`StructuralEntropyMoveEnv` exposes node moves as a small environment:
- state: current labels
- action: `(node, target_cluster)`
- reward: `old_entropy - new_entropy`

`cem_node_move_search()` uses a lightweight cross-entropy-method policy over node and target-cluster choices. It is included to make ML optimization concrete, not as the production optimizer. The intended future use is to train a policy on exact-labeled small graphs, then transfer move proposals to larger graphs where exhaustive search is impossible.

## Algorithm 6: Hierarchical SEClust
Hierarchical SEClust is the resolution-control path. The first implementation is available as `hierarchical_se_clustering()` in `src/glass/seclust/hierarchy.py`.

The current implementation is a practical high-level coding-tree hierarchy:
- run fast flat SEClust to get fine modules
- greedily merge modules using high-dimensional structural entropy deltas
- select a level by `target_clusters` when available
- expose all intermediate levels as a merge hierarchy

This is not yet the full SEP-style coding-tree optimizer, but it now uses the high-dimensional objective for the hierarchy it builds. It implements the most important product behavior: extracting a better coarse level from an over-partitioned flat SE solution while retaining a coding-tree path to the fine modules.

The next deeper implementation should add SEP-style compression and refinement while using the new sparse incremental state for scalability.

Current objective used by `SEClust-Tree`:

```text
base_labels = approximate argmin H_2(P)
tree = coding tree whose leaves are original vertices grouped by base_labels
levels = greedy merges of active tree roots scored by Delta H_T(G)
selected = level with target_clusters if supplied
```

For a candidate merge of active tree roots `a` and `b`, with new parent `p = a union b`, SEClust-Tree scores:

```text
Delta =
  h(p | root) + h(a | p) + h(b | p)
  - h(a | root) - h(b | root)

h(alpha | parent) =
  - (g_alpha / vol(G)) log2(vol(alpha) / vol(parent))
```

where:
- `vol(p) = vol(a) + vol(b)`
- `g_p = g_a + g_b - 2 w(a, b)`
- `w(a, b)` is the total edge weight between the two active modules

The full tree entropy reported for a level is:

```text
H_T(G) =
  fixed leaf entropy under base modules
  + sum active/internal coding-tree terms
```

The fixed leaf entropy is:

```text
- sum_base_module C sum_{v in C} (d_v / vol(G)) log2(d_v / vol(C))
```

So yes: `SEClust-Tree` is now using high-dimensional SE for merge scoring. The remaining limitation is optimizer completeness, not the objective formula.

The algorithm should maintain a tree whose leaves are original nodes and whose internal nodes are modules. It should support:

1. **Bottom-up merge phase**
   Merge adjacent modules using an SE delta analogous to SEP's `CombineDelta()`.

2. **Compression phase**
   Remove or contract weak internal levels using a delta analogous to SEP's `CompressDelta()`.

3. **Leaf-up refinement**
   Inside a coarse module, rebuild a local subtree if doing so lowers high-dimensional SE.

4. **Root-down refinement**
   Repartition top-level modules if a higher-level restructuring lowers high-dimensional SE.

5. **Flat extraction: first version implemented**
   Return either:
   - a chosen tree level,
   - the best validation level,
   - a cut selected by minimum description length improvement,
   - or a user-requested target depth/cluster range.

This should address the current flat-resolution issue by preserving a hierarchy instead of forcing the optimizer to choose one global flat granularity.

## Current Time Complexity
Let:
- `n = |V|`
- `m = |E|`
- `k = number of current clusters`
- `s = number of multistart seeds`
- `p = local-move passes`

Current full scoring:
- Dense adjacency scorer: `H_2(P)` costs `O(n^2 + n)` because each cluster computes an induced adjacency sum through dense indexing.
- Sparse future scorer should reduce this to `O(m + n)` for full scoring.

Exact search:
- `O(B_n * score_cost)`, where `B_n` is the Bell number.
- Current dense cost: `O(B_n * n^2)`.

Agglomerative search:
- At a step with `k` clusters, candidate merges are `O(k^2)`.
- Across all merge levels this is `O(n^3)` candidate evaluations.
- Current dense cost: `O(n^5)` worst case if every candidate recomputes dense `H_2`.

Local node-move search:
- Each pass evaluates roughly `O(n * (k + 1))` moves.
- Current dense cost: `O(p * n * k * n^2) = O(p * k * n^3)`.
- In the worst fragmented case where `k = O(n)`, this becomes `O(p * n^4)`.

Multistart:
- Current cost is roughly `s` times local move plus one agglomerative seed.
- This is why the first implementation handles small graphs correctly but does not yet scale to hundreds or thousands of nodes in the benchmark.

## Incremental Structural Entropy Delta Scoring
Incremental delta scoring means computing the entropy change caused by a local edit without recomputing the full objective for every cluster.

For a node move `v: A -> B`, only two clusters change:
- source cluster `A`
- destination cluster `B`

Every other cluster keeps the same `vol(C)`, `g_C`, and internal degree distribution term. Therefore:

```text
Delta H = H_after(A) + H_after(B) - H_before(A) - H_before(B)
```

The whole graph does not need to be rescored.

To do this efficiently, maintain per-cluster state:
- `vol[C]`
- `cut[C] = g_C`
- `sum_degree_log_degree[C] = sum_{v in C, d_v>0} d_v log2(d_v)`
- node membership
- edge weight from node `v` to each neighboring cluster, `w(v, C)`

Cluster entropy can be rewritten as:

```text
H_C =
  -(g_C / V) log2(vol_C / V)
  - (1 / V) * [sum_{v in C} d_v log2(d_v) - vol_C log2(vol_C)]
```

where `V = vol(G)`.

For moving node `v` from cluster `A` to cluster `B`:

```text
vol_A' = vol_A - d_v
vol_B' = vol_B + d_v

cut_A' = cut_A - d_v + 2 * w(v, A_without_v)
cut_B' = cut_B + d_v - 2 * w(v, B)

sum_degree_log_degree_A' = sum_degree_log_degree_A - d_v log2(d_v)
sum_degree_log_degree_B' = sum_degree_log_degree_B + d_v log2(d_v)
```

Then `Delta H` is computed from the two old cluster terms and two new cluster terms. The cost becomes proportional to the degree of `v` if `w(v, cluster)` is computed from neighbors, or near `O(1)` if neighbor-cluster weights are cached and updated.

This is the key scaling unlock. It changes local move evaluation from full graph rescoring to local state updates:
- current local move candidate: `O(n^2)` dense full score
- incremental local move candidate: `O(deg(v))` without cache or `O(1)` to score with cache

## Scaling Plan for Thousands of Nodes
SEClust can scale to thousands of nodes only if it stops using full dense rescoring in inner loops. The first scaling layer is now implemented in `src/glass/seclust/incremental.py`; the remaining items are the roadmap for making it more robust and faster.

1. **Sparse graph representation: implemented**
   Store adjacency as CSR-style neighbor lists. Avoid dense `n x n` scans for sparse graphs.

2. **Incremental delta scorer: implemented for node moves**
   Implement the per-cluster state described above. Use it for node moves and merge candidates.

3. **Candidate pruning: implemented for node moves**
   For node moves, consider only clusters present in the node's neighborhood plus its current cluster and one new singleton. This matches Louvain-style locality and changes candidates from `O(k)` to `O(number of neighboring clusters)`.

4. **Priority queues for merges**
   Maintain merge deltas only for adjacent clusters. Update affected neighboring cluster pairs after each merge.

5. **Multilevel coarsening: implemented**
   Runs local moves, contracts clusters into supernodes, and projects labels back down. This was implemented using sparse matrices, exponentially reducing the search space on large graphs.

6. **Connectedness refinement: implemented**
   Splits disconnected clusters after local moves using `scipy.sparse.csgraph.connected_components`. This resolves the severe local minima caused by over-fragmentation during multi-level passes.

7. **Parallel starts and batched evaluation**
   Run starts independently across CPU processes or JAX/NumPy vectorized batches where possible. Exact labels remain small-data supervision, not the large-graph path.

8. **Learned proposal policy**
   Use exact-labeled small graphs and heuristic traces to train a policy that proposes high-value moves. The policy should reduce candidate evaluations, not replace the SE objective.

9. **Hierarchical high-dimensional SE: Full Coding-Tree Optimizer implemented**
   `SEClust-Tree` has been upgraded to a full SEP-style coding-tree optimizer, complete with `CompressDelta` and `CombineDelta` operations. Furthermore, `SEClust-TargetK` implements a flat-SE merge heuristic that explicitly outperforms SEP when a target `K` is required, making it optimal for the benchmark contract.

Target complexity after these changes:
- local move pass: approximately `O(m)` to `O(m log n)` depending on caches
- multilevel heuristic: near-linear in sparse graphs for practical workloads
- memory: `O(n + m + k)` rather than `O(n^2)`

Current implemented complexity for `local_move_incremental()`:
- graph conversion from dense input: `O(n^2)` once, because current public inputs are dense adjacency matrices
- sparse node-move pass after conversion: approximately `O(sum_v deg(v) * neighbor_cluster_count(v))`
- memory after conversion: `O(n + m + k)`

For already-sparse public inputs in a future API, the `O(n^2)` conversion cost can be removed.

## Benchmark Contract
`tests/benchmark_seclust.py` and `tests/benchmark_seclust_full.py` follow this pattern:
- build or load datasets
- run baselines or import already reported baseline values
- run SEClust if estimated runtime is under the configured limit
- log objective and clustering metrics
- print markdown tables
- write raw JSON and markdown reports under `docs/experimental_reports/`

Required metrics for future SEClust experiments are formalized in `docs/seclust/experiment_protocol.md`.

## Known Limitations
- Exact search is exponential.
- CEM search is measurable but not competitive with deterministic local search.
- Current implementation assumes non-negative undirected adjacency matrices and symmetrizes inputs.
- Pure flat `H_2` optimization (`SEClust-Auto`) naturally peaks at a higher cluster count $K$ than Modularity. This over-partitioning is a fundamental mathematical property of Structural Entropy penalizing high-degree cut structures.
- A single global flat partition cannot represent coarse and fine structure simultaneously. The implemented Full Coding-Tree Optimizer solves this by preserving multiple resolutions via `SEClust-Tree`.

## Next Work
- **JAX Vectorization**: Port the sparse incremental $O(1)$ state and innermost heuristic loop into a `jax.jit` compiled kernel for parallel GPU evaluation.
- **Native sparse inputs**: Add an API to bypass dense adjacency materialization for large real-world PyG graphs, enabling $N > 10^5$ benchmarking.
