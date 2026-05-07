# SEClust Experiment Protocol

## Purpose
This protocol standardizes how SEClust experiments are run and reported. Every experiment should make it possible to compare clustering quality, objective quality, and runtime against existing graph clustering baselines.

## Required Per-Run Fields
Each algorithm run must log:

| Field | Meaning |
| :--- | :--- |
| `experiment_id` | Stable run group id, usually report name plus date. |
| `dataset` | Dataset name. |
| `dataset_source` | Generator, file path, PyG dataset, or report source. |
| `n_nodes` | Number of nodes. |
| `n_edges` | Number of undirected edges, or total nonzero weighted edges divided by 2. |
| `algorithm` | Algorithm label, e.g. `SEClust-Auto`, `Louvain`, `Infomap`. |
| `seed` | Random seed, or `null` for deterministic algorithms. |
| `status` | `ok`, `skipped`, `failed`, or `baseline_imported`. |
| `skip_reason` | Required if status is `skipped` or `failed`. |
| `estimated_runtime_seconds` | Runtime estimate before execution. |
| `runtime_seconds` | Actual wall-clock runtime if executed. |
| `k` | Number of output clusters. |
| `labels_path` | Optional path to saved labels. |

## Required Metrics
Every completed run should log these metrics:

| Metric | Direction | Required When | Notes |
| :--- | :--- | :--- | :--- |
| `acc` | higher is better | ground-truth labels exist | Clustering accuracy after best label matching. |
| `nmi` | higher is better | ground-truth labels exist | Normalized mutual information. |
| `ari` | higher is better | ground-truth labels exist | Adjusted Rand index. |
| `k` | diagnostic | always | Number of discovered clusters. |
| `modularity` | higher is better | graph exists | Hard modularity of output labels. |
| `structural_entropy` | lower is better | graph exists | Hard 2D SE objective. |
| `map_equation` | lower is better | graph exists | Hard map equation / Infomap-style codelength. |

Skipped runs should still include `k = null`, metric values as `null`, `estimated_runtime_seconds`, and a clear `skip_reason`.

## Metric Formulations
### ACC
Given ground-truth labels `y` and predicted labels `z`, clustering accuracy is:

```text
ACC = max_perm (1 / n) sum_i 1[y_i = perm(z_i)]
```

Use Hungarian matching for the permutation. For very large fragmented outputs, a greedy fallback may be reported only if the report says so explicitly.

### NMI
Use normalized mutual information between ground-truth and predicted labels:

```text
NMI(y, z) = I(y; z) / mean(H(y), H(z))
```

State the exact library implementation when using sklearn or another package.

### ARI
Use adjusted Rand index:

```text
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

State the exact library implementation when using sklearn or another package.

### K
`k` is:

```text
k = number of non-empty clusters in predicted labels
```

### Modularity
For hard partition labels `c_i`:

```text
Q = (1 / 2m) sum_{i,j} [A_ij - (d_i d_j / 2m)] 1[c_i = c_j]
```

where `2m = sum_i d_i`.

### Structural Entropy
Use hard two-dimensional structural entropy:

```text
H_2(P) =
  - sum_C (g_C / vol(G)) log2(vol(C) / vol(G))
  - sum_C sum_{v in C} (d_v / vol(G)) log2(d_v / vol(C))
```

This must be computed by `glass.seclust.structural_entropy()` or an explicitly equivalent implementation.

### Map Equation
For an undirected graph, use random-walk stationary probability:

```text
p_i = d_i / vol(G)
```

For module `M`, define exit probability:

```text
q_M = sum_{i in M, j notin M} p_i * A_ij / d_i
```

Total exit probability:

```text
q = sum_M q_M
```

Map equation codelength:

```text
L(P) = q H(Q) + sum_M (q_M + p_M) H(P_M)
```

where:
- `H(Q)` is entropy over module exits `q_M / q`
- `p_M = sum_{i in M} p_i`
- `H(P_M)` is entropy over the module codebook containing one exit code with mass `q_M` and node visit masses `{p_i | i in M}`, normalized by `q_M + p_M`

Reports should call this field `map_equation`.

## Runtime Policy
Before running SEClust on a dataset, estimate runtime. If estimated runtime exceeds the configured limit, skip and log:

```json
{
  "status": "skipped",
  "skip_reason": "estimated runtime exceeds limit",
  "estimated_runtime_seconds": 660.0,
  "runtime_seconds": null
}
```

Default per-dataset limit:

```text
180 seconds
```

## Report Tables
Tables should bold the best result per dataset and metric:
- ACC, NMI, ARI, modularity: maximum is best.
- structural entropy, map equation, runtime: minimum is best among completed runs.
- K is diagnostic and should not be bolded unless the experiment explicitly targets a known `K`.

Minimum table columns:

```text
Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time(s) | Status
```

## Raw JSON Schema
Reports should write raw rows as JSON with at least:

```json
{
  "experiment_id": "seclust_full_benchmark_20260507",
  "dataset": "Karate",
  "dataset_source": "hardcoded Zachary graph",
  "n_nodes": 34,
  "n_edges": 78,
  "algorithm": "SEClust-Auto",
  "seed": 42,
  "status": "ok",
  "skip_reason": null,
  "estimated_runtime_seconds": 25.8,
  "runtime_seconds": 15.45,
  "acc": 0.382,
  "nmi": 0.474,
  "ari": 0.240,
  "k": 7,
  "modularity": 0.0,
  "structural_entropy": 3.366,
  "map_equation": 0.0
}
```

## Reproducibility Requirements
- Fix random seeds for every stochastic algorithm.
- Record dependency availability and versions where possible.
- Record graph generation parameters.
- Save raw labels for new algorithm runs when the dataset has at least one external baseline.
- Keep imported baseline rows separate from executed rows via `status = baseline_imported`.

## Recommended Experiment Sequence
1. Small exact-labeled graphs: verify SEClust recovers global optimum.
2. Synthetic structural graphs: Karate, Caveman, SBM variants.
3. Real-world topology datasets: Cora, Citeseer, PubMed when local data dependencies are available.
4. Larger graphs after incremental delta scoring lands.
5. Ablations: exact vs heuristic, starts, passes, candidate pruning, connectedness refinement.
