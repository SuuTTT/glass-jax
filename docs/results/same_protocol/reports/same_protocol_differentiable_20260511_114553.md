# Block E: Differentiable Non-Attributed Glass Clustering (20260511_114553)

Protocol: topology-only, no node features; differentiable methods use dense JAX adjacency (capped at N<=5000); discrete baselines reused from topology block JSON (same seeds, same preprocessing).

## Method Legend

**Differentiable (gradient-based, our contribution and seminal baselines):**
- `Glass-SE`: differentiable 2D structural-entropy H2 minimisation (ours)
- `Glass-Mod`: differentiable modularity Q maximisation (ours)
- `Glass-Map`: differentiable map-equation L minimisation (ours)
- `SoftKMeans`: temperature-annealed soft k-means on spectral embedding   (Bregman relaxation; Banerjee et al. 2005)
- `SpectralGrad`: continuous normalised-cut relaxation (Shi & Malik 2000)   minimised via gradient descent instead of eigenvectors

**Discrete (reused from Block A topology block):**
- `Louvain`, `Leiden`, `Infomap`: modularity/flow community detection
- `Spectral`: standard spectral clustering (sklearn, eigenvector-based)
- `SEClust-ConstrainedK`: discrete structural-entropy heuristic (ours)

Datasets: Karate
Differentiable methods: Glass-Map, Glass-Mod, Glass-SE, SoftKMeans, SpectralGrad
Discrete baselines: Infomap, Leiden, Louvain, SEClust-ConstrainedK, Spectral
Seeds: 0, 1

## Results

Bold = best mean NMI per dataset.

| Dataset | Method | Type | OK/N | ACC | NMI | ARI | K | Q | SE | L | Soft-Obj | Time(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Karate | Glass-SE | diff | 2/2 | 0.971 | 0.837 | 0.882 | 2.0 | 0.372 | 3.833 | 4.409 | 3.833 | 0.02 +/- 0.00 |
| Karate | **Glass-Mod** | diff | 2/2 | 0.971 | 0.837 | 0.882 | 2.0 | 0.372 | 3.833 | 4.409 | 0.372 | 0.01 +/- 0.00 |
| Karate | Glass-Map | diff | 2/2 | 0.897 +/- 0.021 | 0.574 +/- 0.000 | 0.621 +/- 0.067 | 2.0 | 0.293 +/- 0.027 | 3.936 +/- 0.054 | 4.628 +/- 0.012 | 4.628 +/- 0.012 | 0.03 +/- 0.00 |
| Karate | SoftKMeans | diff | 2/2 | 0.971 | 0.836 | 0.882 | 2.0 | 0.360 | 3.852 | 4.424 | 0.377 | 0.01 +/- 0.00 |
| Karate | SpectralGrad | diff | 2/2 | 0.529 | 0.000 | 0.000 | 1.0 | 0.000 | 4.704 | 4.704 | 6.058 +/- 0.000 | 0.02 +/- 0.00 |
| Karate | Louvain | disc | 2/2 | 0.691 +/- 0.021 | 0.645 +/- 0.060 | 0.527 +/- 0.020 | 4.0 | 0.418 +/- 0.003 | 3.427 +/- 0.032 | 4.324 +/- 0.015 | --- | 0.00 +/- 0.00 |
| Karate | Leiden | disc | 2/2 | 0.676 | 0.687 | 0.541 | 4.0 | 0.420 | 3.405 | 4.334 | --- | 0.00 +/- 0.00 |
| Karate | Infomap | disc | 2/2 | 0.941 | 0.831 | 0.882 | 3.0 | 0.373 | 3.776 | 4.402 | --- | 0.01 +/- 0.00 |
| Karate | Spectral | disc | 2/2 | 0.971 | 0.836 | 0.882 | 2.0 | 0.360 | 3.852 | 4.424 | --- | 0.03 +/- 0.01 |
| Karate | SEClust-ConstrainedK | disc | 2/2 | 0.971 | 0.837 | 0.882 | 2.0 | 0.372 | 3.833 | 4.409 | --- | 0.23 +/- 0.32 |

## Best-By-Dataset

- Best mean NMI: {"Karate": "Glass-Mod"}
- Best mean structural entropy (lower): {"Karate": "Leiden"}

## Files

- Raw seed-level JSON: `/tmp/glass_diff_smoke3/same_protocol_differentiable_20260511_114553.json`
- Aggregated JSON: `/tmp/glass_diff_smoke3/same_protocol_differentiable_20260511_114553_aggregated.json`

## Notes on Soft-Obj Column

For Glass-SE and SpectralGrad: lower is better (minimisation).
For Glass-Mod: higher is better (sign-corrected to show Q, not -Q).
For Glass-Map: lower is better.
For SoftKMeans: lower is better (within-cluster variance).
Discrete baselines (Louvain, Leiden, Infomap, Spectral, SEClust-ConstrainedK) do not use a differentiable objective; the Soft-Obj column shows `---`.
