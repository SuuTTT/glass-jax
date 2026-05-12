# Block E: Differentiable Non-Attributed Glass Clustering (20260511_114508)

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

Datasets: Karate, SBM-Easy
Differentiable methods: Glass-Mod, Glass-SE, SoftKMeans
Discrete baselines: Infomap, Leiden, Louvain, SEClust-ConstrainedK, Spectral
Seeds: 0

## Results

Bold = best mean NMI per dataset.

| Dataset | Method | Type | OK/N | ACC | NMI | ARI | K | Q | SE | L | Soft-Obj | Time(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Karate | Glass-SE | diff | 0/1 | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Karate | Glass-Mod | diff | 0/1 | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Karate | SoftKMeans | diff | 0/1 | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Karate | Louvain | disc | 1/1 | 0.706 | 0.602 | 0.513 | 4.0 | 0.416 | 3.450 | 4.313 | --- | 0.00 |
| Karate | Leiden | disc | 1/1 | 0.676 | 0.687 | 0.541 | 4.0 | 0.420 | 3.405 | 4.334 | --- | 0.00 |
| Karate | Infomap | disc | 1/1 | 0.941 | 0.831 | 0.882 | 3.0 | 0.373 | 3.776 | 4.402 | --- | 0.01 |
| Karate | Spectral | disc | 1/1 | 0.971 | 0.836 | 0.882 | 2.0 | 0.360 | 3.852 | 4.424 | --- | 0.04 |
| Karate | **SEClust-ConstrainedK** | disc | 1/1 | 0.971 | 0.837 | 0.882 | 2.0 | 0.372 | 3.833 | 4.409 | --- | 0.46 |
| SBM-Easy | Glass-SE | diff | 0/1 | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SBM-Easy | Glass-Mod | diff | 0/1 | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SBM-Easy | SoftKMeans | diff | 0/1 | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SBM-Easy | Louvain | disc | 1/1 | 1.000 | 1.000 | 1.000 | 5.0 | 0.626 | 8.015 | 8.724 | --- | 0.17 |
| SBM-Easy | Leiden | disc | 1/1 | 1.000 | 1.000 | 1.000 | 5.0 | 0.626 | 8.015 | 8.724 | --- | 0.10 |
| SBM-Easy | **Infomap** | disc | 1/1 | 1.000 | 1.000 | 1.000 | 5.0 | 0.626 | 8.015 | 8.724 | --- | 0.11 |
| SBM-Easy | Spectral | disc | 1/1 | 1.000 | 1.000 | 1.000 | 5.0 | 0.626 | 8.015 | 8.724 | --- | 0.20 |
| SBM-Easy | SEClust-ConstrainedK | disc | 1/1 | 1.000 | 1.000 | 1.000 | 5.0 | 0.626 | 8.015 | 8.724 | --- | 0.40 |

## Best-By-Dataset

- Best mean NMI: {"Karate": "SEClust-ConstrainedK", "SBM-Easy": "Infomap"}
- Best mean structural entropy (lower): {"Karate": "Leiden", "SBM-Easy": "Infomap"}

## Files

- Raw seed-level JSON: `/tmp/glass_diff_smoke/same_protocol_differentiable_20260511_114508.json`
- Aggregated JSON: `/tmp/glass_diff_smoke/same_protocol_differentiable_20260511_114508_aggregated.json`

## Notes on Soft-Obj Column

For Glass-SE and SpectralGrad: lower is better (minimisation).
For Glass-Mod: higher is better (sign-corrected to show Q, not -Q).
For Glass-Map: lower is better.
For SoftKMeans: lower is better (within-cluster variance).
Discrete baselines (Louvain, Leiden, Infomap, Spectral, SEClust-ConstrainedK) do not use a differentiable objective; the Soft-Obj column shows `---`.
