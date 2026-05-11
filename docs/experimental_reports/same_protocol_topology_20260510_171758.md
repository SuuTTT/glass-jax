# Same-Protocol Topology Benchmark (20260510_171758)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: LFR
- Methods: Louvain, Leiden, Infomap, Spectral, LabelPropagation, SEClust-ConstrainedK
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LFR | **Infomap** | 7 / 10 | 0.987 +/- 0.010 | 0.963 +/- 0.037 | 0.972 +/- 0.025 | 11.3 +/- 2.3 | 0.526 +/- 0.127 | 0.333 +/- 0.066 | 7.502 +/- 0.355 | 8.294 +/- 0.325 | 0.09 +/- 0.01 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |
| LFR | LabelPropagation | 7 / 10 | 0.801 +/- 0.166 | 0.674 +/- 0.393 | 0.583 +/- 0.382 | 6.7 +/- 3.5 | 0.410 +/- 0.263 | 0.264 +/- 0.053 | 8.012 +/- 0.762 | 8.579 +/- 0.430 | 0.07 +/- 0.02 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |
| LFR | Leiden | 7 / 10 | 0.926 +/- 0.112 | 0.916 +/- 0.115 | 0.862 +/- 0.216 | 10.4 +/- 2.4 | 0.531 +/- 0.122 | 0.302 +/- 0.089 | 7.443 +/- 0.268 | 8.362 +/- 0.419 | 0.07 +/- 0.01 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |
| LFR | Louvain | 7 / 10 | 0.937 +/- 0.108 | 0.916 +/- 0.112 | 0.886 +/- 0.191 | 10.4 +/- 2.4 | 0.528 +/- 0.125 | 0.304 +/- 0.092 | 7.468 +/- 0.307 | 8.349 +/- 0.398 | 0.17 +/- 0.04 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |
| LFR | SEClust-ConstrainedK | 7 / 10 | 0.718 +/- 0.180 | 0.769 +/- 0.153 | 0.626 +/- 0.237 | 10.3 +/- 2.5 | 0.466 +/- 0.115 | 0.412 +/- 0.083 | 7.571 +/- 0.276 | 8.707 +/- 0.455 | 0.41 +/- 0.17 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |
| LFR | Spectral | 7 / 10 | 0.986 +/- 0.010 | 0.957 +/- 0.037 | 0.965 +/- 0.027 | 10.3 +/- 2.5 | 0.525 +/- 0.129 | 0.282 +/- 0.054 | 7.512 +/- 0.359 | 8.299 +/- 0.324 | 0.21 +/- 0.03 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |

## Best-By-Dataset Checks

- Best mean NMI: {"LFR": "Infomap"}
- Best mean structural entropy (lower): {"LFR": "Leiden"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_topology_20260510_171758.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_topology_20260510_171758_aggregated.json`
