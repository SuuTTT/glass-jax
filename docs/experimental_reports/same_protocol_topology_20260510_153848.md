# Same-Protocol Topology Benchmark (20260510_153848)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Coauthor-Physics
- Methods: Louvain, Leiden, Infomap, Spectral, LabelPropagation, SEClust-ConstrainedK
- Seeds: 0, 1, 2
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Coauthor-Physics | Infomap | 3 / 3 | 0.038 +/- 0.004 | 0.282 +/- 0.001 | 0.008 +/- 0.001 | 1162.3 +/- 2.9 | 0.528 +/- 0.003 | 0.422 +/- 0.002 | 9.808 +/- 0.007 | 11.084 +/- 0.001 | 5.34 +/- 0.38 | ok |
| Coauthor-Physics | LabelPropagation | 3 / 3 | 0.494 +/- 0.003 | 0.379 +/- 0.001 | 0.547 +/- 0.007 | 1620.0 +/- 38.2 | 0.342 +/- 0.005 | 0.450 +/- 0.005 | 12.455 +/- 0.032 | 12.785 +/- 0.023 | 19.31 +/- 6.24 | ok |
| Coauthor-Physics | **Leiden** | 3 / 3 | 0.430 +/- 0.017 | 0.481 +/- 0.004 | 0.229 +/- 0.009 | 28.0 +/- 3.0 | 0.685 +/- 0.003 | 0.188 +/- 0.004 | 11.633 +/- 0.094 | 12.405 +/- 0.051 | 3.76 +/- 0.06 | ok |
| Coauthor-Physics | Louvain | 3 / 3 | 0.423 +/- 0.010 | 0.454 +/- 0.007 | 0.217 +/- 0.007 | 22.0 | 0.667 +/- 0.003 | 0.230 +/- 0.011 | 11.694 +/- 0.038 | 12.521 +/- 0.029 | 27.76 +/- 9.78 | ok |
| Coauthor-Physics | SEClust-ConstrainedK | 3 / 3 | 0.282 +/- 0.012 | 0.041 +/- 0.010 | 0.016 +/- 0.003 | 5.0 | 0.469 +/- 0.012 | 0.330 +/- 0.012 | 12.961 +/- 0.027 | 14.033 +/- 0.052 | 18.08 +/- 0.06 | ok |
| Coauthor-Physics | Spectral | 3 / 3 | 0.455 +/- 0.001 | 0.042 +/- 0.001 | -0.050 +/- 0.001 | 5.0 | 0.149 +/- 0.003 | 0.500 +/- 0.005 | 13.923 +/- 0.010 | 14.479 +/- 0.006 | 11.98 +/- 2.09 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Coauthor-Physics": "Leiden"}
- Best mean structural entropy (lower): {"Coauthor-Physics": "Infomap"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_topology_20260510_153848.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_topology_20260510_153848_aggregated.json`
