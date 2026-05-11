# Same-Protocol Scale Benchmark (20260510_185728)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: ogbn-arxiv
- Methods: Louvain, Leiden, SEClust-ConstrainedK
- Seeds: 0, 1, 2, 3, 4
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ogbn-arxiv | **Leiden** | 5 / 5 | 0.358 +/- 0.014 | 0.411 +/- 0.003 | 0.250 +/- 0.008 | 153.8 +/- 10.5 | 0.715 +/- 0.001 | 0.095 +/- 0.006 | 12.594 +/- 0.024 | 13.326 +/- 0.017 | 16.78 +/- 0.39 | ok |
| ogbn-arxiv | Louvain | 5 / 5 | 0.368 +/- 0.016 | 0.399 +/- 0.006 | 0.254 +/- 0.017 | 151.8 +/- 10.3 | 0.709 +/- 0.001 | 0.099 +/- 0.004 | 12.648 +/- 0.029 | 13.398 +/- 0.022 | 98.38 +/- 9.85 | ok |
| ogbn-arxiv | SEClust-ConstrainedK | 5 / 5 | 0.105 +/- 0.009 | 0.110 +/- 0.006 | 0.038 +/- 0.005 | 40.0 | 0.567 +/- 0.009 | 0.392 +/- 0.008 | 12.799 +/- 0.032 | 13.991 +/- 0.050 | 187.76 +/- 1.89 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"ogbn-arxiv": "Leiden"}
- Best mean structural entropy (lower): {"ogbn-arxiv": "Leiden"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_scale_20260510_185728.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_scale_20260510_185728_aggregated.json`
