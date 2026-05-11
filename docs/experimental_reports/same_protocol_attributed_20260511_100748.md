# Same-Protocol Attributed Benchmark (20260511_100748)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Coauthor-CS, Coauthor-Physics
- Methods: RawKMeans, PCAKMeans
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Coauthor-CS | **PCAKMeans** | 10 / 10 | 0.519 +/- 0.002 | 0.577 +/- 0.002 | 0.316 +/- 0.005 | 15.0 | 0.390 +/- 0.001 | 0.537 +/- 0.001 | 11.898 +/- 0.004 | 13.259 +/- 0.004 | 8.32 +/- 0.40 | ok |
| Coauthor-CS | RawKMeans | 10 / 10 | 0.390 +/- 0.085 | 0.348 +/- 0.144 | 0.134 +/- 0.102 | 15.0 | 0.302 +/- 0.114 | 0.744 +/- 0.084 | 12.456 +/- 0.444 | 13.548 +/- 0.132 | 48.81 +/- 1.00 | ok |
| Coauthor-Physics | **PCAKMeans** | 10 / 10 | 0.441 +/- 0.000 | 0.377 +/- 0.000 | 0.140 +/- 0.001 | 5.0 | 0.369 +/- 0.000 | 0.367 +/- 0.000 | 13.167 +/- 0.000 | 14.335 +/- 0.001 | 11.96 +/- 2.45 | ok |
| Coauthor-Physics | RawKMeans | 10 / 10 | 0.430 +/- 0.059 | 0.280 +/- 0.080 | 0.077 +/- 0.098 | 5.0 | 0.281 +/- 0.050 | 0.560 +/- 0.089 | 13.545 +/- 0.182 | 14.598 +/- 0.140 | 28.58 +/- 2.26 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Coauthor-CS": "PCAKMeans", "Coauthor-Physics": "PCAKMeans"}
- Best mean structural entropy (lower): {"Coauthor-CS": "PCAKMeans", "Coauthor-Physics": "PCAKMeans"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260511_100748.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260511_100748_aggregated.json`
