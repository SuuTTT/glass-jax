# Same-Protocol Attributed Benchmark (20260511_070341)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Coauthor-CS, Coauthor-Physics
- Methods: AdjSVDKMeans, GAE, SEClust-ConstrainedK
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Coauthor-CS | AdjSVDKMeans | 10 / 10 | 0.269 +/- 0.005 | 0.115 +/- 0.007 | 0.010 +/- 0.001 | 15.0 | 0.180 +/- 0.009 | 0.493 +/- 0.008 | 12.758 +/- 0.038 | 13.478 +/- 0.016 | 4.45 +/- 0.26 | ok |
| Coauthor-CS | **GAE** | 10 / 10 | 0.470 +/- 0.052 | 0.543 +/- 0.034 | 0.364 +/- 0.077 | 15.0 | 0.433 +/- 0.033 | 0.505 +/- 0.038 | 11.735 +/- 0.119 | 13.023 +/- 0.194 | 7.15 +/- 0.25 | ok |
| Coauthor-CS | SEClust-ConstrainedK | 10 / 10 | 0.187 +/- 0.017 | 0.103 +/- 0.015 | 0.042 +/- 0.010 | 15.0 | 0.535 +/- 0.009 | 0.398 +/- 0.008 | 11.283 +/- 0.033 | 12.480 +/- 0.051 | 7.02 +/- 0.11 | ok |
| Coauthor-Physics | AdjSVDKMeans | 10 / 10 | 0.456 +/- 0.001 | 0.042 +/- 0.001 | -0.050 +/- 0.001 | 5.0 | 0.148 +/- 0.002 | 0.504 +/- 0.009 | 13.926 +/- 0.013 | 14.479 +/- 0.005 | 5.15 +/- 0.33 | ok |
| Coauthor-Physics | **GAE** | 10 / 10 | 0.513 +/- 0.045 | 0.475 +/- 0.027 | 0.278 +/- 0.071 | 5.0 | 0.428 +/- 0.031 | 0.332 +/- 0.039 | 13.057 +/- 0.078 | 14.051 +/- 0.111 | 15.70 +/- 0.60 | ok |
| Coauthor-Physics | SEClust-ConstrainedK | 10 / 10 | 0.278 +/- 0.019 | 0.044 +/- 0.011 | 0.017 +/- 0.005 | 5.0 | 0.469 +/- 0.007 | 0.330 +/- 0.007 | 12.963 +/- 0.016 | 14.034 +/- 0.031 | 16.46 +/- 0.14 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Coauthor-CS": "GAE", "Coauthor-Physics": "GAE"}
- Best mean structural entropy (lower): {"Coauthor-CS": "SEClust-ConstrainedK", "Coauthor-Physics": "SEClust-ConstrainedK"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260511_070341.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260511_070341_aggregated.json`
