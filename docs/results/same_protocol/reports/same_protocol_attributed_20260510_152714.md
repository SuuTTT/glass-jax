# Same-Protocol Attributed Benchmark (20260510_152714)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Coauthor-CS, Coauthor-Physics
- Methods: AdjSVDKMeans, GAE, SEClust-ConstrainedK
- Seeds: 0, 1, 2
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Coauthor-CS | AdjSVDKMeans | 3 / 3 | 0.267 +/- 0.005 | 0.110 +/- 0.003 | 0.010 +/- 0.002 | 15.0 | 0.172 +/- 0.001 | 0.499 +/- 0.004 | 12.794 +/- 0.010 | 13.490 +/- 0.009 | 9.63 +/- 1.39 | ok |
| Coauthor-CS | **GAE** | 3 / 3 | 0.467 +/- 0.010 | 0.556 +/- 0.017 | 0.375 +/- 0.062 | 15.0 | 0.445 +/- 0.028 | 0.476 +/- 0.045 | 11.669 +/- 0.084 | 12.951 +/- 0.164 | 8.40 +/- 0.14 | ok |
| Coauthor-CS | SEClust-ConstrainedK | 3 / 3 | 0.189 +/- 0.007 | 0.101 +/- 0.012 | 0.044 +/- 0.009 | 15.0 | 0.534 +/- 0.010 | 0.399 +/- 0.009 | 11.290 +/- 0.036 | 12.490 +/- 0.057 | 7.45 +/- 0.14 | ok |
| Coauthor-Physics | AdjSVDKMeans | 3 / 3 | 0.455 +/- 0.001 | 0.042 +/- 0.001 | -0.050 +/- 0.001 | 5.0 | 0.149 +/- 0.003 | 0.500 +/- 0.005 | 13.923 +/- 0.010 | 14.479 +/- 0.006 | 9.45 +/- 1.59 | ok |
| Coauthor-Physics | **GAE** | 3 / 3 | 0.507 +/- 0.073 | 0.452 +/- 0.033 | 0.265 +/- 0.120 | 5.0 | 0.422 +/- 0.056 | 0.328 +/- 0.065 | 13.083 +/- 0.127 | 14.062 +/- 0.203 | 18.52 +/- 0.13 | ok |
| Coauthor-Physics | SEClust-ConstrainedK | 3 / 3 | 0.282 +/- 0.012 | 0.041 +/- 0.010 | 0.016 +/- 0.003 | 5.0 | 0.469 +/- 0.012 | 0.330 +/- 0.012 | 12.961 +/- 0.027 | 14.033 +/- 0.052 | 17.37 +/- 0.12 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Coauthor-CS": "GAE", "Coauthor-Physics": "GAE"}
- Best mean structural entropy (lower): {"Coauthor-CS": "SEClust-ConstrainedK", "Coauthor-Physics": "SEClust-ConstrainedK"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260510_152714.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260510_152714_aggregated.json`
