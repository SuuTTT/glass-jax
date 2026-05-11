# Same-Protocol Attributed Benchmark (20260511_065243)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Computers
- Methods: AdjSVDKMeans, GAE, SEClust-ConstrainedK
- Seeds: 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Computers | AdjSVDKMeans | 4 / 4 | 0.236 +/- 0.009 | 0.119 +/- 0.006 | -0.076 +/- 0.001 | 10.0 | 0.288 +/- 0.007 | 0.593 +/- 0.033 | 11.683 +/- 0.051 | 12.864 +/- 0.024 | 4.04 +/- 0.04 | ok |
| Computers | GAE | 4 / 4 | 0.188 +/- 0.001 | 0.037 +/- 0.000 | 0.022 +/- 0.001 | 10.0 | 0.005 +/- 0.000 | 0.888 +/- 0.000 | 12.413 +/- 0.001 | 14.271 +/- 0.002 | 12.69 +/- 0.28 | ok |
| Computers | **SEClust-ConstrainedK** | 4 / 4 | 0.362 +/- 0.014 | 0.287 +/- 0.017 | 0.146 +/- 0.006 | 10.0 | 0.562 +/- 0.011 | 0.303 +/- 0.007 | 10.635 +/- 0.040 | 11.648 +/- 0.056 | 7.38 +/- 0.05 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Computers": "SEClust-ConstrainedK"}
- Best mean structural entropy (lower): {"Computers": "SEClust-ConstrainedK"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260511_065243.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260511_065243_aggregated.json`
