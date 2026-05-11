# Same-Protocol Attributed Benchmark (20260511_071853)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora, Citeseer, PubMed, Photo, Computers
- Methods: AdjSVDKMeans, GAE, SEClust-ConstrainedK
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | AdjSVDKMeans | 10 / 10 | 0.249 +/- 0.008 | 0.071 +/- 0.009 | 0.000 +/- 0.001 | 6.0 | 0.140 +/- 0.016 | 0.552 +/- 0.071 | 10.656 +/- 0.065 | 10.963 +/- 0.040 | 0.45 +/- 0.51 | ok |
| Citeseer | **GAE** | 10 / 10 | 0.405 +/- 0.054 | 0.202 +/- 0.040 | 0.132 +/- 0.039 | 6.0 | 0.581 +/- 0.037 | 0.236 +/- 0.030 | 9.206 +/- 0.099 | 10.050 +/- 0.172 | 0.66 +/- 0.02 | ok |
| Citeseer | SEClust-ConstrainedK | 10 / 10 | 0.233 +/- 0.008 | 0.024 +/- 0.005 | 0.015 +/- 0.004 | 6.0 | 0.671 +/- 0.006 | 0.161 +/- 0.006 | 8.970 +/- 0.016 | 9.642 +/- 0.033 | 0.75 +/- 0.01 | ok |
| Computers | AdjSVDKMeans | 10 / 10 | 0.234 +/- 0.010 | 0.118 +/- 0.008 | -0.077 +/- 0.003 | 10.0 | 0.290 +/- 0.005 | 0.596 +/- 0.023 | 11.676 +/- 0.034 | 12.865 +/- 0.022 | 4.05 +/- 0.08 | ok |
| Computers | GAE | 10 / 10 | 0.189 +/- 0.001 | 0.037 +/- 0.000 | 0.022 +/- 0.001 | 10.0 | 0.005 +/- 0.000 | 0.889 +/- 0.000 | 12.413 +/- 0.001 | 14.271 +/- 0.002 | 12.84 +/- 0.30 | ok |
| Computers | **SEClust-ConstrainedK** | 10 / 10 | 0.350 +/- 0.018 | 0.290 +/- 0.020 | 0.150 +/- 0.015 | 10.0 | 0.561 +/- 0.008 | 0.302 +/- 0.006 | 10.631 +/- 0.026 | 11.645 +/- 0.041 | 7.65 +/- 0.13 | ok |
| Cora | AdjSVDKMeans | 10 / 10 | 0.354 +/- 0.011 | 0.158 +/- 0.006 | 0.022 +/- 0.006 | 7.0 | 0.145 +/- 0.019 | 0.701 +/- 0.072 | 10.377 +/- 0.072 | 10.881 +/- 0.065 | 0.60 +/- 0.85 | ok |
| Cora | **GAE** | 10 / 10 | 0.455 +/- 0.058 | 0.293 +/- 0.075 | 0.203 +/- 0.070 | 7.0 | 0.470 +/- 0.097 | 0.340 +/- 0.072 | 9.168 +/- 0.247 | 10.238 +/- 0.417 | 0.56 +/- 0.05 | ok |
| Cora | SEClust-ConstrainedK | 10 / 10 | 0.282 +/- 0.021 | 0.081 +/- 0.013 | 0.047 +/- 0.009 | 7.0 | 0.612 +/- 0.010 | 0.244 +/- 0.010 | 8.771 +/- 0.028 | 9.656 +/- 0.053 | 0.69 +/- 0.07 | ok |
| Photo | AdjSVDKMeans | 10 / 10 | 0.333 +/- 0.009 | 0.227 +/- 0.011 | 0.025 +/- 0.014 | 8.0 | 0.367 +/- 0.016 | 0.390 +/- 0.017 | 10.759 +/- 0.064 | 11.683 +/- 0.041 | 0.95 +/- 0.31 | ok |
| Photo | GAE | 10 / 10 | 0.232 +/- 0.002 | 0.059 +/- 0.001 | 0.023 +/- 0.001 | 8.0 | 0.040 +/- 0.000 | 0.827 +/- 0.001 | 11.609 +/- 0.002 | 13.398 +/- 0.007 | 6.47 +/- 0.17 | ok |
| Photo | **SEClust-ConstrainedK** | 10 / 10 | 0.458 +/- 0.045 | 0.332 +/- 0.038 | 0.252 +/- 0.044 | 8.0 | 0.643 +/- 0.007 | 0.205 +/- 0.012 | 9.839 +/- 0.019 | 10.614 +/- 0.045 | 3.62 +/- 0.05 | ok |
| PubMed | AdjSVDKMeans | 10 / 10 | 0.398 +/- 0.001 | 0.017 +/- 0.002 | -0.002 +/- 0.000 | 3.0 | 0.092 +/- 0.009 | 0.526 +/- 0.053 | 12.855 +/- 0.034 | 13.252 +/- 0.054 | 4.07 +/- 0.74 | ok |
| PubMed | **GAE** | 10 / 10 | 0.479 +/- 0.008 | 0.054 +/- 0.010 | 0.031 +/- 0.006 | 3.0 | 0.086 +/- 0.104 | 0.564 +/- 0.091 | 12.554 +/- 0.116 | 13.962 +/- 0.411 | 2.65 +/- 0.12 | ok |
| PubMed | SEClust-ConstrainedK | 10 / 10 | 0.368 +/- 0.014 | 0.005 +/- 0.004 | 0.005 +/- 0.004 | 3.0 | 0.441 +/- 0.010 | 0.225 +/- 0.010 | 11.994 +/- 0.016 | 12.837 +/- 0.040 | 6.29 +/- 0.05 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "GAE", "Computers": "SEClust-ConstrainedK", "Cora": "GAE", "Photo": "SEClust-ConstrainedK", "PubMed": "GAE"}
- Best mean structural entropy (lower): {"Citeseer": "SEClust-ConstrainedK", "Computers": "SEClust-ConstrainedK", "Cora": "SEClust-ConstrainedK", "Photo": "SEClust-ConstrainedK", "PubMed": "SEClust-ConstrainedK"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260511_071853.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260511_071853_aggregated.json`
