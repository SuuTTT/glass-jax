# Same-Protocol Attributed Benchmark (20260510_180637)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora, Citeseer, PubMed, Photo, Computers, Coauthor-CS, Coauthor-Physics
- Methods: DeepWalkKMeans, Node2VecKMeans
- Seeds: 0, 1, 2
- Quick mode: True

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | DeepWalkKMeans | 3 / 3 | 0.191 +/- 0.009 | 0.003 +/- 0.002 | 0.001 +/- 0.001 | 6.0 | 0.031 +/- 0.004 | 0.802 +/- 0.004 | 10.624 +/- 0.011 | 12.406 +/- 0.015 | 9.47 +/- 0.08 | ok |
| Citeseer | **Node2VecKMeans** | 3 / 3 | 0.193 +/- 0.003 | 0.004 +/- 0.000 | 0.001 +/- 0.000 | 6.0 | 0.035 +/- 0.003 | 0.799 +/- 0.003 | 10.615 +/- 0.008 | 12.392 +/- 0.013 | 9.54 +/- 0.09 | ok |
| Coauthor-CS | DeepWalkKMeans | 3 / 3 | 0.233 +/- 0.018 | 0.114 +/- 0.008 | 0.069 +/- 0.004 | 15.0 | 0.226 +/- 0.013 | 0.763 +/- 0.017 | 12.577 +/- 0.056 | 14.183 +/- 0.074 | 53.34 +/- 0.27 | ok |
| Coauthor-CS | **Node2VecKMeans** | 3 / 3 | 0.238 +/- 0.006 | 0.115 +/- 0.007 | 0.071 +/- 0.003 | 15.0 | 0.226 +/- 0.009 | 0.765 +/- 0.015 | 12.580 +/- 0.041 | 14.184 +/- 0.053 | 53.66 +/- 0.37 | ok |
| Coauthor-Physics | DeepWalkKMeans | 3 / 3 | 0.549 +/- 0.017 | 0.292 +/- 0.011 | 0.275 +/- 0.005 | 5.0 | 0.381 +/- 0.014 | 0.386 +/- 0.016 | 13.254 +/- 0.055 | 14.204 +/- 0.020 | 101.00 +/- 1.06 | ok |
| Coauthor-Physics | **Node2VecKMeans** | 3 / 3 | 0.548 +/- 0.018 | 0.293 +/- 0.011 | 0.274 +/- 0.007 | 5.0 | 0.379 +/- 0.016 | 0.387 +/- 0.018 | 13.254 +/- 0.058 | 14.212 +/- 0.030 | 102.00 +/- 0.30 | ok |
| Computers | DeepWalkKMeans | 3 / 3 | 0.370 +/- 0.003 | 0.175 +/- 0.004 | 0.138 +/- 0.003 | 10.0 | 0.304 +/- 0.004 | 0.589 +/- 0.003 | 11.605 +/- 0.006 | 12.764 +/- 0.009 | 40.85 +/- 0.92 | ok |
| Computers | **Node2VecKMeans** | 3 / 3 | 0.368 +/- 0.010 | 0.175 +/- 0.004 | 0.132 +/- 0.010 | 10.0 | 0.302 +/- 0.015 | 0.587 +/- 0.008 | 11.594 +/- 0.026 | 12.782 +/- 0.070 | 41.57 +/- 0.70 | ok |
| Cora | **DeepWalkKMeans** | 3 / 3 | 0.169 +/- 0.006 | 0.006 +/- 0.000 | 0.001 +/- 0.001 | 7.0 | 0.021 +/- 0.009 | 0.836 +/- 0.009 | 10.431 +/- 0.025 | 12.255 +/- 0.035 | 7.60 +/- 0.11 | ok |
| Cora | Node2VecKMeans | 3 / 3 | 0.174 +/- 0.005 | 0.006 +/- 0.001 | 0.001 +/- 0.001 | 7.0 | 0.025 +/- 0.005 | 0.833 +/- 0.006 | 10.420 +/- 0.015 | 12.238 +/- 0.020 | 7.68 +/- 0.16 | ok |
| Photo | DeepWalkKMeans | 3 / 3 | 0.362 +/- 0.013 | 0.142 +/- 0.012 | 0.114 +/- 0.010 | 8.0 | 0.302 +/- 0.011 | 0.565 +/- 0.017 | 10.901 +/- 0.036 | 12.213 +/- 0.057 | 22.29 +/- 0.14 | ok |
| Photo | **Node2VecKMeans** | 3 / 3 | 0.371 +/- 0.010 | 0.144 +/- 0.006 | 0.117 +/- 0.006 | 8.0 | 0.301 +/- 0.010 | 0.562 +/- 0.010 | 10.888 +/- 0.030 | 12.222 +/- 0.037 | 22.54 +/- 0.25 | ok |
| PubMed | **DeepWalkKMeans** | 3 / 3 | 0.502 +/- 0.006 | 0.078 +/- 0.006 | 0.079 +/- 0.005 | 3.0 | 0.339 +/- 0.024 | 0.335 +/- 0.027 | 12.187 +/- 0.047 | 13.175 +/- 0.080 | 56.16 +/- 0.85 | ok |
| PubMed | Node2VecKMeans | 3 / 3 | 0.500 +/- 0.008 | 0.076 +/- 0.007 | 0.077 +/- 0.005 | 3.0 | 0.333 +/- 0.021 | 0.342 +/- 0.024 | 12.199 +/- 0.040 | 13.196 +/- 0.073 | 56.61 +/- 0.41 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "Node2VecKMeans", "Coauthor-CS": "Node2VecKMeans", "Coauthor-Physics": "Node2VecKMeans", "Computers": "Node2VecKMeans", "Cora": "DeepWalkKMeans", "Photo": "Node2VecKMeans", "PubMed": "DeepWalkKMeans"}
- Best mean structural entropy (lower): {"Citeseer": "Node2VecKMeans", "Coauthor-CS": "DeepWalkKMeans", "Coauthor-Physics": "DeepWalkKMeans", "Computers": "Node2VecKMeans", "Cora": "Node2VecKMeans", "Photo": "Node2VecKMeans", "PubMed": "DeepWalkKMeans"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260510_180637.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260510_180637_aggregated.json`
