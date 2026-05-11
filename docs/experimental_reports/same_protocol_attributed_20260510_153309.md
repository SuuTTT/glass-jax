# Same-Protocol Attributed Benchmark (20260510_153309)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora, Citeseer, PubMed, Photo, Computers
- Methods: AdjSVDKMeans, GAE, SEClust-ConstrainedK
- Seeds: 0, 1, 2
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | AdjSVDKMeans | 3 / 3 | 0.252 +/- 0.008 | 0.074 +/- 0.010 | 0.000 +/- 0.000 | 6.0 | 0.150 +/- 0.011 | 0.514 +/- 0.052 | 10.612 +/- 0.047 | 10.945 +/- 0.037 | 3.09 +/- 0.07 | ok |
| Citeseer | **GAE** | 3 / 3 | 0.392 +/- 0.042 | 0.199 +/- 0.025 | 0.120 +/- 0.023 | 6.0 | 0.567 +/- 0.024 | 0.239 +/- 0.009 | 9.257 +/- 0.084 | 10.110 +/- 0.083 | 0.73 +/- 0.02 | ok |
| Citeseer | SEClust-ConstrainedK | 3 / 3 | 0.232 +/- 0.002 | 0.021 +/- 0.002 | 0.013 +/- 0.001 | 6.0 | 0.668 +/- 0.004 | 0.164 +/- 0.003 | 8.978 +/- 0.009 | 9.659 +/- 0.016 | 0.75 +/- 0.01 | ok |
| Computers | AdjSVDKMeans | 3 / 3 | 0.236 +/- 0.013 | 0.119 +/- 0.005 | -0.077 +/- 0.002 | 10.0 | 0.290 +/- 0.003 | 0.592 +/- 0.022 | 11.673 +/- 0.027 | 12.857 +/- 0.028 | 8.22 +/- 1.08 | ok |
| Computers | GAE | 3 / 3 | 0.189 +/- 0.001 | 0.037 +/- 0.000 | 0.023 +/- 0.001 | 10.0 | 0.005 +/- 0.000 | 0.889 +/- 0.000 | 12.414 +/- 0.001 | 14.272 +/- 0.002 | 13.57 +/- 0.04 | ok |
| Computers | **SEClust-ConstrainedK** | 3 / 3 | 0.347 +/- 0.005 | 0.300 +/- 0.029 | 0.159 +/- 0.025 | 10.0 | 0.559 +/- 0.007 | 0.304 +/- 0.007 | 10.629 +/- 0.006 | 11.652 +/- 0.029 | 8.11 +/- 0.10 | ok |
| Cora | AdjSVDKMeans | 3 / 3 | 0.354 +/- 0.003 | 0.158 +/- 0.004 | 0.021 +/- 0.003 | 7.0 | 0.149 +/- 0.014 | 0.688 +/- 0.069 | 10.357 +/- 0.078 | 10.868 +/- 0.028 | 2.93 +/- 0.12 | ok |
| Cora | **GAE** | 3 / 3 | 0.405 +/- 0.064 | 0.229 +/- 0.097 | 0.142 +/- 0.073 | 7.0 | 0.368 +/- 0.082 | 0.410 +/- 0.079 | 9.422 +/- 0.208 | 10.667 +/- 0.338 | 0.64 +/- 0.08 | ok |
| Cora | SEClust-ConstrainedK | 3 / 3 | 0.286 +/- 0.030 | 0.082 +/- 0.014 | 0.049 +/- 0.008 | 7.0 | 0.611 +/- 0.015 | 0.246 +/- 0.016 | 8.775 +/- 0.044 | 9.665 +/- 0.081 | 0.75 +/- 0.15 | ok |
| Photo | AdjSVDKMeans | 3 / 3 | 0.335 +/- 0.005 | 0.228 +/- 0.011 | 0.022 +/- 0.017 | 8.0 | 0.364 +/- 0.019 | 0.386 +/- 0.011 | 10.764 +/- 0.070 | 11.674 +/- 0.033 | 3.41 +/- 0.24 | ok |
| Photo | GAE | 3 / 3 | 0.233 +/- 0.001 | 0.059 +/- 0.000 | 0.024 +/- 0.000 | 8.0 | 0.040 +/- 0.000 | 0.827 +/- 0.000 | 11.611 +/- 0.002 | 13.402 +/- 0.006 | 6.78 +/- 0.02 | ok |
| Photo | **SEClust-ConstrainedK** | 3 / 3 | 0.456 +/- 0.055 | 0.323 +/- 0.045 | 0.241 +/- 0.051 | 8.0 | 0.645 +/- 0.006 | 0.209 +/- 0.006 | 9.842 +/- 0.023 | 10.610 +/- 0.039 | 3.79 +/- 0.04 | ok |
| PubMed | AdjSVDKMeans | 3 / 3 | 0.399 +/- 0.002 | 0.018 +/- 0.004 | -0.002 +/- 0.001 | 3.0 | 0.093 +/- 0.016 | 0.540 +/- 0.078 | 12.858 +/- 0.066 | 13.248 +/- 0.028 | 7.31 +/- 1.29 | ok |
| PubMed | **GAE** | 3 / 3 | 0.476 +/- 0.002 | 0.053 +/- 0.002 | 0.029 +/- 0.001 | 3.0 | 0.094 +/- 0.043 | 0.554 +/- 0.037 | 12.530 +/- 0.054 | 13.971 +/- 0.139 | 2.80 +/- 0.03 | ok |
| PubMed | SEClust-ConstrainedK | 3 / 3 | 0.369 +/- 0.013 | 0.006 +/- 0.006 | 0.006 +/- 0.007 | 3.0 | 0.447 +/- 0.007 | 0.219 +/- 0.007 | 11.985 +/- 0.012 | 12.812 +/- 0.030 | 6.32 +/- 0.07 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "GAE", "Computers": "SEClust-ConstrainedK", "Cora": "GAE", "Photo": "SEClust-ConstrainedK", "PubMed": "GAE"}
- Best mean structural entropy (lower): {"Citeseer": "SEClust-ConstrainedK", "Computers": "SEClust-ConstrainedK", "Cora": "SEClust-ConstrainedK", "Photo": "SEClust-ConstrainedK", "PubMed": "SEClust-ConstrainedK"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260510_153309.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260510_153309_aggregated.json`
