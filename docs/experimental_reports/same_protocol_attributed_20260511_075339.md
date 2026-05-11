# Same-Protocol Attributed Benchmark (20260511_075339)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora, Citeseer, Photo
- Methods: DMoN, MinCutPool
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | **DMoN** | 10 / 10 | 0.466 +/- 0.097 | 0.266 +/- 0.080 | 0.225 +/- 0.086 | 5.7 +/- 0.5 | 0.600 +/- 0.169 | 0.188 +/- 0.100 | 9.268 +/- 0.529 | 9.804 +/- 0.475 | 1.33 +/- 0.03 | ok |
| Citeseer | MinCutPool | 10 / 10 | 0.401 +/- 0.049 | 0.230 +/- 0.043 | 0.186 +/- 0.052 | 5.0 +/- 1.2 | 0.473 +/- 0.064 | 0.401 +/- 0.133 | 9.908 +/- 0.217 | 10.465 +/- 0.187 | 2.19 +/- 0.05 | ok |
| Cora | **DMoN** | 10 / 10 | 0.451 +/- 0.088 | 0.291 +/- 0.109 | 0.208 +/- 0.120 | 6.0 +/- 1.2 | 0.459 +/- 0.171 | 0.376 +/- 0.099 | 9.521 +/- 0.553 | 10.135 +/- 0.402 | 0.74 +/- 0.04 | ok |
| Cora | MinCutPool | 10 / 10 | 0.293 +/- 0.017 | 0.012 +/- 0.014 | 0.005 +/- 0.014 | 2.5 +/- 1.6 | 0.064 +/- 0.086 | 0.634 +/- 0.284 | 10.654 +/- 0.317 | 10.976 +/- 0.123 | 1.16 +/- 0.14 | ok |
| Photo | **DMoN** | 10 / 10 | 0.503 +/- 0.148 | 0.356 +/- 0.215 | 0.286 +/- 0.181 | 7.7 +/- 0.7 | 0.429 +/- 0.243 | 0.438 +/- 0.235 | 10.678 +/- 0.855 | 11.294 +/- 0.572 | 5.63 +/- 0.11 | ok |
| Photo | MinCutPool | 10 / 10 | 0.328 +/- 0.057 | 0.172 +/- 0.050 | 0.087 +/- 0.046 | 6.9 +/- 0.6 | 0.274 +/- 0.063 | 0.581 +/- 0.054 | 11.128 +/- 0.204 | 12.170 +/- 0.148 | 9.56 +/- 0.11 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "DMoN", "Cora": "DMoN", "Photo": "DMoN"}
- Best mean structural entropy (lower): {"Citeseer": "DMoN", "Cora": "DMoN", "Photo": "DMoN"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260511_075339.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260511_075339_aggregated.json`
