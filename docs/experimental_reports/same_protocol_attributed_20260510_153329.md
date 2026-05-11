# Same-Protocol Attributed Benchmark (20260510_153329)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora, Citeseer
- Methods: DMoN, MinCutPool
- Seeds: 0, 1, 2
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | **DMoN** | 3 / 3 | 0.337 +/- 0.042 | 0.104 +/- 0.019 | 0.086 +/- 0.021 | 6.0 | 0.404 +/- 0.018 | 0.427 +/- 0.028 | 9.733 +/- 0.062 | 10.857 +/- 0.091 | 0.41 +/- 0.02 | ok |
| Citeseer | MinCutPool | 3 / 3 | 0.258 +/- 0.021 | 0.037 +/- 0.020 | 0.025 +/- 0.017 | 5.3 +/- 1.2 | 0.277 +/- 0.085 | 0.442 +/- 0.011 | 10.192 +/- 0.270 | 11.071 +/- 0.046 | 0.62 +/- 0.01 | ok |
| Cora | **DMoN** | 3 / 3 | 0.261 +/- 0.043 | 0.071 +/- 0.030 | 0.027 +/- 0.036 | 7.0 | 0.235 +/- 0.057 | 0.589 +/- 0.035 | 9.875 +/- 0.185 | 11.151 +/- 0.069 | 0.23 +/- 0.00 | ok |
| Cora | MinCutPool | 3 / 3 | 0.254 +/- 0.007 | 0.017 +/- 0.003 | 0.005 +/- 0.006 | 5.3 +/- 1.2 | 0.196 +/- 0.016 | 0.593 +/- 0.057 | 10.126 +/- 0.056 | 11.216 +/- 0.060 | 0.35 +/- 0.01 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "DMoN", "Cora": "DMoN"}
- Best mean structural entropy (lower): {"Citeseer": "DMoN", "Cora": "DMoN"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260510_153329.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260510_153329_aggregated.json`
