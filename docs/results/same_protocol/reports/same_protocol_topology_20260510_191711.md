# Same-Protocol Topology Benchmark (20260510_191711)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Karate, SBM-Easy, SBM-Noisy, DCSBM, LFR, Cora, Citeseer
- Methods: HCSE
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | **HCSE** | 10 / 10 | 0.177 | 0.246 | 0.022 | 438.0 | 0.347 | 0.000 | 8.906 | 8.906 | 9.80 +/- 0.06 | ok |
| Cora | **HCSE** | 10 / 10 | 0.350 | 0.192 | 0.077 | 79.0 | 0.463 | 0.002 | 9.583 | 9.879 | 13.37 +/- 0.06 | ok |
| DCSBM | **HCSE** | 10 / 10 | 0.312 +/- 0.014 | 0.198 +/- 0.025 | 0.119 +/- 0.024 | 68.9 +/- 5.0 | 0.261 +/- 0.011 | 0.259 +/- 0.023 | 8.572 +/- 0.050 | 9.394 +/- 0.044 | 3.66 +/- 0.49 | ok |
| Karate | **HCSE** | 10 / 10 | 1.000 | 1.000 | 1.000 | 2.0 | 0.371 | 0.132 | 3.833 | 4.409 | 0.00 +/- 0.00 | ok |
| LFR | **HCSE** | 7 / 10 | 0.476 +/- 0.122 | 0.356 +/- 0.042 | 0.285 +/- 0.129 | 2.0 | 0.304 +/- 0.045 | 0.218 +/- 0.058 | 8.630 +/- 0.093 | 9.295 +/- 0.099 | 12.02 +/- 3.63 | ok; skipped: unavailable: LFR generation failed after retries: Could not assi... |
| SBM-Easy | **HCSE** | 10 / 10 | 0.400 +/- 0.000 | 0.580 +/- 0.007 | 0.371 +/- 0.002 | 2.0 | 0.381 +/- 0.002 | 0.124 +/- 0.003 | 9.061 +/- 0.003 | 9.538 +/- 0.009 | 28.68 +/- 0.44 | ok |
| SBM-Noisy | **HCSE** | 10 / 10 | 0.239 +/- 0.011 | 0.010 +/- 0.005 | 0.007 +/- 0.004 | 2.0 | 0.094 +/- 0.004 | 0.486 +/- 0.024 | 9.366 +/- 0.015 | 10.519 +/- 0.022 | 11.99 +/- 2.54 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "HCSE", "Cora": "HCSE", "DCSBM": "HCSE", "Karate": "HCSE", "LFR": "HCSE", "SBM-Easy": "HCSE", "SBM-Noisy": "HCSE"}
- Best mean structural entropy (lower): {"Citeseer": "HCSE", "Cora": "HCSE", "DCSBM": "HCSE", "Karate": "HCSE", "LFR": "HCSE", "SBM-Easy": "HCSE", "SBM-Noisy": "HCSE"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_topology_20260510_191711.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_topology_20260510_191711_aggregated.json`
