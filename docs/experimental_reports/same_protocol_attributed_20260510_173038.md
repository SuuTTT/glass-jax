# Same-Protocol Attributed Benchmark (20260510_173038)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora, Citeseer, PubMed, Photo, Computers, Coauthor-CS, Coauthor-Physics
- Methods: VGAE
- Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Citeseer | **VGAE** | 10 / 10 | 0.469 +/- 0.037 | 0.251 +/- 0.019 | 0.201 +/- 0.027 | 6.0 | 0.686 +/- 0.015 | 0.135 +/- 0.015 | 8.951 +/- 0.058 | 9.520 +/- 0.080 | 0.86 +/- 0.04 | ok |
| Coauthor-CS | **VGAE** | 10 / 10 | 0.544 +/- 0.036 | 0.604 +/- 0.027 | 0.471 +/- 0.042 | 15.0 | 0.431 +/- 0.033 | 0.494 +/- 0.036 | 11.705 +/- 0.129 | 13.036 +/- 0.191 | 8.88 +/- 0.11 | ok |
| Coauthor-Physics | **VGAE** | 10 / 10 | 0.615 +/- 0.036 | 0.543 +/- 0.041 | 0.441 +/- 0.043 | 5.0 | 0.463 +/- 0.022 | 0.249 +/- 0.029 | 12.954 +/- 0.050 | 13.862 +/- 0.105 | 19.58 +/- 0.24 | ok |
| Computers | **VGAE** | 10 / 10 | 0.194 +/- 0.008 | 0.037 +/- 0.000 | 0.023 +/- 0.001 | 10.0 | 0.005 +/- 0.000 | 0.889 +/- 0.001 | 12.409 +/- 0.005 | 14.263 +/- 0.012 | 14.99 +/- 0.09 | ok |
| Cora | **VGAE** | 10 / 10 | 0.598 +/- 0.049 | 0.460 +/- 0.031 | 0.369 +/- 0.044 | 7.0 | 0.648 +/- 0.031 | 0.176 +/- 0.025 | 8.740 +/- 0.096 | 9.405 +/- 0.147 | 0.67 +/- 0.03 | ok |
| Photo | **VGAE** | 10 / 10 | 0.231 +/- 0.002 | 0.058 +/- 0.001 | 0.023 +/- 0.001 | 8.0 | 0.040 +/- 0.001 | 0.827 +/- 0.001 | 11.609 +/- 0.002 | 13.396 +/- 0.006 | 7.59 +/- 0.12 | ok |
| PubMed | **VGAE** | 10 / 10 | 0.389 +/- 0.001 | 0.000 +/- 0.000 | -0.000 +/- 0.000 | 3.0 | -0.133 +/- 0.001 | 0.803 +/- 0.001 | 12.910 +/- 0.002 | 14.669 +/- 0.002 | 3.33 +/- 0.07 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Citeseer": "VGAE", "Coauthor-CS": "VGAE", "Coauthor-Physics": "VGAE", "Computers": "VGAE", "Cora": "VGAE", "Photo": "VGAE", "PubMed": "VGAE"}
- Best mean structural entropy (lower): {"Citeseer": "VGAE", "Coauthor-CS": "VGAE", "Coauthor-Physics": "VGAE", "Computers": "VGAE", "Cora": "VGAE", "Photo": "VGAE", "PubMed": "VGAE"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260510_173038.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260510_173038_aggregated.json`
