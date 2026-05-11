# Same-Protocol Attributed Benchmark (20260510_173512)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Cora
- Methods: DeepWalkKMeans, Node2VecKMeans
- Seeds: 0
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cora | **DeepWalkKMeans** | 1 / 1 | 0.597 | 0.382 | 0.314 | 7.0 | 0.622 | 0.315 | 8.889 | 9.613 | 112.93 | ok |
| Cora | Node2VecKMeans | 1 / 1 | 0.532 | 0.352 | 0.288 | 7.0 | 0.601 | 0.323 | 8.927 | 9.699 | 116.73 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Cora": "DeepWalkKMeans"}
- Best mean structural entropy (lower): {"Cora": "DeepWalkKMeans"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260510_173512.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260510_173512_aggregated.json`
