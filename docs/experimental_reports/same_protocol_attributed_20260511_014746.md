# Same-Protocol Attributed Benchmark (20260511_014746)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: Photo
- Methods: AdjSVDKMeans, GAE, SEClust-ConstrainedK
- Seeds: 9
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Photo | AdjSVDKMeans | 1 / 1 | 0.340 | 0.239 | 0.041 | 8.0 | 0.385 | 0.389 | 10.687 | 11.686 | 3.79 | ok |
| Photo | GAE | 1 / 1 | 0.230 | 0.058 | 0.022 | 8.0 | 0.040 | 0.827 | 11.608 | 13.393 | 7.58 | ok |
| Photo | **SEClust-ConstrainedK** | 1 / 1 | 0.455 | 0.337 | 0.286 | 8.0 | 0.657 | 0.191 | 9.837 | 10.537 | 4.09 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"Photo": "SEClust-ConstrainedK"}
- Best mean structural entropy (lower): {"Photo": "SEClust-ConstrainedK"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_attributed_20260511_014746.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_attributed_20260511_014746_aggregated.json`
