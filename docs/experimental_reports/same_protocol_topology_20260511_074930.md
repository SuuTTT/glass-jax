# Same-Protocol Topology Benchmark (20260511_074930)

Protocol: identical dataset object, target-K where required, seed list, post-hoc labels only for metrics, and raw seed-level JSON retained.

- Datasets: LFR
- Methods: Louvain, Leiden, Infomap, Spectral, LabelPropagation, SEClust-ConstrainedK, HCSE
- Seeds: 4, 5, 7
- Quick mode: False

| Dataset | Method | OK / Runs | ACC | NMI | ARI | K | Modularity | Conductance | SE | MapEq | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LFR | HCSE | 3 / 3 | 0.481 +/- 0.082 | 0.346 +/- 0.163 | 0.224 +/- 0.197 | 2.0 | 0.263 +/- 0.162 | 0.249 +/- 0.170 | 8.690 +/- 0.109 | 9.503 +/- 0.333 | 18.94 +/- 6.30 | ok |
| LFR | Infomap | 3 / 3 | 0.904 +/- 0.159 | 0.661 +/- 0.572 | 0.660 +/- 0.572 | 11.7 +/- 10.5 | 0.376 +/- 0.343 | 0.282 +/- 0.080 | 8.002 +/- 1.100 | 8.513 +/- 0.718 | 0.10 +/- 0.01 | ok |
| LFR | LabelPropagation | 3 / 3 | 0.763 +/- 0.218 | 0.450 +/- 0.508 | 0.387 +/- 0.537 | 8.7 +/- 11.6 | 0.274 +/- 0.353 | 0.264 +/- 0.055 | 8.413 +/- 1.158 | 8.745 +/- 0.830 | 0.05 +/- 0.00 | ok |
| LFR | Leiden | 3 / 3 | 0.773 +/- 0.330 | 0.805 +/- 0.295 | 0.700 +/- 0.491 | 13.3 +/- 4.0 | 0.438 +/- 0.244 | 0.420 +/- 0.254 | 7.694 +/- 0.564 | 8.755 +/- 1.050 | 0.10 +/- 0.05 | ok |
| LFR | Louvain | 3 / 3 | 0.774 +/- 0.328 | 0.819 +/- 0.268 | 0.700 +/- 0.491 | 14.0 +/- 3.6 | 0.437 +/- 0.245 | 0.418 +/- 0.250 | 7.689 +/- 0.556 | 8.753 +/- 1.047 | 0.21 +/- 0.10 | ok |
| LFR | SEClust-ConstrainedK | 3 / 3 | 0.622 +/- 0.243 | 0.707 +/- 0.275 | 0.473 +/- 0.424 | 13.3 +/- 3.2 | 0.404 +/- 0.236 | 0.508 +/- 0.208 | 7.740 +/- 0.534 | 8.975 +/- 0.976 | 0.53 +/- 0.23 | ok |
| LFR | **Spectral** | 3 / 3 | 0.915 +/- 0.136 | 0.917 +/- 0.121 | 0.834 +/- 0.263 | 15.0 +/- 6.1 | 0.428 +/- 0.258 | 0.399 +/- 0.209 | 7.711 +/- 0.629 | 8.640 +/- 0.913 | 0.30 +/- 0.09 | ok |

## Best-By-Dataset Checks

- Best mean NMI: {"LFR": "Spectral"}
- Best mean structural entropy (lower): {"LFR": "Louvain"}

## Files

- Raw seed-level JSON: `docs/experimental_reports/same_protocol_topology_20260511_074930.json`
- Aggregated JSON: `docs/experimental_reports/same_protocol_topology_20260511_074930_aggregated.json`
