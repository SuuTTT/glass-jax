# SEClust Full Benchmark Against Existing Baselines

**Date:** May 7, 2026  
**Project:** glass-jax / `glass.seclust`

## 1. Abstract
This report extends the SEClust benchmark to the datasets used in `tests/benchmark_full.py` and the existing reports:

- `docs/experimental_reports/benchmark_sbm_20260506.md`
- `docs/experimental_reports/real_world_comparison_20260507.md`

Baseline values are copied from those reports. `SEClust-Auto` and `SEClust-Tree` are executed in this run. Any SEClust run with an estimated runtime above 3 minutes is skipped and reported with its estimate.

## 2. Setup
- Synthetic datasets are generated locally with NumPy equivalents of the benchmark definitions.
- Real-world PyG datasets are not installed in this workspace, so Cora and Citeseer SEClust rows are marked unavailable.
- SEClust config: `mode="heuristic"`, `heuristic_starts=6`, `max_passes=10`, seed `42`.
- Runtime limit: `180` seconds per dataset.
- Logged metrics follow `docs/seclust/experiment_protocol.md`: ACC, NMI, ARI, K, modularity, structural entropy, map equation, and runtime.
- Bold values mark the best available result per dataset and metric. K is diagnostic and is not bolded.

## 3. Synthetic Results
| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Estimate (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Karate | Louvain | skip | 0.588 | 0.465 | skip | skip | skip | skip | 0.0048 | skip | baseline from benchmark_sbm_20260506.md |
|  | Infomap | skip | 0.691 | 0.684 | skip | skip | skip | skip | 0.0052 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Mod (JAX) | skip | **0.837** | 0.882 | skip | skip | skip | skip | 0.0239 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Map (JAX) | skip | **0.837** | 0.882 | skip | skip | skip | skip | 0.0158 | skip | baseline from benchmark_sbm_20260506.md |
|  | SEClust-Auto | 0.471 | 0.510 | 0.314 | 6 | 0.342 | **3.427** | 4.612 | 0.0885 | 0.4 | ok |
|  | SEClust-Tree | **0.971** | 0.836 | **0.882** | 2 | **0.357** | 3.849 | **4.464** | 0.0630 | 0.4 | ok |
| Caveman (10x20) | Louvain | skip | **1.000** | **1.000** | skip | skip | skip | skip | 0.0243 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Mod (JAX) | skip | 0.969 | 0.894 | skip | skip | skip | skip | 0.3679 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Map (JAX) | skip | 0.901 | 0.728 | skip | skip | skip | skip | 0.6663 | skip | baseline from benchmark_sbm_20260506.md |
|  | SEClust-Auto | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.4728 | 7.1 | ok |
|  | SEClust-Tree | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.3028 | 4.4 | ok |
| SBM (N=100) | Louvain | skip | 0.970 | 0.973 | skip | skip | skip | skip | 0.0416 | skip | baseline from benchmark_sbm_20260506.md |
|  | Infomap | skip | **1.000** | **1.000** | skip | skip | skip | skip | 0.0281 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Mod (JAX) | skip | 0.970 | 0.973 | skip | skip | skip | skip | 0.0449 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Map (JAX) | skip | 0.857 | 0.708 | skip | skip | skip | skip | 0.1020 | skip | baseline from benchmark_sbm_20260506.md |
|  | SEClust-Auto | 0.720 | 0.850 | 0.731 | 7 | 0.513 | 4.849 | 5.815 | 0.3176 | 1.0 | ok |
|  | SEClust-Tree | **1.000** | **1.000** | **1.000** | 4 | **0.626** | **4.840** | **5.399** | 0.2302 | 2.0 | ok |
| SBM (N=500) | Louvain | skip | **1.000** | **1.000** | skip | skip | skip | skip | 0.1925 | skip | baseline from benchmark_sbm_20260506.md |
|  | Infomap | skip | **1.000** | **1.000** | skip | skip | skip | skip | 0.0600 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Mod (JAX) | skip | **1.000** | **1.000** | skip | skip | skip | skip | 0.8530 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Map (JAX) | skip | 0.897 | 0.888 | skip | skip | skip | skip | 1.0643 | skip | baseline from benchmark_sbm_20260506.md |
|  | SEClust-Auto | **0.936** | 0.945 | 0.931 | 9 | **0.582** | **7.073** | **7.874** | 6.9949 | 19.1 | ok |
|  | SEClust-Tree | 0.604 | 0.737 | 0.481 | 5 | 0.441 | 7.758 | 8.298 | 5.9355 | 11.9 | ok |
| SBM (N=1000) | Louvain | skip | 0.995 | 0.996 | skip | skip | skip | skip | 0.4850 | skip | baseline from benchmark_sbm_20260506.md |
|  | Infomap | skip | **0.998** | **0.998** | skip | skip | skip | skip | 0.1691 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Mod (JAX) | skip | 0.886 | 0.800 | skip | skip | skip | skip | 5.9432 | skip | baseline from benchmark_sbm_20260506.md |
|  | Glass-Map (JAX) | skip | 0.754 | 0.621 | skip | skip | skip | skip | 7.4689 | skip | baseline from benchmark_sbm_20260506.md |
|  | SEClust-Auto | **0.944** | 0.968 | 0.947 | 14 | 0.561 | **7.676** | **8.762** | 12.6195 | 24.0 | ok |
|  | SEClust-Tree | 0.897 | 0.966 | 0.894 | 10 | **0.575** | 7.765 | 8.780 | 13.7013 | 29.4 | ok |

## 4. Real-World Results
| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cora | Louvain (Topology) | 0.372 | **0.439** | **0.236** | skip | skip | skip | skip | skip | baseline from real_world_comparison_20260507.md |
|  | LSEnet (Features + DSI) | **0.387** | 0.266 | 0.164 | skip | skip | skip | skip | skip | baseline from real_world_comparison_20260507.md |
|  | Glass-SE (Pure Topology) | 0.274 | 0.076 | 0.039 | skip | skip | skip | skip | skip | baseline from real_world_comparison_20260507.md |
|  | SEClust-Auto | 0.066 | 0.395 | 0.022 | 473 | **0.604** | **5.940** | **7.015** | 11.2989 | ok |
|  | SEClust-Tree | 0.292 | 0.022 | -0.007 | 7 | 0.036 | 10.717 | 10.717 | 184.2487 | ok |
| Citeseer | Louvain (Topology) | 0.192 | **0.329** | 0.094 | skip | skip | skip | skip | skip | baseline from real_world_comparison_20260507.md |
|  | LSEnet (Features + DSI) | **0.403** | 0.195 | **0.166** | skip | skip | skip | skip | skip | baseline from real_world_comparison_20260507.md |
|  | Glass-SE (Pure Topology) | 0.252 | 0.037 | 0.025 | skip | skip | skip | skip | skip | baseline from real_world_comparison_20260507.md |
|  | SEClust-Auto | 0.064 | 0.311 | 0.013 | 620 | **0.718** | **5.212** | **5.966** | 10.5752 | ok |
|  | SEClust-Tree | 0.222 | 0.012 | 0.005 | 6 | 0.176 | 10.506 | 10.506 | 429.0054 | ok |
| Photo | SEClust-Auto | skip | skip | skip | skip | skip | skip | skip | skip | skipped before dense materialization: 7650 nodes would require ~0.44 GiB dense adjacency |
|  | SEClust-Tree | skip | skip | skip | skip | skip | skip | skip | skip | skipped before dense materialization: 7650 nodes would require ~0.44 GiB dense adjacency |

## 5. Summary
- Completed SEClust runs: `14`.
- Skipped or unavailable SEClust runs: `2`.
- Larger synthetic graphs now use sparse incremental structural entropy delta scoring. Runs are skipped only if the incremental estimator exceeds the 3 minute limit.

Raw results are saved at `docs/experimental_reports/seclust_full_benchmark_20260507.json`.
