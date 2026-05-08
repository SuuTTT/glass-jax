# SEClust Full Benchmark Against Existing Baselines

**Date:** May 8, 2026  
**Project:** glass-jax / `glass.seclust`

## 1. Abstract
This report extends the SEClust benchmark to the datasets used in `tests/benchmark_full.py` and the existing reports:

- `docs/experimental_reports/benchmark_sbm_20260506.md`
- `docs/experimental_reports/real_world_comparison_20260507.md`

Baseline values are copied from those reports. `SEClust-Auto` and `SEClust-Tree` are executed in this run. Any SEClust run with an estimated runtime above 3 minutes is skipped and reported with its estimate.

## 2. Setup
- Synthetic datasets are generated locally with NumPy equivalents of the benchmark definitions.
- Real-world PyG datasets are loaded when `torch_geometric` is available. Dense real-world runs use the same 3 minute guard; Photo is skipped before dense materialization when it exceeds the dense node guard.
- SEClust config: `mode="heuristic"`, `heuristic_starts=6`, `max_passes=10`, seed `42`.
- Runtime limit: `180` seconds per dataset.
- Logged metrics follow `docs/seclust/experiment_protocol.md`: ACC, NMI, ARI, K, modularity, structural entropy, map equation, and runtime.
- Bold values mark the best available result per dataset and metric. K is diagnostic and is not bolded.

## 3. Synthetic Results
| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Estimate (s) | Input (N, E, K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Karate | Louvain | 0.735 | 0.707 | 0.600 | 4 | 0.415 | 3.451 | 4.314 | 0.0032 | skip | 34, 78, 2 |
|  | Leiden | 0.676 | 0.687 | 0.541 | 4 | **0.420** | **3.405** | 4.334 | 0.0031 | skip | 34, 78, 2 |
|  | Infomap | 0.824 | 0.699 | 0.702 | 3 | 0.402 | 3.606 | **4.312** | 0.0064 | skip | 34, 78, 2 |
|  | Glass-Mod (JAX) | **0.971** | **0.837** | 0.882 | 2 | 0.372 | 3.833 | 4.409 | 0.0019 | skip | 34, 78, 2 |
|  | Glass-Map (JAX) | 0.794 | 0.417 | 0.328 | 2 | 0.157 | 4.156 | 4.900 | 0.0076 | skip | 34, 78, 2 |
|  | SEClust-Auto | 0.471 | 0.510 | 0.314 | 6 | 0.342 | 3.427 | 4.612 | 0.0632 | 0.3 | 34, 78, 2 |
|  | SEClust-Tree | **0.971** | 0.836 | **0.882** | 2 | 0.357 | 3.849 | 4.464 | 0.0765 | 0.3 | 34, 78, 2 |
| Caveman (10x20) | Louvain | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.0285 | skip | 200, 1909, 10 |
|  | Leiden | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.0074 | skip | 200, 1909, 10 |
|  | Infomap | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.0152 | skip | 200, 1909, 10 |
|  | Glass-Mod (JAX) | 0.800 | 0.936 | 0.804 | 8 | 0.856 | 4.734 | 4.773 | 0.0178 | skip | 200, 1909, 10 |
|  | Glass-Map (JAX) | 0.800 | 0.936 | 0.804 | 8 | 0.856 | 4.734 | 4.773 | 0.0470 | skip | 200, 1909, 10 |
|  | SEClust-Auto | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.4257 | 3.6 | 200, 1909, 10 |
|  | SEClust-Tree | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.3539 | 3.6 | 200, 1909, 10 |
| SBM (N=100) | Louvain | **1.000** | **1.000** | **1.000** | 4 | **0.626** | **4.840** | **5.399** | 0.0090 | skip | 100, 545, 4 |
|  | Leiden | **1.000** | **1.000** | **1.000** | 4 | **0.626** | **4.840** | **5.399** | 0.0059 | skip | 100, 545, 4 |
|  | Infomap | **1.000** | **1.000** | **1.000** | 4 | **0.626** | **4.840** | **5.399** | 0.0068 | skip | 100, 545, 4 |
|  | Glass-Mod (JAX) | **1.000** | **1.000** | **1.000** | 4 | **0.626** | **4.840** | **5.399** | 0.0038 | skip | 100, 545, 4 |
|  | Glass-Map (JAX) | 0.750 | 0.857 | 0.708 | 3 | 0.521 | 5.263 | 5.729 | 0.0072 | skip | 100, 545, 4 |
|  | SEClust-Auto | 0.720 | 0.850 | 0.731 | 7 | 0.513 | 4.849 | 5.815 | 0.2578 | 1.4 | 100, 545, 4 |
|  | SEClust-Tree | **1.000** | **1.000** | **1.000** | 4 | **0.626** | **4.840** | **5.399** | 0.3003 | 0.9 | 100, 545, 4 |
| SBM (N=500) | Louvain | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 0.1228 | skip | 500, 6091, 5 |
|  | Leiden | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 0.0402 | skip | 500, 6091, 5 |
|  | Infomap | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 0.0570 | skip | 500, 6091, 5 |
|  | Glass-Mod (JAX) | 0.998 | 0.993 | 0.995 | 5 | 0.628 | 7.019 | 7.725 | 0.0843 | skip | 500, 6091, 5 |
|  | Glass-Map (JAX) | 0.984 | 0.961 | 0.961 | 5 | 0.609 | 7.064 | 7.817 | 0.0829 | skip | 500, 6091, 5 |
|  | SEClust-Auto | 0.936 | 0.945 | 0.931 | 9 | 0.582 | 7.073 | 7.874 | 5.7263 | 11.0 | 500, 6091, 5 |
|  | SEClust-Tree | 0.604 | 0.737 | 0.481 | 5 | 0.441 | 7.758 | 8.298 | 4.7568 | 17.5 | 500, 6091, 5 |
| SBM (N=1000) | Louvain | 0.997 | 0.994 | 0.993 | 10 | 0.587 | 7.641 | 8.679 | 0.2011 | skip | 1000, 7299, 10 |
|  | Leiden | **0.999** | **0.998** | **0.998** | 10 | **0.589** | **7.635** | **8.671** | 0.0933 | skip | 1000, 7299, 10 |
|  | Infomap | **0.999** | **0.998** | **0.998** | 10 | **0.589** | **7.635** | **8.671** | 0.0798 | skip | 1000, 7299, 10 |
|  | Glass-Mod (JAX) | 0.749 | 0.907 | 0.772 | 9 | 0.541 | 7.911 | 8.960 | 0.3419 | skip | 1000, 7299, 10 |
|  | Glass-Map (JAX) | 0.781 | 0.849 | 0.731 | 9 | 0.514 | 7.993 | 9.114 | 0.3973 | skip | 1000, 7299, 10 |
|  | SEClust-Auto | 0.944 | 0.968 | 0.947 | 14 | 0.561 | 7.676 | 8.762 | 10.1095 | 26.8 | 1000, 7299, 10 |
|  | SEClust-Tree | 0.897 | 0.966 | 0.894 | 10 | 0.575 | 7.765 | 8.780 | 10.7945 | 14.0 | 1000, 7299, 10 |

## 4. Real-World Results
| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Input (N, E, K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cora | Louvain (Topology) | skip | skip | skip | skip | skip | skip | skip | skip | 2708, 5278, 7 |
|  | LSEnet (Features + DSI) | skip | skip | skip | skip | skip | skip | skip | skip | 2708, 5278, 7 |
|  | Glass-SE (Pure Topology) | skip | skip | skip | skip | skip | skip | skip | skip | 2708, 5278, 7 |
|  | SEClust-Auto | skip | skip | skip | skip | skip | skip | skip | skip | 2708, 5278, 7 |
|  | SEClust-Tree | skip | skip | skip | skip | skip | skip | skip | skip | 2708, 5278, 7 |
| Citeseer | Louvain (Topology) | skip | skip | skip | skip | skip | skip | skip | skip | 3327, 4552, 6 |
|  | LSEnet (Features + DSI) | skip | skip | skip | skip | skip | skip | skip | skip | 3327, 4552, 6 |
|  | Glass-SE (Pure Topology) | skip | skip | skip | skip | skip | skip | skip | skip | 3327, 4552, 6 |
|  | SEClust-Auto | skip | skip | skip | skip | skip | skip | skip | skip | 3327, 4552, 6 |
|  | SEClust-Tree | skip | skip | skip | skip | skip | skip | skip | skip | 3327, 4552, 6 |
| Photo | SEClust-Auto | skip | skip | skip | skip | skip | skip | skip | skip | skip |
|  | SEClust-Tree | skip | skip | skip | skip | skip | skip | skip | skip | skip |

## 5. Summary
- Completed SEClust runs: `10`.
- Skipped or unavailable SEClust runs: `6`.
- Larger synthetic graphs now use sparse incremental structural entropy delta scoring. Runs are skipped only if the incremental estimator exceeds the 3 minute limit.

Raw results are saved at `docs/experimental_reports/seclust_full_benchmark_20260508_000517.json`.
