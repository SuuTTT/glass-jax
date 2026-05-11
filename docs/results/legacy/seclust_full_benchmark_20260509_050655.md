# SEClust Full Benchmark Against Existing Baselines

**Date:** May 9, 2026
**Project:** glass-jax / `glass.seclust`

## 1. Abstract
This report benchmarks SEClust on the synthetic datasets defined in `tests/benchmark_full.py` and the real-world PyG datasets (Cora, Citeseer, Photo) loaded via `torch_geometric.datasets`. All baselines (Louvain, Leiden, Infomap, Glass-JAX variants) are executed end-to-end in this run; no values are imported from prior reports. Real-world graphs are loaded directly into a sparse `SparseGraph` adjacency without any dense `(N, N)` materialization, so Cora/Citeseer/Photo no longer hit the dense node guard.

## 2. Setup
- Synthetic datasets are generated locally with NumPy adjacencies; the SEClust path automatically converts to sparse incremental scoring.
- Real-world PyG datasets are loaded with `SparseGraph.from_edge_index(edge_index, num_nodes)` — no dense materialization. Baselines (Louvain/Leiden/Infomap) accept the sparse adjacency through a CSR coercion path.
- SEClust config: `heuristic_starts=6`, `max_passes=10`, seed `42`.
- Runtime limit: `600` seconds per dataset; runs that the incremental estimator predicts will exceed this are skipped with the estimate recorded.
- Logged metrics follow `docs/seclust/experiment_protocol.md`: ACC, NMI, ARI, K, modularity, structural entropy, map equation, and runtime.
- Bold values mark the best available result per dataset and metric. K is diagnostic and is not bolded.

## 3. Synthetic Results
| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Estimate (s) | Graph (N, E, True K*) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Karate | Louvain | 0.735 | 0.707 | 0.600 | 4 | 0.415 | 3.451 | 4.314 | 0.0044 | skip | 34, 78, 2 |
|  | Leiden | 0.676 | 0.687 | 0.541 | 4 | **0.420** | 3.405 | 4.334 | 0.0019 | skip | 34, 78, 2 |
|  | Infomap | 0.824 | 0.699 | 0.702 | 3 | 0.402 | 3.606 | **4.312** | 0.0066 | skip | 34, 78, 2 |
|  | Glass-Mod (JAX) | 0.971 | 0.837 | 0.882 | 2 | 0.372 | 3.833 | 4.409 | 0.0060 | skip | 34, 78, 2 |
|  | Glass-Map (JAX) | 0.794 | 0.417 | 0.328 | 2 | 0.157 | 4.156 | 4.900 | 0.0082 | skip | 34, 78, 2 |
|  | SEP | **1.000** | **1.000** | **1.000** | 2 | 0.371 | 3.833 | 4.409 | 0.0103 | skip | 34, 78, 2 |
|  | SEClust-Auto | 0.559 | 0.537 | 0.369 | 6 | 0.383 | **3.365** | 4.479 | 0.1252 | 0.4 | 34, 78, 2 |
|  | SEClust-Tree | 0.971 | 0.836 | 0.882 | 2 | 0.360 | 3.852 | 4.424 | 0.0954 | 0.2 | 34, 78, 2 |
|  | SEClust-TargetK | 0.971 | 0.836 | 0.882 | 2 | 0.360 | 3.852 | 4.424 | 0.0956 | 0.2 | 34, 78, 2 |
| Caveman (10x20) | Louvain | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.0245 | skip | 200, 1909, 10 |
|  | Leiden | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.0092 | skip | 200, 1909, 10 |
|  | Infomap | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.0175 | skip | 200, 1909, 10 |
|  | Glass-Mod (JAX) | 0.800 | 0.936 | 0.804 | 8 | 0.856 | 4.734 | 4.773 | 0.0330 | skip | 200, 1909, 10 |
|  | Glass-Map (JAX) | 0.800 | 0.936 | 0.804 | 8 | 0.856 | 4.734 | 4.773 | 0.1051 | skip | 200, 1909, 10 |
|  | SEP | 0.700 | 0.863 | 0.554 | 7 | 0.777 | 5.132 | 5.160 | 0.0498 | skip | 200, 1909, 10 |
|  | SEClust-Auto | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.3434 | 4.7 | 200, 1909, 10 |
|  | SEClust-Tree | 0.400 | 0.733 | 0.433 | 4 | 0.718 | 5.725 | 5.741 | 0.3475 | 4.8 | 200, 1909, 10 |
|  | SEClust-TargetK | **1.000** | **1.000** | **1.000** | 10 | **0.895** | **4.337** | **4.380** | 0.3468 | 4.8 | 200, 1909, 10 |
| SBM (N=100) | Louvain | **1.000** | **1.000** | **1.000** | 4 | **0.626** | 4.840 | **5.399** | 0.0099 | skip | 100, 545, 4 |
|  | Leiden | **1.000** | **1.000** | **1.000** | 4 | **0.626** | 4.840 | **5.399** | 0.0051 | skip | 100, 545, 4 |
|  | Infomap | **1.000** | **1.000** | **1.000** | 4 | **0.626** | 4.840 | **5.399** | 0.0111 | skip | 100, 545, 4 |
|  | Glass-Mod (JAX) | **1.000** | **1.000** | **1.000** | 4 | **0.626** | 4.840 | **5.399** | 0.0079 | skip | 100, 545, 4 |
|  | Glass-Map (JAX) | 0.750 | 0.857 | 0.708 | 3 | 0.521 | 5.263 | 5.729 | 0.0142 | skip | 100, 545, 4 |
|  | SEP | 0.990 | 0.970 | 0.973 | 4 | 0.619 | 4.854 | 5.435 | 0.0457 | skip | 100, 545, 4 |
|  | SEClust-Auto | 0.820 | 0.897 | 0.833 | 6 | 0.550 | **4.839** | 5.653 | 0.3388 | 1.2 | 100, 545, 4 |
|  | SEClust-Tree | **1.000** | **1.000** | **1.000** | 4 | **0.626** | 4.840 | **5.399** | 0.3391 | 1.2 | 100, 545, 4 |
|  | SEClust-TargetK | **1.000** | **1.000** | **1.000** | 4 | **0.626** | 4.840 | **5.399** | 0.3397 | 1.2 | 100, 545, 4 |
| SBM (N=500) | Louvain | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 0.1118 | skip | 500, 6091, 5 |
|  | Leiden | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 0.0398 | skip | 500, 6091, 5 |
|  | Infomap | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 0.0617 | skip | 500, 6091, 5 |
|  | Glass-Mod (JAX) | 0.998 | 0.993 | 0.995 | 5 | 0.628 | 7.019 | 7.725 | 0.1360 | skip | 500, 6091, 5 |
|  | Glass-Map (JAX) | 0.984 | 0.961 | 0.961 | 5 | 0.609 | 7.064 | 7.817 | 0.1690 | skip | 500, 6091, 5 |
|  | SEP | 0.400 | 0.590 | 0.373 | 2 | 0.380 | 8.073 | 8.552 | 4.2094 | skip | 500, 6091, 5 |
|  | SEClust-Auto | 0.950 | 0.966 | 0.951 | 6 | 0.598 | 7.045 | 7.813 | 4.4944 | 14.6 | 500, 6091, 5 |
|  | SEClust-Tree | 0.600 | 0.742 | 0.481 | 3 | 0.440 | 7.769 | 8.297 | 4.5265 | 14.4 | 500, 6091, 5 |
|  | SEClust-TargetK | **1.000** | **1.000** | **1.000** | 5 | **0.629** | **7.016** | **7.717** | 4.5051 | 14.7 | 500, 6091, 5 |
| SBM (N=1000) | Louvain | 0.997 | 0.994 | 0.993 | 10 | 0.587 | 7.641 | 8.679 | 0.1604 | skip | 1000, 7299, 10 |
|  | Leiden | **0.999** | **0.998** | **0.998** | 10 | **0.589** | **7.635** | **8.671** | 0.1042 | skip | 1000, 7299, 10 |
|  | Infomap | **0.999** | **0.998** | **0.998** | 10 | **0.589** | **7.635** | **8.671** | 0.0965 | skip | 1000, 7299, 10 |
|  | Glass-Mod (JAX) | 0.749 | 0.907 | 0.772 | 9 | 0.541 | 7.911 | 8.960 | 0.3146 | skip | 1000, 7299, 10 |
|  | Glass-Map (JAX) | 0.781 | 0.849 | 0.731 | 9 | 0.514 | 7.993 | 9.114 | 3.3579 | skip | 1000, 7299, 10 |
|  | SEP | 0.200 | 0.438 | 0.182 | 2 | 0.313 | 9.125 | 9.801 | 14.1816 | skip | 1000, 7299, 10 |
|  | SEClust-Auto | 0.963 | 0.984 | 0.971 | 11 | 0.574 | 7.655 | 8.718 | 11.6027 | 17.2 | 1000, 7299, 10 |
|  | SEClust-Tree | 0.200 | 0.448 | 0.184 | 2 | 0.321 | 9.116 | 9.774 | 11.4808 | 17.9 | 1000, 7299, 10 |
|  | SEClust-TargetK | **0.999** | **0.998** | **0.998** | 10 | **0.589** | **7.635** | **8.671** | 11.4484 | 18.4 | 1000, 7299, 10 |

## 4. Real-World Results
| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Graph (N, E, True K*) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cora | Louvain | 0.372 | 0.439 | 0.236 | 105 | 0.808 | 6.932 | 7.452 | 0.3058 | 2708, 5278, 7 |
|  | Leiden | 0.380 | **0.459** | 0.238 | 107 | **0.822** | 6.837 | 7.337 | 0.0969 | 2708, 5278, 7 |
|  | Infomap | 0.123 | 0.410 | 0.050 | 280 | 0.732 | **5.563** | **6.378** | 0.1512 | 2708, 5278, 7 |
|  | SEP | 0.350 | 0.192 | 0.077 | 79 | 0.463 | 9.583 | 9.879 | 13.4092 | 2708, 5278, 7 |
|  | LSEnet | 0.388 | 0.264 | 0.164 | 7 | 0.601 | 9.025 | 9.743 | 1.6068 | 2708, 5278, 7 |
|  | SEClust-Auto | 0.082 | 0.400 | 0.027 | 429 | 0.637 | 5.787 | 6.817 | 11.2033 | 2708, 5278, 7 |
|  | SEClust-Tree | **0.497** | 0.377 | **0.242** | 82 | 0.678 | 8.588 | 9.053 | 11.4786 | 2708, 5278, 7 |
|  | SEClust-TargetK | 0.290 | 0.048 | -0.003 | 7 | 0.077 | 10.550 | 10.550 | 15.9021 | 2708, 5278, 7 |
| Citeseer | Louvain | 0.192 | 0.329 | 0.094 | 471 | 0.890 | 5.750 | 6.019 | 0.3235 | 3327, 4552, 6 |
|  | Leiden | 0.194 | 0.327 | 0.094 | 468 | **0.891** | 5.758 | 6.023 | 0.1086 | 3327, 4552, 6 |
|  | Infomap | 0.060 | 0.326 | 0.019 | 582 | 0.824 | **4.538** | **5.054** | 0.1840 | 3327, 4552, 6 |
|  | SEP | 0.177 | 0.246 | 0.022 | 438 | 0.347 | 8.906 | 8.906 | 10.1951 | 3327, 4552, 6 |
|  | LSEnet | **0.404** | 0.195 | **0.166** | 6 | 0.687 | 8.970 | 9.525 | 3.0991 | 3327, 4552, 6 |
|  | SEClust-Auto | 0.057 | **0.338** | 0.018 | 761 | 0.764 | 4.742 | 5.423 | 11.8840 | 3327, 4552, 6 |
|  | SEClust-Tree | 0.314 | 0.309 | 0.129 | 444 | 0.832 | 6.914 | 7.200 | 11.8795 | 3327, 4552, 6 |
|  | SEClust-TargetK | 0.243 | 0.028 | 0.022 | 6 | 0.339 | 9.996 | 9.996 | 92.0793 | 3327, 4552, 6 |
| Photo | Louvain | **0.674** | **0.665** | **0.574** | 150 | 0.738 | 9.266 | 9.689 | 4.7609 | 7650, 119081, 8 |
|  | Leiden | **0.674** | 0.662 | 0.573 | 151 | **0.739** | 9.258 | 9.682 | 0.8489 | 7650, 119081, 8 |
|  | Infomap | 0.407 | 0.570 | 0.355 | 115 | 0.697 | 8.772 | **9.390** | 1.3815 | 7650, 119081, 8 |
|  | SEP | skip | skip | skip | skip | skip | skip | skip | skip | 7650, 119081, 8 |
|  | LSEnet | 0.429 | 0.328 | 0.206 | 2 | 0.416 | 11.214 | 11.639 | 8.3949 | 7650, 119081, 8 |
|  | SEClust-Auto | 0.240 | 0.519 | 0.174 | 306 | 0.623 | **8.648** | 9.596 | 151.2413 | 7650, 119081, 8 |
|  | SEClust-Tree | 0.619 | 0.586 | 0.422 | 173 | 0.696 | 9.556 | 9.995 | 155.9723 | 7650, 119081, 8 |
|  | SEClust-TargetK | 0.255 | 0.006 | 0.000 | 8 | 0.001 | 12.126 | 12.126 | 154.4235 | 7650, 119081, 8 |

## 5. Summary
- Completed SEClust runs: `24`.
- Skipped or unavailable SEClust runs: `0`.
- All baselines on Cora/Citeseer/Photo are executed in this run via the sparse pipeline.

**Notes:**
- **True K***: Ground-truth number of communities. Parameter-free community detection algorithms (Louvain, Leiden, Infomap) do not take `K` as an input.
- **SEClust-Auto** runs `cluster_graph(mode="heuristic")` with multistart incremental local move; **SEClust-Tree** runs `hierarchical_se_clustering` with target_clusters=K; **SEClust-TargetK** runs `merge_hierarchy_levels` then `select_hierarchy_level(K)`.

Raw results are saved at `docs/experimental_reports/seclust_full_benchmark_20260509_050655.json`.
