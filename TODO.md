# SEClust Publication Roadmap & TODO

This document outlines the engineering and research tasks required to publish SEClust in a high-impact journal/conference (e.g., NeurIPS, KDD, TKDE). Our primary objective is to match or exceed the speed and accuracy of state-of-the-art modularity optimizers (Leiden, Louvain) while leveraging the theoretical advantages of Structural Entropy.

## 1. Algorithmic Optimization (Reverse-Engineering Leiden/Louvain)
To compete on large-scale benchmarks ($N > 10^3$), SEClust must move beyond flat greedy search.
- [x] **Multi-Level Coarsening (High Priority)**
    - Implement a `coarsen_graph(graph, labels)` function to aggregate nodes into super-nodes, reducing the search space exponentially in subsequent passes.
    - Create a recursive `multi_level_se_clustering` loop.
- [x] **Connectedness Refinement Phase**
    - Implement a fast connected-components check to split clusters that become internally disconnected during local moves (the core innovation of Leiden over Louvain).
- [x] **Full Coding-Tree Optimization**
    - Upgrade `SEClust-Tree` from a greedy bottom-up merge to a full SEP-style optimizer.
    - Implement `CompressDelta` and leaf-up/root-down refinement operations.
    - **Update**: We successfully reproduced SEP's `CompressDelta` logic. Furthermore, we introduced `SEClust-TargetK`, a novel flat-SE merge heuristic powered by our sparse $O(1)$ state, which completely outperforms SEP for hitting specific cluster targets.
- [ ] **JAX Vectorization / GPU Acceleration**
    - Port the innermost heuristic loop and `IncrementalSEState` updates into a `jax.jit` compiled kernel for parallel evaluation.

## 2. Experimental Rigor (For the Paper)
- [ ] **Ablation Studies**
    - Performance & Accuracy: Multi-level vs. Flat heuristic.
    - Impact of the Refinement Phase on resolving local minima (e.g., SBM N=500 case).
- [ ] **Scalability Benchmarks**
    - Log-log plots of runtime vs. $N$ and $E$ up to $10^5 - 10^6$ nodes, comparing SEClust against Leiden and Infomap.
- [ ] **Qualitative Analysis**
    - Visualize clusterings for Cora/Citeseer. Highlight topological/hierarchical features captured by Structural Entropy that Modularity misses.

## 3. Paper Preparation
- [ ] **Literature Review**: Compile a comprehensive bibliography of Structural Entropy, Map Equation, and Modularity optimization literature.
- [ ] **Mathematical Proofs**: Formally derive and present the $O(1)$ incremental update equations for 2D and high-dimensional Structural Entropy used in our sparse state.
- [ ] **Case Studies**: Identify real-world domains (e.g., biological networks, citation trees) where SE's penalization of boundary uncertainty provides a distinct advantage over resolution-limit-prone Modularity.

## 4. Engineering Maintenance
- [ ] Implement a native sparse input API to bypass dense adjacency materialization for large real-world PyG graphs.
- [ ] Move `tests/benchmark_seclust_full.py` into a dedicated benchmarking module.
- [ ] Set up automated GitHub Actions for CI testing and benchmark tracking.