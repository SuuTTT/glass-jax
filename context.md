# SEClust Handoff Summary & Development Context

## 1. Project Evolution: From `Glass-SE` to `SEClust`
This project began as `Glass-SE`, an attempt to optimize Structural Entropy (SE) using continuous, differentiable relaxations (soft assignments via Sinkhorn/Spectral methods) in JAX. While elegant, continuous relaxations scale poorly and struggle to find sharp, definitive graph cuts. 

We pivoted the core methodology to **discrete heuristic optimization** and rebranded the primary algorithm to **SEClust** (Structural Entropy Clustering). SEClust explicitly minimizes hard 2D and high-dimensional Structural Entropy using highly optimized graph algorithms, explicitly reverse-engineering the success of the Leiden and Louvain algorithms.

## 2. Major Algorithmic Breakthroughs (What has been built)
The project currently possesses a highly advanced, sparse, multi-level structural entropy optimizer. The following core features are fully implemented and verified in the benchmark suite:

*   **Sparse Incremental State ($O(1)$ Updates):** We abandoned $O(N^2)$ dense adjacency scoring. The inner loop now uses `IncrementalSEState` (in `src/glass/seclust/incremental.py`), tracking `vol`, `cut`, and `degree_log_degree` arrays. Node moves and cluster merges are evaluated in $O(1)$ to $O(degree)$ time by strictly evaluating local deltas.
*   **Multi-Level Coarsening (The "Leiden/Louvain Secret"):** To solve the scaling bottleneck for graphs $N > 500$, we implemented a multi-level heuristic. The algorithm runs local greedy node moves, projects the graph down into a smaller "super-node" representation using `scipy.sparse` matrix multiplication ($S^T A S$), and repeats recursively until convergence.
*   **Connectedness Refinement:** To prevent the severe local minima caused by clusters fracturing into disconnected components during multi-level passes, we integrated a fast `connected_components` check. Disconnected clusters are forcibly split before coarsening, ensuring high-quality, continuous communities.
*   **Full Coding-Tree Optimizer (`SEClust-Tree`):** We implemented a native, high-dimensional coding-tree builder (`src/glass/seclust/coding_tree.py`) that exactly mirrors the official SEP baseline. It supports both `CombineDelta` (binary merges) and `CompressDelta` (flattening the tree).
*   **Flat Target-K Heuristic (`SEClust-TargetK`):** Because deep coding trees are sub-optimal for flat 2D benchmark scores, we wrote a specialized heuristic that uses our $O(1)$ state to greedily merge clusters by directly minimizing the Flat 2D SE delta until a target $K$ is reached. It currently matches Leiden's mathematical optimum on large synthetic graphs (e.g., SBM $N=1000$).

## 3. The Benchmark Suite
The project includes a robust, automated benchmark pipeline in `tests/benchmark_seclust_full.py`. 
*   It tests against synthetic datasets (Karate, Caveman, SBM) and real-world PyG citation graphs (Cora, Citeseer, Photo).
*   It compares SEClust against **Louvain**, **Leiden**, **Infomap**, and the official **SEP / LSEnet** baselines.
*   It automatically logs dynamically timestamped Markdown and JSON reports to `docs/experimental_reports/`. (Note: Dense real-world datasets like Cora are currently skipped by a built-in time/memory guardrail until native sparse inputs are supported).

## 4. What to do next (The Roadmap to Publication)
The strategic goal is to publish SEClust in a top-tier journal/conference (NeurIPS, KDD, TKDE). The algorithmic logic is sound and highly competitive; the remaining work is purely focused on extreme scaling, vectorization, and paper writing. 

*(This is mirrored in `TODO.md`)*

**High Priority Engineering:**
1.  **JAX Vectorization:** The multi-level coarsening and incremental state updates are currently written in pure Python/NumPy loop structures. To crush Leiden's execution speed, the `local_move_incremental` inner loop needs to be ported into a `jax.jit` compiled kernel so that candidate evaluations can happen in parallel on GPU/TPU.
2.  **Native Sparse PyG Inputs:** Currently, the benchmark converts inputs to dense matrices before passing them to the SEClust `SparseGraph` constructor. You need to write an API that directly accepts PyG `edge_index` (COO/CSR format) so the system can evaluate massive datasets ($N > 10^5$) without throwing OOM errors on the initial dense conversion.

**Experimental Rigor (For the Paper):**
1.  **Scalability Plots:** Once the sparse input API is built, generate log-log scalability plots ($N$ and $E$ up to $1,000,000$ nodes) comparing SEClust execution time against Leiden and Infomap.
2.  **Ablation Studies:** Run benchmarks proving the necessity of the "Connectedness Refinement Phase" for escaping local minima on noisy SBMs.
3.  **Qualitative Analysis:** Visualize the resulting clusterings on Cora or Citeseer. Find explicit visual examples where Modularity falls prey to the "resolution limit" (merging distinct small communities) while Structural Entropy correctly preserves them due to its penalization of boundary uncertainty.

## 5. Key Files to Know
*   `src/glass/seclust/incremental.py`: The heart of the optimizer. Contains the $O(1)$ math, the sparse graph representation, and the multi-level coarsening loop.
*   `src/glass/seclust/coding_tree.py`: The high-dimensional tree builder.
*   `src/glass/seclust/hierarchy.py`: High-level orchestration for `SEClust-Tree` and `SEClust-TargetK`.
*   `docs/seclust/design.md`: The comprehensive mathematical and architectural documentation for the system.