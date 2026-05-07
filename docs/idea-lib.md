# Idea Library: Enhancing Differentiable Graph Clustering

This document compiles advanced techniques, literature references, and actionable ideas for further enhancing the clustering performance of `glass-jax`. The focus is on preventing degenerate solutions (mode collapse), capturing higher-order structures, and refining the continuous relaxation of discrete objectives like Modularity and the Map Equation.

---

## 1. Advanced Modularity Relaxations

While `glass-jax` currently implements a straightforward trace-based relaxation of Modularity ($Q = \text{Tr}(S^T B S)$), recent literature suggests several improvements:

*   **DMoN (Deep Modularity Networks):** The foundational Google Research paper introduces a specific "Collapse Regularization" term alongside the modularity objective. Instead of just optimizing the trace, it actively penalizes the model when the cluster size distribution deviates from a uniform prior, preventing the "one giant cluster" problem without being as overly restrictive as strict orthogonality constraints.
*   **DCAT (Differentiable Clustering for Graph ATtention, 2024):** Integrates spectral relaxation directly into graph attention mechanisms. This suggests that instead of clustering *after* GNN message passing, the clustering objective could be used to directly guide the message passing (attention weights) itself.
*   **Differentiable Tripartite Modularity (2026):** For heterogeneous graphs, defining modularity via co-paths avoids dense tensor operations. While `glass-jax` focuses on homogeneous graphs, path-based evaluations could inspire future sparse-matrix implementations.

## 2. Overcoming Mode Collapse: Regularization Strategies

A major challenge in continuous graph clustering is "degenerate solutions" where $S$ collapses (all nodes to one cluster, yielding technically zero cut edges).

*   **Orthogonal Regularization (MinCutPool):**
    Introduced by Bianchi et al. (2020), this regularization term forces the columns of the assignment matrix $S$ to be orthogonal:
    $L_o = \left\| \frac{S^\top S}{\|S^\top S\|_F} - \frac{I_K}{\sqrt{K}} \right\|_F$
    *Idea for glass-jax:* Implement $L_o$ as an optional auxiliary loss in `cut.py` and `modularity.py`. While it can sometimes dominate the main objective, it is highly effective at guaranteeing distinct, balanced clusters.
*   **Entropy Loss (DiffPool):**
    Adding a standard entropy minimization term $H(S)$ forces the soft assignments to become "sharp" (closer to one-hot vectors) during the forward pass, acting as an alternative to our current temperature annealing strategy.

## 3. Hierarchical & Optimal Transport Methods

True Louvain and Infomap algorithms build hierarchies (super-nodes). Differentiable approximations are beginning to explore this:

*   **IsoSEL (2025):** Uses "Differentiable Structural Information" to build a hyperbolic Lorentz tree, mimicking a multi-level hierarchy without requiring a predefined number of clusters $K$.
    *Idea for glass-jax:* Implement a `coarsen_graph(A, S)` utility. Once an initial soft assignment $S$ is found, generate a coarsened adjacency matrix $A_{new} = S^T A S$, and recursively apply the `soft_modularity` objective.
*   **DC-GNN (2024/2025):** Replaces the greedy local moving step of Louvain with **Optimal Transport**.
    *Idea for glass-jax:* We already have a rudimentary `sinkhorn.py`. This literature validates expanding its use. Instead of using Sinkhorn just for the final assignment, we can use it iteratively to update node embeddings and "cluster-node" embeddings simultaneously.

## 4. Alternative Forward-Pass Relaxations

*   **Gumbel-Softmax (Concrete Distribution):**
    Currently, `glass-jax` uses `jax.nn.softmax(logits / temp)`. This means the loss function evaluates a highly blended graph. Using the Gumbel-Softmax trick allows the forward pass to sample hard categorical assignments (acting exactly like a discrete graph partition) while maintaining a differentiable backward pass. This could dramatically improve the accuracy of the Soft Map Equation, which relies heavily on precise transition probabilities.

---

## Actionable Next Steps for `glass-jax`

1.  **Implement Auxiliary Regularization:** Add the $L_o$ (Orthogonal Regularization) and DMoN Collapse Regularization as optional parameters. (**Implemented $L_o$**)
2.  **Structural Entropy Integration:** Transition from Map Equation conflation to pure Structural Entropy (1D/2D). (**Implemented**)
3.  **Differentiable Coarsening:** Write a wrapper function that takes an objective, applies it, coarsens the graph via $S^T A S$, and repeats.

### References
*   Tsitsulin, A., et al. (2020). *Graph Clustering with Graph Neural Networks* (DMoN).
*   Bianchi, F. M., et al. (2020). *Spectral Clustering with Graph Neural Networks for Graph Pooling* (MinCutPool).
*   Ying, Z., et al. (2018). *Hierarchical Graph Representation Learning with Differentiable Pooling* (DiffPool).
*   Recent advances (2024-2025) in DCAT, IsoSEL, and DC-GNN for continuous structural entropy and optimal transport clustering.