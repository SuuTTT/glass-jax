# Technical Deep Dive: Mathematics and Implementation of glass-jax

This document provides a comprehensive, from-scratch explanation of the mathematical theory and technical implementation strategies used in the `glass-jax` library.

---

## 1. Core Philosophy: The Differentiable Relaxation
Traditional graph clustering algorithms (Louvain, Infomap) are **discrete and greedy**. They make hard assignments $c_i \in \{1, \dots, K\}$ for each node. This is incompatible with modern neural networks, which require gradients.

`glass-jax` solves this by relaxing the discrete assignment into a **soft assignment matrix** $S \in \mathbb{R}^{N \times K}$, where $S_{ij}$ represents the probability (or logit) of node $i$ belonging to cluster $j$. By expressing graph objectives as continuous functions of $S$, we can use the **Chain Rule** to backpropagate gradients from the topological objective back to the latent node representations.

---

## 2. Mathematical Foundations

### 2.1 Soft Modularity (Louvain Relaxation)
Modularity measures the density of edges inside communities compared to a null model (random edges).

**Discrete Form:**
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

**Continuous Relaxation in JAX:**
We replace the Kronecker delta $\delta(c_i, c_j)$ with the inner product of soft assignment vectors $S_i$ and $S_j$. Let $B$ be the modularity matrix $B_{ij} = A_{ij} - \frac{k_i k_j}{2m}$. The objective becomes:
$$Q = \frac{1}{2m} \text{Tr}(S^T B S)$$

**Implementation Optimization:**
Directly calculating $B$ is memory-intensive ($O(N^2)$). We optimize this using the linearity of the trace:
$$\text{Tr}(S^T B S) = \text{Tr}(S^T A S) - \frac{1}{2m} \text{Tr}(S^T (kk^T) S)$$
In JAX, we compute this via sequential matrix-vector products to maintain $O(NK)$ or $O(N^2)$ efficiency without dense matrix instantiation:
```python
AS = jnp.dot(A, S)
STS_A = jnp.sum(S * AS)  # Efficient Tr(S^T A S)
ST_k = jnp.dot(S.T, k)
STS_k = jnp.sum(ST_k**2) # Efficient Tr(S^T k k^T S)
```

### 2.2 Soft Map Equation (Infomap Relaxation)
The Map Equation minimizes the description length $L(M)$ of a random walk.

**Discrete Form:**
$$L(M) = q_{\text{out}} H(\mathcal{Q}) + \sum_{i=1}^m p^i_{\text{in}} H(\mathcal{P}^i)$$
where $q_{\text{out}}$ is the probability of switching modules and $H$ is the Shannon entropy.

**Continuous Relaxation:**
We compute the stationary distribution $\pi$ via **Power Iteration**. Module exit probabilities $q_{i, \text{out}}$ are relaxed using the soft assignment $S$:
$$q_{i, \text{out}} = \sum_{\alpha} \pi_\alpha S_{\alpha i} \sum_{\beta} P_{\alpha \beta} (1 - S_{\beta i})$$
This allows the "flow" of the random walk to be differentiated with respect to the cluster boundaries.

### 2.3 Structural Entropy (SE)
Structural Entropy measures the uncertainty of the graph's organization based on its hierarchy.

**2D Structural Entropy ($H^2$):**
$$H^2(G) = -\sum_{i=1}^K \frac{g_i}{2m} \log_2 \frac{V_i}{2m} - \sum_{i=1}^K \sum_{j \in i} \frac{d_j}{2m} \log_2 \frac{d_j}{V_i}$$
*   **$V_i$ (Volume):** Sum of degrees in module $i$. $V = \text{diag}(S^T d)$.
*   **$g_i$ (Cut):** Weight of edges leaving module $i$. $g = \text{sum}(S \odot (d - AS))$.

Unlike the Map Equation (flow-centric), SE is **structure-centric**, focusing on the static density of the partition.

---

## 3. Technical Implementation Details

### 3.1 Multi-Start `vmap` Optimization
Because graph landscapes are non-convex, a single gradient descent run often fails. `glass-jax` implements a hardware-accelerated multi-start engine:
*   **Vectorization:** We use `jax.vmap` to wrap the entire optimization unroll. This executes $N$ different initializations (e.g., 1 Spectral + 7 Random) in parallel on the GPU.
*   **XLA Compilation:** The entire optimization loop is compiled into a single XLA kernel using `jax.lax.scan`, eliminating Python interpreter overhead during the optimization steps.

### 3.2 Temperature Annealing
To reach a near-discrete solution while maintaining differentiability, we apply a **Softmax Temperature** $\tau$:
$$S = \text{softmax}(\frac{\text{logits}}{\tau})$$
We anneal $\tau$ from $1.0$ (highly smooth) down to $0.1$ (sharply peaked) during the optimization process. This allows the model to explore the landscape broadly before "locking in" to a specific community structure.

### 3.3 Spectral Initialization
We mitigate the non-convexity by initializing `logits` using the top $K$ eigenvectors of the **Normalized Laplacian** $L_{sym} = I - D^{-1/2} A D^{-1/2}$. This provides the optimizer with a starting point that already respects the global connectivity of the graph.

### 3.4 Static Topologies (Padding & Masking)
JAX/XLA requires fixed-shape arrays to avoid constant recompilation. To handle graphs of different sizes:
1.  **Padding:** Dynamic adjacency matrices are padded with zeros to the nearest power of 2 ($N_{max}$).
2.  **Masking:** A boolean mask array $M$ tracks which nodes are real. Objectives are modified to multiply by $M$ before summation:
    ```python
    S = S * mask[:, None]
    A = A * mask[:, None] * mask[None, :]
    ```

---

## 4. Summary of Computational Complexity

| Operation | Complexity | Note |
| :--- | :--- | :--- |
| **Spectral Init** | $O(N^2 K)$ | SVD/Eigen-decomposition |
| **Soft Modularity** | $O(N^2)$ | Optimized matrix-vector products |
| **Map Equation** | $O(T N^2)$ | Includes $T$ power iterations for $\pi$ |
| **vmap Multi-start**| $O(S \cdot \text{Cost})$ | $S$ starts run in parallel on accelerator |

---
*Developed by Gemini CLI for high-performance topological research.*
