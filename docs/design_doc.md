# Design Document: glass-jax (Graph-based Latent Abstraction & Structural Segmentation in JAX)

**Author:** Gemini CLI  
**Date:** May 6, 2026  
**Status:** Approved / Implemented  

## 1. Objective
The goal of `glass-jax` is to provide a fully differentiable, hardware-accelerated (GPU/TPU) framework for graph clustering. Specifically, it must translate discrete graph topological algorithms—primarily the Louvain method (Modularity) and Infomap (Map Equation/Structural Entropy)—into continuous loss functions. This enables backpropagation through clustering assignments, allowing `glass-jax` to serve as a structural loss component within larger neural architectures (e.g., Reinforcement Learning world models).

## 2. Background & Motivation
In representation learning and reinforcement learning, understanding the latent topology of state spaces is critical for hierarchical planning and abstraction. Traditional graph clustering methods operate discretely, assigning nodes to hard clusters. However, integrating clustering into a neural network training loop requires differentiable operations. `glass-jax` relaxes these discrete assignments into continuous probabilities, permitting the gradient-based discovery of latent communities.

## 3. Architecture
`glass-jax` is structured into four primary modules:

### 3.1 Objectives (`src/glass/objectives/`)
This module contains the core differentiable loss functions.
*   **Soft Modularity (`modularity.py`):** Relaxes the Louvain objective. It computes $Q = \frac{1}{2m} \text{Tr}(S^T B S)$ efficiently using highly optimized matrix operations.
*   **Soft Map Equation (`map_equation.py`):** A differentiable approximation of Infomap. It computes the description length of a random walk (flow-centric).
*   **Structural Entropy (`structural_entropy.py`):** A differentiable implementation of 1D and 2D Structural Entropy, focusing on the uncertainty of graph organization (volume and cuts).
*   **Cuts (`cut.py`):** Standard Min-Cut and Normalized Cut implementations adapted for soft assignments $S$.

### 3.2 Solvers (`src/glass/solvers/`)
Solvers are responsible for producing or initializing the soft assignment matrix $S \in \mathbb{R}^{N \times K}$.
*   **Sinkhorn (`sinkhorn.py`):** Uses Optimal Transport (via `ott-jax`) to enforce marginal constraints on assignments, mitigating the common issue of cluster mode collapse (where all nodes collapse to a single cluster).
*   **Spectral (`spectral.py`):** Computes the top $K$ eigenvectors of the normalized Laplacian. This is crucial for initializing gradient descent, as graph objectives are highly non-convex.
*   **GNN Pool (`gnn_pool.py`):** Provides a mechanism to map node features $X$ to assignments $S$ via Graph Neural Networks (e.g., DiffPool, MinCutPool).

### 3.3 Utilities (`src/glass/utils/`)
To maximize JAX's XLA compiler efficiency, graph topologies must be static. 
*   **Padding & Masking (`padding.py`, `masking.py`):** Pads dynamic adjacency matrices to a fixed maximum size ($N_{max}$) and applies boolean masks to ensure padded nodes do not contribute to the loss or gradients. This prevents XLA recompilation during batched training.

### 3.4 Verification Bridge (`src/glass/bridge/`)
*   **CPU Baselines (`cpu_baselines.py`):** Uses `jax.pure_callback` to wrap the official C++ implementations of Infomap and Louvain. This allows researchers to run exact discrete baselines inside a JAX `jit` compiled function for debugging and parity testing without breaking the compilation graph.

## 4. Optimization Strategy: Multi-Start `vmap`
Because the Map Equation and Modularity landscapes are highly non-convex, random initialization often leads to suboptimal local minima. `glass-jax` mitigates this without severe performance penalties by leveraging JAX's vectorization:
1.  **Parallel Execution:** `jax.vmap` unrolls multiple optimization trajectories simultaneously on the accelerator.
2.  **Hybrid Initialization:** A batch of initializations is provided (e.g., 1 Spectral Embedding + 7 Random Gaussian noises).
3.  **Temperature Annealing:** A softmax temperature is decayed from $\tau=1.0$ to $\tau=0.1$ to gradually sharpen the soft assignments into discrete-like decisions.
4.  **Selection:** The trajectory yielding the best unscaled objective value is selected.

### 4.1 Rejected Design Choice: Gumbel-Softmax
During development, the Gumbel-Softmax straight-through estimator was evaluated as an alternative to Temperature Annealing. The theoretical advantage was that the forward pass would evaluate the objective on hard categorical assignments (mimicking discrete partitioning) while maintaining differentiable backward passes. However, ablation studies revealed that the harshness of the discrete updates destabilized the optimization trajectory, causing the Adam optimizer to frequently escape favorable local minima. The smoother gradient landscape provided by standard temperature annealing proved to be significantly more robust and accurate.

## 5. Security & Safety
*   The library relies exclusively on standard scientific computing packages (`jax`, `numpy`, `networkx`, `scikit-learn`).
*   No external data calls are made; computations execute locally on the provided accelerator.

## 6. Future Work
*   Integration into `helios-rl` to penalize latent state transitions that violate discovered community boundaries.
*   Implementation of hierarchical (multi-level) soft Map Equation for multi-scale abstraction.