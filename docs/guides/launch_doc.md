# Launch Document: glass-jax v0.1.0

**Release Date:** May 6, 2026  
**Project Name:** glass-jax (Graph-based Latent Abstraction & Structural Segmentation in JAX)  
**Primary Maintainer:** Gemini CLI  

## 🚀 Overview
We are thrilled to announce the internal release of **`glass-jax` v0.1.0**. 

`glass-jax` is a foundational research library designed to bridge the gap between discrete network science and continuous deep learning. By providing fully differentiable implementations of classic graph clustering algorithms, `glass-jax` allows researchers to embed structural entropy and modularity directly into the loss functions of neural networks.

This release represents the "Software Matrix" required to inject structural graph priors into Reinforcement Learning (RL) world models, specifically targeting the `helios-rl` pipeline.

## ✨ Key Features

### 1. Differentiable Objectives
You can now backpropagate through graph community assignments:
*   **Soft Map Equation (Infomap):** Optimize for structural entropy by minimizing the description length of a random walk over a soft partition.
*   **Soft Modularity (Louvain):** Maximize the density of links inside communities compared to links between communities, utilizing highly optimized JAX `einsum` operations.
*   **Min-Cut & Normalized Cut:** Standard spectral clustering objectives adapted for soft pooling.

### 2. Multi-Start `vmap` Optimization Engine
Graph clustering is notoriously non-convex. To prevent our differentiable objectives from getting stuck in local minima, `glass-jax` features a parallel multi-start optimization loop:
*   Uses `jax.vmap` to evaluate multiple initializations (Spectral + Random) concurrently on the GPU/TPU.
*   Employs Temperature Annealing to gradually harden soft assignments.
*   *Result:* Drastically outperforms standard greedy algorithms (like CPU Louvain) on tricky real-world graphs like Zachary's Karate Club.

### 3. JIT-Friendly Utilities
Designed for the compiler:
*   **Static Graph Topologies:** Includes utilities (`pad_adjacency_matrix`, `apply_mask`) to handle varying graph sizes within a batched training loop without triggering expensive XLA recompilations.

### 4. Verification Bridge
Trust but verify:
*   Provides `jax.pure_callback` wrappers for the official C++ `infomap` and `python-louvain` libraries. This allows you to run exact, discrete baselines midway through a JAX pipeline for debugging parity.

## 📊 Benchmark Highlights
In our internal evaluations against standard CPU algorithms (Louvain/Infomap), `glass-jax` proved highly competitive:
*   **Accuracy:** Achieved a perfect Adjusted Rand Index (ARI = 1.0) on Medium Stochastic Block Models ($N=500$).
*   **Robustness:** Outperformed standard heuristics on the Karate Club network (ARI 0.882 vs Louvain's 0.509).

*(See `docs/experimental_reports/benchmark_sbm_20260506.md` for the full data.)*

## 🛠️ Getting Started
The repository is ready for immediate integration.

**Installation:**
```bash
pip install -e /path/to/glass-jax
```

**Basic Usage:**
```python
import jax
import jax.numpy as jnp
from glass.objectives.map_equation import soft_map_equation

# Given an adjacency matrix A (N, N) and assignment logits (N, K)
loss = soft_map_equation(A, assignment_logits)
grads = jax.grad(soft_map_equation, argnums=1)(A, assignment_logits)
```

## ⏭️ Next Steps
The immediate next step is the integration of `glass-jax` into the **`helios-rl`** repository. We will utilize the `soft_map_equation` to calculate the structural entropy of latent state transition graphs, driving the discovery of temporal abstractions in agent planning.

---
*“Turning the discrete, irregular world of graph theory into something JAX can digest and differentiate.”*