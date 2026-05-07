# glass-jax
**Graph-based Latent Abstraction & Structural Segmentation in JAX**

`glass-jax` is a research-focused JAX library for differentiable graph clustering and structural entropy optimization. It provides continuous relaxations of popular graph clustering objectives, enabling gradient-based discovery of latent communities.

## Core Features
- **Differentiable Objectives:** 
  - Soft Modularity (Louvain relaxation)
  - Soft Map Equation (Infomap relaxation for structural entropy)
  - Min-Cut and Normalized Cut
- **Assignment Solvers:**
  - Sinkhorn-based assignments (via `ott-jax`) to prevent mode collapse.
  - Spectral embeddings.
  - GNN-based pooling (DiffPool/MinCutPool).
- **JIT-Friendly:** Built with `jax.jit` in mind, using padding and masking for static topologies.
- **Verification Bridge:** `pure_callback` wrappers for official Infomap and Louvain implementations.

## Installation
```bash
pip install -e .
```

## Quick Start
```python
import jax
import jax.numpy as jnp
from glass.objectives.map_equation import soft_map_equation

# Adjacency matrix (N, N)
A = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)

# Soft assignment logits (N, K)
S_logits = jax.random.normal(jax.random.PRNGKey(0), (3, 2))

# Calculate structural entropy (Map Equation)
loss = soft_map_equation(A, S_logits)

# Optimize S_logits using JAX gradients
grads = jax.grad(soft_map_equation, argnums=1)(A, S_logits)
```

## Repository Structure
- `src/glass/objectives/`: Implementation of differentiable loss functions.
- `src/glass/solvers/`: Methods for generating soft assignments.
- `src/glass/utils/`: Padding and masking utilities for static JIT graphs.
- `src/glass/bridge/`: Callbacks to non-differentiable CPU baselines.

## Math Foundations
### Soft Modularity
Maximizes $Q = \frac{1}{2m} \text{Tr}(S^T B S)$, where $B$ is the modularity matrix.

### Soft Map Equation (Infomap)
Approximates the description length $L(M)$ of a random walk over soft community assignments $S$, focusing on the **flow** of information.

### Structural Entropy (SE)
Minimizes the structural uncertainty of the graph organization, focusing on the **static structure** (volume and cuts) of modules.
