# Ablation Study: Enhancing Differentiable Clustering with Regularization and Gumbel-Softmax

**Date:** May 6, 2026  
**Project:** glass-jax (Graph-based Latent Abstraction & Structural Segmentation in JAX)

## 1. Abstract
Following the implementation of advanced graph clustering ideas derived from our literature review, we conducted an ablation study to evaluate the impact of **Orthogonal Regularization** (inspired by MinCutPool) and the **Gumbel-Softmax** trick (for hard categorical sampling in the forward pass) on the performance of `glass-jax`'s differentiable objectives.

## 2. Experimental Setup
The study utilized a synthetic **Stochastic Block Model (SBM)** with the following parameters:
- **Nodes ($N$):** 150
- **Communities ($K$):** 3
- **Edge Probabilities:** $P_{in} = 0.2$ (intra-cluster), $P_{out} = 0.05$ (inter-cluster)

To prevent compiler timeouts and test the raw algorithmic impact, the parallel multi-start optimization (`vmap`) was configured to 2 parallel trajectories (1 spectral, 1 random) over 200 Adam optimization steps with a learning rate of $0.05$.

The ablation tracked four configurations across both the Soft Modularity and Soft Map Equation objectives:
1.  **Baseline:** Standard `glass-jax` with Temperature-Annealed Softmax.
2.  **+ Orthogonal Reg:** Addition of the MinCutPool orthogonal penalty: $L_o = \left\| \frac{S^\top S}{\|S^\top S\|_F} - \frac{I_K}{\sqrt{K}} \right\|_F$ with a weight of 1.0.
3.  **+ Gumbel Softmax:** Replacing standard Softmax with the Gumbel-Softmax straight-through estimator to evaluate the loss on hard, discrete partitions.
4.  **+ Ortho + Gumbel:** A combination of both techniques.

## 3. Results

| Configuration | Objective | ARI | NMI | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Modularity | **0.863** | **0.829** | 0.0273 |
| | Map Eq | 0.157 | 0.296 | 0.0694 |
| **+ Orthogonal Reg** | Modularity | 0.000 | 0.000 | 0.0299 |
| | Map Eq | **0.228** | **0.323** | 0.0619 |
| **+ Gumbel Softmax** | Modularity | 0.642 | 0.635 | 0.1079 |
| | Map Eq | 0.075 | 0.140 | 0.0994 |
| **+ Ortho + Gumbel**| Modularity | 0.000 | 0.000 | 0.1411 |
| | Map Eq | 0.000 | 0.000 | 0.1005 |

## 4. Discussion

### 4.1 The Dominance of Orthogonal Regularization
As predicted in recent literature (e.g., Tsitsulin et al., 2023), strict Orthogonal Regularization can be dangerous. For the **Modularity** objective, adding the orthogonal penalty completely destroyed performance (ARI dropped from 0.863 to 0.000), causing the network to collapse. The orthogonality term heavily dominated the objective, forcing assignments into mathematically orthogonal but structurally incorrect configurations. 

Conversely, for the **Map Equation**, which struggles significantly with mode collapse and local minima on smaller/noisier graphs, the Orthogonal Regularization provided a modest but notable performance boost (ARI increased from 0.157 to 0.228). This suggests that while dangerous for naturally robust objectives like Modularity, it acts as a necessary anchor for highly non-convex entropy objectives.

### 4.2 Gumbel-Softmax Instability
Using the Gumbel-Softmax straight-through estimator degraded performance across the board. The harshness of the discrete updates in the forward pass likely caused the Adam optimizer to bounce out of favorable local minima, especially given the shortened 200-step training schedule. The standard temperature annealing approach used in the baseline provides a much smoother, more stable gradient trajectory.

### 4.3 Conclusion
The standard `glass-jax` baseline (Temperature-Annealed Softmax) remains the most robust and accurate approach for Modularity optimization. However, the ablation study confirms that implementing **Orthogonal Regularization as a hyperparameter** is a valuable addition to the library, particularly as a stabilization mechanism for the Map Equation on difficult or highly noisy graphs. Future implementations should treat the orthogonal weight as a tunable parameter ($\lambda < 1.0$) rather than a static constraint.
