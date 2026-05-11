# Experimental Report: Pure Structural Entropy (SE) vs. Flow-Based Map Equation

**Date:** May 7, 2026  
**Project:** glass-jax (Corrected Structural Entropy Implementation)

## 1. Abstract
Following a review of information-theoretic clustering literature, we identified a previous conflation between the Map Equation (Infomap) and Structural Entropy (SE). This report documents the implementation of the mathematically correct, differentiable 2D Structural Entropy objective and evaluates its performance against flow-based (Map Equation) and density-based (Modularity) objectives.

## 2. Methodology
### 2.1 Mathematical Implementation
We implemented the 2D Structural Entropy ($H^2$) defined by volume and cuts:
$$H^2(G) = -\sum_{i=1}^k \frac{g_i}{2m} \log_2 \frac{V_i}{2m} + H_1(G) + \sum_{i=1}^k \frac{V_i}{2m} \log_2 \frac{V_i}{2m}$$
where $g_i$ is the cut and $V_i$ is the volume of module $i$. The implementation is fully differentiable using soft assignment matrices $S$.

### 2.2 Benchmark Setup
Algorithms were evaluated on:
- **Karate Club:** Real-world social network.
- **Connected Caveman:** Highly structured cliques.
- **Stochastic Block Models (SBM):** Medium-scale synthetic graphs.

Optimization utilized the **Multi-Start `vmap` Engine** (8 parallel trajectories, 500 Adam steps).

## 3. Results (ARI Comparison)

| Dataset | Louvain | Infomap | Glass-Mod | Glass-Map | **Glass-SE (Corrected)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Karate** | 0.509 | 0.684 | 0.882 | 0.403 | **0.882** |
| **Caveman** | 1.000 | 1.000 | 0.804 | 0.728 | **0.894** |
| **SBM (N=100)** | 1.000 | 1.000 | 0.973 | 0.708 | **1.000** |
| **SBM (N=500)** | 1.000 | 1.000 | 1.000 | 0.888 | **1.000** |

## 4. Discussion
### 4.1 Superior Structural Fidelity
The corrected **Glass-SE** objective proved to be the most robust among the differentiable methods. Notably, it achieved a perfect ARI (1.000) on both the $N=100$ and $N=500$ SBM benchmarks, outperforming the Soft Modularity objective on smaller scales. 

### 4.2 Structural Entropy vs. Map Equation
As hypothesized in the literature, the Map Equation (flow-centric) struggled on smaller topologies like the Karate Club (ARI 0.403), whereas Structural Entropy (structure-centric) excelled (ARI 0.882). This confirms that for static graph clustering, minimizing structural uncertainty (SE) is often more reliable than optimizing random-walk description length (Map Equation).

## 5. Conclusion
The implementation of Pure Structural Entropy provides `glass-jax` with its strongest differentiable clustering engine to date. It successfully bridges the gap between discrete graph partitioning and continuous neural optimization with high empirical fidelity.
