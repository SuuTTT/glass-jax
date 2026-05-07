import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
from community import community_louvain

from tests.benchmark import get_karate_club, eval_structural_entropy, run_glass_jax_multistart
from glass.objectives.structural_entropy import two_dimensional_structural_entropy

def main():
    adj, gt_labels, gt_k = get_karate_club()
    
    # Run Louvain
    G = nx.from_numpy_array(adj)
    partition = community_louvain.best_partition(G, random_state=42)
    louvain_labels = np.array([partition[i] for i in range(len(G.nodes))])
    louvain_k = len(np.unique(louvain_labels))
    
    louvain_se = eval_structural_entropy(adj, louvain_labels)
    
    print(f"Louvain:")
    print(f"  Number of communities (K): {louvain_k}")
    print(f"  Structural Entropy: {louvain_se:.4f}")
    
    # Run Glass-SE with K=2
    labels_k2, _ = run_glass_jax_multistart(adj, 2, two_dimensional_structural_entropy, n_iters=500, seed=42)
    se_k2 = eval_structural_entropy(adj, labels_k2)
    
    print(f"\nGlass-SE (Forced K=2):")
    print(f"  Number of communities (K): {len(np.unique(labels_k2))}")
    print(f"  Structural Entropy: {se_k2:.4f}")
    
    # Run Glass-SE with K=4 (matching Louvain)
    labels_k4, _ = run_glass_jax_multistart(adj, 4, two_dimensional_structural_entropy, n_iters=500, seed=42)
    se_k4 = eval_structural_entropy(adj, labels_k4)
    
    print(f"\nGlass-SE (Forced K=4):")
    print(f"  Number of communities (K): {len(np.unique(labels_k4))}")
    print(f"  Structural Entropy: {se_k4:.4f}")

    # Run Glass-SE with K=8 (Over-parameterized)
    labels_k8, _ = run_glass_jax_multistart(adj, 8, two_dimensional_structural_entropy, n_iters=500, seed=42)
    se_k8 = eval_structural_entropy(adj, labels_k8)
    
    print(f"\nGlass-SE (Forced K=8):")
    print(f"  Number of communities (K): {len(np.unique(labels_k8))}")
    print(f"  Structural Entropy: {se_k8:.4f}")

if __name__ == "__main__":
    main()
