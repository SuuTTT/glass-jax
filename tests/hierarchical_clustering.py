import jax
import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
from community import community_louvain
import infomap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from glass.objectives.structural_entropy import two_dimensional_structural_entropy
from glass.solvers.spectral import spectral_embedding
from glass.utils.coarsening import coarsen_graph

# Re-using solvers from benchmarks
from tests.benchmark_realworld import run_glass_se

def run_hierarchical_glass_se(adj, max_depth=2, n_starts=2, seed=42):
    """
    Recursive coarsening to build a hierarchy.
    """
    current_adj = adj
    all_assignments = []
    
    # We coarsen until we reach max_depth
    for depth in range(max_depth):
        n_nodes = current_adj.shape[0]
        # In a real K-agnostic scenario, we'd pick K based on SE optimization.
        # For this experiment, we assume K=4 for simplicity.
        k = min(n_nodes, 4)
        
        # Optimize S for current adj
        # We need logits for this level
        logits = jnp.zeros((n_nodes, k)) 
        
        # Simple optimization (re-use Glass-SE logic internally)
        labels = run_glass_se(current_adj, k, n_iters=100)
        S = np.zeros((n_nodes, k))
        S[np.arange(n_nodes), labels] = 1.0
        S = jnp.array(S)
        
        all_assignments.append(S)
        
        # Coarsen
        current_adj = coarsen_graph(jnp.array(current_adj), S)
        current_adj = np.array(current_adj)
        
    # Project all assignments down to the leaves
    final_S = all_assignments[0]
    for i in range(1, len(all_assignments)):
        final_S = jnp.dot(final_S, all_assignments[i])
        
    labels = np.array(jnp.argmax(final_S, axis=-1))
    return labels

def plot_partition_tree(adj, labels):
    """Simple visualization of the final partition structure."""
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=labels, cmap=plt.cm.jet, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title("Glass-SE Hierarchical Partition")
    plt.savefig("partition_tree.png")
    print("Partition visualization saved to partition_tree.png")

if __name__ == "__main__":
    from tests.benchmark_realworld import get_dataset
    adj, _, gt_labels, _ = get_dataset('Karate')
    
    labels = run_hierarchical_glass_se(adj)
    
    print(f"Hierarchical Glass-SE ARI: {adjusted_rand_score(gt_labels, labels):.3f}")
    plot_partition_tree(adj, labels)
