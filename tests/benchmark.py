import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
from community import community_louvain
import infomap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from glass.objectives.modularity import soft_modularity
from glass.objectives.map_equation import soft_map_equation

# --- Datasets ---

def get_sbm(n_nodes, n_communities, p_in, p_out):
    community_size = n_nodes // n_communities
    labels = np.repeat(np.arange(n_communities), community_size)
    if len(labels) < n_nodes:
        labels = np.concatenate([labels, np.full(n_nodes - len(labels), n_communities - 1)])
    G = nx.stochastic_block_model(
        sizes=[community_size] * (n_communities - 1) + [n_nodes - community_size * (n_communities - 1)],
        p=[[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)],
        seed=42
    )
    return nx.to_numpy_array(G), labels, n_communities

def get_karate_club():
    G = nx.karate_club_graph()
    labels = np.array([1 if G.nodes[i]['club'] == 'Officer' else 0 for i in G.nodes])
    return nx.to_numpy_array(G), labels, 2

def get_caveman_graph(l=10, k=20):
    """Connected caveman graph: l cliques of size k."""
    G = nx.connected_caveman_graph(l, k)
    labels = np.repeat(np.arange(l), k)
    return nx.to_numpy_array(G), labels, l

# --- Algorithms ---

def run_louvain(adj):
    start = time.time()
    G = nx.from_numpy_array(adj)
    partition = community_louvain.best_partition(G)
    labels = np.array([partition[i] for i in range(len(partition))])
    duration = time.time() - start
    return labels, duration

def run_infomap(adj):
    start = time.time()
    im = infomap.Infomap("--two-level --silent")
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows, cols):
        im.add_link(int(r), int(c), float(adj[r, c]))
    im.run()
    labels = np.zeros(adj.shape[0], dtype=np.int32)
    for node in im.tree:
        if node.is_leaf:
            labels[node.node_id] = node.module_id - 1
    duration = time.time() - start
    return labels, duration

def run_glass_jax_multistart(adj, n_communities, objective_fn, n_iters=500, lr=0.05, n_starts=8):
    """
    Runs JAX optimization from multiple starting points in parallel using vmap.
    """
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
    
    from glass.objectives.map_equation import compute_stationary_distribution
    pi = None
    if "map_equation" in objective_fn.__name__:
        pi = compute_stationary_distribution(adj_jax)
    
    # 1. Generate Multiple Initializations
    # Start 0: Spectral Embedding
    from glass.solvers.spectral import spectral_embedding
    emb = spectral_embedding(adj_jax, n_communities)
    spectral_init = jnp.array(emb) * 5.0
    
    # Starts 1 to N: Random Noise
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(lambda k: jax.random.normal(k, (n_nodes, n_communities)) * 0.1)(keys)
    
    # Shape: (n_starts, N, K)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)
    
    optimizer = optax.adam(lr)
    
    # 2. Define single optimization trajectory
    def optimize_single(logits_init):
        opt_state = optimizer.init(logits_init)
        
        def step(state, temp):
            logits, opt_state = state
            def loss_fn(l):
                # Apply softmax internally based on temperature
                if pi is not None:
                    val = objective_fn(adj_jax, l / temp, pi=pi)
                else:
                    val = objective_fn(adj_jax, l / temp)
                return val if "map_equation" in objective_fn.__name__ else -val
                
            loss, grads = jax.value_and_grad(loss_fn)(logits)
            updates, opt_state = optimizer.update(grads, opt_state)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_state), loss
            
        temps = jnp.linspace(1.0, 0.1, n_iters)
        
        # Fast unrolled loop using jax.lax.scan
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]
        
        # Calculate final evaluation loss (without temp scaling to ensure fairness)
        final_loss = objective_fn(adj_jax, final_logits, pi=pi) if pi is not None else objective_fn(adj_jax, final_logits)
        final_loss = final_loss if "map_equation" in objective_fn.__name__ else -final_loss
        
        return final_logits, final_loss

    # 3. Vectorize across initializations
    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    
    # Warmup / Compilation
    _ = vmap_optimize(all_inits)
    
    start = time.time()
    # Execute parallel optimization
    all_final_logits, all_final_losses = vmap_optimize(all_inits)
    all_final_logits.block_until_ready()
    duration = time.time() - start
    
    # 4. Select the best trajectory
    best_idx = jnp.argmin(all_final_losses)
    best_logits = all_final_logits[best_idx]
    
    S = jax.nn.softmax(best_logits / 0.1, axis=-1)
    labels = jnp.argmax(S, axis=-1)
    return np.array(labels), duration


def benchmark():
    datasets = {
        "Karate": get_karate_club(),
        "Caveman(10x20)": get_caveman_graph(10, 20),
        "SBM (N=100)": get_sbm(100, 4, 0.4, 0.02),
        "SBM (N=500)": get_sbm(500, 5, 0.2, 0.01),
        "SBM (N=1000)": get_sbm(1000, 10, 0.1, 0.005),
    }
    
    print(f"{'Dataset':<15} | {'Algo':<15} | {'ARI':<6} | {'NMI':<6} | {'Time(s)':<8}")
    print("-" * 65)
    
    for name, (adj, gt_labels, k) in datasets.items():
        # Louvain
        l_labels, l_time = run_louvain(adj)
        print(f"{name:<15} | Louvain         | {adjusted_rand_score(gt_labels, l_labels):.3f} | {normalized_mutual_info_score(gt_labels, l_labels):.3f} | {l_time:.4f}")
        
        # Infomap
        im_labels, im_time = run_infomap(adj)
        print(f"{name:<15} | Infomap         | {adjusted_rand_score(gt_labels, im_labels):.3f} | {normalized_mutual_info_score(gt_labels, im_labels):.3f} | {im_time:.4f}")
        
        # Glass-Jax Modularity
        gj_m_labels, gj_m_time = run_glass_jax_multistart(adj, k, soft_modularity)
        print(f"{name:<15} | Glass-Mod (JAX) | {adjusted_rand_score(gt_labels, gj_m_labels):.3f} | {normalized_mutual_info_score(gt_labels, gj_m_labels):.3f} | {gj_m_time:.4f}")

        # Glass-Jax Map Equation
        gj_e_labels, gj_e_time = run_glass_jax_multistart(adj, k, soft_map_equation)
        print(f"{name:<15} | Glass-Map (JAX) | {adjusted_rand_score(gt_labels, gj_e_labels):.3f} | {normalized_mutual_info_score(gt_labels, gj_e_labels):.3f} | {gj_e_time:.4f}")
        print("-" * 65)

if __name__ == "__main__":
    benchmark()
