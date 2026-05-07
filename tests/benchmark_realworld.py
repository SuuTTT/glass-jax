import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
from community import community_louvain
import infomap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment

from glass.objectives.structural_entropy import two_dimensional_structural_entropy

def get_pyg_dataset(name):
    from torch_geometric.datasets import Planetoid, Amazon
    from torch_geometric.utils import to_dense_adj, to_undirected
    import torch
    
    if name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='/tmp/dataset', name=name)
    elif name in ['Photo', 'Computers']:
        dataset = Amazon(root='/tmp/dataset', name=name)
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    data = dataset[0]
    edge_index = to_undirected(data.edge_index)
    
    # For large graphs, dense adj might OOM or take too long to compile.
    # We return sparse format for discrete, and dense for JAX (with a warning).
    adj_dense = to_dense_adj(edge_index)[0].numpy()
    labels = data.y.numpy()
    features = data.x.numpy()
    k = dataset.num_classes
    return adj_dense, features, labels, k

def cluster_accuracy(y_true, y_pred):
    """Calculate clustering accuracy."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

def run_louvain(adj, seed=42):
    start = time.time()
    G = nx.from_numpy_array(adj)
    partition = community_louvain.best_partition(G, random_state=seed)
    labels = np.array([partition[i] for i in range(len(G.nodes))])
    duration = time.time() - start
    return labels, duration

def run_glass_se(adj, k, n_iters=100, lr=0.05, n_starts=2, seed=42):
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
    
    from glass.solvers.spectral import spectral_embedding
    emb = spectral_embedding(adj_jax, k)
    spectral_init = jnp.array(emb) * 5.0
    
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(lambda rng: jax.random.normal(rng, (n_nodes, k)) * 0.1)(keys)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)
    
    optimizer = optax.adam(lr)
    
    def optimize_single(logits_init):
        opt_state = optimizer.init(logits_init)
        def step(state, temp):
            logits, opt_state = state
            def loss_fn(l):
                S = jax.nn.softmax(l / temp, axis=-1)
                return two_dimensional_structural_entropy(adj_jax, S, is_logits=False)
            (loss, grads) = jax.value_and_grad(loss_fn)(logits)
            updates, opt_state = optimizer.update(grads, opt_state)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_state), loss
        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]
        S_eval = jax.nn.softmax(final_logits / 0.01, axis=-1)
        eval_loss = two_dimensional_structural_entropy(adj_jax, S_eval, is_logits=False)
        return final_logits, eval_loss

    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    _ = vmap_optimize(all_inits) # Warmup
    
    start = time.time()
    all_final_logits, all_final_losses = vmap_optimize(all_inits)
    all_final_logits.block_until_ready()
    duration = time.time() - start
    
    best_idx = jnp.argmin(all_final_losses)
    best_logits = all_final_logits[best_idx]
    S = jax.nn.softmax(best_logits / 0.01, axis=-1)
    labels = jnp.argmax(S, axis=-1)
    return np.array(labels), duration

def run_lsenet_proxy(adj, features, k, n_iters=100, lr=0.05, n_starts=2, seed=42):
    n_nodes = adj.shape[0]
    n_features = features.shape[1]
    adj_jax = jnp.array(adj)
    features_jax = jnp.array(features)
    
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_starts)
    W_inits = jax.vmap(lambda rng: jax.random.normal(rng, (n_features, k)) * 0.1)(keys)
    
    optimizer = optax.adam(lr)
    
    def optimize_single(W_init):
        opt_state = optimizer.init(W_init)
        def step(state, temp):
            W, opt_state = state
            def loss_fn(w):
                # LSEnet proxy uses features
                logits = jnp.dot(features_jax, w)
                S = jax.nn.softmax(logits / temp, axis=-1)
                # DSI proxy (using full H2 for comparison fairness)
                return two_dimensional_structural_entropy(adj_jax, S, is_logits=False)
            (loss, grads) = jax.value_and_grad(loss_fn)(W)
            updates, opt_state = optimizer.update(grads, opt_state)
            W = optax.apply_updates(W, updates)
            return (W, opt_state), loss
        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (W_init, opt_state), temps)
        final_W = final_state[0]
        logits = jnp.dot(features_jax, final_W)
        S_eval = jax.nn.softmax(logits / 0.01, axis=-1)
        eval_loss = two_dimensional_structural_entropy(adj_jax, S_eval, is_logits=False)
        return final_W, eval_loss

    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    _ = vmap_optimize(W_inits)
    
    start = time.time()
    all_final_W, all_final_losses = vmap_optimize(W_inits)
    all_final_W.block_until_ready()
    duration = time.time() - start
    
    best_idx = jnp.argmin(all_final_losses)
    best_W = all_final_W[best_idx]
    logits = jnp.dot(features_jax, best_W)
    S = jax.nn.softmax(logits / 0.01, axis=-1)
    labels = jnp.argmax(S, axis=-1)
    return np.array(labels), duration


def main():
    # To keep it runnable, we test Cora and Citeseer.
    datasets = ['Cora', 'Citeseer']
    
    print(f"{'Dataset':<10} | {'Algo':<15} | {'ACC':<6} | {'NMI':<6} | {'ARI':<6}")
    print("-" * 55)
    
    for ds_name in datasets:
        adj, features, gt_labels, k = get_pyg_dataset(ds_name)
        
        # Louvain
        l_labels, _ = run_louvain(adj)
        print(f"{ds_name:<10} | Louvain         | {cluster_accuracy(gt_labels, l_labels):.3f} | {normalized_mutual_info_score(gt_labels, l_labels):.3f} | {adjusted_rand_score(gt_labels, l_labels):.3f}")
        
        # LSEnet proxy
        ls_labels, _ = run_lsenet_proxy(adj, features, k)
        print(f"{ds_name:<10} | LSEnet (Proxy)  | {cluster_accuracy(gt_labels, ls_labels):.3f} | {normalized_mutual_info_score(gt_labels, ls_labels):.3f} | {adjusted_rand_score(gt_labels, ls_labels):.3f}")
        
        # Glass-SE
        g_labels, _ = run_glass_se(adj, k)
        print(f"{ds_name:<10} | Glass-SE (Pure) | {cluster_accuracy(gt_labels, g_labels):.3f} | {normalized_mutual_info_score(gt_labels, g_labels):.3f} | {adjusted_rand_score(gt_labels, g_labels):.3f}")
        print("-" * 55)

if __name__ == "__main__":
    main()
