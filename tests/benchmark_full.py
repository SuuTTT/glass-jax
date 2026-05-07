import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import matplotlib.pyplot as plt

from glass.objectives.structural_entropy import two_dimensional_structural_entropy
from glass.solvers.spectral import spectral_embedding
def get_dataset(name):
    from torch_geometric.datasets import Planetoid, Amazon, KarateClub
    # Football is not standard in PyG datasets anymore, skipping it.
    from torch_geometric.utils import to_dense_adj, to_undirected
    import torch

    if name == 'Karate':
        dataset = KarateClub()
    elif name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='/tmp/dataset', name=name)
    elif name in ['Photo', 'Computers']:
        dataset = Amazon(root='/tmp/dataset', name=name)
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    data = dataset[0]
    edge_index = to_undirected(data.edge_index)
    adj_dense = to_dense_adj(edge_index)[0].numpy()
    labels = data.y.numpy()
    features = data.x.numpy()
    k = dataset.num_classes
    return adj_dense, features, labels, k

def cluster_accuracy(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

def run_glass_se(adj, k, n_iters=100, lr=0.05, n_starts=2, seed=42):
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
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
    _ = vmap_optimize(all_inits) 
    all_final_logits, all_final_losses = vmap_optimize(all_inits)
    all_final_logits.block_until_ready()
    
    best_logits = all_final_logits[jnp.argmin(all_final_losses)]
    S = jax.nn.softmax(best_logits / 0.01, axis=-1)
    return np.array(jnp.argmax(S, axis=-1))

def benchmark():
    datasets = ['Karate', 'Cora', 'Citeseer', 'Photo']
    results = []
    
    for ds in datasets:
        adj, features, gt_labels, k = get_dataset(ds)
        print(f"Running {ds}...")
        
        # Pure Glass-SE
        labels = run_glass_se(adj, k)
        results.append({
            'Dataset': ds, 'Algo': 'Glass-SE',
            'ACC': cluster_accuracy(gt_labels, labels),
            'NMI': normalized_mutual_info_score(gt_labels, labels),
            'ARI': adjusted_rand_score(gt_labels, labels)
        })
    
    df = pd.DataFrame(results)
    print("\n--- Summary Table ---")
    print(df.to_markdown(index=False))
    
    # Plotting
    df.plot(x='Dataset', kind='bar', figsize=(10, 6))
    plt.title('Clustering Performance')
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nBenchmark visualization saved to benchmark_results.png")

if __name__ == "__main__":
    benchmark()
