import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from glass.objectives.modularity import soft_modularity
from glass.objectives.map_equation import soft_map_equation

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

def orthogonal_regularization(S):
    """MinCutPool style orthogonal regularization."""
    K = S.shape[1]
    StS = jnp.dot(S.T, S)
    norm_StS = jnp.linalg.norm(StS, ord='fro')
    I_K = jnp.eye(K)
    Lo = jnp.linalg.norm(StS / (norm_StS + 1e-8) - I_K / jnp.sqrt(K), ord='fro')
    return Lo

def run_ablation(adj, n_communities, objective_fn, n_iters=200, lr=0.05, n_starts=2, 
                 use_ortho=False, ortho_weight=1.0):
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
    
    from glass.objectives.map_equation import compute_stationary_distribution
    pi = None
    if "map_equation" in objective_fn.__name__:
        pi = compute_stationary_distribution(adj_jax)
        
    from glass.solvers.spectral import spectral_embedding
    emb = spectral_embedding(adj_jax, n_communities)
    spectral_init = jnp.array(emb) * 5.0
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(lambda k: jax.random.normal(k, (n_nodes, n_communities)) * 0.1)(keys)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)
    
    optimizer = optax.adam(lr)
    
    def optimize_single(logits_init, run_key):
        opt_state = optimizer.init(logits_init)
        
        def step(state, inputs):
            logits, opt_state = state
            temp, step_key = inputs
            
            def loss_fn(l):
                # Standard Temperature-Annealed Softmax (Baseline)
                S = jax.nn.softmax(l / temp, axis=-1)
                
                if pi is not None:
                    obj_val = objective_fn(adj_jax, S, pi=pi, is_logits=False)
                else:
                    obj_val = objective_fn(adj_jax, S, is_logits=False)
                
                loss = obj_val if "map_equation" in objective_fn.__name__ else -obj_val
                
                if use_ortho:
                    loss += ortho_weight * orthogonal_regularization(S)
                    
                return loss
                
            loss, grads = jax.value_and_grad(loss_fn)(logits)
            updates, opt_state = optimizer.update(grads, opt_state)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_state), loss
            
        temps = jnp.linspace(1.0, 0.1, n_iters)
        step_keys = jax.random.split(run_key, n_iters)
        
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), (temps, step_keys))
        final_logits = final_state[0]
        
        # Eval loss
        S_eval = jax.nn.softmax(final_logits / 0.1, axis=-1)
        eval_loss = objective_fn(adj_jax, S_eval, pi=pi, is_logits=False) if pi is not None else objective_fn(adj_jax, S_eval, is_logits=False)
        eval_loss = eval_loss if "map_equation" in objective_fn.__name__ else -eval_loss
        
        return final_logits, eval_loss
        
    run_keys = jax.random.split(key, n_starts)
    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    
    # Warmup
    _ = vmap_optimize(all_inits, run_keys)
    
    start = time.time()
    all_final_logits, all_final_losses = vmap_optimize(all_inits, run_keys)
    all_final_logits.block_until_ready()
    duration = time.time() - start
    
    best_idx = jnp.argmin(all_final_losses)
    best_logits = all_final_logits[best_idx]
    
    S = jax.nn.softmax(best_logits / 0.1, axis=-1)
    labels = jnp.argmax(S, axis=-1)
    return np.array(labels), duration

def main():
    # Use a challenging graph to see the effect of regularization
    adj, gt_labels, k = get_sbm(150, 3, 0.2, 0.05) 
    
    configs = [
        ("Baseline (Default)", False),
        ("+ Orthogonal Reg", True),
    ]
    
    print(f"Graph: SBM (N=150, K=3, P_in=0.2, P_out=0.05)")
    print(f"{'Config':<20} | {'Objective':<10} | {'ARI':<6} | {'NMI':<6} | {'Time(s)':<8}")
    print("-" * 62)
    for name, ortho in configs:
        for obj in [soft_modularity, soft_map_equation]:
            obj_name = "Modularity" if "modularity" in obj.__name__ else "Map Eq"
            labels, duration = run_ablation(adj, k, obj, use_ortho=ortho)
            ari = adjusted_rand_score(gt_labels, labels)
            nmi = normalized_mutual_info_score(gt_labels, labels)
            print(f"{name:<20} | {obj_name:<10} | {ari:.3f} | {nmi:.3f} | {duration:.4f}")

if __name__ == "__main__":
    main()
