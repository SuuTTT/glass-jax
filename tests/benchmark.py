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
from glass.objectives.structural_entropy import two_dimensional_structural_entropy

# --- Metrics Evaluators ---
def eval_modularity(adj, labels):
    n_nodes = adj.shape[0]
    n_communities = max(np.max(labels) + 1, 1)
    S = np.zeros((n_nodes, n_communities))
    S[np.arange(n_nodes), labels] = 1.0
    return float(soft_modularity(jnp.array(adj), jnp.array(S), is_logits=False))

def eval_map_equation(adj, labels):
    n_nodes = adj.shape[0]
    n_communities = max(np.max(labels) + 1, 1)
    S = np.zeros((n_nodes, n_communities))
    S[np.arange(n_nodes), labels] = 1.0
    return float(soft_map_equation(jnp.array(adj), jnp.array(S), is_logits=False))

def eval_structural_entropy(adj, labels):
    n_nodes = adj.shape[0]
    n_communities = max(np.max(labels) + 1, 1)
    S = np.zeros((n_nodes, n_communities))
    S[np.arange(n_nodes), labels] = 1.0
    return float(two_dimensional_structural_entropy(jnp.array(adj), jnp.array(S), is_logits=False))

# --- Datasets ---

def get_sbm(n_nodes, n_communities, p_in, p_out, seed):
    community_size = n_nodes // n_communities
    labels = np.repeat(np.arange(n_communities), community_size)
    if len(labels) < n_nodes:
        labels = np.concatenate([labels, np.full(n_nodes - len(labels), n_communities - 1)])
    G = nx.stochastic_block_model(
        sizes=[community_size] * (n_communities - 1) + [n_nodes - community_size * (n_communities - 1)],
        p=[[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)],
        seed=seed
    )
    return nx.to_numpy_array(G), labels, n_communities

def get_karate_club():
    G = nx.karate_club_graph()
    labels = np.array([1 if G.nodes[i]['club'] == 'Officer' else 0 for i in G.nodes])
    return nx.to_numpy_array(G), labels, 2

def get_caveman_graph(l=10, k=10):
    G = nx.connected_caveman_graph(l, k)
    labels = np.repeat(np.arange(l), k)
    return nx.to_numpy_array(G), labels, l

# --- Algorithms ---

def run_louvain(adj, seed=42):
    start = time.time()
    G = nx.from_numpy_array(adj)
    partition = community_louvain.best_partition(G, random_state=seed)
    labels = np.array([partition[i] for i in range(len(G.nodes))])
    duration = time.time() - start
    return labels, duration

def run_infomap(adj, seed=42):
    start = time.time()
    im = infomap.Infomap(f"--two-level --silent --seed {seed}")
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

import math
import heapq

class PartitionTreeNode:
    def __init__(self, ID, partition, vol, g):
        self.ID = ID
        self.partition = partition
        self.vol = vol
        self.g = g
        self.merged = False

def CombineDelta(node1, node2, cut_v, g_vol):
    v1, v2 = node1.vol, node2.vol
    g1, g2 = node1.g, node2.g
    v12 = v1 + v2

    term1 = (v1 - g1) * math.log2(v12 / v1) if v1 > 0 and v12 > 0 else 0
    term2 = (v2 - g2) * math.log2(v12 / v2) if v2 > 0 and v12 > 0 else 0
    term3 = -2 * cut_v * math.log2(g_vol / v12) if v12 > 0 and g_vol > 0 else 0

    return (term1 + term2 + term3) / g_vol

def run_sep(adj, k, seed=42):
    start = time.time()
    num_nodes = adj.shape[0]
    g_vol = np.sum(adj)
    node_vol = np.sum(adj, axis=1)

    nodes_dict = {}
    adj_table = {}
    for i in range(num_nodes):
        nodes_dict[i] = PartitionTreeNode(ID=i, partition=[i], vol=node_vol[i], g=node_vol[i])
        adj_table[i] = set(np.where(adj[i] > 0)[0])

    min_heap = []
    for i in range(num_nodes):
        for j in adj_table[i]:
            if j > i:
                cut_v = adj[i, j]
                diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                heapq.heappush(min_heap, (diff, i, j, cut_v))

    id_counter = num_nodes
    unmerged_count = num_nodes
    while unmerged_count > k and min_heap:
        diff, id1, id2, cut_v = heapq.heappop(min_heap)

        if nodes_dict[id1].merged or nodes_dict[id2].merged:
            continue

        nodes_dict[id1].merged = nodes_dict[id2].merged = True
        new_id = id_counter
        id_counter += 1

        n1, n2 = nodes_dict[id1], nodes_dict[id2]
        new_partition = n1.partition + n2.partition
        new_vol = n1.vol + n2.vol
        new_g = n1.g + n2.g - 2 * cut_v

        new_node = PartitionTreeNode(ID=new_id, partition=new_partition, g=new_g, vol=new_vol)
        nodes_dict[new_id] = new_node
        unmerged_count -= 1

        adj_table[new_id] = adj_table[id1].union(adj_table[id2])
        adj_table[new_id].discard(id1)
        adj_table[new_id].discard(id2)

        for neighbor_id in adj_table[new_id]:
            if not nodes_dict[neighbor_id].merged:
                p1 = np.array(new_node.partition)
                p2 = np.array(nodes_dict[neighbor_id].partition)
                new_cut_v = np.sum(adj[np.ix_(p1, p2)])

                new_diff = CombineDelta(nodes_dict[neighbor_id], new_node, new_cut_v, g_vol)
                heapq.heappush(min_heap, (new_diff, min(neighbor_id, new_id), max(neighbor_id, new_id), new_cut_v))
                adj_table[neighbor_id].add(new_id)

    labels = np.zeros(num_nodes, dtype=int)
    cluster_idx = 0
    for _, node in nodes_dict.items():
        if not node.merged:
            for n in node.partition:
                labels[n] = cluster_idx
            cluster_idx += 1

    duration = time.time() - start
    return labels, duration

def run_lsenet_baseline(adj, n_communities, n_iters=150, lr=0.05, n_starts=4, seed=42):
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
    features = adj_jax

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_starts)
    W_inits = jax.vmap(lambda k: jax.random.normal(k, (n_nodes, n_communities)) * 0.1)(keys)

    optimizer = optax.adam(lr)

    def optimize_single(W_init):
        opt_state = optimizer.init(W_init)

        def step(state, temp):
            W, opt_state = state
            def loss_fn(w):
                logits = jnp.dot(features, w)
                S = jax.nn.softmax(logits / temp, axis=-1)
                val = two_dimensional_structural_entropy(adj_jax, S, is_logits=False)
                return val

            (loss, grads) = jax.value_and_grad(loss_fn)(W)
            updates, opt_state = optimizer.update(grads, opt_state)
            W = optax.apply_updates(W, updates)
            return (W, opt_state), loss

        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (W_init, opt_state), temps)
        final_W = final_state[0]

        logits = jnp.dot(features, final_W)
        S_eval = jax.nn.softmax(logits / 0.1, axis=-1)
        eval_loss = two_dimensional_structural_entropy(adj_jax, S_eval, is_logits=False)
        return final_W, eval_loss

    vmap_optimize = jax.jit(jax.vmap(optimize_single))

    # Warmup
    _ = vmap_optimize(W_inits)

    start = time.time()
    all_final_W, all_final_losses = vmap_optimize(W_inits)
    all_final_W.block_until_ready()
    duration = time.time() - start

    best_idx = jnp.argmin(all_final_losses)
    best_W = all_final_W[best_idx]

    logits = jnp.dot(features, best_W)
    S = jax.nn.softmax(logits / 0.1, axis=-1)
    labels = jnp.argmax(S, axis=-1)
    return np.array(labels), duration

def run_glass_jax_multistart(adj, n_communities, objective_fn, n_iters=500, lr=0.05, n_starts=4, seed=42):
    n_nodes = adj.shape[0]
    adj_jax = jnp.array(adj)
    
    from glass.objectives.map_equation import compute_stationary_distribution
    pi = None
    if "map_equation" in objective_fn.__name__:
        pi = compute_stationary_distribution(adj_jax)
    
    from glass.solvers.spectral import spectral_embedding
    emb = spectral_embedding(adj_jax, n_communities)
    spectral_init = jnp.array(emb) * 5.0
    
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(lambda k: jax.random.normal(k, (n_nodes, n_communities)) * 0.1)(keys)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)
    
    optimizer = optax.adam(lr)
    
    def optimize_single(logits_init):
        opt_state = optimizer.init(logits_init)
        
        def step(state, temp):
            logits, opt_state = state
            def loss_fn(l):
                S = jax.nn.softmax(l / temp, axis=-1)
                
                if pi is not None:
                    val = objective_fn(adj_jax, S, pi=pi, is_logits=False)
                else:
                    val = objective_fn(adj_jax, S, is_logits=False)
                
                if "map_equation" in objective_fn.__name__ or "structural_entropy" in objective_fn.__name__:
                    return val
                else:
                    return -val # Maximize modularity
                
            (loss, grads) = jax.value_and_grad(loss_fn)(logits)
            updates, opt_state = optimizer.update(grads, opt_state)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_state), loss
            
        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]
        
        # Eval loss
        S_eval = jax.nn.softmax(final_logits / 0.1, axis=-1)
        if pi is not None:
            eval_loss = objective_fn(adj_jax, S_eval, pi=pi, is_logits=False)
        else:
            eval_loss = objective_fn(adj_jax, S_eval, is_logits=False)
        eval_loss = eval_loss if ("map_equation" in objective_fn.__name__ or "structural_entropy" in objective_fn.__name__) else -eval_loss
        
        return final_logits, eval_loss

    vmap_optimize = jax.jit(jax.vmap(optimize_single))
    
    # Warmup
    _ = vmap_optimize(all_inits)
    
    start = time.time()
    all_final_logits, all_final_losses = vmap_optimize(all_inits)
    all_final_logits.block_until_ready()
    duration = time.time() - start
    
    best_idx = jnp.argmin(all_final_losses)
    best_logits = all_final_logits[best_idx]
    
    S = jax.nn.softmax(best_logits / 0.01, axis=-1)
    labels = jnp.argmax(S, axis=-1)
    return np.array(labels), duration


def format_ci(data):
    mean = np.mean(data)
    std = np.std(data)
    ci = 1.96 * std / np.sqrt(len(data))
    return f"{mean:.3f}±{ci:.3f}"

def benchmark():
    seeds = [42, 100, 2026, 777, 999]
    
    datasets_fns = [
        ("Karate", lambda seed: get_karate_club()), # deterministic
        ("Caveman(10x10)", lambda seed: get_caveman_graph(10, 10)), # deterministic
        ("SBM (Clean)", lambda seed: get_sbm(150, 3, 0.3, 0.05, seed)),
        ("SBM (Noisy)", lambda seed: get_sbm(150, 3, 0.15, 0.08, seed)),
    ]
    
    algorithms = [
        ("Louvain", lambda adj, k, seed: run_louvain(adj, seed)),
        ("Infomap", lambda adj, k, seed: run_infomap(adj, seed)),
        ("SEP", lambda adj, k, seed: run_sep(adj, k, seed)),
        ("LSEnet", lambda adj, k, seed: run_lsenet_baseline(adj, k, seed=seed)),
        ("Glass-Mod", lambda adj, k, seed: run_glass_jax_multistart(adj, k, soft_modularity, seed=seed)),
        ("Glass-Map", lambda adj, k, seed: run_glass_jax_multistart(adj, k, soft_map_equation, seed=seed)),
        ("Glass-SE", lambda adj, k, seed: run_glass_jax_multistart(adj, k, two_dimensional_structural_entropy, seed=seed))
    ]
    
    print(f"{'Dataset':<15} | {'Algo':<12} | {'ARI':<12} | {'NMI':<12} | {'Modularity':<12} | {'Map Eq':<12} | {'Struct Ent':<12} | {'Time(s)':<12}")
    print("-" * 115)
    
    for ds_name, ds_fn in datasets_fns:
        for algo_name, algo_fn in algorithms:
            metrics = {'ari': [], 'nmi': [], 'mod': [], 'map': [], 'se': [], 'time': []}
            
            for seed in seeds:
                adj, gt_labels, k = ds_fn(seed)
                # Remap gt_labels to contiguous integers
                _, gt_labels = np.unique(gt_labels, return_inverse=True)
                
                try:
                    labels, duration = algo_fn(adj, k, seed)
                    
                    # Ensure labels don't exceed adj shape or something weird
                    _, labels = np.unique(labels, return_inverse=True)
                    
                    ari = adjusted_rand_score(gt_labels, labels)
                    nmi = normalized_mutual_info_score(gt_labels, labels)
                    
                    mod = eval_modularity(adj, labels)
                    mapeq = eval_map_equation(adj, labels)
                    se = eval_structural_entropy(adj, labels)
                    
                    metrics['ari'].append(ari)
                    metrics['nmi'].append(nmi)
                    metrics['mod'].append(mod)
                    metrics['map'].append(mapeq)
                    metrics['se'].append(se)
                    metrics['time'].append(duration)
                except Exception as e:
                    print(f"Error on {ds_name} with {algo_name}: {e}")
                    pass
                
            if len(metrics['ari']) > 0:
                print(f"{ds_name:<15} | {algo_name:<12} | {format_ci(metrics['ari']):<12} | {format_ci(metrics['nmi']):<12} | {format_ci(metrics['mod']):<12} | {format_ci(metrics['map']):<12} | {format_ci(metrics['se']):<12} | {format_ci(metrics['time']):<12}", flush=True)
        print("-" * 115, flush=True)

if __name__ == "__main__":
    benchmark()
