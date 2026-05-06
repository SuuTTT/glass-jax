import jax
import jax.numpy as jnp
import numpy as np

def run_infomap_callback(adj):
    """
    CPU-based Infomap baseline using jax.pure_callback.
    
    Args:
        adj: Adjacency matrix (N, N) as a JAX array.
        
    Returns:
        Cluster assignments (N,) as a JAX array.
    """
    def _infomap(adj_np):
        import infomap
        im = infomap.Infomap("--two-level --silent")
        
        # Add links
        rows, cols = np.where(adj_np > 0)
        for r, c in zip(rows, cols):
            im.add_link(int(r), int(c), float(adj_np[r, c]))
            
        im.run()
        
        # Get modules
        modules = np.zeros(adj_np.shape[0], dtype=np.int32)
        for node in im.tree:
            if node.is_leaf:
                modules[node.node_id] = node.module_id - 1 # 0-indexed
        return modules

    # Define result shape and dtype
    result_shape = jax.ShapeDtypeStruct((adj.shape[0],), jnp.int32)
    
    return jax.pure_callback(_infomap, result_shape, adj)

def run_louvain_callback(adj):
    """CPU-based Louvain baseline."""
    def _louvain(adj_np):
        import networkx as nx
        from community import community_louvain
        G = nx.from_numpy_array(adj_np)
        partition = community_louvain.best_partition(G)
        modules = np.array([partition[i] for i in range(len(partition))], dtype=np.int32)
        return modules

    result_shape = jax.ShapeDtypeStruct((adj.shape[0],), jnp.int32)
    return jax.pure_callback(_louvain, result_shape, adj)
