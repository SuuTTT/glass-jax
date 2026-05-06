import jax
import jax.numpy as jnp
import jraph

def min_cut_pool(adj, x, W_s):
    """
    MinCutPool assignment matrix calculation.
    S = softmax(GNN(A, X))
    
    Args:
        adj: Adjacency matrix (N, N).
        x: Node features (N, D).
        W_s: Weight matrix for assignment (D, K).
        
    Returns:
        Soft assignment matrix S (N, K).
    """
    # Simple GCN: H = A X W
    # Here we just do the projection for the stub
    s_logits = jnp.dot(jnp.dot(adj, x), W_s)
    return s_logits

def diff_pool(adj, x, W_s):
    """
    DiffPool assignment matrix calculation.
    """
    s_logits = jnp.dot(jnp.dot(adj, x), W_s)
    return s_logits
