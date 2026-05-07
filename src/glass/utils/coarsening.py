import jax
import jax.numpy as jnp

def coarsen_graph(A, S):
    """
    Coarsen graph A using soft assignment S.
    A_new = S^T * A * S
    
    Args:
        A: Adjacency matrix (N, N).
        S: Soft assignment matrix (N, K).
        
    Returns:
        Coarsened adjacency matrix (K, K).
    """
    # S^T A S
    return jnp.dot(jnp.dot(S.T, A), S)

def get_node_volumes(A):
    """Calculate node volumes (degrees)."""
    return jnp.sum(A, axis=-1)

def coarsen_assignments(S, S_new):
    """
    Hierarchical assignment projection.
    If S maps N->K, and S_new maps K->M, 
    the combined mapping is S_combined = S * S_new (N->M).
    """
    return jnp.dot(S, S_new)
