import jax.numpy as jnp

def pad_adjacency_matrix(A, max_N):
    """
    Pad adjacency matrix A to (max_N, max_N).
    
    Args:
        A: Adjacency matrix of shape (N, N).
        max_N: Target size.
        
    Returns:
        Padded matrix of shape (max_N, max_N) and mask of shape (max_N,).
    """
    N = A.shape[0]
    padded_A = jnp.zeros((max_N, max_N), dtype=A.dtype)
    padded_A = padded_A.at[:N, :N].set(A)
    
    mask = jnp.zeros(max_N, dtype=jnp.float32)
    mask = mask.at[:N].set(1.0)
    
    return padded_A, mask

def pad_assignment_matrix(S, max_N):
    """Pad soft assignment matrix S to (max_N, K)."""
    N, K = S.shape
    padded_S = jnp.zeros((max_N, K), dtype=S.dtype)
    padded_S = padded_S.at[:N, :].set(S)
    return padded_S
