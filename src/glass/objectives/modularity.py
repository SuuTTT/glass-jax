import jax
import jax.numpy as jnp

def soft_modularity(A, S, mask=None):
    """
    Calculate the soft modularity Q for a given adjacency matrix A and soft assignment S.
    
    Args:
        A: Adjacency matrix of shape (N, N).
        S: Soft assignment matrix of shape (N, K).
        mask: Optional mask of shape (N,) for real/padded nodes.
        
    Returns:
        Scalar modularity value Q.
    """
    # Apply softmax to ensure rows sum to 1
    S = jax.nn.softmax(S, axis=-1)
    
    if mask is not None:
        S = S * mask[:, None]
        A = A * mask[:, None] * mask[None, :]
        
    # Degrees
    k = jnp.sum(A, axis=-1)
    two_m = jnp.sum(k)
    
    # Avoid division by zero
    two_m = jnp.where(two_m == 0, 1.0, two_m)
    
    # B = A - (k k^T) / 2m
    # Instead of explicitly forming B (which is N x N), we can compute Tr(S^T B S)
    # Tr(S^T B S) = Tr(S^T A S) - Tr(S^T (k k^T / 2m) S)
    #             = Tr(S^T A S) - (1/2m) * Tr(S^T k k^T S)
    #             = Tr(S^T A S) - (1/2m) * (S^T k)^T (S^T k)
    
    STS_A = jnp.einsum('ik,jk,ij->', S, S, A)
    ST_k = jnp.dot(S.T, k)
    STS_k = jnp.sum(ST_k**2)
    
    Q = (STS_A - (STS_k / two_m)) / two_m
    return Q
