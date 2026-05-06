import jax
import jax.numpy as jnp

def min_cut(A, S, mask=None):
    """
    Min-Cut objective.
    
    Args:
        A: Adjacency matrix (N, N).
        S: Soft assignment matrix (N, K).
        mask: Optional mask (N,).
    """
    S = jax.nn.softmax(S, axis=-1)
    if mask is not None:
        S = S * mask[:, None]
        A = A * mask[:, None] * mask[None, :]
        
    # cut = Tr(S^T (D - A) S)
    D = jnp.diag(jnp.sum(A, axis=-1))
    L = D - A
    
    return jnp.trace(jnp.dot(jnp.dot(S.T, L), S))

def normalized_cut(A, S, mask=None):
    """
    Normalized Cut objective.
    
    Args:
        A: Adjacency matrix (N, N).
        S: Soft assignment matrix (N, K).
        mask: Optional mask (N,).
    """
    S = jax.nn.softmax(S, axis=-1)
    if mask is not None:
        S = S * mask[:, None]
        A = A * mask[:, None] * mask[None, :]
        
    d = jnp.sum(A, axis=-1)
    D = jnp.diag(d)
    L = D - A
    
    # ncut = sum_i [ (S_i^T L S_i) / (S_i^T D S_i) ]
    num = jnp.diag(jnp.dot(jnp.dot(S.T, L), S))
    den = jnp.diag(jnp.dot(jnp.dot(S.T, D), S))
    
    return jnp.sum(num / (den + 1e-8))
