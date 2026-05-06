import jax
import jax.numpy as jnp

def spectral_embedding(A, K, mask=None):
    """
    Compute spectral embeddings using the top K eigenvectors of the Laplacian.
    
    Args:
        A: Adjacency matrix (N, N).
        K: Number of clusters (embedding dimension).
        mask: Optional mask (N,).
    """
    if mask is not None:
        A = A * mask[:, None] * mask[None, :]
        
    d = jnp.sum(A, axis=-1)
    D = jnp.diag(d)
    L = D - A
    
    # Normalized Laplacian: L_rw = D^-1 L = I - D^-1 A
    D_inv = jnp.diag(1.0 / (d + 1e-8))
    L_rw = jnp.eye(A.shape[0]) - jnp.dot(D_inv, A)
    
    # Eigh for symmetric matrices. L_rw is not necessarily symmetric.
    # L_sym = D^-1/2 L D^-1/2 is symmetric.
    D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(d + 1e-8))
    L_sym = jnp.eye(A.shape[0]) - jnp.dot(jnp.dot(D_inv_sqrt, A), D_inv_sqrt)
    
    vals, vecs = jnp.linalg.eigh(L_sym)
    
    # Return the K eigenvectors corresponding to the smallest eigenvalues
    return vecs[:, :K]
