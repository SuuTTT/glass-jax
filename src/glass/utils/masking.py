import jax.numpy as jnp

def apply_mask(A, mask):
    """Apply mask to adjacency matrix."""
    return A * mask[:, None] * mask[None, :]

def get_real_nodes_count(mask):
    """Return number of real nodes from mask."""
    return jnp.sum(mask)
