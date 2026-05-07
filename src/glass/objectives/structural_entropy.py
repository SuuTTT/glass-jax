import jax
import jax.numpy as jnp

def one_dimensional_structural_entropy(A, mask=None, eps=1e-8):
    """
    Calculate 1D Structural Entropy H_1(G).
    H_1(G) = -sum_i (d_i / 2m) * log2(d_i / 2m)
    """
    if mask is not None:
        A = A * mask[:, None] * mask[None, :]
        
    d = jnp.sum(A, axis=-1)
    two_m = jnp.sum(d)
    
    p = d / (two_m + eps)
    p = jnp.clip(p, eps, 1.0)
    
    if mask is not None:
        p = p * mask
        
    h1 = -jnp.sum(p * jnp.log2(p))
    return h1

def two_dimensional_structural_entropy(A, S, mask=None, is_logits=True, eps=1e-8):
    """
    Differentiable 2D Structural Entropy H^2(G).
    H^2(G) = -sum_i (g_i / 2m) * log2(V_i / 2m) - sum_i sum_{j in i} (d_j / 2m) * log2(d_j / V_i)
    """
    if is_logits:
        S = jax.nn.softmax(S, axis=-1)
        
    if mask is not None:
        S = S * mask[:, None]
        A = A * mask[:, None] * mask[None, :]
        
    d = jnp.sum(A, axis=-1) # (N,)
    two_m = jnp.sum(d) # scalar
    
    # V_i: Volume of module i (sum of degrees of nodes in module i)
    # V_i = sum_j S_ji * d_j
    V = jnp.dot(d, S) # (K,)
    
    # g_i: Cut of module i (sum of weights of edges leaving module i)
    # g_i = sum_j S_ji * (d_j - sum_l A_jl * S_li)
    # sum_l A_jl * S_li is (A S)_ji
    AS = jnp.dot(A, S) # (N, K)
    g = jnp.sum(S * (d[:, None] - AS), axis=0) # (K,)
    
    # Term 1: -sum_i (g_i / 2m) * log2(V_i / 2m)
    p_vol = V / (two_m + eps)
    p_cut = g / (two_m + eps)
    
    term1 = -jnp.sum(p_cut * jnp.log2(jnp.clip(p_vol, eps, 1.0)))
    
    # Term 2: -sum_i sum_j (S_ji * d_j / 2m) * log2(d_j / V_i)
    # log2(d_j / V_i) = log2(d_j / 2m) - log2(V_i / 2m)
    # Term 2 = -sum_j (d_j / 2m) * log2(d_j / 2m) * (sum_i S_ji) + sum_i (sum_j S_ji * d_j / 2m) * log2(V_i / 2m)
    # Since sum_i S_ji = 1:
    # Term 2 = H_1(G) + sum_i (V_i / 2m) * log2(V_i / 2m)
    
    h1 = one_dimensional_structural_entropy(A, mask=mask, eps=eps)
    
    p_vol_clamped = jnp.clip(p_vol, eps, 1.0)
    term2 = h1 + jnp.sum(p_vol * jnp.log2(p_vol_clamped))
    
    return term1 + term2
