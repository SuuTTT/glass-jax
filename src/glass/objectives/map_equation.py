import jax
import jax.numpy as jnp

def entropy(p, axis=-1, keepdims=False):
    """Safe entropy calculation."""
    p = jnp.clip(p, 1e-12, 1.0)
    return -jnp.sum(p * jnp.log2(p), axis=axis, keepdims=keepdims)

def compute_stationary_distribution(A, num_iters=100, eps=1e-8):
    """Compute stationary distribution via power iteration."""
    N = A.shape[0]
    # Row-normalize to get transition matrix P
    d = jnp.sum(A, axis=-1)
    P = A / (d[:, None] + eps)
    
    # Start with uniform distribution
    pi = jnp.ones(N) / N
    
    def body_fn(i, val):
        return jnp.dot(val, P)
    
    pi = jax.lax.fori_loop(0, num_iters, body_fn, pi)
    return pi

def soft_map_equation(A, S, mask=None, pi=None):
    """
    Differentiable Map Equation (Soft Infomap).
    
    Args:
        A: Adjacency matrix (N, N).
        S: Soft assignment matrix (N, K).
        mask: Optional mask (N,).
        pi: Optional pre-computed stationary distribution (N,).
    """
    N, K = S.shape
    S = jax.nn.softmax(S, axis=-1)
    
    if mask is not None:
        S = S * mask[:, None]
        A = A * mask[:, None] * mask[None, :]
        
    if pi is None:
        pi = compute_stationary_distribution(A)
    
    # Transition matrix
    d = jnp.sum(A, axis=-1)
    P = A / (d[:, None] + 1e-8)
    
    # Total visit probability per module
    # pi_m: (K,)
    pi_m = jnp.dot(pi, S)
    
    # Exit probabilities per module
    # q_i_out = sum_{alpha, beta} pi_alpha * P_alpha_beta * S_alpha_i * (1 - S_beta_i)
    # q_i_out = sum_alpha pi_alpha * S_alpha_i * sum_beta P_alpha_beta * (1 - S_beta_i)
    # sum_beta P_alpha_beta * (1 - S_beta_i) = 1 - (P S)_alpha_i
    
    PS = jnp.dot(P, S) # (N, K)
    q_m_out = jnp.sum(pi[:, None] * S * (1 - PS), axis=0) # (K,)
    
    q_total_out = jnp.sum(q_m_out)
    
    # H(Q)
    h_q = entropy(q_m_out / (q_total_out + 1e-8))
    
    # Term 1: q_total_out * H(Q)
    term1 = q_total_out * h_q
    
    # Term 2: sum_i p_i_in * H(P_i)
    # p_i_in = q_i_out + pi_i
    p_m_in = q_m_out + pi_m
    
    # H(P_i) is entropy of (q_i_out, {pi_alpha * S_alpha_i}) normalized by p_m_in
    # We can compute this more efficiently
    
    # sum_{alpha in i} (pi_alpha * S_alpha_i / p_m_in) * log2(pi_alpha * S_alpha_i / p_m_in)
    # = (1/p_m_in) * [ sum_alpha pi_alpha S_alpha_i (log2 pi_alpha S_alpha_i - log2 p_m_in) ]
    # = (1/p_m_in) * [ sum_alpha pi_alpha S_alpha_i log2 pi_alpha S_alpha_i - pi_m_i log2 p_m_in ]
    
    pi_S = pi[:, None] * S
    pi_S_log = pi_S * jnp.log2(jnp.clip(pi_S, 1e-12, 1.0))
    sum_pi_S_log = jnp.sum(pi_S_log, axis=0)
    
    h_pi_part = -(sum_pi_S_log - pi_m * jnp.log2(jnp.clip(p_m_in, 1e-12, 1.0)))
    
    # Exit part: -(q_i_out / p_m_in) * log2(q_i_out / p_m_in)
    q_log_part = -(q_m_out * jnp.log2(jnp.clip(q_m_out / (p_m_in + 1e-8), 1e-12, 1.0)))
    
    term2 = jnp.sum(h_pi_part + q_log_part)
    
    return term1 + term2
