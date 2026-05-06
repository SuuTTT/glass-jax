import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def sinkhorn_assignment(logits, epsilon=1e-2, threshold=1e-3, iterations=100):
    """
    Compute soft assignment using Sinkhorn iterations to avoid mode collapse.
    
    Args:
        logits: Input logits of shape (N, K).
        epsilon: Entropic regularization parameter.
        threshold: Convergence threshold.
        iterations: Max number of Sinkhorn iterations.
        
    Returns:
        Soft assignment matrix S of shape (N, K).
    """
    N, K = logits.shape
    
    # We want to map N nodes to K clusters.
    # Cost matrix is -logits (we want to maximize the "score")
    cost_matrix = -logits
    
    # Define geometry
    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    
    # Define marginals (uniform by default to prevent mode collapse)
    a = jnp.ones(N) / N
    b = jnp.ones(K) / K
    
    # Solve linear OT problem
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn(threshold=threshold, max_iterations=iterations)
    out = solver(prob)
    
    # The transportation matrix P is our soft assignment (N x K)
    # P_ij is the probability of node i being assigned to cluster j.
    # Note: Sinkhorn output P sums to 1 globally. We might want to re-scale 
    # so each row sums to 1 if we treat it as cluster probabilities.
    # However, for OT, a and b are marginals. So row sums are 'a' and col sums are 'b'.
    # If a = 1/N, then row sums are 1/N. We multiply by N to get row sums = 1.
    
    return out.matrix * N
