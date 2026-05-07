import jax
import jax.numpy as jnp
import flax.linen as nn

class GNNEncoder(nn.Module):
    """
    A simple GCN-style encoder to project node features into community assignments.
    """
    hidden_dim: int
    num_communities: int

    @nn.compact
    def __call__(self, x, adj):
        # Simplistic GNN layer: S = softmax(A * ReLU(X * W1) * W2)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(jnp.dot(adj, h))
        logits = nn.Dense(self.num_communities)(h)
        return logits
