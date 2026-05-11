# Using Glass-SE / Glass-GNN on RL Transition Matrices

This note describes how to use the differentiable Glass-SE and Glass-GNN objectives as auxiliary losses for TD-MPC2-style latent world models. The goal is to discover reusable temporal abstractions by clustering latent states according to transition structure, not according to external labels.

## Setting

Let a world model produce latent states

```text
z_t = h(o_t),        z_{t+1} = d(z_t, a_t)
```

where `h` is the encoder and `d` is the latent dynamics model. TD-MPC2-style training already uses prediction, reward, value, policy, and consistency losses. Glass-SE adds a structural loss over the transition graph induced by the latent trajectory batch.

## Build A Transition Matrix

For a batch of latent states `Z = {z_i}` and observed or predicted transitions `(z_i, a_i, z_i^+)`, build a soft transition matrix `P`.

Prototype-based version:

```text
c_i = softmax(-||z_i - p_k||^2 / tau_p)
c_i^+ = softmax(-||z_i^+ - p_k||^2 / tau_p)
P = normalize_rows(sum_i c_i^T c_i^+)
```

Batch-state version:

```text
P_ij = softmax_j(-||d(z_i, a_i) - stopgrad(z_j)||^2 / tau_t)
```

Use a symmetrized adjacency for undirected structural entropy:

```text
A = (P + P^T) / 2
```

Use the directed `P` directly for flow-oriented map-equation experiments when the objective supports directed flow.

## Glass-SE Auxiliary Loss

Given an adjacency `A` and assignment logits `S_logits`, use the differentiable structural objective:

```python
import jax
import jax.numpy as jnp
from glass.objectives.structural_entropy import two_dimensional_structural_entropy

def glass_se_loss(A, S_logits):
    return two_dimensional_structural_entropy(A, S_logits)
```

The TD-MPC2-style total loss becomes:

```text
L = L_tdmpc2
  + lambda_se * H2(A, S)
  + lambda_bal * L_balance(S)
  + lambda_temp * L_temporal_consistency(S_t, S_{t+1})
```

Recommended auxiliary terms:

- `L_balance`: prevents all states collapsing into one cluster.
- `L_temporal_consistency`: encourages adjacent latent states to stay in compatible abstract regions unless the transition crosses a bottleneck.
- entropy annealing: starts with soft assignments and gradually sharpens `S`.

## Glass-GNN Variant

Glass-GNN replaces free assignment logits with a graph encoder:

```text
S_logits = GNN(A, X)
```

where node features `X` can be latent states, rewards, value estimates, action embeddings, or concatenated transition statistics. This is useful when topology alone is insufficient and the abstraction should depend on latent dynamics semantics.

Use Glass-SE when the transition graph is the source of truth. Use Glass-GNN when state attributes should influence the abstraction.

## TD-MPC2 Integration Points

Recommended order:

1. Train the world model without Glass-SE for a short warm-up.
2. Build `P` from latent rollouts in the replay batch.
3. Add Glass-SE with a small coefficient, for example `lambda_se in [1e-4, 1e-2]`.
4. Anneal assignment temperature from soft to sharper clusters.
5. Log cluster count, structural entropy, transition cut mass, and option-level dwell time.

Practical safeguards:

- Stop-gradient through one side of pairwise transition matching to avoid degenerate latent contraction.
- Cap graph size per batch or use prototypes to avoid `O(B^2)` memory.
- Do not use environment labels or task success labels for clustering loss selection.
- Report whether the transition matrix is observed, predicted, or mixed.

## Experiment To Add

Add a paper-side experiment block comparing:

- no abstraction loss,
- modularity loss on the transition graph,
- map-equation loss on the transition graph,
- Glass-SE loss on the transition graph,
- Glass-GNN loss using latent features.

Primary RL metrics:

- episode return,
- planning success rate,
- model prediction error,
- option/cluster dwell time,
- transition cut mass,
- structural entropy of the learned transition graph,
- wall-clock overhead.

This block should be treated as an additional RL/representation experiment, not as a replacement for the graph-clustering benchmark.
