# Differentiable vs Discrete Clustering — Why Both Matter

*Logged: 2026-05-11, after Block E (differentiable) benchmark run*

---

## Observation from Block E results

In the standalone topology-only benchmark (Block E), discrete methods (Leiden, Louvain, Infomap, Spectral) consistently beat differentiable methods (Glass-SE, Glass-Mod, Glass-Map, SoftKMeans) on NMI — and run orders of magnitude faster.

Example (DCSBM, best NMI):
- Leiden: 0.614, ~0.06s
- Glass-SE: 0.443, ~1.28s

---

## Why discrete methods win in this setting

1. **Decades of engineering** — Louvain/Leiden exploit integer structure with greedy local moves that implicitly perform billions of efficient "gradient steps" per second. The differentiable methods here run 500 Adam steps on a soft relaxation then argmax. This is a fundamentally weaker optimiser for the discrete combinatorial problem.

2. **Relaxation gap** — Minimising a soft/temperature-annealed objective does not guarantee a good discrete assignment after argmax. `SpectralGrad` illustrates this perfectly: it minimises the NCut relaxation well in continuous space, but the discrete output is near-random (NMI ≈ 0 on most datasets).

3. **Block E is the hardest setting** — topology only, no node features, standalone optimisation. This is not what differentiable clustering objectives are designed for.

---

## Why differentiable clustering still matters

The actual use case is **a differentiable loss inside a GNN training loop**, not standalone graph optimisation:

```
node features → GNN encoder → soft assignment S → Glass-SE(A, S) → backprop into encoder
```

A discrete method like Louvain cannot provide gradients back into the encoder weights. A differentiable objective can. This is why every modern deep clustering paper (DMoN, MinCutPool, etc.) uses a soft loss — not because it gives better clusters standalone, but because it enables learning better node representations end-to-end.

Block E (standalone differentiable) is a **soundness test**: can the objective landscape recover known communities without a GNN? `Glass-SE` winning on Karate and SBM-Easy (NMI = 0.886 and 1.000) confirms the inductive bias is correct. Failure on noisier graphs (DCSBM, LFR, citation networks) is expected — that is precisely where a learned representation becomes critical.

---

## Summary table

| Property | Discrete (Louvain/Leiden) | Differentiable (Glass-SE/Mod/Map) |
|---|---|---|
| Standalone NMI | ✅ High | ⚠️ Moderate |
| Runtime | ✅ Fast (ms–s) | ❌ Slow (s–min) |
| Backpropagatable | ❌ No | ✅ Yes |
| Works with node features | ❌ No (post-hoc) | ✅ Yes (end-to-end) |
| Scalable (N > 5k dense) | ✅ Yes | ❌ No (dense JAX cap) |
| Purpose | Final clustering | Training loss / representation learning |

**Punchline:** discrete methods are better *optimisers* for the discrete problem; differentiable objectives are better *losses* for the representation learning problem. They are not competing — they operate in different parts of the pipeline.

---

## Related results

- Block E raw data: `docs/results/same_protocol/raw/same_protocol_differentiable_20260511_131124.json`
- Block E report: `docs/results/same_protocol/reports/same_protocol_differentiable_20260511_131124.md`
- Discrete baseline source: `docs/results/same_protocol/raw/same_protocol_gap_closure_20260511.json`
