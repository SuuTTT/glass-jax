"""JAX-JIT kernel for SEClust per-node move-delta scoring.

Companion to ``IncrementalSEState.move_delta_batch`` (numpy version).
The JAX kernel uses static padded shapes so a single compile is reused
across calls with different numbers of neighbors / candidates.

Usage:
    from glass.seclust.jit_kernel import jit_move_delta_batch

    deltas = jit_move_delta_batch(
        nbrs_padded, weights_padded, valid_nbr_mask,
        cands_padded, valid_cand_mask, source_id,
        labels, volume, cut, dlogd, active,
        node_degree, node_dlogd, graph_volume, eps,
    )

The caller is responsible for padding to a fixed ``(MAX_DEG,)`` and
``(MAX_CAND,)`` and supplying validity masks. Padded entries should
have ``label[-1] = some unused id`` (we mask before any reads).

The JIT version is *additional*; the numpy ``move_delta_batch`` stays
the default for one-shot calls. JIT pays off when the same shapes are
reused many times (e.g., a multistart-style outer loop).
"""

from __future__ import annotations

from functools import partial

import numpy as np


def _build_jax_kernel():
    """Lazy import + compile so we don't pay the JAX startup cost
    when the kernel is never used."""

    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnums=())
    def _kernel(
        nbrs,             # (MAX_DEG,) int32, padded with 0 (use mask)
        weights,          # (MAX_DEG,) float32, padded with 0
        nbr_mask,         # (MAX_DEG,) bool, valid neighbors
        cands,            # (MAX_CAND,) int32, padded
        cand_mask,        # (MAX_CAND,) bool
        source,           # () int32
        labels,           # (N,) int32 (full label vector)
        volume,           # (capacity,) float32
        cut,              # (capacity,) float32
        dlogd,            # (capacity,) float32
        active,           # (capacity,) bool
        node_degree,      # ()
        node_dlogd,       # ()
        graph_volume,     # ()
        eps,              # ()
    ):
        # Edge weight to each candidate cluster.
        nbr_clusters = labels[nbrs]                         # (MAX_DEG,)
        masked_w = jnp.where(nbr_mask, weights, 0.0)        # (MAX_DEG,)
        # (MAX_CAND, MAX_DEG)
        match = (nbr_clusters[None, :] == cands[:, None]).astype(jnp.float32)
        w_to_cand = jnp.sum(match * masked_w[None, :], axis=1)
        # (scalar) edge weight to source
        w_source = jnp.sum(jnp.where(nbr_clusters == source, masked_w, 0.0))

        def cluster_entropy_vec(vol, c, dl):
            mask = (graph_volume > eps) & (vol > eps)
            safe_vol = jnp.where(mask, vol, 1.0)
            log_vol_over_V = jnp.log2(jnp.maximum(safe_vol / jnp.maximum(graph_volume, eps), eps))
            log_vol = jnp.log2(jnp.maximum(safe_vol, eps))
            boundary = -(jnp.maximum(c, 0.0) / jnp.maximum(graph_volume, eps)) * log_vol_over_V
            internal = -((dl - safe_vol * log_vol) / jnp.maximum(graph_volume, eps))
            return jnp.where(mask, boundary + internal, 0.0)

        # Source before/after.
        source_vol_before = volume[source]
        source_cut_before = cut[source]
        source_dlogd_before = dlogd[source]
        source_before = cluster_entropy_vec(source_vol_before, source_cut_before, source_dlogd_before)
        source_after = cluster_entropy_vec(
            source_vol_before - node_degree,
            source_cut_before - node_degree + 2.0 * w_source,
            source_dlogd_before - node_dlogd,
        )

        # Per-candidate before/after, vectorised.
        cand_vol_before = volume[cands]
        cand_cut_before = cut[cands]
        cand_dlogd_before = dlogd[cands]
        cand_active = active[cands]

        target_before = jnp.where(
            cand_active,
            cluster_entropy_vec(cand_vol_before, cand_cut_before, cand_dlogd_before),
            0.0,
        )
        target_after = cluster_entropy_vec(
            cand_vol_before + node_degree,
            cand_cut_before + node_degree - 2.0 * w_to_cand,
            cand_dlogd_before + node_dlogd,
        )

        deltas = source_after + target_after - source_before - target_before
        # Source-to-source is a no-op.
        deltas = jnp.where(cands == source, 0.0, deltas)
        # Padded candidates get +inf so argmin ignores them.
        deltas = jnp.where(cand_mask, deltas, jnp.inf)
        return deltas

    return _kernel


_JIT_KERNEL = None


def jit_move_delta_batch(state, node, candidates, max_deg=None, max_cand=None):
    """Numpy-friendly wrapper around the JAX kernel.

    Pads neighbors/candidates to ``(max_deg,)`` / ``(max_cand,)`` and
    returns a length-``len(candidates)`` numpy array of deltas.

    On the first call the JAX kernel is built and JIT-compiled. The
    caller is responsible for passing consistent ``max_deg``/``max_cand``
    across calls if shape stability is desired (otherwise JAX recompiles
    per shape).
    """

    global _JIT_KERNEL
    if _JIT_KERNEL is None:
        _JIT_KERNEL = _build_jax_kernel()

    import jax.numpy as jnp

    candidates = np.asarray(candidates, dtype=np.int32)
    nbrs = np.asarray(state.graph.neighbors[node], dtype=np.int32)
    weights = np.asarray(state.graph.weights[node], dtype=np.float32)
    if max_deg is None:
        max_deg = int(nbrs.size)
    if max_cand is None:
        max_cand = int(candidates.size)

    nbrs_padded = np.zeros(max_deg, dtype=np.int32)
    weights_padded = np.zeros(max_deg, dtype=np.float32)
    nbr_mask = np.zeros(max_deg, dtype=bool)
    if nbrs.size:
        nbrs_padded[: nbrs.size] = nbrs
        weights_padded[: nbrs.size] = weights
        nbr_mask[: nbrs.size] = True

    cands_padded = np.zeros(max_cand, dtype=np.int32)
    cand_mask = np.zeros(max_cand, dtype=bool)
    cands_padded[: candidates.size] = candidates
    cand_mask[: candidates.size] = True

    deltas = _JIT_KERNEL(
        jnp.asarray(nbrs_padded),
        jnp.asarray(weights_padded),
        jnp.asarray(nbr_mask),
        jnp.asarray(cands_padded),
        jnp.asarray(cand_mask),
        jnp.int32(int(state.labels[node])),
        jnp.asarray(state.labels.astype(np.int32)),
        jnp.asarray(state.volume.astype(np.float32)),
        jnp.asarray(state.cut.astype(np.float32)),
        jnp.asarray(state.degree_log_degree.astype(np.float32)),
        jnp.asarray(state.active),
        jnp.float32(float(state.graph.degrees[node])),
        jnp.float32(float(state.node_degree_log_degree[node])),
        jnp.float32(float(state.graph.volume)),
        jnp.float32(float(state.eps)),
    )
    return np.asarray(deltas)[: candidates.size].astype(float)
