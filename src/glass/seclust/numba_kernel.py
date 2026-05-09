"""Numba-JIT kernel for SEClust per-node move-delta scoring.

Companion to ``IncrementalSEState.move_delta_batch``. Numba avoids the
host-device dispatch overhead that makes JAX JIT slower than numpy on
CPU at this kernel granularity (move-delta is ~1 µs of pure compute,
swamped by JAX's per-call Python overhead). LLVM-compiled native code
gets us ~25-50× speedup over pure Python on CPU; numpy gets us ~5×;
this gets us close to the JIT ceiling on CPU.

The numba kernel is loaded lazily (so the package still imports
cleanly when numba is unavailable). If numba is not installed, the
public ``numba_move_delta_batch`` falls back to the numpy version.
"""

from __future__ import annotations

import math
import numpy as np


_NUMBA_AVAILABLE = None
_NUMBA_KERNEL = None


def _ensure_numba_kernel():
    global _NUMBA_AVAILABLE, _NUMBA_KERNEL
    if _NUMBA_AVAILABLE is False:
        return None
    if _NUMBA_KERNEL is not None:
        return _NUMBA_KERNEL
    try:
        from numba import njit
    except ImportError:
        _NUMBA_AVAILABLE = False
        return None
    _NUMBA_AVAILABLE = True

    @njit(cache=True, fastmath=True)
    def _kernel(
        nbrs,             # (deg,) int64
        weights,          # (deg,) float64
        cands,            # (n_cand,) int64
        source,           # int64
        labels,           # (N,) int64
        volume,           # (capacity,) float64
        cut,              # (capacity,) float64
        dlogd,            # (capacity,) float64
        active,           # (capacity,) bool
        node_degree,      # float64
        node_dlogd,       # float64
        graph_volume,     # float64
        eps,              # float64
    ):
        deg = nbrs.shape[0]
        n_cand = cands.shape[0]

        # Per-candidate edge weight to that cluster, plus weight to source.
        w_to_cand = np.zeros(n_cand, dtype=np.float64)
        w_source = 0.0
        for i in range(deg):
            cl = labels[nbrs[i]]
            w = weights[i]
            if cl == source:
                w_source += w
            for c_idx in range(n_cand):
                if cands[c_idx] == cl:
                    w_to_cand[c_idx] += w

        # Source before/after.
        def _ent(vol, c, dl):
            if graph_volume <= eps or vol <= eps:
                return 0.0
            cc = c if c > 0.0 else 0.0
            log_vol_over_V = math.log2(max(vol / max(graph_volume, eps), eps))
            log_vol = math.log2(max(vol, eps))
            return -(cc / max(graph_volume, eps)) * log_vol_over_V \
                   - (dl - vol * log_vol) / max(graph_volume, eps)

        source_vol_before = volume[source]
        source_cut_before = cut[source]
        source_dlogd_before = dlogd[source]
        source_before = _ent(source_vol_before, source_cut_before, source_dlogd_before)
        source_after = _ent(
            source_vol_before - node_degree,
            source_cut_before - node_degree + 2.0 * w_source,
            source_dlogd_before - node_dlogd,
        )

        deltas = np.zeros(n_cand, dtype=np.float64)
        for ci in range(n_cand):
            tgt = cands[ci]
            if tgt == source:
                deltas[ci] = 0.0
                continue
            cv = volume[tgt]
            cc = cut[tgt]
            cdl = dlogd[tgt]
            tb = _ent(cv, cc, cdl) if active[tgt] else 0.0
            ta = _ent(
                cv + node_degree,
                cc + node_degree - 2.0 * w_to_cand[ci],
                cdl + node_dlogd,
            )
            deltas[ci] = source_after + ta - source_before - tb
        return deltas

    _NUMBA_KERNEL = _kernel
    return _kernel


def numba_move_delta_batch(state, node, candidates):
    """Numpy-friendly numba-jitted batched move-delta.

    Falls back to ``state.move_delta_batch`` if numba is unavailable or
    fails to compile (e.g., older numba versions).
    """

    kernel = _ensure_numba_kernel()
    if kernel is None:
        return state.move_delta_batch(node, candidates)

    candidates = np.asarray(candidates, dtype=np.int64)
    nbrs = np.asarray(state.graph.neighbors[node], dtype=np.int64)
    weights = np.asarray(state.graph.weights[node], dtype=np.float64)
    return kernel(
        nbrs,
        weights,
        candidates,
        np.int64(int(state.labels[node])),
        np.asarray(state.labels, dtype=np.int64),
        np.asarray(state.volume, dtype=np.float64),
        np.asarray(state.cut, dtype=np.float64),
        np.asarray(state.degree_log_degree, dtype=np.float64),
        np.asarray(state.active),
        np.float64(float(state.graph.degrees[node])),
        np.float64(float(state.node_degree_log_degree[node])),
        np.float64(float(state.graph.volume)),
        np.float64(float(state.eps)),
    )
