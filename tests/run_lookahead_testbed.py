"""Fast testbed runner for the SEClust lookahead variants.

Runs every test case in ``tests/lookahead_testbed.py`` against:
- ``greedy``       — pure one-step greedy (baseline)
- ``mpc(beta,w)``  — receding-horizon beam search
- ``td(top_w)``    — TD bootstrap with greedy rollout as value fn
- ``tdlambda(...)``— TD($\\lambda$) blend

Prints a single table per test case with quality metrics and runtime.
Each run completes in under a second on the testbed.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from glass.seclust import (
    multistart_incremental_se_heuristic,
    merge_to_target_with_mpc,
    merge_to_target_with_td_bootstrap,
    merge_to_target_with_td_lambda,
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lookahead_testbed import get_testbed


SEED = 42


def _accuracy(y_true, y_pred):
    """Best-permutation accuracy (Hungarian)."""

    from scipy.optimize import linear_sum_assignment
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(int(y_true.max()), int(y_pred.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in zip(*ind)) / y_pred.size)


def _modularity(adj, labels):
    """Hard modularity Q for an undirected graph."""

    deg = adj.sum(axis=1)
    V = float(deg.sum())
    if V <= 1e-12:
        return 0.0
    score = 0.0
    for c in np.unique(labels):
        mask = labels == c
        internal = float(adj[np.ix_(mask, mask)].sum())
        d = float(deg[mask].sum())
        score += internal - (d * d / V)
    return float(score / V)


def _evaluate(case, labels):
    return {
        "K": int(np.unique(labels).size),
        "ACC": _accuracy(case.labels, labels),
        "NMI": float(normalized_mutual_info_score(case.labels, labels)),
        "ARI": float(adjusted_rand_score(case.labels, labels)),
        "Q":   _modularity(case.adjacency, labels),
    }


def main():
    cases = get_testbed()
    strategies = [
        ("greedy",            ("mpc", {"beta": 1, "beam_width": 1})),
        ("mpc(b=3,w=4)",      ("mpc", {"beta": 3, "beam_width": 4})),
        ("mpc(b=8,w=8)",      ("mpc", {"beta": 8, "beam_width": 8})),
        ("td(w=4)",           ("td",  {"top_w": 4})),
        ("td(w=8)",           ("td",  {"top_w": 8})),
        ("td(w=16)",          ("td",  {"top_w": 16})),
        ("tdλ(b=3,w=8,λ=.5)", ("tdlambda", {"beta": 3, "top_w": 8, "lam": 0.5})),
        ("tdλ(b=3,w=8,λ=.8)", ("tdlambda", {"beta": 3, "top_w": 8, "lam": 0.8})),
    ]

    for case in cases:
        # One multistart base partition shared across all strategies (so the
        # comparison is purely of the merge phase).
        base, _ = multistart_incremental_se_heuristic(
            case.adjacency, starts=4, max_passes=8, seed=SEED,
        )
        K_local = int(np.unique(base).size)
        print(f"\n=== {case.name}  N={case.adjacency.shape[0]}  "
              f"K_local={K_local} -> target K={case.target_k}  "
              f"({case.description}) ===")
        print(f"  {'strategy':<22} {'K':>3}  {'SE':>8}  {'Q':>7}  {'ACC':>5}  {'NMI':>5}  {'ARI':>6}  {'time':>7}")
        for label, (mode, kwargs) in strategies:
            t0 = time.time()
            if mode == "mpc":
                labels, se = merge_to_target_with_mpc(case.adjacency, base, case.target_k, **kwargs)
            elif mode == "td":
                labels, se = merge_to_target_with_td_bootstrap(case.adjacency, base, case.target_k, **kwargs)
            elif mode == "tdlambda":
                labels, se = merge_to_target_with_td_lambda(case.adjacency, base, case.target_k, **kwargs)
            else:
                raise ValueError(mode)
            elapsed = time.time() - t0
            m = _evaluate(case, labels)
            print(f"  {label:<22} {m['K']:>3}  {se:>8.4f}  {m['Q']:>7.4f}  "
                  f"{m['ACC']:>5.3f}  {m['NMI']:>5.3f}  {m['ARI']:>6.3f}  {elapsed:>6.2f}s")


if __name__ == "__main__":
    main()
