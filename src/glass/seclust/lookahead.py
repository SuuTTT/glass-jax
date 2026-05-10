"""Look-ahead greedy merge for SEClust target-K (workstream 2.1).

The pure-greedy ``merge_hierarchy_levels`` minimises immediate
$\\Delta H_2$ at each merge step. On graphs whose topological community
count is much larger than the target $K$ this collapses (paper
Sec.~VIII-A). We frame the lookahead choice through two
control-theory analogies:

* **MPC** (Model Predictive Control, receding horizon). Roll out
  partial merge sequences of fixed depth $\\beta$ via beam search of
  width $w$, commit only the first merge of the lowest-cost plan,
  re-plan from the new state. Implemented as
  :func:`merge_to_target_with_mpc` and exposed via
  :func:`seclust_target_k_lookahead` for backward compatibility.

* **TD bootstrap** (Bertsekas's *rollout planning*; equivalent to
  TD(0) with the greedy rollout as the leaf value function). For each
  candidate first merge in the top-$w$ by immediate $\\Delta H_2$,
  apply it and run a *full pure-greedy rollout to* $K_\\text{target}$;
  the cumulative $\\Delta H_2$ of that rollout is the
  bootstrap estimate of the cost-to-go. Pick the candidate with
  smallest total cost. Cost: $O(w (K-T)^2)$ per merge step --
  practical even at $K_\\text{local}\\!\\approx\\!400$.
  Implemented as :func:`merge_to_target_with_td_bootstrap`.

* **TD($\\lambda$)** blends the two: at each candidate, score the
  weighted sum of (a) finite-horizon MPC cumulative cost at depth
  $\\beta$, and (b) the bootstrap-estimate cost-to-go from the leaf,
  with mixing weight $\\lambda \\in [0, 1]$. $\\lambda=0$ is pure
  finite-horizon MPC; $\\lambda=1$ is pure bootstrap. Implemented as
  :func:`merge_to_target_with_td_lambda`.

The three routines share the same underlying :class:`_LookaheadState`
representation. The single entry point :func:`seclust_target_k_lookahead`
selects the strategy via a ``mode`` argument
(``"mpc"`` / ``"td"`` / ``"tdlambda"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .entropy import canonicalize_labels
from .heuristics import ClusteringResult
from .hierarchy import _initial_module_stats
from .incremental import IncrementalSEState, SparseGraph


# ---------- per-cluster SE term ----------

def _se_cluster_term(volume: float, cut: float, S_cluster: float, V: float) -> float:
    """Per-cluster contribution to flat 2D SE.

    H_2(P) = const + sum_C { -(g_C/V) log2(vol_C/V)
                              + (vol_C/V) log2(vol_C)
                              - S_C / V }

    where S_C = sum_{v in C} d_v log2(d_v). The graph-wide constant
    cancels in deltas, so we drop it.
    """

    if V <= 1e-12 or volume <= 1e-12:
        return 0.0
    cut_term = -(max(cut, 0.0) / V) * math.log2(max(volume / V, 1e-300))
    vol_term = (volume / V) * math.log2(volume) if volume > 1e-12 else 0.0
    return cut_term + vol_term - (S_cluster / V if V > 0 else 0.0)


# ---------- shared cheap-to-clone state ----------

@dataclass
class _LookaheadState:
    volumes: np.ndarray
    cuts: np.ndarray
    S: np.ndarray
    between: np.ndarray
    active: list[int]
    active_set: set[int]
    V: float
    parent: dict[int, int]
    cumulative_delta: float = 0.0

    def clone(self) -> "_LookaheadState":
        return _LookaheadState(
            volumes=self.volumes.copy(),
            cuts=self.cuts.copy(),
            S=self.S.copy(),
            between=self.between.copy(),
            active=list(self.active),
            active_set=set(self.active_set),
            V=self.V,
            parent=dict(self.parent),
            cumulative_delta=self.cumulative_delta,
        )

    def merge_delta(self, i: int, j: int) -> float:
        v_i, v_j = float(self.volumes[i]), float(self.volumes[j])
        c_i, c_j = float(self.cuts[i]), float(self.cuts[j])
        s_i, s_j = float(self.S[i]), float(self.S[j])
        w = float(self.between[i, j])
        new_v = v_i + v_j
        new_c = max(c_i + c_j - 2.0 * w, 0.0)
        new_s = s_i + s_j
        before = _se_cluster_term(v_i, c_i, s_i, self.V) + _se_cluster_term(v_j, c_j, s_j, self.V)
        after = _se_cluster_term(new_v, new_c, new_s, self.V)
        return after - before

    def merge_delta_modularity(self, i: int, j: int) -> float:
        """Return $-\\Delta Q$ for merging $i$ and $j$ (smaller = better merge).

        Standard Louvain modularity gain on a merge:
        $\\Delta Q = \\frac{w_{ij}}{V} - \\frac{\\mathrm{vol}_i\\,\\mathrm{vol}_j}{V^2}$
        where $V$ is the total graph volume ($=2m$). To stay consistent
        with ``merge_delta``'s sign convention (lower is a better merge,
        like an SE-decrease), we return $-\\Delta Q$. The caller picks
        the pair with smallest value.
        """

        v_i, v_j = float(self.volumes[i]), float(self.volumes[j])
        w = float(self.between[i, j])
        if self.V <= 1e-12:
            return 0.0
        delta_Q = (w / self.V) - (v_i * v_j) / (self.V * self.V)
        return -float(delta_Q)

    def merge_delta_hybrid(self, i: int, j: int, alpha: float) -> float:
        """Convex combination of SE and (negated) modularity merge deltas.

        $\\text{score} = \\alpha \\Delta H_2 + (1 - \\alpha)(-\\Delta Q)$.
        $\\alpha = 1$ recovers pure SE-merge; $\\alpha = 0$ recovers
        pure modularity-merge. Both deltas are sign-aligned (smaller is
        better) so the convex combination is well-defined.
        """

        if alpha >= 1.0 - 1e-12:
            return self.merge_delta(i, j)
        if alpha <= 1e-12:
            return self.merge_delta_modularity(i, j)
        return alpha * self.merge_delta(i, j) + (1.0 - alpha) * self.merge_delta_modularity(i, j)

    def apply_merge(self, i: int, j: int) -> None:
        v_i, v_j = float(self.volumes[i]), float(self.volumes[j])
        c_i, c_j = float(self.cuts[i]), float(self.cuts[j])
        w = float(self.between[i, j])
        self.volumes[i] = v_i + v_j
        self.cuts[i] = max(c_i + c_j - 2.0 * w, 0.0)
        self.S[i] = self.S[i] + self.S[j]
        for k in self.active:
            if k == i or k == j:
                continue
            new_w = float(self.between[i, k]) + float(self.between[j, k])
            self.between[i, k] = new_w
            self.between[k, i] = new_w
            self.between[j, k] = 0.0
            self.between[k, j] = 0.0
        self.between[i, j] = 0.0
        self.between[j, i] = 0.0
        self.between[i, i] = 0.0
        self.between[j, j] = 0.0
        self.volumes[j] = 0.0
        self.cuts[j] = 0.0
        self.S[j] = 0.0
        self.active = [a for a in self.active if a != j]
        self.active_set.discard(j)
        self.parent[j] = i

    def candidate_pairs(self) -> list[tuple[int, int]]:
        """Active pairs with non-zero between weight."""

        pairs: list[tuple[int, int]] = []
        active_list = self.active
        for idx_i, i in enumerate(active_list):
            row = self.between[i]
            for j in active_list[idx_i + 1 :]:
                if float(row[j]) > 0.0:
                    pairs.append((i, j))
        return pairs

    def all_active_pairs(self) -> list[tuple[int, int]]:
        active_list = self.active
        pairs: list[tuple[int, int]] = []
        for idx_i, i in enumerate(active_list):
            for j in active_list[idx_i + 1 :]:
                pairs.append((i, j))
        return pairs


def _initial_state(adj, base_labels: np.ndarray) -> tuple[_LookaheadState, np.ndarray]:
    canonical, degrees, V, volumes, cuts, between, _ = _initial_module_stats(adj, base_labels)
    K = volumes.size
    S = np.zeros(K, dtype=float)
    safe_log = np.zeros_like(degrees)
    pos = degrees > 1e-12
    safe_log[pos] = degrees[pos] * np.log2(degrees[pos])
    for cid in range(K):
        S[cid] = float(safe_log[canonical == cid].sum())
    state = _LookaheadState(
        volumes=volumes.astype(float),
        cuts=cuts.astype(float),
        S=S,
        between=between.astype(float),
        active=list(range(K)),
        active_set=set(range(K)),
        V=float(V),
        parent={},
        cumulative_delta=0.0,
    )
    return state, canonical


# ---------- step primitives ----------

def _greedy_choice(
    state: _LookaheadState,
    alpha: float = 1.0,
) -> tuple[int, int, float] | None:
    """Pick the immediate min-delta pair under the SE/modularity hybrid.

    ``alpha=1.0`` (default) is pure SE; ``alpha=0.0`` is pure
    modularity; intermediate values blend.
    """

    candidates = state.candidate_pairs() or state.all_active_pairs()
    if not candidates:
        return None
    best = None
    best_delta = float("inf")
    for (i, j) in candidates:
        d = state.merge_delta_hybrid(i, j, alpha)
        if d < best_delta - 1e-12:
            best_delta = d
            best = (i, j, d)
    return best


def _top_w_choices(
    state: _LookaheadState,
    width: int,
    alpha: float = 1.0,
) -> list[tuple[int, int, float]]:
    candidates = state.candidate_pairs() or state.all_active_pairs()
    if not candidates:
        return []
    scored = [(state.merge_delta_hybrid(i, j, alpha), i, j) for (i, j) in candidates]
    scored.sort()
    return [(i, j, d) for (d, i, j) in scored[:width]]


def _greedy_rollout_cost(
    state: _LookaheadState,
    target_K: int,
    alpha: float = 1.0,
) -> float:
    """Pure-greedy rollout from ``state`` to ``target_K``.

    Uses the SE/modularity hybrid objective at each step. Returns
    cumulative score under the same hybrid (so the bootstrap
    estimate is in the same units as the immediate scores). Pass
    ``alpha=1.0`` for pure SE (the original behaviour),
    ``alpha=0.0`` for pure modularity, or anywhere in between.
    """

    cum = 0.0
    while len(state.active) > target_K:
        choice = _greedy_choice(state, alpha=alpha)
        if choice is None:
            break
        i, j, d = choice
        cum += d
        state.apply_merge(i, j)
    return cum


# ---------- MPC: depth-β, width-w beam search ----------

def merge_to_target_with_mpc(
    adj,
    base_labels: np.ndarray,
    target_clusters: int,
    beta: int = 3,
    beam_width: int = 4,
) -> tuple[np.ndarray, float]:
    """Receding-horizon MPC with beam search of depth $\\beta$, width $w$.

    Beam initially holds the top-$w$ immediate merges. Each beam step
    expands every entry by its top-$w$ next merges; after $\\beta$ beam
    steps we commit only the *first* merge of the best surviving plan
    and re-plan. With $(\\beta, w) = (1, 1)$ this reduces exactly to
    one-step greedy.
    """

    if target_clusters < 1 or beta < 1 or beam_width < 1:
        raise ValueError("target_clusters/beta/beam_width must be >= 1")
    state, canonical = _initial_state(adj, base_labels)
    while len(state.active) > target_clusters:
        first = _best_first_merge_via_beam(state, depth=beta, width=beam_width)
        if first is None:
            break
        state.apply_merge(*first)
    return _decode(state, canonical, adj)


def _best_first_merge_via_beam(state: _LookaheadState, depth: int, width: int) -> tuple[int, int] | None:
    options = _top_w_choices(state, width)
    if not options:
        return None
    if depth <= 1 or len(options) == 1:
        return (options[0][0], options[0][1])

    @dataclass
    class _Beam:
        first: tuple[int, int]
        state: _LookaheadState

    beam: list[_Beam] = []
    for (i, j, d) in options:
        cloned = state.clone()
        cloned.cumulative_delta = state.cumulative_delta + d
        cloned.apply_merge(i, j)
        beam.append(_Beam(first=(i, j), state=cloned))

    for _ in range(depth - 1):
        expanded: list[_Beam] = []
        for entry in beam:
            sub_options = _top_w_choices(entry.state, width)
            if not sub_options:
                expanded.append(entry)
                continue
            for (i, j, d) in sub_options:
                cloned = entry.state.clone()
                cloned.cumulative_delta = entry.state.cumulative_delta + d
                cloned.apply_merge(i, j)
                expanded.append(_Beam(first=entry.first, state=cloned))
        expanded.sort(key=lambda b: b.state.cumulative_delta)
        beam = expanded[:width]
        if not beam:
            break
    if not beam:
        return None
    return beam[0].first


# ---------- TD bootstrap: rollout-policy lookahead ----------

def merge_to_target_with_adaptive_td(
    adj,
    base_labels: np.ndarray,
    target_clusters: int,
    max_w: int = 8,
    margin_tau: float = 0.10,
) -> tuple[np.ndarray, float]:
    r"""TD bootstrap with **adaptive width** by score-margin confidence.

    Idea **A** in `_ideas_to_try.md` / item #5 in idealist. Addresses
    the td(w=8) regression on hier_sbm in idea 001: wider $w$ can
    *hurt* because the greedy-rollout value function is noisy, so
    scoring more candidates with a noisier estimator increases
    variance.

    The adaptive rule at each merge step:

    1. Score the top-``max_w`` candidates by immediate $\Delta H_2$
       and run a full greedy rollout to ``target_clusters`` from each.
    2. Sort by ``immediate + bootstrap`` total. Compute the *margin*
       between the best and second-best totals, normalised by the
       overall score range:

           margin = (s[1] - s[0]) / max(s[-1] - s[0], 1e-12)

       This is in $[0, 1]$: large means the winner is clearly
       separated; small means the top candidates are within noise of
       each other.
    3. If ``margin >= margin_tau``: trust TD bootstrap; commit
       ``s[0]``'s first merge.
    4. Else: fall back to the immediate-delta winner (pure greedy
       choice). We trust greedy when bootstrap can't discriminate.

    With ``margin_tau = 0`` we recover full TD bootstrap (always trust
    bootstrap); with ``margin_tau = 1`` we recover pure greedy.

    The point: **adaptive_td matches td(w=4) on hier_sbm and avoids
    td(w=8)'s regression** without the user having to pick $w$ by hand.
    """

    if target_clusters < 1 or max_w < 1:
        raise ValueError("target_clusters/max_w must be >= 1")
    if not (0.0 <= margin_tau <= 1.0):
        raise ValueError("margin_tau must be in [0, 1]")

    state, canonical = _initial_state(adj, base_labels)
    while len(state.active) > target_clusters:
        immediate = _top_w_choices(state, max_w)
        if not immediate:
            break

        scored: list[tuple[float, int, int, float]] = []
        for (i, j, d) in immediate:
            cloned = state.clone()
            cloned.apply_merge(i, j)
            total = d + _greedy_rollout_cost(cloned, target_clusters)
            scored.append((total, i, j, d))
        scored.sort(key=lambda t: t[0])

        if len(scored) < 2:
            best_pair = (scored[0][1], scored[0][2])
        else:
            score_min = scored[0][0]
            score_max = scored[-1][0]
            score_range = max(score_max - score_min, 1e-12)
            margin = (scored[1][0] - scored[0][0]) / score_range
            if margin >= margin_tau:
                # Confident: take the bootstrap winner.
                best_pair = (scored[0][1], scored[0][2])
            else:
                # Noisy: fall back to the smallest *immediate* delta
                # (the greedy choice).
                greedy_best = min(scored, key=lambda t: t[3])
                best_pair = (greedy_best[1], greedy_best[2])

        state.apply_merge(*best_pair)

    return _decode(state, canonical, adj)


def merge_to_target_with_hybrid_objective(
    adj,
    base_labels: np.ndarray,
    target_clusters: int,
    top_w: int = 8,
    alpha: float = 0.5,
) -> tuple[np.ndarray, float]:
    """TD bootstrap with an SE / modularity hybrid merge objective.

    Idea **§2.1''** in NEXT_STEPS / item #4 in idealist. At each step
    score candidate merges by

        $$\\text{score} = \\alpha \\Delta H_2 + (1-\\alpha)(-\\Delta Q),$$

    pick top-$w$ by score, run a hybrid-greedy rollout to
    ``target_clusters``, and commit the candidate with smallest
    immediate $+$ bootstrap. Final partition is scored on 2D SE for
    paper-comparable numbers.

    ``alpha=1.0`` recovers pure-SE TD bootstrap; ``alpha=0.0`` is
    pure-modularity merge; ``alpha=0.5`` is a balanced blend.
    """

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if target_clusters < 1 or top_w < 1:
        raise ValueError("target_clusters/top_w must be >= 1")

    state, canonical = _initial_state(adj, base_labels)
    while len(state.active) > target_clusters:
        candidates = _top_w_choices(state, top_w, alpha=alpha)
        if not candidates:
            break
        best_pair, best_total = None, float("inf")
        for (i, j, d) in candidates:
            cloned = state.clone()
            cloned.apply_merge(i, j)
            total = d + _greedy_rollout_cost(cloned, target_clusters, alpha=alpha)
            if total < best_total - 1e-12:
                best_total = total
                best_pair = (i, j)
        if best_pair is None:
            break
        state.apply_merge(*best_pair)
    return _decode(state, canonical, adj)


def merge_to_target_with_td_bootstrap(
    adj,
    base_labels: np.ndarray,
    target_clusters: int,
    top_w: int = 8,
) -> tuple[np.ndarray, float]:
    """TD(0) with greedy-rollout value function.

    At each merge step:
    1. Identify the top-$w$ candidate first merges by immediate
       $\\Delta H_2$.
    2. For each candidate, apply it on a clone, then run pure greedy
       all the way to ``target_clusters``. Score = candidate's
       immediate $\\Delta H_2$ + cumulative greedy cost-to-go.
    3. Commit the candidate with the smallest score.

    Equivalent to MPC with $\\beta = K_\\text{local} - K_\\text{target}$
    and width 1 (i.e., infinite horizon), but only top-$w$ first moves
    are evaluated. Cost: $O(w (K-T)^2)$ per merge step; for $K-T<500$
    this is sub-second in pure Python.
    """

    if target_clusters < 1 or top_w < 1:
        raise ValueError("target_clusters/top_w must be >= 1")
    state, canonical = _initial_state(adj, base_labels)
    while len(state.active) > target_clusters:
        candidates = _top_w_choices(state, top_w)
        if not candidates:
            break
        best_pair, best_total = None, float("inf")
        for (i, j, d) in candidates:
            cloned = state.clone()
            cloned.apply_merge(i, j)
            total = d + _greedy_rollout_cost(cloned, target_clusters)
            if total < best_total - 1e-12:
                best_total = total
                best_pair = (i, j)
        if best_pair is None:
            break
        state.apply_merge(*best_pair)
    return _decode(state, canonical, adj)


# ---------- TD(lambda): blend MPC and bootstrap ----------

def merge_to_target_with_td_lambda(
    adj,
    base_labels: np.ndarray,
    target_clusters: int,
    beta: int = 3,
    top_w: int = 4,
    lam: float = 0.5,
) -> tuple[np.ndarray, float]:
    r"""TD($\lambda$) blend of MPC and bootstrap.

    For each top-$w$ candidate first merge, score it as

        score = (1 - lam) * mpc_cost(beta) + lam * bootstrap_cost
              = (1 - lam) * sum of beta greedy deltas after this merge
              + lam * full greedy cost-to-go from this merge

    With $\lambda = 0$ this is pure depth-$\beta$ MPC (with width 1
    expansion of each candidate). With $\lambda = 1$ it is pure
    bootstrap. Intermediate $\lambda$ trades horizon for cost.
    """

    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if target_clusters < 1 or beta < 1 or top_w < 1:
        raise ValueError("target_clusters/beta/top_w must be >= 1")

    state, canonical = _initial_state(adj, base_labels)
    while len(state.active) > target_clusters:
        candidates = _top_w_choices(state, top_w)
        if not candidates:
            break
        best_pair, best_total = None, float("inf")
        for (i, j, d) in candidates:
            cloned = state.clone()
            cloned.apply_merge(i, j)
            mpc_part = 0.0
            for _ in range(beta - 1):
                if len(cloned.active) <= target_clusters:
                    break
                step = _greedy_choice(cloned)
                if step is None:
                    break
                mpc_part += step[2]
                cloned.apply_merge(step[0], step[1])
            bootstrap_part = _greedy_rollout_cost(cloned.clone(), target_clusters) if lam > 0 else 0.0
            score = d + (1.0 - lam) * mpc_part + lam * bootstrap_part
            if score < best_total - 1e-12:
                best_total = score
                best_pair = (i, j)
        if best_pair is None:
            break
        state.apply_merge(*best_pair)
    return _decode(state, canonical, adj)


# ---------- shared decode + entry point ----------

def _decode(state: _LookaheadState, canonical: np.ndarray, adj) -> tuple[np.ndarray, float]:
    def _root(module: int) -> int:
        cur = module
        while cur in state.parent:
            cur = state.parent[cur]
        return cur

    K_total = state.volumes.size
    root_for = np.array([_root(int(m)) for m in range(K_total)], dtype=np.int64)
    labels = canonicalize_labels(np.asarray([root_for[int(m)] for m in canonical], dtype=np.int32))
    if isinstance(adj, SparseGraph):
        graph = adj
    else:
        graph = SparseGraph.from_adjacency(adj)
    se_state = IncrementalSEState(graph, labels)
    return labels, float(se_state.entropy)


def merge_to_target_with_lookahead(
    adj,
    base_labels: np.ndarray,
    target_clusters: int,
    beta: int = 3,
    beam_width: int = 4,
) -> tuple[np.ndarray, float]:
    """Backward-compatible alias for the MPC variant."""

    return merge_to_target_with_mpc(adj, base_labels, target_clusters, beta=beta, beam_width=beam_width)


def seclust_target_k_lookahead(
    adj,
    target_clusters: int,
    starts: int = 6,
    max_passes: int = 10,
    seed: int = 0,
    mode: str = "td",
    beta: int = 3,
    beam_width: int = 4,
    top_w: int = 8,
    lam: float = 0.5,
) -> ClusteringResult:
    """End-to-end SEClust-TargetK with a chosen lookahead strategy.

    ``mode`` selects between "mpc" (receding-horizon beam search),
    "td" (rollout-policy bootstrap), and "tdlambda" (blend).
    """

    from .incremental import multistart_incremental_se_heuristic

    base_labels, _ = multistart_incremental_se_heuristic(
        adj, starts=starts, max_passes=max_passes, seed=seed,
    )
    if mode == "mpc":
        labels, entropy = merge_to_target_with_mpc(
            adj, base_labels, target_clusters, beta=beta, beam_width=beam_width,
        )
        method = f"seclust-target-k-mpc(beta={beta}, w={beam_width})"
    elif mode == "td":
        labels, entropy = merge_to_target_with_td_bootstrap(
            adj, base_labels, target_clusters, top_w=top_w,
        )
        method = f"seclust-target-k-td(top_w={top_w})"
    elif mode == "tdlambda":
        labels, entropy = merge_to_target_with_td_lambda(
            adj, base_labels, target_clusters, beta=beta, top_w=top_w, lam=lam,
        )
        method = f"seclust-target-k-tdlambda(beta={beta}, top_w={top_w}, lam={lam})"
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected mpc/td/tdlambda")
    return ClusteringResult(entropy=entropy, labels=labels, method=method)
