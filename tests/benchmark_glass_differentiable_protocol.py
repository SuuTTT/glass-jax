"""Block E: Differentiable Non-Attributed Glass Clustering benchmark.

Evaluates differentiable graph clustering objectives (Glass-SE, Glass-Mod,
Glass-Map) alongside seminal differentiable clustering baselines (SoftKMeans,
SpectralGrad) on topology-only graph datasets.

Discrete baselines (Louvain, Leiden, Infomap, Spectral, SEClust-ConstrainedK)
are loaded from an existing topology block JSON to avoid redundant computation.
They are re-tagged with block=differentiable and included in the merged output.

Seminal differentiable baselines:
- SoftKMeans:   temperature-annealed soft k-means on spectral embedding of
                adjacency (Bregman clustering relaxation; Duchi et al.).
- SpectralGrad: continuous relaxation of normalized cut (Shi & Malik 2000)
                minimized via gradient descent (not eigenvector decomposition).
- Glass-SE:     differentiable 2D structural-entropy H2 minimization (ours).
- Glass-Mod:    differentiable modularity Q maximization (gradient-based).
- Glass-Map:    differentiable map-equation L minimization (gradient-based).

Usage:
    cd /workspace/glass-jax

    # Quick smoke test
    python tests/benchmark_glass_differentiable_protocol.py \\
        --datasets Karate,SBM-Easy \\
        --methods Glass-SE,Glass-Mod,SoftKMeans \\
        --seeds 0 \\
        --output-dir /tmp/glass_diff_smoke

    # Full Block E (large datasets skip dense differentiable methods automatically)
    python tests/benchmark_glass_differentiable_protocol.py \\
        --datasets Karate,SBM-Easy,SBM-Noisy,DCSBM,LFR,Cora,Citeseer,PubMed,Photo,Computers \\
        --methods Glass-SE,Glass-Mod,Glass-Map,SoftKMeans,SpectralGrad \\
        --seeds 0,1,2,3,4,5,6,7,8,9 \\
        --reuse-discrete docs/results/same_protocol/raw/same_protocol_gap_closure_20260511.json \\
        --output-dir docs/results/same_protocol/raw

Output files (in --output-dir):
    same_protocol_differentiable_<timestamp>.json
    same_protocol_differentiable_<timestamp>_aggregated.json
    same_protocol_differentiable_<timestamp>.md

The merged result also copies to docs/results/same_protocol/reports/ (Markdown)
and docs/results/same_protocol/aggregated/ (aggregated JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Ensure tests/ is on sys.path so we can import from benchmark_seclust_full
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from benchmark_seclust_full import (  # noqa: E402
    DatasetCase,
    adjusted_rand_index,
    clustering_accuracy,
    hard_map_equation,
    hard_modularity,
    karate_graph,
    normalized_mutual_info,
    run_hcse,
    run_infomap,
    run_leiden,
    run_louvain,
    sparse_structural_entropy,
    structural_entropy,
    _adj_as_csr,
)
from benchmark_same_protocol import (  # noqa: E402
    HCSE_DENSE_MAX_NODES,
    SkipMethod,
    canonicalize_labels,
    load_case,
    make_sbm,
    mean_conductance,
    n_edges,
    n_nodes,
    rss_mb,
    run_label_propagation,
    run_spectral,
    run_seclust_same_protocol,
    RunConfig,
)
from glass.seclust import SparseGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_NAME = "differentiable"

# Dense JAX adjacency cap: at 5000 nodes a float32 dense matrix is ~100 MB
# and vmapped 4-start optimisation is memory-feasible on CPU/GPU.
DIFF_DENSE_MAX_NODES = 5_000

# Differentiable method hyper-parameters (sensible defaults; overridden by CLI)
DEFAULT_N_ITERS = 500
DEFAULT_N_STARTS = 4
DEFAULT_LR = 0.05

# Differentiable method names handled by this script
GLASS_DIFF_METHODS = [
    "Glass-SE",
    "Glass-Mod",
    "Glass-Map",
    "SoftKMeans",
    "SpectralGrad",
]

# Discrete method names that are re-tagged from an existing topology block JSON
DISCRETE_REUSE_METHODS = [
    "Louvain",
    "Leiden",
    "Infomap",
    "Spectral",
    "SEClust-ConstrainedK",
]

ALL_BLOCK_E_METHODS = GLASS_DIFF_METHODS + DISCRETE_REUSE_METHODS

BLOCK_E_DATASETS = [
    "Karate",
    "SBM-Easy",
    "SBM-Noisy",
    "DCSBM",
    "LFR",
    "Cora",
    "Citeseer",
    "PubMed",
    "Photo",
    "Computers",
]

# ---------------------------------------------------------------------------
# JAX objectives (lazy import to allow --dry-run without JAX)
# ---------------------------------------------------------------------------


def _jax_imports():
    import jax
    import jax.numpy as jnp
    import optax

    from glass.objectives.map_equation import (
        compute_stationary_distribution,
        soft_map_equation,
    )
    from glass.objectives.modularity import soft_modularity
    from glass.objectives.structural_entropy import two_dimensional_structural_entropy

    return jax, jnp, optax, soft_modularity, soft_map_equation, two_dimensional_structural_entropy, compute_stationary_distribution


# ---------------------------------------------------------------------------
# Dense adjacency helper
# ---------------------------------------------------------------------------


def _to_dense(adj) -> np.ndarray:
    """Convert adjacency to a float32 dense numpy array."""
    csr = _adj_as_csr(adj)
    dense = np.asarray(csr.toarray(), dtype=np.float32)
    return dense


# ---------------------------------------------------------------------------
# Glass differentiable objective runner (Glass-SE / Glass-Mod / Glass-Map)
# ---------------------------------------------------------------------------


def run_glass_differentiable(
    case: DatasetCase,
    objective_tag: str,
    seed: int,
    n_iters: int = DEFAULT_N_ITERS,
    n_starts: int = DEFAULT_N_STARTS,
    lr: float = DEFAULT_LR,
) -> tuple[np.ndarray, float, float, dict]:
    """Multistart JAX optimisation for a differentiable graph objective.

    Returns
    -------
    labels : np.ndarray  (hard assignment after argmax)
    runtime_seconds : float
    soft_objective : float  (best soft loss before hardening)
    extra : dict
    """
    jax, jnp, optax, soft_modularity, soft_map_equation, two_dim_se, compute_pi = _jax_imports()

    nn = n_nodes(case)
    if nn > DIFF_DENSE_MAX_NODES:
        raise SkipMethod(
            f"{objective_tag} dense JAX optimisation capped at N<={DIFF_DENSE_MAX_NODES}; "
            f"this graph has N={nn}"
        )

    adj_dense = _to_dense(case.adjacency)
    adj_jax = jax.device_put(jnp.array(adj_dense))
    k = int(case.k or 2)

    # Stationary distribution (needed for map equation)
    if objective_tag == "Glass-Map":
        pi = compute_pi(adj_jax)
    else:
        pi = None

    def make_objective():
        if objective_tag == "Glass-SE":
            def obj(logits, temp):
                S = jax.nn.softmax(logits / temp, axis=-1)
                return two_dim_se(adj_jax, S, is_logits=False)
            sign = 1.0
        elif objective_tag == "Glass-Mod":
            def obj(logits, temp):
                S = jax.nn.softmax(logits / temp, axis=-1)
                return -soft_modularity(adj_jax, S, is_logits=False)  # negate to minimise
            sign = -1.0  # report actual Q (positive when good)
        elif objective_tag == "Glass-Map":
            def obj(logits, temp):
                S = jax.nn.softmax(logits / temp, axis=-1)
                return soft_map_equation(adj_jax, S, pi=pi, is_logits=False)
            sign = 1.0
        else:
            raise ValueError(objective_tag)
        return obj, sign

    obj_fn, sign = make_objective()

    # Spectral init (best single initialisation for structured graphs)
    from glass.solvers.spectral import spectral_embedding
    spectral_emb = spectral_embedding(adj_jax, k)
    spectral_init = jnp.array(spectral_emb) * 5.0

    key = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(
        lambda rk: jax.random.normal(rk, (nn, k)) * 0.1
    )(rng_keys)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)

    optimizer = optax.adam(lr)

    def optimise_single(logits_init):
        opt_state = optimizer.init(logits_init)

        def step(state, temp):
            logits, opt_st = state
            loss, grads = jax.value_and_grad(obj_fn)(logits, temp)
            updates, opt_st = optimizer.update(grads, opt_st)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_st), loss

        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]

        # Eval at low temperature
        eval_loss = obj_fn(final_logits, 0.01)
        return final_logits, eval_loss

    vmap_opt = jax.jit(jax.vmap(optimise_single))

    # JIT warmup
    t_jit_start = time.time()
    _ = vmap_opt(all_inits)
    jit_compile_seconds = time.time() - t_jit_start

    # Timed run
    t_start = time.time()
    all_logits, all_losses = vmap_opt(all_inits)
    all_logits.block_until_ready()
    runtime = time.time() - t_start

    best_idx = int(jnp.argmin(all_losses))
    best_logits = all_logits[best_idx]
    soft_obj = float(all_losses[best_idx]) * sign  # restore sign for reporting

    # Harden via argmax at T=0.01
    S_hard = jax.nn.softmax(best_logits / 0.01, axis=-1)
    labels = np.asarray(jnp.argmax(S_hard, axis=-1), dtype=np.int32)

    extra = {
        "soft_objective": soft_obj,
        "objective_name": objective_tag,
        "n_starts": n_starts,
        "n_iters": n_iters,
        "lr": lr,
        "jit_compile_seconds": round(jit_compile_seconds, 3),
    }
    return labels, runtime, soft_obj, extra


# ---------------------------------------------------------------------------
# SoftKMeans: differentiable k-means on spectral embedding
# ---------------------------------------------------------------------------


def run_soft_kmeans(
    case: DatasetCase,
    seed: int,
    n_iters: int = DEFAULT_N_ITERS,
    n_starts: int = DEFAULT_N_STARTS,
    lr: float = DEFAULT_LR,
) -> tuple[np.ndarray, float, float, dict]:
    """Differentiable soft k-means on adjacency spectral embedding.

    The spectral embedding (top-k Laplacian eigenvectors) is fixed.
    Soft assignment logits are optimised to minimise the temperature-annealed
    soft k-means objective:

        L(S) = sum_i sum_k S_ik * ||embed_i - mu_k(S)||^2

    where mu_k(S) = (S_:k . embed) / (sum_i S_ik + eps) is the soft centroid.
    This is the Bregman soft clustering relaxation (Banerjee et al. 2005).
    """
    jax, jnp, optax, *_ = _jax_imports()

    nn = n_nodes(case)
    if nn > DIFF_DENSE_MAX_NODES:
        raise SkipMethod(
            f"SoftKMeans dense spectral capped at N<={DIFF_DENSE_MAX_NODES}; N={nn}"
        )

    from glass.solvers.spectral import spectral_embedding

    adj_dense = _to_dense(case.adjacency)
    adj_jax = jax.device_put(jnp.array(adj_dense))
    k = int(case.k or 2)

    # Fixed spectral embedding of adjacency (top-K eigenvectors of D^-1/2 A D^-1/2)
    emb = spectral_embedding(adj_jax, k)  # (N, k)

    def soft_kmeans_loss(logits, temp):
        S = jax.nn.softmax(logits / temp, axis=-1)  # (N, K)
        S_sum = S.sum(axis=0)  # (K,)
        mu = (S.T @ emb) / (S_sum[:, None] + 1e-8)  # (K, d)
        dists = jnp.sum((emb[:, None, :] - mu[None, :, :]) ** 2, axis=-1)  # (N, K)
        return jnp.sum(S * dists)

    key = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(
        lambda rk: jax.random.normal(rk, (nn, k)) * 0.1
    )(rng_keys)
    # One spectral-guided init: assign each node to its nearest spectral centroid
    kmeans_init = jnp.array(emb) * 5.0  # (N, K)
    all_inits = jnp.concatenate([kmeans_init[None, ...], random_inits], axis=0)

    optimizer = optax.adam(lr)

    def optimise_single(logits_init):
        opt_state = optimizer.init(logits_init)

        def step(state, temp):
            logits, opt_st = state
            loss, grads = jax.value_and_grad(soft_kmeans_loss)(logits, temp)
            updates, opt_st = optimizer.update(grads, opt_st)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_st), loss

        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]
        eval_loss = soft_kmeans_loss(final_logits, 0.01)
        return final_logits, eval_loss

    vmap_opt = jax.jit(jax.vmap(optimise_single))

    t_jit_start = time.time()
    _ = vmap_opt(all_inits)
    jit_compile_seconds = time.time() - t_jit_start

    t_start = time.time()
    all_logits, all_losses = vmap_opt(all_inits)
    all_logits.block_until_ready()
    runtime = time.time() - t_start

    best_idx = int(jnp.argmin(all_losses))
    best_logits = all_logits[best_idx]
    soft_obj = float(all_losses[best_idx])

    S_hard = jax.nn.softmax(best_logits / 0.01, axis=-1)
    labels = np.asarray(jnp.argmax(S_hard, axis=-1), dtype=np.int32)

    extra = {
        "soft_objective": soft_obj,
        "objective_name": "SoftKMeans",
        "n_starts": n_starts,
        "n_iters": n_iters,
        "lr": lr,
        "jit_compile_seconds": round(jit_compile_seconds, 3),
        "embedding": "spectral_adjacency",
        "note": "Differentiable soft k-means on spectral embedding (Bregman relaxation)",
    }
    return labels, runtime, soft_obj, extra


# ---------------------------------------------------------------------------
# SpectralGrad: differentiable normalized cut (Shi & Malik 2000 relaxation)
# ---------------------------------------------------------------------------


def run_spectral_grad(
    case: DatasetCase,
    seed: int,
    n_iters: int = DEFAULT_N_ITERS,
    n_starts: int = DEFAULT_N_STARTS,
    lr: float = DEFAULT_LR,
) -> tuple[np.ndarray, float, float, dict]:
    """Gradient descent on the continuous normalized-cut relaxation.

    Objective:
        L(S) = trace(S^T L_sym S)   where L_sym = I - D^{-1/2} A D^{-1/2}

    with soft assignment S = softmax(logits / T).

    Unlike standard spectral clustering (which computes eigenvectors analytically),
    this is a fully differentiable continuous relaxation of the NCut cost optimised
    via gradient descent with temperature annealing (Shi & Malik 2000 relaxation
    driven by gradient-based search rather than eigenvector decomposition).
    """
    jax, jnp, optax, *_ = _jax_imports()

    nn = n_nodes(case)
    if nn > DIFF_DENSE_MAX_NODES:
        raise SkipMethod(
            f"SpectralGrad dense capped at N<={DIFF_DENSE_MAX_NODES}; N={nn}"
        )

    adj_dense = _to_dense(case.adjacency)
    adj_jax = jax.device_put(jnp.array(adj_dense))
    k = int(case.k or 2)

    # Normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    d = adj_jax.sum(axis=-1)  # (N,)
    d_inv_sqrt = 1.0 / jnp.sqrt(d + 1e-8)
    # L_sym = I - d_inv_sqrt * A * d_inv_sqrt (broadcast)
    L_sym = jnp.eye(nn) - d_inv_sqrt[:, None] * adj_jax * d_inv_sqrt[None, :]
    L_sym = jax.device_put(L_sym)

    def ncut_loss(logits, temp):
        S = jax.nn.softmax(logits / temp, axis=-1)  # (N, K)
        # NCut = trace(S^T L_sym S)
        LS = L_sym @ S  # (N, K)
        return jnp.sum(S * LS)  # trace(S^T L_sym S)

    # Spectral init: Laplacian eigenvectors  (via adjacency spectral embedding)
    from glass.solvers.spectral import spectral_embedding
    emb = spectral_embedding(adj_jax, k)
    spectral_init = jnp.array(emb) * 5.0

    key = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(key, n_starts - 1)
    random_inits = jax.vmap(
        lambda rk: jax.random.normal(rk, (nn, k)) * 0.1
    )(rng_keys)
    all_inits = jnp.concatenate([spectral_init[None, ...], random_inits], axis=0)

    optimizer = optax.adam(lr)

    def optimise_single(logits_init):
        opt_state = optimizer.init(logits_init)

        def step(state, temp):
            logits, opt_st = state
            loss, grads = jax.value_and_grad(ncut_loss)(logits, temp)
            updates, opt_st = optimizer.update(grads, opt_st)
            logits = optax.apply_updates(logits, updates)
            return (logits, opt_st), loss

        temps = jnp.linspace(1.0, 0.01, n_iters)
        final_state, _ = jax.lax.scan(step, (logits_init, opt_state), temps)
        final_logits = final_state[0]
        eval_loss = ncut_loss(final_logits, 0.01)
        return final_logits, eval_loss

    vmap_opt = jax.jit(jax.vmap(optimise_single))

    t_jit_start = time.time()
    _ = vmap_opt(all_inits)
    jit_compile_seconds = time.time() - t_jit_start

    t_start = time.time()
    all_logits, all_losses = vmap_opt(all_inits)
    all_logits.block_until_ready()
    runtime = time.time() - t_start

    best_idx = int(jnp.argmin(all_losses))
    best_logits = all_logits[best_idx]
    soft_obj = float(all_losses[best_idx])

    S_hard = jax.nn.softmax(best_logits / 0.01, axis=-1)
    labels = np.asarray(jnp.argmax(S_hard, axis=-1), dtype=np.int32)

    extra = {
        "soft_objective": soft_obj,
        "objective_name": "SpectralGrad",
        "n_starts": n_starts,
        "n_iters": n_iters,
        "lr": lr,
        "jit_compile_seconds": round(jit_compile_seconds, 3),
        "note": (
            "Continuous normalised-cut relaxation (Shi & Malik 2000) "
            "minimised via gradient descent with temperature annealing"
        ),
    }
    return labels, runtime, soft_obj, extra


# ---------------------------------------------------------------------------
# Evaluation helpers (same schema as benchmark_same_protocol.py)
# ---------------------------------------------------------------------------


def evaluate_labels(
    case: DatasetCase,
    labels: np.ndarray,
    method: str,
    seed: int,
    runtime_seconds: float,
    status: str = "ok",
    extra: dict | None = None,
) -> dict:
    labels = canonicalize_labels(labels)
    row: dict[str, Any] = {
        "block": BLOCK_NAME,
        "dataset": case.name,
        "method": method,
        "seed": int(seed),
        "status": status,
        "source": case.source,
        "n_nodes": n_nodes(case),
        "n_edges": n_edges(case),
        "true_k": int(case.k) if case.k is not None else None,
        "pred_k": int(np.unique(labels).size),
        "runtime_seconds": float(runtime_seconds),
        "peak_rss_mb": rss_mb(),
    }
    if case.labels is not None:
        row["acc"] = clustering_accuracy(case.labels, labels)
        row["nmi"] = normalized_mutual_info(case.labels, labels)
        row["ari"] = adjusted_rand_index(case.labels, labels)
    try:
        row["modularity"] = hard_modularity(case.adjacency, labels)
    except Exception:
        row["modularity"] = None
    try:
        row["mean_conductance"] = mean_conductance(case.adjacency, labels)
    except Exception:
        row["mean_conductance"] = None
    try:
        if isinstance(case.adjacency, SparseGraph) or sp.issparse(case.adjacency):
            row["structural_entropy"] = float(sparse_structural_entropy(case.adjacency, labels))
        else:
            row["structural_entropy"] = float(structural_entropy(case.adjacency, labels))
    except Exception:
        row["structural_entropy"] = None
    try:
        row["map_equation"] = hard_map_equation(case.adjacency, labels)
    except Exception:
        row["map_equation"] = None
    if extra:
        row.update(extra)
    return row


def skip_row(case: DatasetCase, method: str, seed: int, reason: str) -> dict:
    return {
        "block": BLOCK_NAME,
        "dataset": case.name,
        "method": method,
        "seed": int(seed),
        "status": f"skipped: {reason}",
        "source": case.source,
        "n_nodes": n_nodes(case) if case.adjacency is not None else None,
        "n_edges": n_edges(case) if case.adjacency is not None else None,
        "true_k": int(case.k) if case.k is not None else None,
        "pred_k": None,
        "runtime_seconds": None,
        "peak_rss_mb": rss_mb(),
    }


def failed_row(case: DatasetCase, method: str, seed: int, exc: Exception) -> dict:
    row = skip_row(case, method, seed, f"failed: {type(exc).__name__}: {exc}")
    row["status"] = f"failed: {type(exc).__name__}: {exc}"
    return row


# ---------------------------------------------------------------------------
# Dispatch: run a single differentiable method
# ---------------------------------------------------------------------------


def run_diff_method(
    case: DatasetCase,
    method: str,
    seed: int,
    n_iters: int,
    n_starts: int,
    lr: float,
    dry_run: bool = False,
) -> dict:
    if case.adjacency is None or case.labels is None or case.k is None:
        return skip_row(case, method, seed, case.source or "missing adjacency/labels/k")
    if dry_run:
        return skip_row(case, method, seed, "dry run")

    t0 = time.time()
    try:
        if method in {"Glass-SE", "Glass-Mod", "Glass-Map"}:
            labels, runtime, soft_obj, extra = run_glass_differentiable(
                case, method, seed, n_iters=n_iters, n_starts=n_starts, lr=lr
            )
        elif method == "SoftKMeans":
            labels, runtime, soft_obj, extra = run_soft_kmeans(
                case, seed, n_iters=n_iters, n_starts=n_starts, lr=lr
            )
        elif method == "SpectralGrad":
            labels, runtime, soft_obj, extra = run_spectral_grad(
                case, seed, n_iters=n_iters, n_starts=n_starts, lr=lr
            )
        else:
            raise ValueError(f"Unknown differentiable method: {method}")

        row = evaluate_labels(case, labels, method, seed, runtime, extra=extra)
    except SkipMethod as exc:
        row = skip_row(case, method, seed, str(exc))
    except Exception as exc:
        row = failed_row(case, method, seed, exc)

    row["wall_seconds_including_scoring"] = float(time.time() - t0)
    return row


# ---------------------------------------------------------------------------
# Load discrete topology rows from an existing block-A JSON
# ---------------------------------------------------------------------------


def load_discrete_rows(
    json_path: Path,
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
) -> list[dict]:
    """Re-tag matching topology rows as block=differentiable for merging."""
    if not json_path.exists():
        print(f"[warn] --reuse-discrete path not found: {json_path}", flush=True)
        return []

    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    raw_rows: list[dict] = payload.get("rows", [])
    selected: list[dict] = []
    for row in raw_rows:
        if (
            row.get("block") in {"topology", "differentiable"}
            and row.get("dataset") in datasets
            and row.get("method") in methods
            and row.get("seed") in seeds
        ):
            retagged = dict(row)
            retagged["block"] = BLOCK_NAME
            retagged["_reused_from_block"] = row.get("block", "topology")
            retagged["_reused_from_file"] = str(json_path)
            selected.append(retagged)

    unique_keys = set((r["dataset"], r["method"], r["seed"]) for r in selected)
    print(
        f"[reuse] Loaded {len(selected)} discrete rows covering "
        f"{len(unique_keys)} (dataset, method, seed) triples from {json_path}",
        flush=True,
    )
    return selected


# ---------------------------------------------------------------------------
# Aggregation (mirrors benchmark_same_protocol.aggregate_rows)
# ---------------------------------------------------------------------------

_NUMERIC_KEYS = [
    "acc", "nmi", "ari",
    "modularity", "mean_conductance", "structural_entropy", "map_equation",
    "soft_objective",
    "runtime_seconds", "wall_seconds_including_scoring", "peak_rss_mb",
    "pred_k",
]


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["block"], row["dataset"], row["method"])
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for (block, dataset, method), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "block": block,
            "dataset": dataset,
            "method": method,
            "n_runs": len(group),
            "n_ok": sum(1 for r in group if str(r.get("status")) in {"ok", "baseline_executed"}),
            "statuses": sorted(set(str(r.get("status")) for r in group)),
            "n_nodes": group[0].get("n_nodes"),
            "n_edges": group[0].get("n_edges"),
            "true_k": group[0].get("true_k"),
            "is_differentiable": group[0].get("objective_name") is not None,
            "_reused_from_block": group[0].get("_reused_from_block"),
        }
        for key_name in _NUMERIC_KEYS:
            values = [r.get(key_name) for r in group if isinstance(r.get(key_name), (int, float))]
            if values:
                arr = np.asarray(values, dtype=float)
                item[f"{key_name}_mean"] = float(arr.mean())
                item[f"{key_name}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(item)
    return out


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "---"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_mean(row: dict, key: str, digits: int = 3) -> str:
    mean = row.get(f"{key}_mean")
    if mean is None:
        return "---"
    std = row.get(f"{key}_std", 0.0) or 0.0
    if abs(float(std)) > 1e-12:
        return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"
    return f"{float(mean):.{digits}f}"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    rows: list[dict],
    cfg_meta: dict,
    output_dir: Path,
    also_copy_to_canonical: bool = True,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"same_protocol_differentiable_{timestamp}"

    raw_path = output_dir / f"{prefix}.json"
    agg_path = output_dir / f"{prefix}_aggregated.json"
    md_path = output_dir / f"{prefix}.md"

    aggregated = aggregate_rows(rows)

    payload = {
        "metadata": {
            "timestamp": timestamp,
            "block": BLOCK_NAME,
            **cfg_meta,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    agg_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- Markdown report ----
    nmi_best: dict[str, str] = {}
    se_best: dict[str, str] = {}
    for dataset in sorted(set(r["dataset"] for r in aggregated)):
        cands_nmi = [r for r in aggregated if r["dataset"] == dataset and r.get("nmi_mean") is not None]
        if cands_nmi:
            nmi_best[dataset] = max(cands_nmi, key=lambda r: float(r["nmi_mean"]))["method"]
        cands_se = [r for r in aggregated if r["dataset"] == dataset and r.get("structural_entropy_mean") is not None]
        if cands_se:
            se_best[dataset] = min(cands_se, key=lambda r: float(r["structural_entropy_mean"]))["method"]

    methods_shown = sorted(set(r["method"] for r in aggregated))
    diff_methods = [m for m in methods_shown if m in GLASS_DIFF_METHODS]
    disc_methods = [m for m in methods_shown if m in DISCRETE_REUSE_METHODS]

    lines = [
        f"# Block E: Differentiable Non-Attributed Glass Clustering ({timestamp})",
        "",
        "Protocol: topology-only, no node features; differentiable methods use "
        "dense JAX adjacency (capped at N<=5000); discrete baselines reused from "
        "topology block JSON (same seeds, same preprocessing).",
        "",
        "## Method Legend",
        "",
        "**Differentiable (gradient-based, our contribution and seminal baselines):**",
        "- `Glass-SE`: differentiable 2D structural-entropy H2 minimisation (ours)",
        "- `Glass-Mod`: differentiable modularity Q maximisation (ours)",
        "- `Glass-Map`: differentiable map-equation L minimisation (ours)",
        "- `SoftKMeans`: temperature-annealed soft k-means on spectral embedding "
        "  (Bregman relaxation; Banerjee et al. 2005)",
        "- `SpectralGrad`: continuous normalised-cut relaxation (Shi & Malik 2000) "
        "  minimised via gradient descent instead of eigenvectors",
        "",
        "**Discrete (reused from Block A topology block):**",
        "- `Louvain`, `Leiden`, `Infomap`: modularity/flow community detection",
        "- `Spectral`: standard spectral clustering (sklearn, eigenvector-based)",
        "- `SEClust-ConstrainedK`: discrete structural-entropy heuristic (ours)",
        "",
        f"Datasets: {', '.join(cfg_meta.get('datasets', []))}",
        f"Differentiable methods: {', '.join(diff_methods)}",
        f"Discrete baselines: {', '.join(disc_methods)}",
        f"Seeds: {', '.join(str(s) for s in cfg_meta.get('seeds', []))}",
        "",
        "## Results",
        "",
        "Bold = best mean NMI per dataset.",
        "",
        "| Dataset | Method | Type | OK/N | ACC | NMI | ARI | K | Q | SE | L | "
        "Soft-Obj | Time(s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    # Group aggregated rows by dataset for display
    datasets_order = [d for d in BLOCK_E_DATASETS if d in set(r["dataset"] for r in aggregated)]
    method_order = GLASS_DIFF_METHODS + DISCRETE_REUSE_METHODS

    for dataset in datasets_order:
        ds_rows = [r for r in aggregated if r["dataset"] == dataset]
        ds_rows_sorted = sorted(
            ds_rows,
            key=lambda r: (
                method_order.index(r["method"]) if r["method"] in method_order else 999,
            ),
        )
        for row in ds_rows_sorted:
            method = row["method"]
            is_diff = row.get("is_differentiable") or method in GLASS_DIFF_METHODS
            mtype = "diff" if is_diff else "disc"
            display_method = f"**{method}**" if nmi_best.get(dataset) == method else method
            status = "; ".join(row["statuses"])[:60]
            soft_obj_str = fmt_mean(row, "soft_objective")
            lines.append(
                "| "
                + " | ".join([
                    dataset,
                    display_method,
                    mtype,
                    f"{row['n_ok']}/{row['n_runs']}",
                    fmt_mean(row, "acc"),
                    fmt_mean(row, "nmi"),
                    fmt_mean(row, "ari"),
                    fmt_mean(row, "pred_k", 1),
                    fmt_mean(row, "modularity"),
                    fmt_mean(row, "structural_entropy"),
                    fmt_mean(row, "map_equation"),
                    soft_obj_str,
                    fmt_mean(row, "runtime_seconds", 2),
                ])
                + " |"
            )

    lines += [
        "",
        "## Best-By-Dataset",
        "",
        f"- Best mean NMI: {json.dumps(nmi_best, sort_keys=True)}",
        f"- Best mean structural entropy (lower): {json.dumps(se_best, sort_keys=True)}",
        "",
        "## Files",
        "",
        f"- Raw seed-level JSON: `{raw_path}`",
        f"- Aggregated JSON: `{agg_path}`",
        "",
        "## Notes on Soft-Obj Column",
        "",
        "For Glass-SE and SpectralGrad: lower is better (minimisation).",
        "For Glass-Mod: higher is better (sign-corrected to show Q, not -Q).",
        "For Glass-Map: lower is better.",
        "For SoftKMeans: lower is better (within-cluster variance).",
        "Discrete baselines (Louvain, Leiden, Infomap, Spectral, SEClust-ConstrainedK) "
        "do not use a differentiable objective; the Soft-Obj column shows `---`.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Copy to canonical locations ----
    if also_copy_to_canonical:
        repo_root = Path(__file__).parent.parent
        canon_raw = repo_root / "docs/results/same_protocol/raw"
        canon_agg = repo_root / "docs/results/same_protocol/aggregated"
        canon_rep = repo_root / "docs/results/same_protocol/reports"
        for d in (canon_raw, canon_agg, canon_rep):
            d.mkdir(parents=True, exist_ok=True)
        import shutil

        def _copy_if_different(src: Path, dst: Path) -> None:
            if src.resolve() != dst.resolve():
                shutil.copy2(str(src), str(dst))

        _copy_if_different(raw_path, canon_raw / raw_path.name)
        _copy_if_different(agg_path, canon_agg / agg_path.name)
        _copy_if_different(md_path, canon_rep / md_path.name)
        print(f"Canonical raw JSON: {canon_raw / raw_path.name}", flush=True)
        print(f"Canonical aggregated: {canon_agg / agg_path.name}", flush=True)
        print(f"Canonical report: {canon_rep / md_path.name}", flush=True)

    return raw_path, agg_path, md_path


def write_checkpoint(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "same_protocol_differentiable_checkpoint.json"
    payload = {
        "metadata": {"checkpoint": True, "rows_completed": len(rows)},
        "rows": rows,
    }
    ckpt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value or value.strip().lower() in {"", "default"}:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        default=None,
        help=f"Comma-separated datasets. Default: {','.join(BLOCK_E_DATASETS)}",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help=f"Comma-separated differentiable methods to RUN. Default: {','.join(GLASS_DIFF_METHODS)}. "
        "Discrete baselines are added automatically from --reuse-discrete.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--reuse-discrete",
        default=None,
        metavar="JSON_PATH",
        help="Path to a topology block JSON file from which to reuse discrete "
        "baseline rows (Louvain, Leiden, Infomap, Spectral, SEClust-ConstrainedK).",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/results/same_protocol/raw",
        help="Directory for raw JSON + aggregated JSON + Markdown report.",
    )
    parser.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS)
    parser.add_argument("--n-starts", type=int, default=DEFAULT_N_STARTS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Small smoke test: 2 seeds, 100 iters, 2 starts.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    datasets = parse_csv_list(args.datasets, BLOCK_E_DATASETS)
    diff_methods = parse_csv_list(args.methods, GLASS_DIFF_METHODS)

    n_iters = args.n_iters
    n_starts = args.n_starts
    lr = args.lr

    if args.quick:
        seeds = seeds[:2]
        datasets = [d for d in datasets if d in {"Karate", "SBM-Easy"}] or datasets[:1]
        diff_methods = [m for m in diff_methods if m in {"Glass-SE", "Glass-Mod", "SoftKMeans"}]
        n_iters = 100
        n_starts = 2

    output_dir = Path(args.output_dir)

    print(
        f"Block E – Differentiable clustering | "
        f"datasets={datasets} | diff_methods={diff_methods} | seeds={seeds}",
        flush=True,
    )

    all_rows: list[dict] = []

    # 1. Load discrete baseline rows from existing topology block JSON
    if args.reuse_discrete:
        discrete_rows = load_discrete_rows(
            Path(args.reuse_discrete),
            datasets=datasets,
            methods=DISCRETE_REUSE_METHODS,
            seeds=seeds,
        )
        all_rows.extend(discrete_rows)
    else:
        print(
            "[info] No --reuse-discrete path provided; discrete baselines will not "
            "be included. Pass --reuse-discrete <path-to-topology-json> to merge them.",
            flush=True,
        )

    # 2. Run differentiable methods
    total = len(datasets) * len(diff_methods) * len(seeds)
    completed = 0

    for dataset_name in datasets:
        for seed in seeds:
            case = load_case(dataset_name, seed=seed, quick=args.quick)
            print(
                f"[dataset] {case.name} seed={seed} source={case.source} "
                f"N={n_nodes(case) if case.adjacency is not None else 'NA'}",
                flush=True,
            )
            for method in diff_methods:
                completed += 1
                print(
                    f"[{completed}/{total}] {case.name} / {method} / seed={seed}",
                    flush=True,
                )
                row = run_diff_method(
                    case,
                    method,
                    seed,
                    n_iters=n_iters,
                    n_starts=n_starts,
                    lr=lr,
                    dry_run=args.dry_run,
                )
                all_rows.append(row)
                write_checkpoint(all_rows, output_dir)
                status = row.get("status", "?")
                t = row.get("runtime_seconds")
                t_str = f"{t:.2f}s" if isinstance(t, float) else "---"
                print(f"  -> {status} time={t_str}", flush=True)

    # 3. Write outputs
    cfg_meta = {
        "datasets": datasets,
        "diff_methods": diff_methods,
        "discrete_methods_reused": DISCRETE_REUSE_METHODS,
        "seeds": seeds,
        "n_iters": n_iters,
        "n_starts": n_starts,
        "lr": lr,
        "quick": args.quick,
        "dense_cap_nodes": DIFF_DENSE_MAX_NODES,
        "reuse_discrete_source": args.reuse_discrete,
    }
    raw_path, agg_path, md_path = write_outputs(
        all_rows, cfg_meta, output_dir, also_copy_to_canonical=(output_dir != output_dir)
    )
    # Always also copy to canonical
    repo_root = Path(__file__).parent.parent
    canon_raw = repo_root / "docs/results/same_protocol/raw"
    canon_agg = repo_root / "docs/results/same_protocol/aggregated"
    canon_rep = repo_root / "docs/results/same_protocol/reports"
    for d in (canon_raw, canon_agg, canon_rep):
        d.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(raw_path), str(canon_raw / raw_path.name))
    shutil.copy2(str(agg_path), str(canon_agg / agg_path.name))
    shutil.copy2(str(md_path), str(canon_rep / md_path.name))

    print(f"\nSaved raw JSON      : {raw_path}", flush=True)
    print(f"Saved aggregated    : {agg_path}", flush=True)
    print(f"Saved Markdown      : {md_path}", flush=True)
    print(f"Canonical raw       : {canon_raw / raw_path.name}", flush=True)
    print(f"Canonical report    : {canon_rep / md_path.name}", flush=True)
    print("\nBlock E done.", flush=True)


if __name__ == "__main__":
    main()
