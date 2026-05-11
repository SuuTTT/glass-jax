# SEClust — Continuation Context

Single-page handoff for picking up the SEClust / TPAMI work cold.
Last updated 2026-05-10 (post idea-015 ogbn-arxiv landing,
appendix-with-full-proofs draft, submit-ready paper version).

## Two repos

| Repo | Purpose | Branch | Last commit topic |
| --- | --- | --- | --- |
| `/workspace/glass-jax` | Code: `glass.seclust` package + benchmarks | `main` | tree clustering, ogbn-arxiv runner |
| `/workspace/SEClust-paper` | LaTeX paper + `idea_lib/` provenance | `main` | §VII theory, §VI-G ogbn-arxiv, Appendix A proofs |

The paper repo is the artifact-of-truth for any narrative/claim;
the code repo is the artifact-of-truth for any number that ends up
in a table.

## What this work is

A **TPAMI submission** on **discrete structural-entropy clustering**.
The pitch the paper makes:

1. **Algorithmic.** A sparse incremental SE optimiser
   (`IncrementalSEState`) with $O(\deg v)$ per-node moves and a
   `numba`-JIT'd kernel (36× speedup). Four variants share the loss
   but differ in how $K$ is picked: Auto (free), Tree (hierarchical),
   TargetK (post-merge), MultiLevel (coarsen-refine), and the
   constrained-K multistart (the new headline variant from idea 007).
2. **Empirical.** Beats HCSE on Dasgupta cost / dendrogram purity on
   every real-world graph; matches Leiden ACC on synthetic SBMs;
   constrained-K rescues the TargetK collapse on Photo
   (Q $0.001 \to 0.663$). Scales to **ogbn-arxiv ($N\!=\!169{,}343$)**
   in tractable wallclock — the largest discrete-SE demonstration in
   the literature.
3. **Theoretical.** Two new theorems in §VII anchor the empirical
   gains in **structural information theory** and *make this a TPAMI
   paper* rather than just a TKDE paper:
   - **Theorem 2** (*constrained-K SE distortion lower bound*):
     $\delta_K(G) \geq \rho_K(G)\log_2(K^*/K) - o(1)$. Validated at
     two orders of magnitude in $N$ (prefactor $\rho_K \approx 0.20$
     on Photo, $0.21$ on ogbn-arxiv).
   - **Theorem 4** (*SE/$Q$ equivalence in the dense regime*):
     $\Delta H_2 = -c(P,G)\Delta Q + O(d_{\max}/V_G)$. Explains why
     the lookahead and hybrid-α policy ablations are noops on Cora.
   - **Theorem 3** (*algorithmic match on planted SBMs*) wraps the
     two together to give a recovery threshold for SEClust-ConstrainedK.

   **Appendix A** contains the full proofs (added 2026-05-10). The
   `theory.tex` body keeps tight sketches that point at the appendix.

## Current empirical results (5-seed where reported)

| Dataset | $N$ | Best ACC | Best NMI | Best ARI | SEClust win |
| --- | ---: | ---: | ---: | ---: | --- |
| Karate | 34 | 1.000 | — | — | tied |
| Caveman | 200 | 1.000 | — | — | tied |
| SBM-100/500/1000 | 100/500/1000 | 1.000 | — | — | exact (TargetK + MultiLevel) |
| Cora | 2,708 | **0.551**±0.035 | 0.395 | **0.325**±0.033 | Glass-SE GNN (idea 002) |
| Citeseer | 3,327 | **0.414**±0.036 | — | — | Glass-SE GNN |
| Photo | 7,650 | 0.512±0.050 (ConstrainedK) | — | — | best Q at K=8 |
| **ogbn-arxiv** | **169,343** | 0.114 (ConstrainedK, K=40) | — | — | scale demo |

Hierarchical (Dasgupta cost / DP, real-world):

| Dataset | SEClust-Tree | HCSE | Improvement |
| --- | ---: | ---: | ---: |
| Cora | $3.6\!\times\!10^6$ | $1.4\!\times\!10^7$ | **3.9×** |
| Citeseer | (similar) | — | **12.6×** |
| Photo | wins | wins | — |

## Repo state — paper

- **25 pages, zero undefined references** as of the last `pdflatex`
  pass.
- Section structure: abstract, intro, related, preliminaries,
  design space (§IV from idea 017), method (with constrained-K
  subsection), theory (§VII, sketches), experiments (with §VI-G
  ogbn-arxiv subsection), comparison, discussion, conclusion,
  **Appendix A (full proofs)**.
- `IEEEtran` journal class (TPAMI/TKDE compatible). Title line
  reads: *"SEClust: Sparse Incremental Structural Entropy Clustering
  with Coding-Tree and Target-K Variants."*
- `refs.bib` includes `li2016structural`, `pan2021hcse`,
  `sun2024lsenet`, `zeng2025hypcse`, `traag2019leiden`,
  `blondel2008louvain`, `rosvall2008mapequation`, `dasgupta2016cost`,
  `hu2020ogb`. **No undefined cite keys.**
- `idea_lib/idealist.md` is the prioritised backlog; `STATUS.md` is
  the strategic summary; `README.md` is the catalog of 14 numbered
  ideas (001–020 with gaps).

### Compiling the paper

```bash
cd /workspace/SEClust-paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# → main.pdf
```

## Repo state — code

- **`src/glass/seclust/`**:
  - `incremental.py` — `IncrementalSEState`, `move_delta`,
    `multistart_incremental_se_heuristic`, `constrained_k_multistart`,
    `local_move_incremental(alpha)` (hybrid SE/Q).
  - `numba_kernel.py` — `@njit(cache=True, fastmath=True)` move-delta
    kernel; **36× microbench, 7× end-to-end on Photo**. This is the
    production fast path.
  - `jit_kernel.py` — JAX-JIT version (kept for GPU port).
  - `lookahead.py` — MPC, TD bootstrap, TD($\lambda$), adaptive-w,
    Boltzmann, hybrid α policies.
  - `multilevel.py` — coarsen-refine wrapper with monotone check.
  - `coding_tree.py`, `hierarchy.py` — high-dimensional tree builder.
  - `metrics.py` — `dasgupta_cost`, `dendrogram_purity`.
  - `sync_kernel.py` — synchronous batched local-move (idea 018,
    JAX-GPU port WIP).
- **`tests/`**:
  - `benchmark_seclust_full.py` — main 5-seed bench (Karate, Caveman,
    SBMs, Cora, Citeseer, Photo). Variants: Auto, Tree, TargetK,
    MultiLevel, ConstrainedK. ~1h 02m wallclock with numba.
  - `benchmark_hierarchical.py` — Dasgupta + DP scoring.
  - `benchmark_ogbn_arxiv.py` — standalone ogbn-arxiv runner.
  - `lookahead_testbed.py` — 7 small graphs for fast iteration.
  - `aggregated_to_latex.py` — JSON → LaTeX tables.

### Running benchmarks

```bash
cd /workspace/glass-jax
python tests/benchmark_seclust_full.py --seeds 0,7,17,23,42
python tests/benchmark_hierarchical.py
python tests/benchmark_ogbn_arxiv.py    # ~25 min
# Outputs land in docs/experimental_reports/<name>_<ts>.{json,md}
```

## Idea catalog cheat sheet

001–014 are tactical (lookahead, GNN, multilevel, hier. metrics,
multi-seed, JIT, constrained-K, spectral seeding, hybrid loss,
adaptive-w, Boltzmann, hybrid move, Dasgupta-as-objective,
continuous Euclidean SE).

015 is **ogbn-arxiv scale demo** — landed.

017 is **paper-framework unification** — landed (gave us §IV design
space and the theorem-anchored §VII).

018 is **synchronous batched kernel** — partial; numpy version ships,
JAX-GPU `compute_AP` step validated 3.2× over numpy on Photo, full
GPU pipeline is the half-day follow-up.

019, 020, 016 are the **theory-track ideas**: distortion invariant,
SE/Q equivalence, SBM recovery. Sketches in §VII; **full proofs in
Appendix A**.

## What's done (TPAMI scorecard)

| Milestone | Status |
| --- | --- |
| Tier 1: speed + win on every real-world hierarchical metric | ✅ |
| 5-seed re-run with mean±std reporting | ✅ |
| Constrained-K multistart fixes TargetK collapse | ✅ |
| §IV design-space framework unification | ✅ |
| §VII theorems 1–4 + corollary (sketches) | ✅ |
| **Appendix A full proofs** | ✅ |
| §VI-G ogbn-arxiv scale demo + Theorem 2 validation | ✅ |
| Submit-ready abstract, contributions, IEEE keywords | ✅ |

## What's still open

| Item | Effort | Priority |
| --- | --- | --- |
| 5-seed run on ogbn-arxiv | ~2 h wallclock | nice-to-have |
| JAX-GPU full pipeline (idea 018) | ~half day | would cut ogbn-arxiv 21 min → 1 min |
| Idea 013: Dasgupta cost as the optimiser objective | M | closes scoring/optimising loop |
| Idea 014: continuous Euclidean SE variant | L | only if 016/019/020 fail review |
| Glass-SE GNN on ogbn-arxiv (uses node features) | M | might beat Leiden's ACC=0.374 |

The TPAMI submission package is **complete enough to submit**;
the open items above are reviewer-response material, not blockers.

## How to continue

The user's pattern is `next idea` / `yes` to push through
`idealist.md` Tier-3 work autonomously. The next backlog item per
`idealist.md` is **idea 013 (Dasgupta-cost as optimiser objective)**.
Alternatively the user may want one of:
- **idea 018 follow-up** (JAX-GPU port to flip the GPU-vs-numba
  result) — the explainer doc at
  `idea_lib/_explainer_K_local_numba_gpu.md` documents why the
  current GPU port is slower.
- **a 5-seed ogbn-arxiv run** for tighter confidence intervals.
- **a strict-form proof refinement of §VII Theorem 1's monotonicity
  for non-balanced SBMs** (the appendix proves the balanced case).

Always read `idea_lib/STATUS.md` and `idealist.md` first; that's
where the strategy lives.

## User preferences (durable, from prior sessions)

- **Push autonomously** — *"avoid interrupt and ask for my approval,
  just keep work until finish it"*. After landing an idea, propose
  the next one as the default action.
- **TPAMI target.** The user explicitly aimed at TPAMI in May 2026
  and asked for new theory in **structural information theory** (this
  is what 019/020/016 + Appendix A are for).
- **Both repos must move together** — code commit and paper commit
  per landed idea.
- **Honest negative results count.** Mixed/failed ideas (008, 009,
  010, 012, 018-partial) stay in the catalog as documented evidence,
  not buried.

## File index — the ones to read first

| If you want to know… | Read |
| --- | --- |
| Where the paper stands | `/workspace/SEClust-paper/main.pdf` (compile if missing) |
| What every idea did | `/workspace/SEClust-paper/idea_lib/README.md` |
| The TPAMI plan | `/workspace/SEClust-paper/idea_lib/idealist.md` |
| The strategic summary | `/workspace/SEClust-paper/idea_lib/STATUS.md` |
| The full theorem proofs | `/workspace/SEClust-paper/sections/appendix.tex` |
| The optimiser code | `/workspace/glass-jax/src/glass/seclust/incremental.py` |
| The JIT kernel | `/workspace/glass-jax/src/glass/seclust/numba_kernel.py` |
| The headline ogbn-arxiv result | `/workspace/SEClust-paper/idea_lib/015_ogbn_arxiv_sweep/README.md` |
