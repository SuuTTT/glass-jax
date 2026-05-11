# Same-Protocol Benchmark Progress Report

Generated: 2026-05-11 07:21 UTC

This report consolidates the completed fast-iteration benchmark artifacts for the SEClust journal revision. It is intended as a run-status and evidence-quality report, not the final camera-ready experimental section.

## Executive Status

The benchmark now has a clean same-protocol backbone across topology-only, attributed, and scale settings. The strongest completed evidence is the 10-seed topology core and the 10-seed attributed core; both include SEClust-ConstrainedK and recognized baselines under fixed seeds and shared metrics. The scale run covers ogbn-arxiv with 5 seeds and excludes SEClust-Auto because the first Auto scale row took about 831 seconds.

| Block | Rows | OK | Skipped | Failed | Datasets | Methods | Seed range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attributed | 334 | 334 | 0 | 0 | 7 | 8 | 0-9 |
| scale | 15 | 15 | 0 | 0 | 1 | 3 | 0-4 |
| topology | 790 | 769 | 21 | 0 | 12 | 7 | 0-9 |

## Artifact Manifest

| Artifact | Rows | Status | Datasets | Methods | Seeds | File |
| --- | --- | --- | --- | --- | --- | --- |
| topology-core-10seed | 720 | ok:702, skipped:18 | 12 | 6 | 0-9 | `same_protocol_topology_20260510_201438.json` |
| topology-hcse-10seed-subset | 70 | ok:67, skipped:3 | 7 | 1 | 0-9 | `same_protocol_topology_20260510_191711.json` |
| scale-ogbn-arxiv-5seed-no-auto | 15 | ok:15 | 1 | 3 | 0-4 | `same_protocol_scale_20260510_185728.json` |
| attributed-core-10seed-noncoauthor | 150 | ok:150 | 5 | 3 | 0-9 | `same_protocol_attributed_20260511_071853.json` |
| attributed-core-10seed-coauthor | 60 | ok:60 | 2 | 3 | 0-9 | `same_protocol_attributed_20260511_070341.json` |
| attributed-vgae-10seed | 70 | ok:70 | 7 | 1 | 0-9 | `same_protocol_attributed_20260510_173038.json` |
| attributed-node2vec-deepwalk-3seed | 42 | ok:42 | 7 | 2 | 0-2 | `same_protocol_attributed_20260510_180637.json` |
| attributed-dmon-mincut-3seed-pilot | 12 | ok:12 | 2 | 2 | 0-2 | `same_protocol_attributed_20260510_153329.json` |

## Dataset-Level Winners

NMI and ARI are the primary label-aware clustering metrics here. SEClust columns show SEClust-ConstrainedK when present; `NA` means the method was not run in that block/dataset artifact.

| Block | Dataset | Best NMI | Best ARI | SEClust NMI | SEClust ARI | SEClust sec |
| --- | --- | --- | --- | --- | --- | --- |
| topology | Citeseer | LabelPropagation (0.340) | Louvain (0.095) | 0.024 | 0.015 | 0.76 |
| topology | Coauthor-CS | Leiden (0.612) | Leiden (0.444) | 0.103 | 0.042 | 7.48 |
| topology | Coauthor-Physics | Leiden (0.483) | LabelPropagation (0.542) | 0.044 | 0.017 | 17.25 |
| topology | Computers | Leiden (0.535) | LabelPropagation (0.342) | 0.290 | 0.150 | 8.14 |
| topology | Cora | Leiden (0.461) | Leiden (0.251) | 0.081 | 0.047 | 0.67 |
| topology | DCSBM | Leiden (0.614) | Leiden (0.654) | 0.207 | 0.186 | 0.27 |
| topology | Karate | HCSE (1.000) | HCSE (1.000) | 0.732 | 0.777 | 0.05 |
| topology | LFR | Infomap (0.963) | Infomap (0.972) | 0.769 | 0.626 | 0.35 |
| topology | Photo | Leiden (0.657) | Leiden (0.564) | 0.332 | 0.252 | 3.90 |
| topology | PubMed | Leiden (0.204) | Louvain (0.101) | 0.005 | 0.005 | 6.61 |
| topology | SBM-Easy | Infomap (1.000) | Infomap (1.000) | 0.998 | 0.998 | 0.39 |
| topology | SBM-Noisy | Spectral (0.201) | Spectral (0.193) | 0.027 | 0.018 | 0.37 |
| attributed | Citeseer | VGAE (0.251) | VGAE (0.201) | 0.024 | 0.015 | 0.75 |
| attributed | Coauthor-CS | VGAE (0.604) | VGAE (0.471) | 0.103 | 0.042 | 7.02 |
| attributed | Coauthor-Physics | VGAE (0.543) | VGAE (0.441) | 0.044 | 0.017 | 16.46 |
| attributed | Computers | SEClust-ConstrainedK (0.290) | SEClust-ConstrainedK (0.150) | 0.290 | 0.150 | 7.65 |
| attributed | Cora | VGAE (0.460) | VGAE (0.369) | 0.081 | 0.047 | 0.69 |
| attributed | Photo | SEClust-ConstrainedK (0.332) | SEClust-ConstrainedK (0.252) | 0.332 | 0.252 | 3.62 |
| attributed | PubMed | DeepWalkKMeans (0.078) | DeepWalkKMeans (0.079) | 0.005 | 0.005 | 6.29 |
| scale | ogbn-arxiv | Leiden (0.411) | Louvain (0.254) | 0.110 | 0.038 | 187.8 |

## Win Counts

| Scope | Winner counts |
| --- | --- |
| attributed:ARI | VGAE:4, SEClust-ConstrainedK:2, DeepWalkKMeans:1 |
| attributed:NMI | VGAE:4, SEClust-ConstrainedK:2, DeepWalkKMeans:1 |
| scale:ARI | Louvain:1 |
| scale:NMI | Leiden:1 |
| topology:ARI | Leiden:4, Louvain:2, LabelPropagation:2, Infomap:2, HCSE:1, Spectral:1 |
| topology:NMI | Leiden:7, Infomap:2, LabelPropagation:1, HCSE:1, Spectral:1 |

## Method-Level Mean Overview

Means below average dataset-method means, not raw rows, so large-seed artifacts do not dominate solely by row count.

| Block | Method | Datasets | Mean NMI | Mean ARI | Mean modularity | Mean sec |
| --- | --- | --- | --- | --- | --- | --- |
| topology | HCSE | 7 | 0.369 | 0.268 | 0.317 | 11.36 |
| topology | Infomap | 12 | 0.487 | 0.327 | 0.554 | 1.09 |
| topology | LabelPropagation | 12 | 0.455 | 0.318 | 0.462 | 3.54 |
| topology | Leiden | 12 | 0.546 | 0.422 | 0.635 | 0.76 |
| topology | Louvain | 12 | 0.534 | 0.415 | 0.628 | 5.11 |
| topology | SEClust-ConstrainedK | 12 | 0.301 | 0.261 | 0.496 | 3.85 |
| topology | Spectral | 12 | 0.314 | 0.261 | 0.243 | 4.12 |
| attributed | AdjSVDKMeans | 7 | 0.107 | -0.010 | 0.195 | 2.82 |
| attributed | DMoN | 2 | 0.088 | 0.056 | 0.320 | 0.32 |
| attributed | DeepWalkKMeans | 7 | 0.116 | 0.097 | 0.229 | 41.53 |
| attributed | GAE | 7 | 0.238 | 0.151 | 0.292 | 6.58 |
| attributed | MinCutPool | 2 | 0.027 | 0.015 | 0.237 | 0.49 |
| attributed | Node2VecKMeans | 7 | 0.116 | 0.096 | 0.229 | 41.94 |
| attributed | SEClust-ConstrainedK | 7 | 0.126 | 0.075 | 0.562 | 6.07 |
| attributed | VGAE | 7 | 0.279 | 0.218 | 0.306 | 7.99 |
| scale | Leiden | 1 | 0.411 | 0.250 | 0.715 | 16.78 |
| scale | Louvain | 1 | 0.399 | 0.254 | 0.709 | 98.38 |
| scale | SEClust-ConstrainedK | 1 | 0.110 | 0.038 | 0.567 | 187.8 |

## Current Interpretation

The current fast-iteration data is sufficient to support protocol design decisions and to identify where SEClust is competitive, but it is not yet enough for a final TPAMI/TKDE/TNNLS submission claim. The main issue is not missing attributed-core rows anymore; it is evidence strictness: several neural baselines are intentionally short-trained, Node2Vec/DeepWalk are only 3-seed quick runs, and DMoN/MinCut are only a 3-seed pilot on Cora/Citeseer.

SEClust-ConstrainedK is now fairly compared against classical graph baselines on topology datasets and against `AdjSVDKMeans`/`GAE`/`VGAE` on attributed datasets with common seeds. The scale result establishes feasibility on ogbn-arxiv for constrained-k SEClust but does not yet establish a full large-scale superiority claim because only Louvain, Leiden, and SEClust-ConstrainedK were included in the final no-Auto scale run.

## Remaining Journal-Quality Gaps

- Replace quick neural settings with final settings: increase GAE/VGAE epochs and run Node2Vec/DeepWalk with full walk/training budgets where feasible.
- Expand DMoN and MinCutPool beyond the current 3-seed Cora/Citeseer pilot or explicitly classify them as secondary/pilot baselines.
- Decide whether HCSE remains in the main table or appendix. It completed on seven topology datasets but was not run for larger real attributed graphs.
- Resolve LFR generation instability for seeds 4, 5, and 7. Those rows are currently marked skipped in topology artifacts rather than failed.
- Add statistical testing for the final tables: paired Wilcoxon or paired t-test across seeds, with Holm correction, plus effect sizes.
- Freeze final protocol in the paper: dataset versions, preprocessing, seed list, metrics, hardware, stopping criteria, and exact baseline hyperparameters.

## Recommended Next Runs

- Full neural attributed pass: `VGAE`, `GAE`, `Node2VecKMeans`, and `DeepWalkKMeans` with final training budgets on the seven attributed datasets.
- Scale appendix pass: ogbn-arxiv with at least Louvain, Leiden, SEClust-ConstrainedK, and one embedding baseline if runtime is acceptable.
- Robustness appendix: sensitivity to `K`, SEClust starts/passes, and ablations for the structural-entropy objective.

## Source Artifacts

All referenced raw JSON and Markdown artifacts are under `/workspace/glass-jax/docs/experimental_reports`. The paper-side protocol document is `/workspace/SEClust-paper/SAME_PROTOCOL_EXPERIMENT_PROTOCOL.md`.
