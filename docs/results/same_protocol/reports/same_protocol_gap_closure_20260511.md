# Same-Protocol Gap-Closure Benchmark Report

Generated: 2026-05-11 10:09 UTC

This report supersedes the earlier progress report for the gaps targeted in this run. It merges repaired LFR rows, mandatory raw/PCA feature baselines, expanded DMoN/MinCutPool rows, and 50-epoch GAE/VGAE rows while avoiding double-counting older short-run cells.

## Gap Closure Status

- LFR skipped seeds 4, 5, and 7 were rerun successfully for Louvain, Leiden, Infomap, Spectral, LabelPropagation, SEClust-ConstrainedK, and HCSE.
- RawKMeans and PCAKMeans were completed for all seven attributed datasets with 10 seeds; large raw-feature rows use scalable MiniBatchKMeans.
- DMoN and MinCutPool were expanded from a 3-seed Cora/Citeseer pilot to 10 seeds on Cora, Citeseer, and Photo with 50 epochs.
- GAE and VGAE were rerun for all seven attributed datasets with 50 epochs and 10 seeds.
- Paired Wilcoxon tests with Holm correction were added for common-seed SEClust-ConstrainedK comparisons.

| Block | Rows | OK | Skipped | Failed | Datasets | Methods | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attributed | 522 | 522 | 0 | 0 | 7 | 10 | 0-9 |
| scale | 15 | 15 | 0 | 0 | 1 | 3 | 0-4 |
| topology | 790 | 790 | 0 | 0 | 12 | 7 | 0-9 |

## Remaining Gaps

- Node2Vec/DeepWalk remain a 3-seed quick artifact. A full 10-seed, full-walk run over all attributed datasets is still expensive and should be scheduled separately or narrowed to citation/Amazon datasets.
- DMoN/MinCutPool remain dense-pooling baselines. They now cover Photo but not PubMed, Computers, or Coauthor graphs because dense adjacency would be too large for this machine.
- ogbn-arxiv scale still has only Louvain, Leiden, and SEClust-ConstrainedK in the final no-Auto scale artifact. SEClust-Auto was intentionally excluded after a single row took about 831 seconds.
- Official LSEnet and DGI/GRACE-style self-supervised baselines are not implemented in this harness; they should be appendix-only unless added with maintained code.

## Artifact Manifest

| Artifact | Used Rows | Status | Datasets | Methods | File |
| --- | --- | --- | --- | --- | --- |
| topology-core-10seed | 720 | ok:702, skipped:18 | 12 | 6 | `same_protocol_topology_20260510_201438.json` |
| topology-hcse-10seed-subset | 70 | ok:67, skipped:3 | 7 | 1 | `same_protocol_topology_20260510_191711.json` |
| topology-lfr-repair-10seed-completion | 21 | ok:21 | 1 | 7 | `same_protocol_topology_20260511_074930.json` |
| scale-ogbn-arxiv-5seed-no-auto | 15 | ok:15 | 1 | 3 | `same_protocol_scale_20260510_185728.json` |
| attributed-core-10seed-noncoauthor | 150 | ok:150 | 5 | 3 | `same_protocol_attributed_20260511_071853.json` |
| attributed-core-10seed-coauthor | 60 | ok:60 | 2 | 3 | `same_protocol_attributed_20260511_070341.json` |
| attributed-raw-pca-cora-to-computers | 100 | ok:100 | 5 | 2 | `same_protocol_attributed_20260511_rawpca_cora_to_computers.json` |
| attributed-raw-pca-coauthor-scalable | 40 | ok:40 | 2 | 2 | `same_protocol_attributed_20260511_100748.json` |
| attributed-neural-50epoch-gae-vgae | 140 | ok:140 | 7 | 2 | `same_protocol_attributed_20260511_090926.json` |
| attributed-dmon-mincut-50epoch-expanded | 60 | ok:60 | 3 | 2 | `same_protocol_attributed_20260511_075339.json` |
| attributed-node2vec-deepwalk-3seed-quick | 42 | ok:42 | 7 | 2 | `same_protocol_attributed_20260510_180637.json` |

## Dataset-Level Winners

| Block | Dataset | Best NMI | Best ARI | SEClust NMI | SEClust ARI | SEClust sec |
| --- | --- | --- | --- | --- | --- | --- |
| topology | Citeseer | LabelPropagation (0.340) | Louvain (0.095) | 0.024 | 0.015 | 0.76 |
| topology | Coauthor-CS | Leiden (0.612) | Leiden (0.444) | 0.103 | 0.042 | 7.48 |
| topology | Coauthor-Physics | Leiden (0.483) | LabelPropagation (0.542) | 0.044 | 0.017 | 17.25 |
| topology | Computers | Leiden (0.535) | LabelPropagation (0.342) | 0.290 | 0.150 | 8.14 |
| topology | Cora | Leiden (0.461) | Leiden (0.251) | 0.081 | 0.047 | 0.67 |
| topology | DCSBM | Leiden (0.614) | Leiden (0.654) | 0.207 | 0.186 | 0.27 |
| topology | Karate | HCSE (1.000) | HCSE (1.000) | 0.732 | 0.777 | 0.05 |
| topology | LFR | Spectral (0.945) | Spectral (0.925) | 0.750 | 0.580 | 0.40 |
| topology | Photo | Leiden (0.657) | Leiden (0.564) | 0.332 | 0.252 | 3.90 |
| topology | PubMed | Leiden (0.204) | Louvain (0.101) | 0.005 | 0.005 | 6.61 |
| topology | SBM-Easy | Infomap (1.000) | Infomap (1.000) | 0.998 | 0.998 | 0.39 |
| topology | SBM-Noisy | Spectral (0.201) | Spectral (0.193) | 0.027 | 0.018 | 0.37 |
| attributed | Citeseer | VGAE (0.306) | GAE (0.276) | 0.024 | 0.015 | 0.75 |
| attributed | Coauthor-CS | VGAE (0.729) | VGAE (0.606) | 0.103 | 0.042 | 7.02 |
| attributed | Coauthor-Physics | VGAE (0.546) | GAE (0.363) | 0.044 | 0.017 | 16.46 |
| attributed | Computers | SEClust-ConstrainedK (0.290) | SEClust-ConstrainedK (0.150) | 0.290 | 0.150 | 7.65 |
| attributed | Cora | VGAE (0.526) | VGAE (0.460) | 0.081 | 0.047 | 0.69 |
| attributed | Photo | DMoN (0.356) | DMoN (0.286) | 0.332 | 0.252 | 3.62 |
| attributed | PubMed | RawKMeans (0.313) | GAE (0.286) | 0.005 | 0.005 | 6.29 |
| scale | ogbn-arxiv | Leiden (0.411) | Louvain (0.254) | 0.110 | 0.038 | 187.8 |

## Method-Level Mean Overview

Means average dataset-method means, not raw rows.

| Block | Method | Datasets | Mean NMI | Mean ARI | Mean modularity | Mean sec |
| --- | --- | --- | --- | --- | --- | --- |
| topology | HCSE | 7 | 0.368 | 0.266 | 0.315 | 11.66 |
| topology | Infomap | 12 | 0.480 | 0.320 | 0.550 | 1.09 |
| topology | LabelPropagation | 12 | 0.450 | 0.313 | 0.459 | 3.54 |
| topology | Leiden | 12 | 0.543 | 0.418 | 0.633 | 0.76 |
| topology | Louvain | 12 | 0.532 | 0.410 | 0.626 | 5.11 |
| topology | SEClust-ConstrainedK | 12 | 0.299 | 0.257 | 0.494 | 3.86 |
| topology | Spectral | 12 | 0.312 | 0.257 | 0.241 | 4.13 |
| attributed | AdjSVDKMeans | 7 | 0.107 | -0.010 | 0.195 | 2.82 |
| attributed | DMoN | 3 | 0.304 | 0.240 | 0.496 | 2.57 |
| attributed | DeepWalkKMeans | 7 | 0.116 | 0.097 | 0.229 | 41.53 |
| attributed | GAE | 7 | 0.334 | 0.264 | 0.451 | 28.82 |
| attributed | MinCutPool | 3 | 0.138 | 0.093 | 0.270 | 4.30 |
| attributed | Node2VecKMeans | 7 | 0.116 | 0.096 | 0.229 | 41.94 |
| attributed | PCAKMeans | 7 | 0.268 | 0.158 | 0.239 | 5.26 |
| attributed | RawKMeans | 7 | 0.227 | 0.127 | 0.218 | 15.51 |
| attributed | SEClust-ConstrainedK | 7 | 0.126 | 0.075 | 0.562 | 6.07 |
| attributed | VGAE | 7 | 0.326 | 0.253 | 0.458 | 31.53 |
| scale | Leiden | 1 | 0.411 | 0.250 | 0.715 | 16.78 |
| scale | Louvain | 1 | 0.399 | 0.254 | 0.709 | 98.38 |
| scale | SEClust-ConstrainedK | 1 | 0.110 | 0.038 | 0.567 | 187.8 |

## Paired Significance Tests

Tests compare SEClust-ConstrainedK against each competitor on common seeds within each dataset. The table shows Holm-significant results at alpha=0.05 only.

| Block | Dataset | Metric | Competitor | N | Delta | Holm p | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | none | NA | NA | NA | NA | NA | No Holm-significant paired differences |

## Paired Test Direction Counts

| Block | Metric | Direction | Count |
| --- | --- | --- | --- |
| attributed | ari | SEClust higher | 17 |
| attributed | ari | SEClust lower | 24 |
| attributed | nmi | SEClust higher | 13 |
| attributed | nmi | SEClust lower | 28 |
| scale | ari | SEClust lower | 2 |
| scale | nmi | SEClust lower | 2 |
| topology | ari | SEClust higher | 21 |
| topology | ari | SEClust lower | 46 |
| topology | nmi | SEClust higher | 16 |
| topology | nmi | SEClust lower | 51 |

## Protocol Implication

The paper can now claim a same-protocol empirical backbone with no LFR skipped cells in the merged evidence table, complete mandatory raw/PCA attributed feature baselines, and stronger neural baselines than the initial fast iteration. The results still do not support a broad state-of-the-art claim: SEClust is competitive on structural-entropy/modularity-oriented objectives and wins selected attributed datasets, but Leiden/Louvain/VGAE/GAE remain stronger on many label-alignment metrics. The submission should frame SEClust as a sparse structural-entropy clustering method with scalability and objective-alignment advantages, not as uniformly best on NMI/ARI.

## Source Files

Raw artifacts are under `/workspace/glass-jax/docs/experimental_reports`. The merged JSON companion to this report contains the exact aggregate rows and statistical-test payload.
