# Changelog

All notable changes to the `glass-jax` project are documented here, synchronized with the repository's commit history.

## [2026-05-08] - Current Sprint
### Added
- **Leiden Baseline Integration**: Integrated `leidenalg` and `igraph` into the benchmark suite for comparison against state-of-the-art modularity methods.
- **Improved Metrics Reporting**: Updated tables to show `Graph (N, E, True K*)` and added footnotes explaining parameter-free algorithm behavior and skip reasons.
- **Project Governance**: Created `CHANGELOG.md` and `TODO.md` to track publication roadmap.

## [2026-05-07] - Transition to SEClust
### Added
- **High-Level Hierarchy**: Implemented hierarchical structural entropy clustering in `src/glass/seclust/hierarchy.py`.
- **Benchmark Expansion**: Created `tests/benchmark_seclust_full.py` and ran comprehensive evaluations.
- **Official Baselines**: Successfully integrated and uploaded `SEP` (Structural Entropy Partition) and `LSEnet` baselines for comparison.
- **Real-World Evaluation**: Executed benchmarks on real-world topology datasets (Cora, Citeseer).

### Changed
- **Rebranding**: Renamed the primary algorithm and internal modules from **Glass-SE** to **SEClust**.
- **Discrete Algorithms**: Shifted from soft-continuous relaxation to discrete heuristic optimization for better stability.

### Fixed
- **Baselines Upload**: Corrected directory structure and file issues for `SEP` and `LSEnet`.

## [2026-05-06] - Project Inception (Glass-SE)
### Added
- **Glass-SE Core**: Initial implementation of Structural Entropy (SE) objectives and solvers in JAX.
- **Design & Architecture**: Initial design phase with Gemini, establishing the project's mathematical and architectural foundation.
- **Initial Commit**: Repository initialization.
