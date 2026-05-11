# Same-Protocol Benchmark Results

This directory is the canonical result layout for the SEClust paper benchmark.

## Subdirectories

- `raw/`: seed-level JSON files, including merged reports with full row payloads.
- `aggregated/`: aggregated JSON files produced by the benchmark harness.
- `reports/`: Markdown summaries generated from raw and aggregate files.
- `checkpoints/`: recoverable long-run checkpoints. Do not cite these as final results.

## Primary Paper Artifacts

- `reports/same_protocol_gap_closure_20260511.md`
- `raw/same_protocol_gap_closure_20260511.json`
- `reports/same_protocol_benchmark_progress_20260511.md`
- `raw/same_protocol_benchmark_progress_20260511.json`

## Compatibility

The benchmark harness still writes first to `docs/experimental_reports/`. The artifacts here are synchronized copies with a stable hierarchy for paper references.
