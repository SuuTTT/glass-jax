# Result Artifact Layout

Canonical result artifacts are organized by protocol rather than by creation time.

## Same-Protocol Results

- `same_protocol/raw/`: raw seed-level JSON payloads.
- `same_protocol/aggregated/`: aggregate JSON files grouped by block, dataset, and method.
- `same_protocol/reports/`: human-readable Markdown reports.
- `same_protocol/checkpoints/`: long-run checkpoints; use these only for recovery, not paper tables.

Current paper-facing reports:

- `same_protocol/reports/same_protocol_gap_closure_20260511.md`
- `same_protocol/reports/same_protocol_benchmark_progress_20260511.md`

Current paper-facing merged JSON:

- `same_protocol/raw/same_protocol_gap_closure_20260511.json`

## Legacy Results

`legacy/` contains older benchmark artifacts from before the same-protocol harness. These may be useful for diagnosis or appendix context, but should not support main paper claims unless explicitly rerun under the same protocol.

## Compatibility Path

The harness still writes to `docs/experimental_reports/` by default. Files are copied into this organized hierarchy for paper-facing references.
