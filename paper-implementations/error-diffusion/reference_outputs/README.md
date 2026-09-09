# GAED Reference Outputs

This folder preserves the final benchmark and adaptive-branch diagnostics generated during the GAED paper-fidelity review on `2026-04-08`.

Only the final retained outputs are kept here. Earlier intermediate runs and superseded variants were removed.

## Benchmark

- `benchmark_kodak_full2`
  Final full-resolution 2-image Kodak benchmark using the paper-aligned stronger edge trigger and strict boundary handling.

## Adaptive-Branch Diagnostics

- `adaptive_branch_kodim01`
  Final crop-level adaptive-branch analysis on `kodim01` corresponding to the retained benchmark configuration.

## Key Files

Inside `benchmark_kodak_full2`:
- `summary.json`: aggregate metrics and trend deltas
- `metrics.csv`: per-image, per-variant metrics
- `*_halftone.png`: rendered halftones
- `*_reconstructed.png`: Gaussian reconstructions

Inside `adaptive_branch_kodim01`:
- `summary.json`: crop-level adaptive activation and `R`-ranking statistics
- `adaptive_branch_panel.png`: visual debug panel
