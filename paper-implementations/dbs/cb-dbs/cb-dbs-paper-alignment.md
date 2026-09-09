# CB-DBS Paper Alignment Audit

Paper:

- Je-Ho Lee, Jan P. Allebach
- *Colorant-based direct binary search halftoning*
- Journal of Electronic Imaging, 2002

Implementation under audit:

- [cb_dbs.py](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/CISLAB/HALFTONING/paper-implementations/dbs/cb-dbs/cb_dbs.py)

## Verified Matches

- `C'`, `M'`, and `CM` preprocessing follows the paper's piecewise definitions:
  - `C' = C`, `M' = M` when `C + M <= 1`
  - `C' = 1 - M`, `M' = 1 - C` when `C + M > 1`
- `CM` total-dot pattern is generated before color reassignment.
- `C/M` initialization uses random thresholding with probability `C' / (C' + M')`.
- `C/M` refinement is swap-only, constrained to preserve `g_C' + g_M' = g_CM`.
- `C/M` neighborhood is `7x7`, matching the paper.
- `Y` is halftoned independently.
- The local trial-update machinery computes exact squared-error deltas against the filtered error field.
- Final `C` and `M` reconstruction follows the paper's `C or B` / `M or B` rule.

## Fixed During Audit

- Blue overlap reconstruction now follows Eq. (5): zero pixels of `g_CM` become `B` when `C + M >= 1`.
- Default pass caps were raised from the earlier speed shortcut to `10` with early stopping, matching the paper's statement that convergence typically takes about `10` iterations.
- Tests were added for:
  - preprocessing equations
  - exact local delta-energy evaluation
  - preservation of the `g_C' + g_M' = g_CM` constraint
  - binary final-plane reconstruction
  - blue-overlap reconstruction from zero `g_CM` pixels at or above `C + M = 1`

## Remaining Limits

These are paper-fidelity limits that are not fully eliminable from the paper text alone:

- The paper delegates the exact monochrome DBS setup to Refs. `25/26`. The current implementation uses a standard toggle-and-swap monochrome DBS objective with the same filtered squared-error form, but the paper itself does not print that full monochrome algorithm.
- The paper's experimental RGB-to-CMY conversion used an HP printer-driver color map. This repository implementation uses direct `RGB -> CMY` complement because that printer profile is not available locally.
- The paper references Näsänen-model HVS filtering but does not print a discrete kernel table. The implementation uses an explicit Näsänen-style discretization with `S = 2000`.

## Practical Conclusion

The implementation is now structurally faithful to the algorithm described in the paper, with the main remaining gaps caused by external dependencies the paper does not fully specify in-line:

- printer calibration / color map
- exact referenced monochrome DBS implementation details
- exact HVS discretization choice
