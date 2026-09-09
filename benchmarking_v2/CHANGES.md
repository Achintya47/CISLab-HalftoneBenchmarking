# What changed, and why

This document is the delta between the existing per-method
`paper-implementations/*` + `benchmarking/*` pipeline and this new
`benchmarking_v2/` package. Nothing in `benchmarking/` or
`paper-implementations/` was modified — `benchmarking_v2` is purely
additive and imports the existing adapters.

## 1. Reconstruction operator was inconsistent across the codebase — unified

| Location | Reconstruction used |
|---|---|
| `GAED.evaluate()` | Gaussian blur, `sigma=1.0` |
| `efficient_halftoning_drl` `stabilized` variant reward | Gaussian, `sigma=1.5` |
| `efficient_halftoning_drl` `paper` variant reward | Näsänen kernel, `S=2000` |
| `cb_dbs` / `hcb_dbs` viewing blur | Näsänen kernel, `S=2000` |
| `benchmarking/metrics.py::viewed()` (used by `benchmarking/run.py`) | Näsänen kernel, `S=2000` |
| **`benchmark_specifications.pdf`, Sec. 5.2** | **Gaussian blur, `sigma=1.2`, full stop** |
| **`benchmarking_v2/reconstruction.py`** | **Gaussian blur, `sigma=1.2`, always** |

`benchmarking_v2` never touches the algorithm-internal sigmas/kernels above
— those remain whatever each algorithm was designed/trained against, since
they're optimization-time choices. But every metric this package reports
(`PSNR`, `SSIM`, `LPIPS`, `ΔE00`) reconstructs through the single spec-mandated
operator, so numbers are comparable across DBS / Error Diffusion / Ordered
Dithering / Deep Learning regardless of what HVS model each one trains
against internally. This was the specific inconsistency called out in the
request.

## 2. Anisotropy Index scoped correctly

Spec Sec. 5.1: *"All full-reference metrics **except anisotropy** will be
computed on a reconstructed continuous-tone image."* `metrics.py`
enforces this: `anisotropy_index()` always takes the raw halftone, never
`reconstruct(halftone)`. This mirrors the ring-variance-of-power-spectrum
style anisotropy scoring already used ad hoc in several
`paper-implementations/*` diagnostic scripts, but standardized into a single
implementation with a single normalization.

## 3. A 4th algorithm family was missing — added

The spec (Sec. 1) requires DBS, Error Diffusion, Ordered Dithering, and
Deep-Learning Halftoning. The repository had implementations for the first,
second, and fourth (in two variants each), but **no Ordered Dithering
implementation at all**. `algorithms/ordered_dithering.py` adds a from-scratch
Bayer-matrix ordered ditherer (grayscale + per-channel color), wired into
the registry the same way as everything else.

## 4. Dataset fetch + stress-variant generation was per-method, ad hoc, Kodak-only — generalized

The existing benchmark (`benchmarking/run.py`) is hardcoded to
`datasets/kodak/images/kodim*.png` and a single constant-gray synthetic
diagnostic. Nothing in the repo implements the spec's 5-content-family
sampling or Table-1 stress variants. `datasets.py` + `stress_variants.py`
add:

- stratified sampling across the 5 content families (`edge_text`,
  `scenery_gradient`, `texture`, `color_skin`, `pattern`), each mapped to
  the Kaggle dataset keys named in the spec (Sec. 8 links), with a
  local-copy-first / kagglehub-best-effort / synthetic-fallback loading
  chain (see `README.md`);
- all 15 Table-1 stress-variant definitions (3 per family), each carrying
  the exact numeric parameters from Table 1 (blur lengths/sigmas, JPEG
  qualities, bit depths, contrast/saturation levels, rotation angle, line
  widths, etc.), centralized in `spec.py` so a spec revision is a one-line
  constant change rather than a hunt through algorithm code.

## 5. Checkpoint policy generalized for the RL row

`benchmarking/adapters.py`'s `EfficientDRLAdapter` (unmodified, reused
as-is) hard-fails if a checkpoint file is missing — appropriate for its
original "never silently substitute a random model for a released result"
design goal. `benchmarking_v2/checkpoints.py` adds a resolution ladder in
front of that constructor (explicit path -> released checkpoint -> cached
local bootstrap -> short bootstrap-train-then-cache) so a benchmark run
never has to hard-fail or manually re-run 200k-step training just to
produce a Deep-Learning row, while still preferring a real trained/released
checkpoint whenever one is available, and always recording in `run.json` /
adapter metadata which path was actually used.

## 6. Metric surface expanded to match the spec's two tracks

`benchmarking/metrics.py` computes `viewed_mse/psnr/ssim` (Nasanen-based)
and CIEDE2000; it has no LPIPS and no Anisotropy Index computation of its
own (those exist only as one-off scripts under specific
`paper-implementations/*` diagnostics). `benchmarking_v2/metrics.py` adds:

- PSNR/SSIM against the spec's Gaussian `R(H)`,
- optional LPIPS (`pip install lpips`; degrades to `NaN` + a recorded reason
  rather than crashing the run if unavailable — matches the "not in
  requirements.txt today" reality without silently dropping the column),
- a single canonical Anisotropy Index implementation,
- CIEDE2000, reused conceptually from `benchmarking/metrics.py::color_metrics`
  but recomputed against the sigma=1.2 reconstruction instead of the Nasanen one.

## 7. Everything is registry-based, not hardcoded, for extension

`benchmarking/run.py` has an `if name == "..."` chain for adapters and a
single fixed dataset root. `benchmarking_v2` uses three independent
registries (`algorithms.ALGORITHM_REGISTRY`, `spec.FAMILY_VARIANTS` /
`stress_variants.FAMILY_VARIANT_FUNCS`, `datasets.KAGGLE_SLUGS`) so adding a
method, a dataset, or a stress variant never requires touching
`run_benchmark.py`. See `EXTENDING.md`.

## 8. Fixed a numba disk-cache crash triggered by dynamic module loading

Reported crash (Windows, running `benchmarking_v2` against a full repo
checkout where the DBS test suite had already been run at some point):

```
ModuleNotFoundError: No module named 'cb_dbs'
```

Root cause (reproduced and regression-tested, see
`tests/test_numba_dynamic_load_workaround.py`): `cb_dbs.py` marks
`_evaluate_local_update_impl` with `@njit(cache=True)`. Numba's on-disk
cache for that function lives next to the source file and is keyed by
(source file, function name, numba version) — but the cached artifact
embeds whatever **import name** was active the first time it was ever
compiled on that machine, and reloading it later calls
`importlib.import_module(that_baked_in_name)`. `benchmarking.adapters
._load_module` (unmodified, reused as-is) always loads `cb_dbs.py` under a
synthetic name (`halftoning_bench_cb_dbs`); if the file was ever compiled
under its plain name `cb_dbs` in an earlier, unrelated process (e.g.
running `paper-implementations/dbs/cb-dbs/test_cb_dbs.py` directly, which
does `from cb_dbs import ...`), the resulting stale disk-cache entry
permanently poisons every later dynamic load — regardless of machine, OS,
or dataset, as long as the stale `__pycache__/*.nbi/.nbc` files exist next
to `cb_dbs.py`.

Fix: `algorithms/build_dbs` wraps `CBDBSAdapter(...)` construction in
`_numba_compat.force_no_disk_cache()`, which monkeypatches `numba.njit` for
the duration of the (one-time) module load to force `cache=False`. This
means the hot loop is always freshly JIT-compiled in-process — never
round-tripped through numba's disk cache at all — so no stale cache can
ever be read (or written) via this project's adapters. One-time
recompilation cost is roughly 100-300ms per process; negligible next to a
benchmark run. `paper-implementations/dbs/cb-dbs/cb_dbs.py` itself is
untouched — the fix lives entirely on the loading side in
`benchmarking_v2`.

## 9. Small/fast test runs, and skipping Deep Learning without a checkpoint

The default config runs the full ~50-image protocol with no progress
output, which on a CPU-only machine (a) takes a while and (b) can look
hung. Two additions:

- `run_benchmark.py` now logs progress per algorithm build and per
  evaluated item (`[build] dbs ready (0.1s)`, `[12/200] dbs edge_text base
  psnr=26.51 (3.7s)`, a per-method timing line, and a final `[done]` line
  with total elapsed time), so a long run is visibly alive. `--quiet`
  suppresses it.
- New CLI overrides (`--images-per-family`, `--only`, `--disable`, `--seed`,
  `--no-lpips`) let you shrink a run or drop the Deep-Learning method
  without hand-editing TOML, e.g.
  `--images-per-family 2 --disable deep_learning --no-lpips`. `--disable`
  raises immediately with the valid method list if given an unknown name,
  rather than silently no-op'ing.
- New `config/quick_test.toml`: a ready-made 10-base-image config with
  `[methods.deep_learning].enabled = false`, for a fast first run without
  needing a checkpoint or paying the bootstrap-training cost at all.

`tests/test_benchmark_v2_smoke.py::test_cli_overrides_scale_down_and_disable_methods`
exercises exactly this combination (small image count + Deep Learning
excluded) as a regression test.

## 10. Added visual comparison panels (original vs. binary halftone vs. reconstruction)

Nothing previously rendered a visual sanity-check of a halftone next to its
reconstruction; `benchmarking_v2/visualize.py` + wiring in `run_benchmark.py`
add, per run:

- one PNG per method (`comparisons/<method>_comparison.png`) with up to
  `comparisons_per_method` (default 5) sample rows of
  Original / Halftone (binary) / Reconstructed (the exact `R(H)` used for
  every reported PSNR/SSIM/LPIPS number), sampled to spread across content
  families and adapting down automatically if fewer items exist;
- one bonus cross-method panel (`comparisons/all_methods_comparison.png`)
  putting every enabled method's halftone/reconstruction of the *same*
  sample image side by side.

Sample selection (`visualize.select_sample_item_indices`) and image capture
are unit- and integration-tested (`tests/test_visualize.py`,
`tests/test_benchmark_v2_smoke.py`).

## Section 7 (print-and-scan) — out of scope

Explicitly out of scope per the request; `run_benchmark.py` has no
print/scan step. If it's ever needed, it would slot in as a post-processing
stage consuming the same `metrics.csv` item identifiers, reported
separately per spec Sec. 7.3 ("not collapsed into a single scalar").
