# Halftoning Benchmark v2

A generalized dataset -> stress-variant -> benchmark -> report pipeline that
implements `benchmark_specifications.pdf` on top of the existing
`paper-implementations/*` algorithm code, plus one algorithm that the repo
was missing (Ordered Dithering).

This package is additive: it does not modify `benchmarking/` or any
`paper-implementations/*` file. It sits next to them and reuses their
adapters directly (`benchmarking.adapters.{CBDBSAdapter, GAEDAdapter,
EfficientDRLAdapter}`).

## Run it

```bash
python -m benchmarking_v2.run_benchmark --config benchmarking_v2/config/default.toml
```

This prints live progress (`[build] dbs ready (0.1s)`, `[12/200] dbs edge_text base psnr=26.51 (3.7s)`, ...) so a run in progress is never silent. Use `--quiet` to suppress it and only print the final output path.

### Quick / small test runs (no DRL checkpoint needed)

The default config is the full ~50-image / 200-item protocol, which is
slow on CPU (DBS in particular). For a fast local sanity check with fewer
images and Deep Learning skipped entirely (no checkpoint, no bootstrap
training cost):

```bash
# Option A: a ready-made small config (10 base images, DRL disabled)
python -m benchmarking_v2.run_benchmark --config benchmarking_v2/config/quick_test.toml

# Option B: override the default config from the CLI instead
python -m benchmarking_v2.run_benchmark \
  --images-per-family 2 \
  --disable deep_learning \
  --no-lpips
```

Other CLI overrides (all optional, all composable, none require editing the TOML):

| Flag | Effect |
|---|---|
| `--images-per-family N` | overrides every `[families.*].count`; `N=2` → 10 base images, `N=1` → 5 |
| `--only METHOD [METHOD ...]` | run only these method keys from `[methods.*]` |
| `--disable METHOD [METHOD ...]` | disable specific method keys, e.g. `--disable deep_learning`; errors out with the valid method list if you typo a name |
| `--seed N` | override the config's seed |
| `--no-lpips` | force-disable LPIPS even if the config enables it |
| `--output-dir PATH` | write results elsewhere |
| `--quiet` | suppress the per-item progress log |

`--only` and `--disable` both refer to your config's `[methods.*]` table
keys (the ones on the left of `[methods.dbs]`, `[methods.deep_learning]`,
etc.), not the underlying `algorithm =` registry name, so they work even if
you've renamed/duplicated a method entry.

Outputs land in `<output_dir>/`:

- `manifest.json` — exactly which images (or synthetic stand-ins) were sampled per family
- `metrics.csv` — one row per (method, image, variant): every metric, both tracks
- `summary.json` — per-method means, plus a `_by_variant` breakdown
- `leaderboard.md` — a Table-2-style leaderboard
- `run.json` — provenance: git commit, python/platform, reconstruction sigma, whether any synthetic fallback images were used, LPIPS availability

## What this run actually evaluates

Per the request ("50 images from 5 datasets, 3 stress variants each"), the
default config samples **10 base images per content family x 5 families =
50 base images**, applies that family's **3** Table-1 stress variants to
each (`spec.FAMILY_VARIANTS`), giving **150 stress items**, for **200
evaluation items** total per method. This is an exact 1/3 scale-down of the
full spec protocol (150 base / 450 stress items) — same structure, smaller
N, both configurable in `config/default.toml` (`[families.<name>].count`).

Section 7 (print-and-scan) is explicitly out of scope, as requested.

## Algorithms benchmarked (4 families, per spec Sec. 1)

| Family | Adapter | Source |
|---|---|---|
| Direct Binary Search | `algorithms.build_dbs` | reuses `benchmarking.adapters.CBDBSAdapter` -> `paper-implementations/dbs/cb-dbs/cb_dbs.py` |
| Error Diffusion | `algorithms.build_error_diffusion` | reuses `benchmarking.adapters.GAEDAdapter` -> `paper-implementations/error-diffusion/GAED_implementation.py` |
| Ordered Dithering | `algorithms.build_ordered_dithering` | **new** — `algorithms/ordered_dithering.py`, a from-scratch Bayer-matrix implementation. Nothing in the repo implemented this family before; the spec requires all four. |
| Deep-Learning Halftoning | `algorithms.build_deep_learning` | reuses `benchmarking.adapters.EfficientDRLAdapter` -> `paper-implementations/drl/.../efficient_halftoning_drl.py` (`paper` variant), with checkpoint reuse-or-bootstrap-train (see below) |

HCB-DBS and the MARL implementation are not wired in by default, but they
follow the exact same `MethodAdapter` shape and can be registered in
`algorithms/__init__.py` in one line each — see `EXTENDING.md`.

## The reconstruction-sigma inconsistency this project fixes

Different pieces of the existing codebase reconstruct halftones with
different perceptual models before scoring them:

- `GAED.evaluate()` uses a Gaussian blur, `sigma=1.0`
- `efficient_halftoning_drl`'s `stabilized` variant HVS term uses `sigma=1.5`
- `benchmarking/metrics.py::viewed()` (used by the *original*
  `benchmarking/run.py`) uses a **Nasanen** HVS kernel (`S=2000`), not a
  plain Gaussian at all
- the spec (`Sec. 5.2`) requires: *"apply a Gaussian observer blur with
  fixed standard deviation sigma = 1.2"* for every full-reference metric
  except Anisotropy

`benchmarking_v2/reconstruction.py::reconstruct()` is the **single**
reconstruction operator used anywhere in this package's metric computation,
hard-set to `sigma=1.2`, completely independent of whatever HVS model an
algorithm used internally for its own training-time reward. Internal HVS
kernels are left untouched — they affect what the algorithm optimizes for,
not how the benchmark scores the result. Anisotropy Index is computed on
the *raw* halftone per spec Sec. 5.1, never on `R(H)`.

## Checkpoints for the Deep-Learning row (no from-scratch retrain each run)

`benchmarking_v2/checkpoints.py::ensure_efficient_drl_checkpoint()` resolves
a checkpoint in this order, and only ever trains as a last resort:

1. an explicit `checkpoint = "..."` path in the config,
2. a released, checksum-verified checkpoint via
   `benchmarking.checkpoints.fetch_checkpoint("efficient-drl-paper-v1")`,
3. a bootstrap checkpoint already cached under
   `<run_dir>/bootstrap_checkpoints/` from a previous run,
4. a **short** bootstrap-training loop (`bootstrap_iterations`, default
   200, on synthetic crops), whose result is cached for next time.

Step 4 exists purely so the pipeline never hard-fails for lack of a
checkpoint. It is not a substitute for the real paper training protocol —
point `checkpoint = "..."` at a properly trained/released checkpoint (see
the main repo's `train_efficient_halftoning_drl.py --variant paper ...`,
200k iterations) for a paper-faithful Deep-Learning row. `run.json` /
adapter metadata always records which path was actually used.

## Datasets

See `datasets.py`. Real dataset download (Kaggle, per Sec. 8 of the spec)
requires network + Kaggle credentials this sandboxed environment doesn't
have; the loader is layered so it works the same way in a normal
researcher's environment:

1. use a local copy under `datasets/<key>/...` if present (matches what the
   rest of this repo already does for `datasets/kodak`),
2. else try `kagglehub.dataset_download(...)` best-effort,
3. else fall back to a deterministic, clearly-tagged synthetic generator so
   the full pipeline (stress variants, metrics, reporting) is always
   exercisable offline. `manifest.json` and `run.json` both record
   `synthetic: true` per item so a synthetic-fallback run is never mistaken
   for a real benchmark result.

## Metrics

- PSNR / SSIM / LPIPS: computed on `reconstruct(halftone)` vs. reference (spec Sec. 5.1). LPIPS is optional (`pip install lpips`); if unavailable it's reported as `NaN` with the reason recorded in `run.json`, never silently skipped.
- CIEDE2000 (`delta_e00`): color track only, also on `R(H)`.
- Anisotropy Index: ring-variance of the *raw* halftone's normalized power spectrum — the one metric the spec keeps off the reconstruction operator.
- Runtime: reported per spec Sec. 2 footnote — diagnostic only, not part of ranking (the leaderboard sorts by PSNR).

## Visual comparisons: original vs. binary halftone vs. reconstruction

Every run also writes `<output_dir>/comparisons/`:

- `<method>_comparison.png` — one per enabled method, up to `comparisons_per_method` (default **5**) sample rows, each showing **Original | Halftone (binary) | Reconstructed** (this benchmark's σ=1.2 operator, exactly what PSNR/SSIM/LPIPS were computed from). Samples are spread across content families (one per family's `base` item first, if the family count matches; extra slots fall back to stress variants) and **adapt down automatically** if fewer than `comparisons_per_method` items exist.
- `all_methods_comparison.png` — a bonus panel: the *same* sample image across every enabled method, one row per method, so you can compare e.g. DBS's dispersed dots vs. Ordered Dithering's Bayer crosshatch vs. Error Diffusion's blue-noise-like pattern directly, side by side.

Control it without touching the config:

```bash
python -m benchmarking_v2.run_benchmark --comparisons-per-method 8   # more samples
python -m benchmarking_v2.run_benchmark --comparisons-per-method 0   # disable entirely
```

or in TOML: `comparisons_per_method = 5` (top-level key, alongside `compute_lpips`).

## Known-issue fix: numba disk-cache crash on dynamic loading

If you ever see:

```
ModuleNotFoundError: No module named 'cb_dbs'
```

with a traceback through `numba/core/caching.py` / `numba/core/environment.py`,
that's not an environment problem on your machine — it's a pre-existing
interaction between numba's `@njit(cache=True)` disk cache and this
project's dynamic module loading (`benchmarking.adapters._load_module`
loads `cb_dbs.py` under a synthetic module name; numba's cache keys itself
by source file, but bakes in whatever module name was active *the first
time* that file was ever compiled anywhere on the machine — e.g. from
running `paper-implementations/dbs/cb-dbs/test_cb_dbs.py` directly).

`benchmarking_v2/algorithms/__init__.py::build_dbs` already works around
this by forcing fresh in-process JIT compilation (never touching numba's
disk cache) via `_numba_compat.force_no_disk_cache()`. It's transparent —
you don't need to do anything — but if you register a new adapter that
dynamically loads another `@njit`-decorated module (e.g. HCB-DBS, which has
the same pattern), wrap its construction the same way; see
`EXTENDING.md`. `tests/test_numba_dynamic_load_workaround.py` reproduces
the original crash and asserts the fix holds.



```
benchmarking_v2/
  spec.py              constants transcribed from the PDF (sigma, families, Table 1 params)
  reconstruction.py    the single R(H) operator (sigma=1.2)
  metrics.py           PSNR / SSIM / LPIPS / Anisotropy / DeltaE00
  datasets.py          fetch + stratified sample, with synthetic fallback
  stress_variants.py   Table 1, one function per (family, A/B/C)
  checkpoints.py       checkpoint reuse-or-bootstrap-train for RL methods
  algorithms/
    base.py            adapter Protocol
    ordered_dithering.py   new algorithm
    __init__.py         registry: name -> adapter factory
  run_benchmark.py     orchestrator / CLI
  config/default.toml  the run config
```

See `CHANGES.md` for a line-by-line account of what was reused vs. added vs.
reconciled, and `EXTENDING.md` for how to add a new algorithm, dataset,
stress variant, or metric.
