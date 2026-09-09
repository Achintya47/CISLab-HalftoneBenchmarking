# Extending the benchmark

Three independent registries. Adding to any of them never requires touching
`run_benchmark.py`.

## Add a new algorithm

1. Write an adapter class exposing:
   - `.name: str`, `.family: str`, `.stochastic: bool`, `.supports_color: bool`
   - `.run(image: np.ndarray, seed: int) -> benchmarking.model.HalftoneResult`
     (`image` is an `HxWx3` float array in `[0, 1]`)

   `algorithms/ordered_dithering.py` is a complete, minimal example. If your
   method is grayscale-only, set `supports_color = False` and leave
   `HalftoneResult.rgb_output` as `None` — the orchestrator skips the color
   track automatically (see `GAEDAdapter` reused via `build_error_diffusion`).

2. Register a one-line factory in `algorithms/__init__.py`:

   ```python
   def build_my_method(config: dict, device: str) -> Any:
       return _tag_family(MyMethodAdapter(**config), "my_family", supports_color=True)

   ALGORITHM_REGISTRY["my_method"] = build_my_method
   ```

3. Add a `[methods.my_method]` table to a config TOML:

   ```toml
   [methods.my_method]
   algorithm = "my_method"
   enabled = true
   # ...adapter-specific kwargs, forwarded verbatim to build_my_method...
   ```

Existing repo methods not wired in by default (HCB-DBS, MARL) follow this
exact recipe — e.g. for HCB-DBS:

```python
from benchmarking.adapters import _load_module
from .._numba_compat import force_no_disk_cache

def build_hcb_dbs(config: dict, device: str) -> Any:
    # hcb_dbs.py has the same @njit(cache=True) pattern as cb_dbs.py, and
    # is loaded the same dynamic way -- wrap it the same way to avoid the
    # stale-cache crash documented in README.md / CHANGES.md item 8.
    with force_no_disk_cache():
        module = _load_module("paper-implementations/dbs/hcb-dbs/hcb_dbs.py", "hcb_dbs_v2")
    # wrap module.run_hcb_dbs(...) in a class with .run(image, seed) -> HalftoneResult,
    # following benchmarking.adapters.HCBDBSAdapter as a template
    ...

ALGORITHM_REGISTRY["hcb_dbs"] = build_hcb_dbs
```

## Add a new dataset / content family

1. Add the family to `spec.CONTENT_FAMILIES` with its `dataset_keys` and
   relevant spec properties.
2. Add its 3 stress-variant definitions to `spec.FAMILY_VARIANTS` (numeric
   parameters go in their own module-level dict, like `spec.ADDITIVE_NOISE`).
3. Implement the 3 corresponding functions in `stress_variants.py`
   (`(rgb, rng) -> rgb`) and add them, in order, to
   `stress_variants.FAMILY_VARIANT_FUNCS[<family>]`.
4. If the dataset has a real source, add its Kaggle slug (or swap in your
   own loader) to `datasets.KAGGLE_SLUGS` / `_try_kagglehub_download`; the
   local-copy-first / synthetic-fallback chain in `sample_family_images`
   needs no changes.
5. Add `[families.<name>]` with a `count` to your config TOML.

## Add a new metric

Add a function to `metrics.py` following the shape of `anisotropy_index()`
or `lpips_distance()`, then include it in `grayscale_full_reference_metrics`
/ `color_full_reference_metrics`. Decide up front whether it belongs on
`reconstruct(halftone)` or the raw halftone (see spec Sec. 5.1 — only
Anisotropy is currently the raw-halftone exception) and document that choice
next to the function, the same way `reconstruction.py` documents its own
sigma choice.

## Change the reconstruction operator

One line: `spec.RECONSTRUCTION_SIGMA`. Every metric that should follow the
spec's `R(H)` reads it through `reconstruction.reconstruct()`, so a revision
propagates everywhere automatically. If a future spec version needs a
different *kind* of operator (not just a different sigma), change the body
of `reconstruction.reconstruct()` — it's the single point of truth by
design.

## Change how many images / which counts are sampled

Purely a config change — `[families.<name>].count` in the TOML, no code
edits. `spec.DEFAULT_IMAGES_PER_FAMILY` is only the fallback when a family
isn't listed in the config at all.

## Enable a print-and-scan (spec Sec. 7) pass later

Not implemented (explicitly out of scope for this iteration). It would
consume `metrics.csv`'s `(method, family, image_id, variant)` keys to know
which rendered halftone to print/scan, and write a parallel
`printing_metrics.csv` reported separately per spec Sec. 7.3 rather than
merged into `summary.json`.
