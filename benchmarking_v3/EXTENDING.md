# Extending benchmarking_v3

## Use a different (or your own) ICC profile

No code changes needed:

```bash
python -m benchmarking_v3.run_benchmark --cmyk-profile /path/to/profile.icc --rendering-intent perceptual
```

or in TOML:

```toml
[icc]
cmyk_profile_path = "/path/to/profile.icc"
rendering_intent = "perceptual"          # perceptual | relative_colorimetric | saturation | absolute_colorimetric
black_point_compensation = true
```

`ICCPipeline` (`icc.py`) probes the profile's capabilities automatically at
construction time — if it supports a full bidirectional RGB<->CMYK round
trip, every conversion in the run uses genuine ICC management; if not, both
directions fall back to the naive analytic conversion together (see
`README.md`'s "Why there's a fallback" section for the full reasoning). You
can inspect which path a given run actually took via
`<output_dir>/icc_fingerprint.json`'s `use_icc` field, or by watching for
the `[icc] NAIVE FALLBACK` / `[icc] ICC-managed` line in the run's log
output.

## Add a new halftoning algorithm

Same registry pattern as `benchmarking_v2`, but the adapter shape is
per-plane, not per-RGB-image:

1. Write an adapter class exposing `.name` and
   `.run_plane(plane: np.ndarray, seed: int) -> np.ndarray` (`plane` and
   the return value are both `HxW` float arrays in `[0, 1]`). If your
   algorithm's reference implementation is RGB-facing rather than
   grayscale-facing, look for its underlying single-channel primitive first
   (most halftoning algorithms have one under the hood, even if their
   public API takes RGB) — see `algorithms.py`'s module docstring for how
   this was done for the four already-registered methods.
2. Register a one-line factory in `algorithms.py`:

   ```python
   def build_my_method(config: dict, device: str) -> PlaneAlgorithm:
       return MyMethodPlaneAlgorithm(**config)

   ALGORITHM_REGISTRY_V3["my_method"] = build_my_method
   ```

3. Add a `[methods.my_method]` table to a config TOML.

### Adding HCB-DBS specifically

HCB-DBS is a *joint* colorant-placement algorithm — its core idea is
optimizing multiple dot colors together to prevent them from landing on the
same pixel. Naively calling its color-agnostic building blocks
(`constrained_monochrome_dbs`) once per CMYK plane independently would work
mechanically, but it throws away exactly the property that makes HCB-DBS
what it is (its explicit hierarchy avoids exactly the kind of inter-plane
dot collisions that four fully-independent planes can't avoid). If you want
it in `benchmarking_v3` anyway, the honest options are:

- register it as an independent-per-plane approximation, clearly labeled
  as such (e.g. `hcb_dbs_naive_independent`) so its leaderboard entry isn't
  confused with a faithful HCB-DBS reproduction, or
- extend `algorithms.py`'s interface to support a *joint* 4-plane call
  (`run_planes(planes: dict[str, np.ndarray], seed) -> dict[str, np.ndarray]`)
  as an alternative to `run_plane`, and have `run_benchmark.py` call
  whichever one an algorithm exposes — a larger, more invasive change than
  anything else in this document, which is why it wasn't done as part of
  this iteration.

## Add a new metric

Add it to `metrics.py`, following `rgb_reconstruction_metrics()` (if it
belongs on the ICC-round-tripped RGB reconstruction) or
`multichannel_anisotropy_index()` (if it belongs on the raw per-plane
halftones, per spec Sec. 5.1's anisotropy exception). Wire it into the row
dict built in `run_benchmark.py::evaluate_item`.

## Change how CMYK planes are aggregated into a single Anisotropy Index

`multichannel_anisotropy_index()` currently takes a plain mean across the 4
channels. If you want, say, a K-weighted score (since black ink usually
dominates perceived tone) or to report all 4 channel scores separately
instead of collapsing them, that function is the only place to change —
`run_benchmark.py` just calls it and puts the result in one CSV column.

## Change the reconstruction operator

Same as `benchmarking_v2`: edit `spec.RECONSTRUCTION_SIGMA` in
`benchmarking_v2/spec.py` (v3 imports it from there, doesn't duplicate it),
or change `benchmarking_v2.reconstruction.reconstruct()`'s implementation
directly if a future spec version needs a different *kind* of operator, not
just a different sigma.

## Add a printing constraint (spec Sec. 7) pass

Still out of scope for this iteration, same as `benchmarking_v2`. Would
consume the same `(method, family, image_id, variant)` keys from
`metrics.csv`, but would need the *halftone planes themselves* (not just
the final RGB), which `run_benchmark.py::evaluate_item` currently discards
after reconstruction except for the up-to-5 comparison samples — extending
`capture=` to persist all halftone planes (not just sample ones) to disk
would be the natural first step if this becomes needed.
