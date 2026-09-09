# What changed vs. benchmarking_v2

`benchmarking_v3` is an extension, not a rewrite: everything in
`benchmarking_v2` that already matched the spec (dataset sampling, stress
variants, manifest format, the sigma=1.2 reconstruction operator,
checkpoint-resolution policy, CLI ergonomics) is reused unmodified. This
document covers only what's new or different.

## 1. A print-realistic pre/post pipeline was added around halftoning

`benchmarking_v2` halftones RGB/luma directly. `benchmarking_v3` inserts,
around the exact same halftoning algorithms:

```
RGB --[NEW ICC conversion]--> CMYK --[NEW separation]--> 4 planes
   --[SPEC halftoning, unchanged]--> 4 halftones
   --[SPEC reconstruction, unchanged]--> 4 continuous planes
   --[NEW combination]--> CMYK --[NEW ICC conversion]--> RGB
   --[SPEC Lab + metrics, same definitions]-->
```

This is closer to how a real printer actually processes an image (it never
sees RGB) and exercises a failure mode `benchmarking_v2` cannot: colorant
planes halftoned independently can fall out of phase with each other,
producing visible color fringing in the reconstruction even when every
individual plane's tone reproduction is fine (see the `pattern` family
sample rows in `comparisons/*_cmyk_comparison.png` -- a black/white stripe
target reconstructs with a visible red/blue tint, a real analogue of screen
misregistration in offset printing).

## 2. New centralized ICC color-management module (`icc.py`)

Nothing in `benchmarking_v2` does color management at all (its color track
just compares RGB arrays directly). `icc.py` adds a single,
profile-agnostic `ICCPipeline` built on `PIL.ImageCms`/Little CMS, with:

- a frozen `ICCProfileSpec` (profile path, rendering intent, black point
  compensation, source profile) as the one thing that determines every
  conversion in a run;
- automatic, bidirectional capability probing, so a one-directional
  profile (like the bundled CC0 one -- see `icc_profiles/README.md`) is
  detected and both directions consistently fall back to an exactly-
  invertible naive conversion, rather than silently mixing real ICC one way
  and made-up math the other way;
- a `fingerprint()` provenance record (profile path + SHA-256, intent, BPC,
  Little CMS version, whether ICC or fallback was actually used) written to
  every run's `run.json` and `icc_fingerprint.json`.

See `README.md`'s "How the CMYK/ICC conversion actually works" section for
the full explanation, including why a fallback exists at all (freely
redistributable *bidirectional* CMYK ICC profiles essentially don't exist)
and exactly what was verified (not assumed) about the bundled profile's
capabilities.

## 3. Algorithm interface changed: RGB image -> per-plane grayscale

`benchmarking_v2`'s adapters take an `HxWx3` RGB image and hand back a
`luma_output` (plus optional `rgb_output` for color-capable methods).
`benchmarking_v3`'s adapters (`algorithms.py`) instead expose
`.run_plane(plane: HxW, seed) -> HxW`, called once per CMYK channel. This
isn't a wrapper around the v2 adapters -- each one calls directly into the
same underlying single-plane primitive the v2 adapter itself was built on
top of (`monochrome_dbs`, `GAED.halftone_variant`, `ordered_dither_channel`,
`infer_halftone`), since those primitives already operated on arbitrary
grayscale planes; only the RGB-facing wrapper layer differs. See
`algorithms.py`'s module docstring for the full mapping and for why
HCB-DBS specifically is not included (it's a joint multi-colorant
algorithm, not decomposable into independent per-plane calls without
changing what it is).

## 4. New metrics module: RGB/Lab is the primary (only) track

`benchmarking_v2` splits into a grayscale track (PSNR/SSIM/LPIPS/Anisotropy
on luma) and a secondary color track (RGB PSNR/SSIM + Delta E00) for
color-capable methods only. `benchmarking_v3` has one track: every method
processes CMYK, so every method's reconstruction gets converted all the way
back to RGB (spec step 8) before scoring -- there's no grayscale-only
method left to special-case. `metrics.py` reuses
`benchmarking_v2.metrics.anisotropy_index` and `.lpips_distance` directly
(same definitions) but adds `rgb_reconstruction_metrics()` (PSNR/SSIM/LPIPS/
dE00 on the full RGB+Lab image) and `multichannel_anisotropy_index()` (mean
anisotropy across the 4 raw halftone planes, since there are 4 of them now
instead of 1 luma image).

## 5. Reconstruction operator: reused verbatim, applied per-plane

Spec step 6 asks for the same Gaussian sigma=1.2 observer as
`benchmarking_v2`. `reconstruction.py` re-exports
`benchmarking_v2.reconstruction.reconstruct` rather than reimplementing
it -- there's nothing CMYK-specific about a 2D Gaussian blur, it's simply
called once per colorant plane (4 calls) instead of once per luma image (1
call).

## 6. Visualization: per-plane panels

`benchmarking_v2/visualize.py`'s Original/Halftone/Reconstructed panel
becomes a 6-column panel here: Original (RGB), C halftone, M halftone, Y
halftone, K halftone, Reconstructed (RGB). Sample selection
(`select_sample_item_indices`, spread across content families, adapts down
automatically) is reused verbatim from `benchmarking_v2.visualize` -- only
what gets drawn per sample changed.

## 7. Dataset/stress-variant/manifest/checkpoint machinery: unchanged

Per the brief ("the existing dataset/stress-variant framework, method
adapters, manifest, and benchmark orchestration can otherwise be
retained"), `run_benchmark.py` imports `benchmarking_v2.datasets`,
`.stress_variants`, and `.checkpoints` directly rather than forking them.
The three-variant structure described in the spec (Base/Original, Stress
Variant, Perturbed/Degraded Variant) is realized exactly as
`benchmarking_v2` already does it: one base item plus that content
family's three Table-1 stress variants, all following the identical
downstream (now CMYK-aware) pipeline.

## Section 7 (print-and-scan) -- still out of scope

The spec's physical print-and-scan layer (TVI/dot gain, channel
misregistration, banding, trapping, dot loss) remains explicitly out of
scope, same as `benchmarking_v2`. `benchmarking_v3` gets you *closer* to
that world (real CMYK, real ICC color management, per-channel halftoning)
but does not simulate a physical press.
