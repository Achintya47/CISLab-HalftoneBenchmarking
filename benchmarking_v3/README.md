# Halftoning Benchmark v3 — Printer-Style CMYK/ICC Benchmark

`benchmarking_v3` extends `benchmarking_v2`'s spec-driven benchmark with a
print-realistic pre/post pipeline: instead of halftoning RGB/luma directly,
every image is put through a genuine ICC color-managed **RGB → CMYK**
conversion first, each of the four colorant planes (Cyan, Magenta, Yellow,
Black) is halftoned **independently**, reconstructed, recombined, and
converted **back through the same ICC transform to RGB** for evaluation.
This is what actually happens in a real print workflow — a printer never
sees RGB, it sees four (or more) separate ink planes — so this is a more
realistic stress test than scoring in RGB/luma space directly.

It reuses `benchmarking_v2`'s dataset sampling, stress variants, manifest
format, checkpoint-resolution policy, and reconstruction operator
unchanged; only the pre/post color pipeline and the algorithm call
convention (RGB image → per-plane grayscale) are new. See `CHANGES.md` for
the itemized delta.

## Run it

```bash
# Fast local check: 10 base images, Deep Learning disabled
python -m benchmarking_v3.run_benchmark --config benchmarking_v3/config/quick_test.toml

# Full protocol (~50 base images)
python -m benchmarking_v3.run_benchmark --config benchmarking_v3/config/default.toml

# Scale/skip from the CLI, same flags as benchmarking_v2, plus ICC ones:
python -m benchmarking_v3.run_benchmark \
  --images-per-family 2 --disable deep_learning --no-lpips \
  --cmyk-profile /path/to/your/printer_profile.icc \
  --rendering-intent perceptual
```

Outputs (per run, in `<output_dir>/`): `manifest.json`, `metrics.csv`,
`summary.json`, `leaderboard.md`, `run.json` (full provenance, including the
ICC fingerprint), `icc_fingerprint.json` (the same fingerprint on its own),
and `comparisons/<method>_cmyk_comparison.png` — Original RGB, each of the
four halftoned colorant planes, and the final reconstructed RGB, up to 5
sample images per method.

## The pipeline, step by step

```
                              [SPEC]                         [SPEC]
Original RGB --> Base / Stress / Perturbed variant --> (unchanged, held aside as the reference)
     |
     | [NEW] step 2
     v
RGB --ICC--> CMYK  (source: sRGB, destination: your chosen CMYK ICC profile,
     |                     with a configured rendering intent + BPC)
     | [NEW] step 3-4
     v
{ C, M, Y, K }  --  four independent grayscale planes (channel extraction,
     |                not RGB-luminance conversion -- each plane already IS
     |                a grayscale image)
     | [SPEC] step 5
     v
{ C_halftone, M_halftone, Y_halftone, K_halftone }  -- each plane run through
     |                the SAME chosen halftoning algorithm, independently
     | [SPEC] step 6
     v
{ C_recon, M_recon, Y_recon, K_recon }  -- Gaussian observer, sigma=1.2,
     |                per plane (identical operator to benchmarking_v2)
     | [NEW] step 7
     v
Reconstructed CMYK  -- recombine the 4 reconstructed planes
     | [NEW] step 8
     v
Reconstructed RGB  -- CMYK--ICC-->RGB, through the SAME ICC profile/intent/BPC
     | [SPEC] step 9-10
     v
CIELAB (both images) --> PSNR / SSIM / LPIPS / Anisotropy / dE00
```

Two things are worth calling out explicitly:

- **The original RGB image is never touched.** Step 1's base/stress/
  perturbed variants are still produced in RGB (reusing
  `benchmarking_v2.stress_variants` exactly), and that RGB is what every
  final metric compares against. CMYK only exists between steps 2 and 8.
- **The same `ICCPipeline` instance is used for both the forward (RGB->CMYK)
  and inverse (CMYK->RGB) legs**, built once per benchmark run and shared
  across every method and item. This is what "a fixed, benchmark-wide ICC
  RGB->CMYK conversion" means in practice -- every method is scored against
  exactly the same colorant decomposition and exactly the same recombination
  transform, so differences in the leaderboard reflect the halftoning
  algorithms, not incidental differences in how each one's CMYK was derived.

## How the CMYK/ICC conversion actually works (`icc.py`)

This is the part of the spec that needed the most care, so here's the full
explanation of what's really going on, not just the interface.

### What "ICC-managed" means

A real color-managed conversion doesn't map RGB to CMYK with a formula --
it goes through a **device-independent connection space** (CIE Lab or XYZ,
called the "Profile Connection Space", PCS):

```
RGB values -- (source profile's device->PCS table) --> Lab/XYZ
Lab/XYZ    -- (destination profile's PCS->device table) --> CMYK values
```

Each ICC profile is a file that describes one device's (or one assumed
color space's) mapping to/from that connection space, typically as
multi-dimensional lookup tables built from real measurements (e.g.
spectrophotometer readings of a printed test chart, for a real printer
profile). `PIL.ImageCms` is a thin Python wrapper around **Little CMS
(lcms2)** -- the same open-source color engine used by GIMP, ImageMagick,
Firefox/Chrome, and most print RIPs -- so `benchmarking_v3` is doing exactly
the same kind of conversion real color-managed software does, not a
simulation of it.

Two extra knobs matter for how that mapping behaves:

- **Rendering intent** -- what to do about colors the destination can't
  reproduce (its gamut is smaller than the source's, which is *always*
  true going RGB->CMYK, since screens can show colors printers can't). This
  project exposes all four standard ICC intents: `perceptual` (compress the
  whole gamut smoothly, favors "still looks natural" over accuracy),
  `relative_colorimetric` (clip out-of-gamut colors to the nearest
  reproducible one, rescale white point -- the default here, and the most
  common choice for photographic reproduction), `saturation` (favor vivid
  colors over accuracy -- used for charts/graphics), and
  `absolute_colorimetric` (like relative, but doesn't rescale white point --
  used for proofing one device's output on another).
- **Black point compensation (BPC)** -- whether to rescale the darkest
  reproducible tone so shadow detail doesn't get crushed (printers usually
  can't reach as deep a black as a screen can produce). On by default here.

Both are set once per run (`[icc].rendering_intent`,
`[icc].black_point_compensation`) and recorded in every run's provenance --
per the spec's requirement for "a frozen CMYK profile, rendering intent,
black-point compensation, and conversion-library version."

### Why there's a fallback, and why it's not a shortcut

A **full print-characterization ICC profile** -- the kind a real press or
proofer would use -- has lookup tables in *both* directions: device->PCS
(`AtoB`, needed to convert CMYK ink readings *into* Lab, used for e.g.
color measurement/verification) and PCS->device (`BtoA`, needed to convert
a Lab/RGB-derived color *into* CMYK ink amounts, which is what step 2
needs). These real profiles (SWOP2006, FOGRA39/51, GRACoL2013, or one
generated from your own printer+paper+ink via a spectrophotometer) are
almost never freely redistributable -- they're built from proprietary
characterization data or licensed from bodies like Idealliance/Fogra.

So this project ships one real, CC0-licensed CMYK ICC profile
(`icc_profiles/CGATS001Compat-v2-micro.icc`, from
[Compact-ICC-Profiles](https://github.com/saucecontrol/Compact-ICC-Profiles))
so the pipeline works out of the box -- but it's honest about what that
profile actually is: a compact "legacy CMYK content assumption" profile
(ICC device class `scnr`), which only has the `AtoB` (device->PCS) table.
Verified directly, not assumed:

```python
>>> ImageCms.buildTransform(srgb, cmyk_profile, "RGB", "CMYK", ...)   # step 2 direction
PyCMSError: cannot build transform          # no BtoA table -- can't build this direction
>>> ImageCms.buildTransform(cmyk_profile, srgb, "CMYK", "RGB", ...)   # step 8 direction
<ImageCmsTransform ...>                     # AtoB table exists -- this direction works fine
```

If `ICCPipeline` silently used real ICC math for CMYK->RGB (because it
happens to work) but a made-up formula for RGB->CMYK (because it has no
choice), the two legs of the pipeline would be **inconsistent with each
other** -- the "reconstructed CMYK" fed into step 8 wouldn't be something
step 2's transform could have produced, making the round trip meaningless.

Instead, `ICCPipeline` **probes both directions up front** (`_build()` in
`icc.py`), and:

- if **both** build successfully -> every conversion, both directions, goes
  through genuine ICC transforms (`use_icc = True`);
- if **either** fails -> **both** directions fall back to a documented,
  exactly-invertible analytic conversion (`use_icc = False`), so forward
  and inverse always agree with each other, even though neither is
  ICC-managed.

The fallback (`_naive_rgb_to_cmyk` / `_naive_cmyk_to_rgb`) is the standard
textbook "full GCR" (Gray Component Replacement) formula:

```
K  = min(1-R, 1-G, 1-B)                      # as much black as the darkest channel allows
C  = ((1-R) - K) / (1-K)                     # remaining color relative to what's left after K
M  = ((1-G) - K) / (1-K)
Y  = ((1-B) - K) / (1-K)

# inverse:
R = (1-C)(1-K)     G = (1-M)(1-K)     B = (1-Y)(1-K)
```

which is exactly, algebraically, invertible (see the one-line derivation in
`icc.py`'s docstrings and
`tests/test_icc.py::test_naive_full_gcr_conversion_is_exactly_invertible`,
which checks it to `1e-9`) -- a safe, transparent, deterministic default
when no ICC-capable destination profile is configured, exactly mirroring
this project's existing "local dataset -> kagglehub -> synthetic fallback"
layering philosophy (see `benchmarking_v2/README.md`).

**To get a fully ICC-managed round trip**, point `[icc].cmyk_profile_path`
(or `--cmyk-profile`) at your own real bidirectional output-class ICC
profile -- nothing else needs to change; `ICCPipeline` detects the
capability automatically and switches over. `run.json`'s `icc.use_icc`
field always tells you, after the fact, which path a given run actually
used.

### "Standard" and "flexible to match any ICC profile"

Concretely, that means:

- **No profile is special-cased by name or hardcoded path.** Any `.icc`/
  `.icm` file works; `ICCProfileSpec.cmyk_profile_path` accepts any path.
- **No assumption about profile capability.** Both directions are probed,
  not assumed, so a one-directional profile degrades safely instead of
  crashing or silently producing wrong output.
- **The knobs that matter in real color management are exposed**, not
  buried: rendering intent (all 4 ICC-standard intents), black point
  compensation, and -- recorded, even though not yet configurable --
  Little CMS's own version number, so a provenance record fully pins down
  what conversion actually happened.
- **The conversion library is the industry-standard one** (Little CMS via
  `PIL.ImageCms`), not a custom color-math reimplementation, so a real
  bidirectional profile gets genuinely correct, industry-standard behavior.

## Algorithms (per-plane adapters)

| Method | Underlying single-plane primitive reused |
|---|---|
| DBS | `cb_dbs.monochrome_dbs` (already takes an arbitrary density plane) |
| Error Diffusion | `GAED.halftone_variant` (already accepts a bare 2D array) |
| Ordered Dithering | `ordered_dithering.ordered_dither_channel` (already per-channel) |
| Deep-Learning | `efficient_halftoning_drl.infer_halftone` (already single-channel) |

Each of these already had a single-plane/grayscale core underneath the RGB
wrapper used in `benchmarking_v2` -- v3 calls straight into that core for
each of the 4 CMYK planes, instead of reusing the RGB-facing adapters. See
`algorithms.py`'s module docstring for why, and note that **HCB-DBS is
intentionally not wired in**: it's a joint hierarchical colorant-placement
algorithm that optimizes multiple ink colors *together* specifically to
prevent overlap between them, so it doesn't decompose into 4 independent
per-plane calls without changing what the algorithm fundamentally is. See
`EXTENDING.md` if you want to explore adapting it anyway.

## Metrics

Identical metric *definitions* to `benchmarking_v2` (PSNR, SSIM, LPIPS,
Anisotropy Index, dE00), reused directly from `benchmarking_v2.metrics`
where the definition doesn't change -- but a different evaluation space:
v3's primary track is RGB/Lab (there's no separate grayscale/color split,
since by the time metrics run, CMYK has already been fully round-tripped
back to RGB). Anisotropy Index is the one exception carried over from the
spec: it's computed on the *raw halftone planes* (mean across the 4
channels), never on the reconstruction -- same rule as `benchmarking_v2`,
just applied per-plane instead of per-luma-image.

## Checkpoints

Identical resolution ladder to `benchmarking_v2`
(explicit path -> released checkpoint -> cached bootstrap -> short
bootstrap-train), reused directly from `benchmarking_v2.checkpoints`. See
`benchmarking_v2/README.md`'s "Checkpoints" section.

## Files

```
benchmarking_v3/
  icc.py                the RGB<->CMYK ICC pipeline (this doc's main subject)
  icc_profiles/          the bundled CC0 CMYK profile + its LICENSE/README
  cmyk_pipeline.py       channel separation/combination (steps 3-4, 7)
  algorithms.py          per-plane halftoning adapters (step 5)
  reconstruction.py      re-exports benchmarking_v2's sigma=1.2 operator (step 6)
  metrics.py             RGB/Lab metrics (steps 9-10), reusing benchmarking_v2 where possible
  visualize.py           Original | C | M | Y | K | Reconstructed comparison panels
  run_benchmark.py       orchestrator / CLI (step 1, 11; wires everything else together)
  config/default.toml, config/quick_test.toml
```
