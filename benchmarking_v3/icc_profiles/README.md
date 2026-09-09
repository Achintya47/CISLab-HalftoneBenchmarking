# ICC profiles

## `CGATS001Compat-v2-micro.icc`

- **Source**: https://github.com/saucecontrol/Compact-ICC-Profiles
- **License**: CC0 (public domain dedication) — see `LICENSE` in this folder
- **What it is**: a compact, standard 4-channel CMYK ICC profile approximating
  the appearance of "generic SWOP/CGATS001-compatible" CMYK content. It's
  the kind of profile applications embed when they need to *assume* a color
  space for legacy/untagged CMYK files, not a full commercial press
  characterization profile.
- **Device class**: `scnr` (input/scanner) — it has a device→PCS (`AtoB`)
  table but no PCS→device (`BtoA`) table. Practically: it can genuinely
  ICC-convert **CMYK → RGB/Lab**, but **not RGB → CMYK** (verified directly
  via `ImageCms.buildTransform`, see `tests/test_icc.py`).
- **Why it's here anyway**: it's real, small, freely redistributable, and
  lets `benchmarking_v3` ship with a working, non-trivial, standards-based
  ICC profile out of the box rather than requiring everyone to go find
  their own before the pipeline runs at all. `benchmarking_v3/icc.py`
  detects its one-directional limitation automatically and falls back to a
  documented, exactly-invertible analytic conversion for *both* directions
  so the forward and inverse legs of the pipeline never disagree with each
  other (see that module's docstring, and `benchmarking_v3/README.md`'s
  "How the ICC conversion works" section, for the full reasoning).

## Using your own profile

Point `[icc].cmyk_profile_path` (or `--cmyk-profile`) at any `.icc`/`.icm`
file — nothing in this code is specific to the bundled one. If your profile
is a real bidirectional **output/printer** ("prtr" device class) profile —
e.g. a licensed SWOP2006_Coated3v2, FOGRA39/51, or GRACoL2013 profile, or
one exported from your own printer + paper + ink combination via a
spectrophotometer — `ICCPipeline` will detect that both directions build
successfully and use genuine ICC management for the full round trip
automatically. No code or config changes beyond the path are needed.
