# HCB-DBS Implementation

This directory contains a compact reference implementation of Algorithm 0 for HCB-DBS.

The implementation is intentionally small and explicit:

- `hcb_dbs.py` contains the full pipeline: MBVCD conversion, visibility-priority grouping, constrained monochrome DBS for hierarchical dot positioning, recursive color assignment, and RGB preview rendering.
- `hcb-dbs-review.md` records the issues found in the extracted scaffold that this version replaces.

## Design choices

Algorithm 0 leaves several hooks underspecified. This implementation makes those assumptions explicit instead of hiding them in scattered helper files:

- Dot-color grouping uses a fixed visibility-priority partition: `("K", "B")`, `("R", "M")`, `("G", "C")`, `("Y",)`.
- Hierarchical dot positioning applies constrained monochrome DBS at every group level, not just random initialization.
- Recursive color assignment optimizes the HVS-filtered split objective with exact energy recomputation for each accepted move.
- The code is a reference implementation optimized for readability and faithfulness, not for speed.

## Usage

```python
import numpy as np
from hcb_dbs import HCBDBSConfig, run_hcb_dbs

rgb = np.random.rand(64, 64, 3).astype(np.float32)
result = run_hcb_dbs(rgb, HCBDBSConfig(random_seed=0))

preview = result.preview_rgb
dot_maps = result.final_maps
```

`run_hcb_dbs` expects an `HxWx3` RGB array with values in `[0, 1]` and returns a `HCBDBSResult` with:

- `densities`: MBVCD densities for `K/B/R/G/C/M/Y`
- `groups`: the top-level hierarchy used during dot positioning
- `final_maps`: final binary dot maps per color
- `preview_rgb`: an additive RGB visualization of the dot-color halftone

## Diagnostics

The Kodak plus constant-gray diagnostic workflow lives in
`benchmark_hcb_dbs_diagnostics.py`.

Default output layout:

- `output/kodak-diagnostics-maxside-128/README.md`
- `output/kodak-diagnostics-maxside-128/kodak_metrics.csv`
- `output/kodak-diagnostics-maxside-128/summary.json`
- `output/kodak-diagnostics-maxside-128/method_outputs/preview_rgb`
- `output/kodak-diagnostics-maxside-128/method_outputs/rendered_rgb`
- `output/kodak-diagnostics-maxside-128/method_outputs/viewed_rgb`
- `output/kodak-diagnostics-maxside-128/diagnostics`
- `output/kodak-diagnostics-maxside-128/constant_gray_diagnostics`

Run it with:

```bash
python paper-implementations/dbs/hcb-dbs/benchmark_hcb_dbs_diagnostics.py
```

Notes:

- The default benchmark uses Kodak images resized to `max-side 128` because HCB-DBS is much slower than the DRL forward pass.
- FFT and radial diagnostics are computed on luminance because HCB-DBS is a color halftoning method.
