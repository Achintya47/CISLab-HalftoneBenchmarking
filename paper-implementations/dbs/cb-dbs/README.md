# CB-DBS

This folder contains a from-scratch Python implementation of:

- Je-Ho Lee, Jan P. Allebach
- *Colorant-based direct binary search halftoning*
- Journal of Electronic Imaging, 2002

Files:

- `cb_dbs.py`: CB-DBS pipeline and objective implementation
- `render_cb_dbs.py`: full-image renderer and artifact writer

Implementation notes:

- `C` and `M` are jointly processed using the constrained colorant formulation from the paper.
- `Y` is halftoned independently with monochrome DBS, as described in the paper.
- The total `CM` pattern is produced first, then recolored by swap-only optimization in a `7x7` neighborhood.
- The HVS kernel is a Näsänen-style low-pass kernel parameterized by `hvs_kernel_size`, `hvs_scale_factor`, and `hvs_luminance`.
- The default pass caps are `10` for both monochrome and `C/M` refinement, with early stopping when an iteration accepts no changes.

Example:

```bash
python paper-implementations/dbs/cb-dbs/render_cb_dbs.py \
  --input datasets/kodak/images/kodim04.png datasets/kodak/images/kodim05.png \
  --output-dir paper-implementations/dbs/cb-dbs/output/kodak-full
```
