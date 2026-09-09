# Halftoning Bench

A reproducible research repository for digital halftoning: paper-faithful
reference implementations across four algorithm families (Direct Binary
Search, Error Diffusion, Ordered Dithering, Deep-RL Halftoning), plus two
generations of a spec-driven benchmark that score them on a common,
content-stratified protocol -- one in RGB/luma space, one through a full
printer-style CMYK/ICC pipeline.

**Almost every directory below has its own `README.md`** with far more
detail than this file goes into -- this file's job is just to tell you
which one to read for what you're trying to do.

## Repository layout

```
.
├── README.md                    <- you are here
├── CONTRIBUTING.md               dev setup, PR checklist
├── LICENSE
├── pyproject.toml, requirements.txt
├── .github/workflows/ci.yml      Linux x Windows, Python 3.11 x 3.12
│
├── paper-implementations/       one folder per algorithm, paper-faithful reference code
│   ├── dbs/
│   │   ├── cb-dbs/                CB-DBS (Colorant-Based Direct Binary Search)   -- has its own README.md
│   │   └── hcb-dbs/                HCB-DBS (Hierarchical Colorant-Based DBS)      -- has its own README.md
│   ├── error-diffusion/            GAED (Gradient-Adaptive Error Diffusion)       -- has its own README.md
│   └── drl/
│       ├── efficient-halftoning-via-deep-reinforcement-learning-implementation/  -- has its own README.md
│       └── halftoning-with-multiagent-drl-implementation/                        -- has its own README.md
│
├── benchmarking/                 the ORIGINAL Kodak-only benchmark + shared adapters/metrics
│                                  (kept because benchmarking_v2/v3 both reuse pieces of it directly:
│                                   CBDBSAdapter, GAEDAdapter, EfficientDRLAdapter, checkpoint fetching)
│
├── benchmarking_v2/               CURRENT RGB/luma benchmark -- spec-driven, 4 method families,
│   ├── README.md                  content-stratified sampling, 15 stress variants, visual comparison
│   ├── CHANGES.md                 panels. START HERE for the general benchmark.
│   └── EXTENDING.md               README = what it is & how to run it; CHANGES = what was fixed/added
│                                  and why; EXTENDING = how to add an algorithm/dataset/metric.
│
├── benchmarking_v3/                PRINTER-STYLE CMYK/ICC benchmark -- same spec, but every image goes
│   ├── README.md                   through a real ICC RGB<->CMYK conversion first, and each of the 4
│   ├── CHANGES.md                  colorant planes (C/M/Y/K) is halftoned independently. Same 3-file
│   ├── EXTENDING.md                doc pattern as benchmarking_v2 (README/CHANGES/EXTENDING).
│   └── icc_profiles/README.md      the bundled ICC profile's provenance/license + how to use your own
│
├── checkpoints/README.md          checkpoint metadata format (binaries are release assets, not Git blobs)
├── datasets/                       local dataset copies go here (gitignored; synthetic fallback if empty)
└── tests/                          shared unit + integration tests for everything above
```

## Which benchmark do I run?

**`benchmarking_v2/`** if you want the standard RGB/luma benchmark.
**`benchmarking_v3/`** if you want the printer-realistic version (real ICC
color management, CMYK colorant-plane halftoning). Both share the same
underlying dataset sampling, stress variants, and CLI shape; `benchmarking_v3/CHANGES.md`
documents exactly what it adds on top of `benchmarking_v2`.

```bash
# v2 -- RGB/luma
python -m benchmarking_v2.run_benchmark --config benchmarking_v2/config/quick_test.toml

# v3 -- printer-style CMYK/ICC
python -m benchmarking_v3.run_benchmark --config benchmarking_v3/config/quick_test.toml
```

`benchmarking/` (no suffix) is the earlier Kodak-only benchmark that v2/v3
both reuse adapters and utilities from directly -- new benchmark *runs*
should go through v2 or v3, but `benchmarking/adapters.py`,
`benchmarking/metrics.py`, and `benchmarking/checkpoints.py` are still
live, load-bearing code, not legacy cruft.

## Quick start

```bash
git clone <this-repo>
cd halftoning_bench

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

CPU execution is enough for everything except full-scale (200k-step) DRL
training -- classical methods, unit tests, and both benchmarks' bootstrap
DRL checkpoint path all run fine on CPU. For a smaller/faster PyTorch
install:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

See `CONTRIBUTING.md` for the full dev workflow and PR checklist.

## Testing and CI

```bash
python -m pytest                                                              # full suite
python -m compileall -q benchmarking benchmarking_v2 benchmarking_v3 paper-implementations  # fast syntax check
```

GitHub Actions (`.github/workflows/ci.yml`) runs the suite on a Linux x
Windows, Python 3.11 x 3.12 matrix on every push/PR, and never downloads
research datasets or real checkpoints -- tests exercise the deterministic
synthetic-image and bootstrap-checkpoint fallback paths instead (see
`benchmarking_v2/README.md`), so CI is fully offline-reproducible.

## Datasets and checkpoints

Neither is required to run the test suite or a quick-test benchmark run --
both degrade gracefully (synthetic images; a short bootstrap-trained
checkpoint) so the pipeline is always runnable end to end. For real
results: see `benchmarking_v2/README.md`'s "Datasets" section (local copy
first, `kagglehub` best-effort second) and `checkpoints/README.md` (the
checkpoint metadata format + resolution ladder).

## Reproducibility boundaries

This repository provides reproducible code, a fixed benchmark protocol,
deterministic synthetic fallbacks, and CI. Exact paper-result reproduction
for the DRL methods still depends on external factors: the real dataset
splits, a completed long-horizon (200k-step) training run (typically
requiring a CUDA GPU), and -- for `benchmarking_v3`'s CMYK/ICC pipeline --
a licensed bidirectional printer ICC profile if you want fully ICC-managed
(rather than naive-fallback) color conversion; see
`benchmarking_v3/README.md`. Section 7 (physical print-and-scan) evaluation
is explicitly out of scope for both benchmarks.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT -- see [`LICENSE`](LICENSE). If your organization requires a different
license or has an existing CLA, replace this before publishing; nothing
elsewhere in the repo assumes MIT specifically. Note that
`benchmarking_v3/icc_profiles/` bundles one third-party file under its own
CC0 license -- see `benchmarking_v3/icc_profiles/README.md`.
