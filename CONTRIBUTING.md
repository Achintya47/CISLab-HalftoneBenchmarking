# Contributing

## Setup

```bash
git clone <this-repo>
cd halftoning_bench
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For a smaller, faster PyTorch install, grab the CPU-only wheel first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## Before opening a PR

```bash
python -m compileall -q benchmarking benchmarking_v2 benchmarking_v3 paper-implementations tests
python -m pytest -v
```

Both must pass. CI (`.github/workflows/ci.yml`) runs the same two commands
on a Linux x Windows, Python 3.11 x 3.12 matrix and never needs network
access or real datasets/checkpoints -- the test suite exercises the
deterministic synthetic-image / bootstrap-checkpoint fallback paths (see
`benchmarking_v2/README.md` and `benchmarking_v2/datasets.py`), so it's
fully offline-reproducible. Keep new tests that way: no dataset downloads,
no dependence on a real trained checkpoint, no non-deterministic assertions.

For a quick manual sanity check beyond the test suite:

```bash
python -m benchmarking_v2.run_benchmark --config benchmarking_v2/config/quick_test.toml
python -m benchmarking_v3.run_benchmark --config benchmarking_v3/config/quick_test.toml
```

## Where things live

Read `README.md`'s "Repository layout" section first -- every non-trivial
directory has its own `README.md` (and, for the two benchmark packages,
`CHANGES.md` + `EXTENDING.md`) that goes into far more depth than this file
does. `EXTENDING.md` in `benchmarking_v2/` and `benchmarking_v3/` is the
right starting point for adding a new algorithm, dataset, stress variant,
or metric -- both are written as extension registries specifically so that
kind of change doesn't require touching the orchestrator.

## Style

- Keep functions/modules narrated with a short docstring explaining *why*,
  not just what -- this codebase leans on that heavily (see almost any
  existing module) because several design decisions here (reconstruction
  sigma, checkpoint fallback policy, ICC fallback policy) are non-obvious
  and have already caused confusion once; the comment is cheaper than
  someone re-deriving the reasoning later.
- `ruff check .` runs in CI as a non-blocking lint pass for now.
- Prefer adding to an existing registry (algorithms, stress variants,
  datasets, metrics) over adding a new special case in the orchestrator.

## Reporting a bug

Include: the exact command you ran, the full traceback, and -- if it's
pipeline-specific -- the relevant `run.json` / `icc_fingerprint.json` from
the output directory, since most non-obvious failures in this repo so far
(e.g. the numba dynamic-loading cache bug, see
`benchmarking_v2/README.md`'s "Known-issue fix" section) turned out to be
environment/history-dependent in ways the provenance files make obvious
immediately.
