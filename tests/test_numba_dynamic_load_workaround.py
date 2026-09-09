from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CB_DBS_DIR = REPO_ROOT / "paper-implementations" / "dbs" / "cb-dbs"


def _clear_module_state() -> None:
    for name in list(sys.modules):
        if name in ("cb_dbs",) or name.startswith("halftoning_bench_cb_dbs"):
            del sys.modules[name]
    shutil.rmtree(CB_DBS_DIR / "__pycache__", ignore_errors=True)


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")
def test_stale_disk_cache_from_a_plain_import_does_not_break_dynamic_loading() -> None:
    """Reproduces, then verifies the fix for, a real crash:

    `benchmarking.adapters.CBDBSAdapter` loads cb_dbs.py dynamically under a
    synthetic module name. If that same file was ever compiled once while
    imported under its plain name `cb_dbs` (exactly what
    `paper-implementations/dbs/cb-dbs/test_cb_dbs.py` does with
    `from cb_dbs import ...`), numba's on-disk cache for its
    `@njit(cache=True)` function is permanently keyed to that name, and any
    later dynamic load crashes with `ModuleNotFoundError: No module named
    'cb_dbs'` -- even on a different machine or fresh checkout, as long as
    the stale `__pycache__/*.nbi/.nbc` files are present. This is what a
    user hit running `benchmarking_v2` for the first time against a repo
    checkout where the test suite (or any plain `import cb_dbs`) had run
    beforehand.
    """
    _clear_module_state()
    sys.path.insert(0, str(CB_DBS_DIR))
    try:
        import cb_dbs  # plain import: poisons the on-disk numba cache with modname="cb_dbs"

        kernel = cb_dbs.nasanen_kernel(7, 2000.0, 100.0)
        binary = np.zeros((6, 6), dtype=np.float32)
        target = np.full((6, 6), 0.3, dtype=np.float32)
        state = cb_dbs.build_objective_state(binary, target, kernel)
        cb_dbs.evaluate_local_update(state.filtered_error, kernel, [(1, 1, 1.0)], binary.shape)
    finally:
        sys.path.remove(str(CB_DBS_DIR))
    assert any(p.suffix == ".nbi" for p in (CB_DBS_DIR / "__pycache__").glob("*"))

    # Now go through the actual dynamic-loading path benchmarking_v2 uses.
    # Force a fresh module load (as if this were a brand-new process) so we
    # exercise the real bug rather than a warm sys.modules cache.
    for name in list(sys.modules):
        if name.startswith("halftoning_bench_cb_dbs"):
            del sys.modules[name]

    from benchmarking_v2.algorithms import build_algorithm

    algorithm = build_algorithm("dbs", {"mono_passes": 1, "cm_passes": 1}, device="cpu")
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    result = algorithm.run(image, seed=0)  # would raise ModuleNotFoundError without the fix
    assert result.luma_output.shape == (8, 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
