"""Workaround for a numba on-disk-cache bug triggered by this project's
dynamic module loading.

## The bug

`benchmarking.adapters._load_module` loads `cb_dbs.py` (and `hcb_dbs.py`)
via `importlib.util.spec_from_file_location` under a synthetic module name
(`halftoning_bench_cb_dbs`), which is how the same source file can be
loaded independently by several tools without colliding in `sys.modules`.

`cb_dbs.py` marks one hot function `@njit(cache=True)`. Numba's disk cache
for a `cache=True` function lives in a `__pycache__/` directory *next to the
source file* and is keyed by (source file, function name, numba version) —
**not** by which module name was used to import it. But the cached artifact
itself embeds the import name that was active *at compile time*, and
reloading it later calls `importlib.import_module(that_baked_in_name)` to
rebuild the function's global environment.

So: if `cb_dbs.py` is EVER compiled once while imported under its plain
name (e.g. running `paper-implementations/dbs/cb-dbs/test_cb_dbs.py`
directly, which does `from cb_dbs import ...`), the resulting on-disk cache
is permanently keyed to `modname="cb_dbs"`. Every later run that instead
loads the same file dynamically (as `benchmarking.adapters._load_module`
does, under `halftoning_bench_cb_dbs`) hits that stale cache entry and
crashes with:

    ModuleNotFoundError: No module named 'cb_dbs'

This is a pre-existing fragility of numba's caching + Python's dynamic
import machinery interacting badly — nothing about a user's dataset,
config, or OS is at fault, and it is reproducible deterministically (see
`tests/test_numba_dynamic_load_workaround.py`).

## The fix

Force `cache=False` for every `@njit(...)` call made *while a dynamically
loaded algorithm module is being exec'd*, via a context manager that
monkeypatches `numba.njit` for the duration of the load. This means the
function is always freshly JIT-compiled in-process instead of round-
tripped through numba's disk cache at all, so no stale on-disk cache can
ever be read (or written) by code going through `benchmarking_v2`'s
adapters. Recompilation costs on the order of ~100-300ms once per process,
which is negligible next to a benchmark run.
"""

from __future__ import annotations

import contextlib
import functools


@contextlib.contextmanager
def force_no_disk_cache():
    """Disable numba's on-disk cache for the duration of the `with` block.

    Safe to use even if numba isn't installed (classical, non-DBS methods
    don't need it) or already imported elsewhere -- it patches and restores
    `numba.njit` in place and never touches `sys.modules`.
    """
    try:
        import numba
    except ImportError:
        yield
        return

    original_njit = numba.njit

    @functools.wraps(original_njit)
    def patched_njit(*args, **kwargs):
        kwargs["cache"] = False
        return original_njit(*args, **kwargs)

    numba.njit = patched_njit
    try:
        yield
    finally:
        numba.njit = original_njit
