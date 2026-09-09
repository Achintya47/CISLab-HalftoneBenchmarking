from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).with_name("hcb_dbs.py")
SPEC = importlib.util.spec_from_file_location("hcb_dbs_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mbvcd_densities_are_bounded_and_partition_coverage() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.random((6, 7, 3), dtype=np.float32)
    densities = MODULE.mbvcd_exact(rgb)
    assert set(densities) == set(MODULE.DOT_COLORS)
    total = sum(densities.values())
    assert np.all(total >= 0.0)
    assert np.all(total <= 1.0 + 1e-6)


def test_hierarchy_produces_disjoint_binary_dot_maps() -> None:
    rgb = np.tile(np.linspace(0, 1, 6, dtype=np.float32), (6, 1))
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    config = MODULE.HCBDBSConfig(monochrome_passes=1, color_passes=1, random_seed=0)
    result = MODULE.run_hcb_dbs(rgb, config)
    total = np.zeros((6, 6), dtype=np.float32)
    for color in MODULE.DOT_COLORS:
        dot_map = result.final_maps[color]
        assert set(np.unique(dot_map)).issubset({0.0, 1.0})
        total += dot_map
    assert np.all(total <= 1.0)
    assert result.rendered_rgb.shape == (6, 6, 3)

