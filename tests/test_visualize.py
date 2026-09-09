from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking_v2.visualize import select_sample_item_indices  # noqa: E402


def test_selects_one_base_item_per_family_when_families_match_count() -> None:
    families = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    variants = ["base", "x", "y", "z"] * 3
    selected = select_sample_item_indices(families, variants, 3)
    assert len(selected) == 3
    assert {families[i] for i in selected} == {"a", "b", "c"}
    assert all(variants[i] == "base" for i in selected)


def test_adapts_down_when_fewer_items_than_requested() -> None:
    families = ["a", "a"]
    variants = ["base", "x"]
    selected = select_sample_item_indices(families, variants, 5)
    assert len(selected) == 2  # never crashes / never pads with duplicates or invalid indices
    assert set(selected) == {0, 1}


def test_fills_extra_slots_with_non_base_variants_after_one_per_family() -> None:
    families = ["a", "a", "a", "a", "b", "b", "b", "b"]
    variants = ["base", "v1", "v2", "v3"] * 2
    selected = select_sample_item_indices(families, variants, 4)
    assert len(selected) == 4
    # first pass: one "base" per family (2 items) -> second pass fills remaining 2 from either family
    base_count = sum(1 for i in selected if variants[i] == "base")
    assert base_count == 2


def test_returns_empty_for_zero_or_negative_count() -> None:
    assert select_sample_item_indices(["a"], ["base"], 0) == []
    assert select_sample_item_indices(["a"], ["base"], -1) == []


def test_returns_empty_for_no_items() -> None:
    assert select_sample_item_indices([], [], 5) == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
