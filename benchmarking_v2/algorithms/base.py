from __future__ import annotations

from typing import Protocol

import numpy as np

from benchmarking.model import HalftoneResult


class MethodAdapterV2(Protocol):
    """Same shape as `benchmarking.adapters.MethodAdapter`.

    Re-declared here (rather than imported) so `benchmarking_v2` algorithm
    adapters have zero hard dependency on `benchmarking/adapters.py`
    internals beyond the classes it re-exports; new algorithms can implement
    this Protocol directly without touching the original package at all.
    """

    name: str
    family: str  # one of: "dbs", "error_diffusion", "ordered_dithering", "deep_learning"
    stochastic: bool
    supports_color: bool

    def run(self, image: np.ndarray, seed: int) -> HalftoneResult: ...
