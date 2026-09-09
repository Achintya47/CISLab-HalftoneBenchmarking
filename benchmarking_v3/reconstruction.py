"""Spec step 6 ("Binary CMYK -> Gaussian Observer sigma=1.2 -> Continuous
CMYK") is the exact same 2D Gaussian reconstruction operator used
throughout benchmarking_v2 -- there is nothing CMYK-specific about it, it's
just applied once per colorant plane instead of once per luma image. Reused
verbatim rather than forked, per the instruction that the reconstruction
operator can be retained as-is.
"""

from benchmarking_v2.reconstruction import RECONSTRUCTION_SIGMA, reconstruct

__all__ = ["RECONSTRUCTION_SIGMA", "reconstruct"]
