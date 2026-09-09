"""Spec step 5: halftoning algorithms, adapted to operate on a single
CMYK colorant plane (HxW, float [0,1]) rather than an HxWx3 RGB image.

Why this is a *different* adapter layer than benchmarking_v2's, not a
wrapper around it: benchmarking_v2's `CBDBSAdapter` internally derives its
own C/M/Y from RGB via a direct complement (`preprocess_cmy` inside
`cb_dbs.py`) and jointly optimizes C/M against each other -- that is
CB-DBS's *own* colorant model, tuned for that specific algorithm. Here, the
four planes are already fixed by the benchmark's own ICC pipeline before
any algorithm sees them, and each plane must be halftoned independently
(spec: "four independent grayscale inputs"). So v3 calls straight into
each algorithm's underlying single-plane primitive instead:

| Method             | Underlying single-plane primitive reused                          |
|---------------------|--------------------------------------------------------------------|
| DBS                 | `cb_dbs.monochrome_dbs` (already takes an arbitrary density plane) |
| Error Diffusion     | `GAED.halftone_variant` (already accepts a bare 2D [0,1] array)    |
| Ordered Dithering   | `ordered_dithering.ordered_dither_channel` (already per-channel)   |
| Deep-Learning       | `efficient_halftoning_drl.infer_halftone` (already single-channel) |

HCB-DBS is intentionally NOT registered here: it is a *joint* hierarchical
colorant-placement algorithm (it optimizes 7 dot colors together precisely
to avoid overlap between colorants), which doesn't decompose into 4
independent per-plane calls without changing what the algorithm *is*. See
EXTENDING.md for how you'd approach adding it anyway if needed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Protocol

import numpy as np

from benchmarking.adapters import REPO_ROOT, _load_module
from benchmarking_v2 import checkpoints as ckpt
from benchmarking_v2._numba_compat import force_no_disk_cache
from benchmarking_v2.algorithms.ordered_dithering import ordered_dither_channel


class PlaneAlgorithm(Protocol):
    name: str

    def run_plane(self, plane: np.ndarray, seed: int) -> np.ndarray: ...


class DBSPlaneAlgorithm:
    name = "dbs"

    def __init__(self, passes: int = 4, window_size: int = 3, hvs_kernel_size: int = 11, hvs_scale_factor: float = 2000.0, hvs_luminance: float = 100.0, tolerance: float = 1e-8) -> None:
        # See benchmarking_v2/_numba_compat.py: cb_dbs.py's @njit(cache=True)
        # function can crash on dynamic loading if the file was ever
        # compiled under a different import name elsewhere on this machine.
        with force_no_disk_cache():
            self.module = _load_module("paper-implementations/dbs/cb-dbs/cb_dbs.py", "cb_dbs_v3")
        self.kernel = self.module.nasanen_kernel(hvs_kernel_size, hvs_scale_factor, hvs_luminance)
        self.passes = passes
        self.window_size = window_size
        self.tolerance = tolerance

    def run_plane(self, plane: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        result = self.module.monochrome_dbs(
            target_density=plane.astype(np.float32),
            kernel=self.kernel,
            window_size=self.window_size,
            passes=self.passes,
            tolerance=self.tolerance,
            rng=rng,
        )
        return result.astype(np.float64)


class ErrorDiffusionPlaneAlgorithm:
    name = "error_diffusion"

    def __init__(self, variant: str = "gaed", boundary_mode: str = "renormalize") -> None:
        module = _load_module("paper-implementations/error-diffusion/GAED_implementation.py", "gaed_v3")
        self.model = module.GAED(boundary_mode=boundary_mode)
        self.variant = variant

    def run_plane(self, plane: np.ndarray, seed: int) -> np.ndarray:
        # GAED.halftone_variant accepts a bare 2D [0,1] float array directly
        # (its internal _to_gray_float is a no-op for already-float input) --
        # no RGB faking needed.
        return self.model.halftone_variant(plane.astype(np.float64), variant=self.variant).astype(np.float64)


class OrderedDitheringPlaneAlgorithm:
    name = "ordered_dithering"

    def __init__(self, matrix_size: int = 8) -> None:
        self.matrix_size = matrix_size

    def run_plane(self, plane: np.ndarray, seed: int) -> np.ndarray:
        return ordered_dither_channel(plane.astype(np.float32), self.matrix_size).astype(np.float64)


class DeepLearningPlaneAlgorithm:
    name = "deep_learning"

    def __init__(self, module: Any, model: Any, config: Any, device: Any) -> None:
        self.module = module
        self.model = model
        self.config = config
        self.device = device

    def run_plane(self, plane: np.ndarray, seed: int) -> np.ndarray:
        import torch

        contone = torch.from_numpy(plane.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        generator = torch.Generator(device=self.device.type if self.device.type == "cuda" else "cpu")
        generator.manual_seed(seed)
        noise = torch.randn(contone.shape, generator=generator, device=self.device, dtype=contone.dtype)
        result = self.module.infer_halftone(self.model, contone, self.config, noise=noise)
        return result["halftone"][0, 0].detach().cpu().numpy().astype(np.float64)


def build_dbs(config: Dict[str, Any], device: str) -> PlaneAlgorithm:
    return DBSPlaneAlgorithm(
        passes=int(config.get("passes", 4)),
        window_size=int(config.get("window_size", 3)),
        hvs_kernel_size=int(config.get("hvs_kernel_size", 11)),
        hvs_scale_factor=float(config.get("hvs_scale_factor", 2000.0)),
        hvs_luminance=float(config.get("hvs_luminance", 100.0)),
    )


def build_error_diffusion(config: Dict[str, Any], device: str) -> PlaneAlgorithm:
    return ErrorDiffusionPlaneAlgorithm(variant=str(config.get("variant", "gaed")), boundary_mode=str(config.get("boundary_mode", "renormalize")))


def build_ordered_dithering(config: Dict[str, Any], device: str) -> PlaneAlgorithm:
    return OrderedDitheringPlaneAlgorithm(matrix_size=int(config.get("matrix_size", 8)))


def build_deep_learning(config: Dict[str, Any], device: str) -> PlaneAlgorithm:
    """Same checkpoint-resolution ladder as benchmarking_v2 (explicit path ->
    released checkpoint -> cached bootstrap -> short bootstrap-train),
    reused directly from `benchmarking_v2.checkpoints` rather than
    reimplemented."""
    import torch

    module = _load_module(
        "paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/efficient_halftoning_drl.py",
        "efficient_drl_v3",
    )
    run_dir = Path(config.get("run_dir", REPO_ROOT / "benchmarking_v3" / "output" / "deep_learning"))
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))

    def config_factory():
        base = module.paper_reference_config()
        overrides = {k: v for k, v in config.items() if k in base.__dataclass_fields__}
        return module.DRLHalftoningConfig(**{**base.__dict__, **overrides})

    explicit = Path(config["checkpoint"]) if config.get("checkpoint") else None
    checkpoint_path, provenance = ckpt.ensure_efficient_drl_checkpoint(
        module=module,
        config_factory=config_factory,
        run_dir=run_dir,
        device=resolved_device,
        explicit_checkpoint=explicit,
        released_checkpoint_name=config.get("released_checkpoint_name", "efficient-drl-paper-v1"),
        bootstrap_iterations=int(config.get("bootstrap_iterations", 200)),
        seed=int(config.get("seed", 0)),
    )
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = module.DRLHalftoningConfig(**checkpoint_data["config"])
    model = module.build_reference_model(model_config, device=resolved_device)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()
    algorithm = DeepLearningPlaneAlgorithm(module, model, model_config, resolved_device)
    algorithm.checkpoint_provenance = provenance  # type: ignore[attr-defined]
    return algorithm


ALGORITHM_REGISTRY_V3 = {
    "dbs": build_dbs,
    "error_diffusion": build_error_diffusion,
    "ordered_dithering": build_ordered_dithering,
    "deep_learning": build_deep_learning,
}


def build_plane_algorithm(name: str, config: Dict[str, Any], device: str = "auto") -> PlaneAlgorithm:
    if name not in ALGORITHM_REGISTRY_V3:
        raise ValueError(f"Unknown v3 algorithm {name!r}. Registered: {sorted(ALGORITHM_REGISTRY_V3)}. HCB-DBS is intentionally excluded -- see EXTENDING.md.")
    return ALGORITHM_REGISTRY_V3[name](config, device)
