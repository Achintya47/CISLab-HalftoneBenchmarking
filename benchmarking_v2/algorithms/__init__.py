"""Extensible registry mapping a benchmark method name -> adapter factory.

To add a new algorithm to the benchmark, you do NOT edit the orchestrator.
You write an adapter class exposing `.name`, `.family`, `.stochastic`,
`.supports_color`, and `.run(image, seed) -> HalftoneResult` (see
`benchmarking_v2/algorithms/base.py` and `ordered_dithering.py` for a
minimal example), then register a one-line factory here. See
EXTENDING.md for the full walkthrough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from benchmarking.adapters import CBDBSAdapter, EfficientDRLAdapter, GAEDAdapter
from .ordered_dithering import OrderedDitheringAdapter
from .. import checkpoints as ckpt
from .._numba_compat import force_no_disk_cache


def _tag_family(adapter: Any, family: str, supports_color: bool) -> Any:
    adapter.family = family
    adapter.supports_color = supports_color
    return adapter


def build_dbs(config: Dict[str, Any], device: str) -> Any:
    # See _numba_compat.py: cb_dbs.py's @njit(cache=True) function can crash
    # with "ModuleNotFoundError: No module named 'cb_dbs'" if the same file
    # was ever compiled under a different import name in an earlier process
    # (e.g. running its own test file directly). Force fresh in-process JIT
    # compilation instead of touching numba's on-disk cache at all.
    with force_no_disk_cache():
        adapter = CBDBSAdapter(mono_passes=int(config.get("mono_passes", 4)), cm_passes=int(config.get("cm_passes", 4)))
    return _tag_family(adapter, "dbs", True)


def build_error_diffusion(config: Dict[str, Any], device: str) -> Any:
    return _tag_family(GAEDAdapter(variant=str(config.get("variant", "gaed"))), "error_diffusion", False)


def build_ordered_dithering(config: Dict[str, Any], device: str) -> Any:
    return _tag_family(OrderedDitheringAdapter(matrix_size=int(config.get("matrix_size", 8))), "ordered_dithering", True)


def build_deep_learning(config: Dict[str, Any], device: str) -> Any:
    """Deep-Learning family adapter: efficient_halftoning_drl (paper variant),
    with checkpoint reuse-or-bootstrap-train (see checkpoints.py)."""
    import torch

    from benchmarking.adapters import _load_module, REPO_ROOT

    module = _load_module(
        "paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/efficient_halftoning_drl.py",
        "efficient_drl_v2",
    )
    run_dir = Path(config.get("run_dir", REPO_ROOT / "benchmarking_v2" / "output" / "deep_learning"))
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)

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
    adapter = EfficientDRLAdapter(checkpoint_path, device=device, required_variant=str(config.get("required_variant", "paper")))
    adapter.metadata_provenance = provenance  # type: ignore[attr-defined]
    return _tag_family(adapter, "deep_learning", False)


ALGORITHM_REGISTRY: Dict[str, Callable[[Dict[str, Any], str], Any]] = {
    "dbs": build_dbs,
    "error_diffusion": build_error_diffusion,
    "ordered_dithering": build_ordered_dithering,
    "deep_learning": build_deep_learning,
}


def build_algorithm(name: str, config: Dict[str, Any], device: str = "auto") -> Any:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}'. Registered: {sorted(ALGORITHM_REGISTRY)}")
    return ALGORITHM_REGISTRY[name](config, device)
