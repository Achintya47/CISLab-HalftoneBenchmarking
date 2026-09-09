from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np

from .metrics import normalize_rgb, rgb_to_luma
from .model import HalftoneResult


REPO_ROOT = Path(__file__).resolve().parents[1]


class MethodAdapter(Protocol):
    name: str
    stochastic: bool

    def run(self, image: np.ndarray, seed: int) -> HalftoneResult: ...


def _load_module(relative_path: str, name: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"halftoning_bench_{name}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_call(function: Any, device: Any = None) -> tuple[Any, float]:
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


class GAEDAdapter:
    name = "gaed"
    stochastic = False

    def __init__(self, *, variant: str = "gaed", boundary_mode: str = "renormalize") -> None:
        module = _load_module("paper-implementations/error-diffusion/GAED_implementation.py", "gaed")
        self.variant = variant
        self.model = module.GAED(boundary_mode=boundary_mode)
        self.boundary_mode = boundary_mode

    def run(self, image: np.ndarray, seed: int) -> HalftoneResult:
        gray = rgb_to_luma(image)
        output, elapsed = _time_call(lambda: self.model.halftone_variant(gray, variant=self.variant))
        return HalftoneResult(self.name, output.astype(np.float64), elapsed, seed, metadata={"variant": self.variant, "boundary_mode": self.boundary_mode})


class CBDBSAdapter:
    name = "cb_dbs"
    stochastic = True

    def __init__(self, *, mono_passes: int = 10, cm_passes: int = 10) -> None:
        self.module = _load_module("paper-implementations/dbs/cb-dbs/cb_dbs.py", "cb_dbs")
        self.mono_passes = mono_passes
        self.cm_passes = cm_passes

    def run(self, image: np.ndarray, seed: int) -> HalftoneResult:
        rgb = normalize_rgb(image).astype(np.float32)
        config = self.module.CBDBSConfig(mono_passes=self.mono_passes, cm_passes=self.cm_passes, random_seed=seed)
        result, elapsed = _time_call(lambda: self.module.run_cb_dbs(rgb, config))
        rendered = np.asarray(result.rendered_rgb, dtype=np.float64)
        return HalftoneResult(self.name, rgb_to_luma(rendered), elapsed, seed, rendered, np.asarray(result.preview_rgb), asdict(config))


class _TorchAdapter:
    stochastic = True

    def __init__(self, checkpoint: str | Path, device: str, required_variant: str | None = None) -> None:
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = REPO_ROOT / self.checkpoint_path
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Required checkpoint is missing: {self.checkpoint_path}")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.required_variant = required_variant
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

    def _run_model(self, image: np.ndarray, seed: int) -> HalftoneResult:
        torch = self.torch
        gray = rgb_to_luma(image).astype(np.float32)
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            payload, elapsed = _time_call(lambda: self.module.infer_halftone(self.model, tensor, self.config), self.device)
        output = payload["halftone"][0, 0].detach().cpu().numpy().astype(np.float64)
        return HalftoneResult(self.name, output, elapsed, seed, metadata={"checkpoint": str(self.checkpoint_path), "checkpoint_sha256": self.checkpoint_sha256, "config": asdict(self.config)})

    def run(self, image: np.ndarray, seed: int) -> HalftoneResult:
        return self._run_model(image, seed)


class EfficientDRLAdapter(_TorchAdapter):
    name = "efficient_drl_paper"

    def __init__(self, checkpoint: str | Path, device: str = "auto", required_variant: str = "paper", adapter_name: str = "efficient_drl_paper") -> None:
        self.name = adapter_name
        super().__init__(checkpoint, device, required_variant)
        self.module = _load_module("paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/efficient_halftoning_drl.py", "efficient_drl")
        checkpoint_data = self.torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config = self.module.DRLHalftoningConfig(**checkpoint_data["config"])
        if required_variant and self.config.variant != required_variant:
            raise ValueError(f"Checkpoint variant is {self.config.variant!r}, expected {required_variant!r}")
        self.model = self.module.build_reference_model(self.config, device=self.device)
        self.model.load_state_dict(checkpoint_data["model_state"])
        self.model.eval()


def build_adapter(name: str, config: dict[str, Any], device: str = "auto") -> MethodAdapter:
    if name == "gaed":
        return GAEDAdapter(variant=str(config.get("variant", "gaed")), boundary_mode=str(config.get("boundary_mode", "renormalize")))
    if name == "cb_dbs":
        return CBDBSAdapter(mono_passes=int(config.get("mono_passes", 10)), cm_passes=int(config.get("cm_passes", 10)))
    if name == "efficient_drl_paper":
        return EfficientDRLAdapter(str(config["checkpoint"]), device=device, required_variant=str(config.get("required_variant", "paper")))
    raise ValueError(f"Unknown benchmark method: {name}")
