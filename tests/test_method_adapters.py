from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest

from benchmarking.adapters import CBDBSAdapter, EfficientDRLAdapter, GAEDAdapter, HCBDBSAdapter, MARLAdapter, _load_module
from benchmarking.model import validate_result


def _image() -> np.ndarray:
    x = np.linspace(0, 1, 8, dtype=np.float32)
    return np.repeat(np.tile(x, (8, 1))[..., None], 3, axis=2)


def test_classical_adapters_return_valid_results() -> None:
    image = _image()
    adapters = [GAEDAdapter(), CBDBSAdapter(mono_passes=1, cm_passes=1), HCBDBSAdapter(monochrome_passes=1, color_passes=1)]
    for adapter in adapters:
        result = adapter.run(image, 0)
        validate_result(result, (8, 8))


def test_torch_checkpoint_adapters(tmp_path: Path) -> None:
    efficient = _load_module("paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/efficient_halftoning_drl.py", "test_efficient")
    efficient_config = efficient.paper_reference_config()
    efficient_config = efficient.DRLHalftoningConfig(**{**efficient_config.__dict__, "channels": 4, "num_res_blocks": 1})
    efficient_model = efficient.build_reference_model(efficient_config)
    efficient_path = tmp_path / "efficient.pt"
    torch.save({"config": efficient_config.__dict__, "model_state": efficient_model.state_dict()}, efficient_path)

    marl = _load_module("paper-implementations/drl/halftoning-with-multiagent-drl-implementation/multiagent_drl.py", "test_marl")
    marl_config = marl.paper_reference_config()
    marl_config = marl.MultiAgentDRLConfig(**{**marl_config.__dict__, "channels": 4, "num_res_blocks": 1})
    marl_model = marl.build_reference_model(marl_config)
    marl_path = tmp_path / "marl.pt"
    torch.save({"config": marl_config.__dict__, "model_state": marl_model.state_dict()}, marl_path)

    for adapter in (EfficientDRLAdapter(efficient_path, device="cpu"), MARLAdapter(marl_path, device="cpu")):
        result = adapter.run(_image(), 0)
        validate_result(result, (8, 8))
        assert set(np.unique(result.luma_output)).issubset({0.0, 1.0})


def test_missing_checkpoint_is_a_hard_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required checkpoint"):
        MARLAdapter(tmp_path / "missing.pt", device="cpu")
