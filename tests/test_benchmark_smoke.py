from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from benchmarking.adapters import _load_module
from benchmarking.run import run_benchmark


def test_all_method_synthetic_benchmark(tmp_path: Path) -> None:
    images = tmp_path / "kodak"
    images.mkdir()
    gradient = np.tile(np.linspace(0, 255, 8, dtype=np.uint8), (8, 1))
    Image.fromarray(np.repeat(gradient[..., None], 3, axis=2)).save(images / "kodim01.png")

    efficient = _load_module("paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/efficient_halftoning_drl.py", "smoke_efficient")
    ec = efficient.paper_reference_config()
    ec = efficient.DRLHalftoningConfig(**{**ec.__dict__, "channels": 4, "num_res_blocks": 1})
    ep = tmp_path / "efficient.pt"
    torch.save({"config": ec.__dict__, "model_state": efficient.build_reference_model(ec).state_dict()}, ep)
    marl = _load_module("paper-implementations/drl/halftoning-with-multiagent-drl-implementation/multiagent_drl.py", "smoke_marl")
    mc = marl.paper_reference_config()
    mc = marl.MultiAgentDRLConfig(**{**mc.__dict__, "channels": 4, "num_res_blocks": 1})
    mp = tmp_path / "marl.pt"
    torch.save({"config": mc.__dict__, "model_state": marl.build_reference_model(mc).state_dict()}, mp)

    config = tmp_path / "smoke.toml"
    config.write_text(
        f'''protocol_version = 1
dataset_root = "{images.as_posix()}"
output_dir = "{(tmp_path / 'output').as_posix()}"
seeds = [0]
[timing]
warmup = 0
repeats = 1
[constant_gray]
size = 8
levels = [0.5]
[methods.gaed]
variant = "gaed"
[methods.cb_dbs]
mono_passes = 1
cm_passes = 1
[methods.hcb_dbs]
monochrome_passes = 1
color_passes = 1
[methods.efficient_drl_paper]
checkpoint = "{ep.as_posix()}"
required_variant = "paper"
[methods.marl_paper]
checkpoint = "{mp.as_posix()}"
''', encoding="utf-8")
    output = run_benchmark(config, device="cpu")
    for filename in ("run.json", "metrics.csv", "summary.json", "leaderboard.md"):
        assert (output / filename).is_file()
    first_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    second_output = run_benchmark(config, device="cpu", output_override=tmp_path / "output-2")
    second_summary = json.loads((second_output / "summary.json").read_text(encoding="utf-8"))
    for summary in (first_summary, second_summary):
        assert set(summary) == {"constant_gray", "constant_gray_secondary", "kodak", "kodak_secondary"}
        assert set(summary["kodak"]) == {"gaed", "cb_dbs", "hcb_dbs", "efficient_drl_paper", "marl_paper"}
    for method in first_summary["kodak"]:
        for key, value in first_summary["kodak"][method].items():
            if key != "runtime_sec":
                assert second_summary["kodak"][method][key] == value
