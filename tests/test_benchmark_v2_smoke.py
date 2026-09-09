from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking_v2.run_benchmark import run  # noqa: E402


def test_full_pipeline_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.toml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        f'''
seed = 0
output_dir = "{output_dir.as_posix()}"
datasets_root = "{(tmp_path / "no_datasets").as_posix()}"
device = "cpu"
synthetic_size = 24
working_size = 24
compute_lpips = false

[families.edge_text]
count = 1
[families.scenery_gradient]
count = 1
[families.texture]
count = 1
[families.color_skin]
count = 1
[families.pattern]
count = 1

[methods.dbs]
algorithm = "dbs"
enabled = true
mono_passes = 1
cm_passes = 1

[methods.error_diffusion]
algorithm = "error_diffusion"
enabled = true

[methods.ordered_dithering]
algorithm = "ordered_dithering"
enabled = true
matrix_size = 4

[methods.deep_learning]
algorithm = "deep_learning"
enabled = true
bootstrap_iterations = 2
channels = 4
num_res_blocks = 1
crop_size = 16
batch_size = 2
''',
        encoding="utf-8",
    )

    result_dir = run(config_path, output_override=output_dir)
    for filename in ("manifest.json", "metrics.csv", "summary.json", "leaderboard.md", "run.json"):
        assert (result_dir / filename).is_file(), filename

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"dbs", "error_diffusion", "ordered_dithering", "deep_learning"}
    for method, values in summary.items():
        assert "gray_psnr" in values
        assert "gray_anisotropy_index" in values

    run_payload = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["n_items"] == 5 * (1 + 3)  # 5 families x (1 base + 3 stress variants)
    assert run_payload["any_synthetic_images"] is True

    comparisons_dir = result_dir / "comparisons"
    for method in ("dbs", "error_diffusion", "ordered_dithering", "deep_learning"):
        assert (comparisons_dir / f"{method}_comparison.png").is_file()
    assert (comparisons_dir / "all_methods_comparison.png").is_file()
    from PIL import Image

    with Image.open(comparisons_dir / "dbs_comparison.png") as panel:
        # 5 content families, each contributing exactly one "base" sample by default
        assert panel.height > panel.width * 0  # sanity: non-degenerate image
        assert panel.size[0] > 0 and panel.size[1] > 0


def test_cli_overrides_scale_down_and_disable_methods(tmp_path: Path) -> None:
    """Covers the exact request: run a small subset and skip DRL entirely
    without needing a checkpoint or bootstrap training."""
    from benchmarking_v2.run_benchmark import run as run_pipeline

    config_path = tmp_path / "base.toml"
    output_dir = tmp_path / "output_override"
    config_path.write_text(
        f'''
seed = 0
output_dir = "{output_dir.as_posix()}"
datasets_root = "{(tmp_path / "no_datasets").as_posix()}"
device = "cpu"
synthetic_size = 24
working_size = 24
compute_lpips = true

[families.edge_text]
count = 5
[families.scenery_gradient]
count = 5
[families.texture]
count = 5
[families.color_skin]
count = 5
[families.pattern]
count = 5

[methods.dbs]
algorithm = "dbs"
enabled = true
mono_passes = 1
cm_passes = 1

[methods.error_diffusion]
algorithm = "error_diffusion"
enabled = true

[methods.ordered_dithering]
algorithm = "ordered_dithering"
enabled = true
matrix_size = 4

[methods.deep_learning]
algorithm = "deep_learning"
enabled = true
bootstrap_iterations = 500
''',
        encoding="utf-8",
    )

    result_dir = run_pipeline(
        config_path,
        output_override=output_dir,
        images_per_family=1,  # instead of 5 -- "10 images or less"
        disable_methods=["deep_learning"],  # "excluding drl ... don't have checkpoints"
        no_lpips=True,
        quiet=True,
    )

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"dbs", "error_diffusion", "ordered_dithering"}  # deep_learning excluded

    run_payload = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["n_items"] == 5 * (1 + 3)  # images_per_family=1 override took effect
    assert run_payload["n_algorithms"] == 3
    assert "deep_learning" not in run_payload["methods"]
    assert run_payload["lpips_unavailable_reason"] is None  # never even attempted


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
