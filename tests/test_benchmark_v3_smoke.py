from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking_v3.run_benchmark import run  # noqa: E402


def _write_config(path: Path, output_dir: Path, datasets_root: Path) -> None:
    path.write_text(
        f'''
seed = 0
output_dir = "{output_dir.as_posix()}"
datasets_root = "{datasets_root.as_posix()}"
device = "cpu"
synthetic_size = 24
working_size = 24
compute_lpips = false
comparisons_per_method = 2

[icc]
cmyk_profile_path = "benchmarking_v3/icc_profiles/CGATS001Compat-v2-micro.icc"
rendering_intent = "relative_colorimetric"
black_point_compensation = true

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
passes = 1

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


def test_full_cmyk_pipeline_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.toml"
    output_dir = tmp_path / "output"
    _write_config(config_path, output_dir, tmp_path / "no_datasets")

    result_dir = run(config_path, output_override=output_dir)

    for filename in ("manifest.json", "metrics.csv", "summary.json", "leaderboard.md", "run.json", "icc_fingerprint.json"):
        assert (result_dir / filename).is_file(), filename

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"dbs", "error_diffusion", "ordered_dithering", "deep_learning"}
    for method, values in summary.items():
        for key in ("psnr", "ssim", "delta_e00", "anisotropy_index"):
            assert key in values, f"{method} missing {key}"

    run_payload = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["n_items"] == 5 * (1 + 3)
    assert run_payload["icc"]["use_icc"] is False  # shipped profile is one-directional -> naive fallback, both legs
    assert run_payload["icc"]["cmyk_profile_sha256"]

    for method in ("dbs", "error_diffusion", "ordered_dithering", "deep_learning"):
        assert (result_dir / "comparisons" / f"{method}_cmyk_comparison.png").is_file()

    # Sanity check the metrics.csv actually has CMYK-pipeline-specific columns
    import csv

    with (result_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == run_payload["n_items"] * run_payload["n_algorithms"]
    assert "delta_e00" in rows[0]
    assert "anisotropy_index" in rows[0]
    assert float(rows[0]["reconstruction_sigma"]) == 1.2


def test_cli_style_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "base.toml"
    output_dir = tmp_path / "out2"
    _write_config(config_path, output_dir, tmp_path / "no_datasets")

    result_dir = run(
        config_path,
        output_override=output_dir,
        images_per_family=1,
        disable_methods=["deep_learning"],
        no_lpips=True,
        quiet=True,
    )
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert "deep_learning" not in summary
    assert set(summary) == {"dbs", "error_diffusion", "ordered_dithering"}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
