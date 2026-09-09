"""Checkpoint resolution for RL-based halftoning methods.

Priority order (highest first):

  1. An explicit `--checkpoint` path passed on the CLI / config.
  2. A previously released, checksum-verified checkpoint registered in
     `checkpoints/*.json` (via `benchmarking.checkpoints.fetch_checkpoint`),
     e.g. the paper-faithful 200k-step run once it exists.
  3. A local bootstrap checkpoint already produced by a prior
     `benchmarking_v2` run, cached under `<run_dir>/bootstrap_checkpoints/`.
  4. As a last resort: run a SHORT bootstrap training loop and cache the
     result for next time.

Step 4 is intentionally not a substitute for the full paper training
protocol (`train_efficient_halftoning_drl.py --variant paper ...`,
200k iterations) -- it exists so `run_benchmark.py` never hard-fails just
because nobody has trained a checkpoint yet, matching the request that "if
no checkpoint then training will run". Anyone who cares about a
paper-faithful DRL row should point `--checkpoint` at a real trained/
released one; the provenance JSON always records which path was used.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _bootstrap_train(
    module,
    config,
    device: torch.device,
    iterations: int,
    seed: int,
) -> "torch.nn.Module":
    torch.manual_seed(seed)
    model = module.build_reference_model(config, device=device)
    optimizer, scheduler = module.build_optimizer_and_scheduler(model, iterations, config)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        # Synthetic training crops: smooth random fields stand in for real
        # photographs during a lightweight bootstrap. This is sufficient to
        # produce a *non-degenerate* policy for smoke-testing the pipeline;
        # it is NOT a substitute for training on the real dataset.
        base = rng.random((config.batch_size, 1, config.crop_size, config.crop_size)).astype(np.float32)
        contone = torch.from_numpy(base).to(device)
        module.train_step(model, optimizer, scheduler, contone, config)
    return model


def ensure_efficient_drl_checkpoint(
    *,
    module,
    config_factory,
    run_dir: Path,
    device: torch.device,
    explicit_checkpoint: Optional[Path] = None,
    released_checkpoint_name: Optional[str] = None,
    bootstrap_iterations: int = 200,
    seed: int = 0,
) -> tuple[Path, str]:
    """Returns (checkpoint_path, provenance_tag) where provenance_tag is one
    of "explicit", "released", "bootstrap_reused", "bootstrap_trained"."""

    if explicit_checkpoint is not None:
        if not explicit_checkpoint.is_file():
            raise FileNotFoundError(f"Explicit checkpoint not found: {explicit_checkpoint}")
        return explicit_checkpoint, "explicit"

    if released_checkpoint_name is not None:
        try:
            from benchmarking.checkpoints import fetch_checkpoint

            path = fetch_checkpoint(released_checkpoint_name)
            return path, "released"
        except Exception:
            pass  # fall through to bootstrap path

    bootstrap_dir = run_dir / "bootstrap_checkpoints"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = bootstrap_dir / "efficient_drl_bootstrap.pt"

    if bootstrap_path.is_file():
        return bootstrap_path, "bootstrap_reused"

    config = config_factory()
    model = _bootstrap_train(module, config, device, bootstrap_iterations, seed)
    torch.save({"config": asdict(config), "model_state": model.state_dict()}, bootstrap_path)
    (bootstrap_dir / "efficient_drl_bootstrap.json").write_text(
        json.dumps({"note": "lightweight bootstrap checkpoint; not a paper-faithful trained model", "iterations": bootstrap_iterations, "config": asdict(config)}, indent=2),
        encoding="utf-8",
    )
    return bootstrap_path, "bootstrap_trained"
