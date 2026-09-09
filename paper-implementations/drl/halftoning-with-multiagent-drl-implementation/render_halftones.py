from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from multiagent_drl import (
    MultiAgentDRLConfig,
    build_reference_model,
    compute_reward,
    infer_halftone,
    list_images,
    load_grayscale_image,
    save_triptych,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render halftones from a saved MARL checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint produced by train_multiagent_drl.py.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing input images.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional newline-delimited manifest relative to input-root or repo root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write rendered outputs.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device.")
    parser.add_argument(
        "--binary-mode",
        choices=["threshold", "sample", "posthoc_blue_noise"],
        default="posthoc_blue_noise",
        help="Binary realization mode.",
    )
    parser.add_argument("--max-side", type=int, default=256, help="Resize inputs so the longest side is at most this value. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for stochastic modes.")
    return parser.parse_args()


def resolve_paths(root: Path, manifest: Path | None) -> list[Path]:
    if manifest is None:
        return list_images(root)
    repo_root = Path.cwd().resolve()
    paths: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidate = Path(line)
        if candidate.is_absolute():
            path = candidate
        else:
            root_candidate = (root / candidate).resolve()
            repo_candidate = (repo_root / candidate).resolve()
            path = root_candidate if root_candidate.exists() else repo_candidate
        if not path.exists():
            raise FileNotFoundError(f"Missing input image listed in manifest: {line}")
        paths.append(path)
    return paths


def maybe_resize(image: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return image
    height, width = image.shape[-2:]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[MultiAgentDRLConfig, dict[str, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"Checkpoint missing config payload: {checkpoint_path}")
    config = MultiAgentDRLConfig(**config_payload)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model_state payload: {checkpoint_path}")
    return config, model_state


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    config, model_state = load_checkpoint(args.checkpoint, device)
    model = build_reference_model(config, device=device)
    model.load_state_dict(model_state)
    model.eval()

    input_paths = resolve_paths(args.input_root, args.manifest)
    if not input_paths:
        raise ValueError("No input images found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    generator_device = device.type if device.type == "cuda" else "cpu"

    for index, image_path in enumerate(input_paths):
        contone = load_grayscale_image(image_path).unsqueeze(0).to(device)
        contone = maybe_resize(contone, args.max_side)
        noise_generator = torch.Generator(device=generator_device)
        noise_generator.manual_seed(args.seed + index * 1009)
        noise = torch.randn(contone.shape, generator=noise_generator, device=device, dtype=contone.dtype)
        sample_generator = None
        if args.binary_mode == "sample":
            sample_generator = torch.Generator(device=generator_device)
            sample_generator.manual_seed(args.seed + index * 2027)
        result = infer_halftone(
            model,
            contone,
            config,
            noise=noise,
            binary_mode=args.binary_mode,
            sample_generator=sample_generator,
        )
        stem = image_path.stem
        save_triptych(result, args.output_dir / f"{stem}.png")
        reward, tone_error, ssim = compute_reward(result["halftone"], result["contone"], config)
        payload = {
            "image": str(image_path),
            "binary_mode": args.binary_mode,
            "reward": float(reward.item()),
            "tone_error": float(tone_error.item()),
            "ssim": float(ssim.item()),
            "prob_mean": float(result["probability"].mean().item()),
            "half_mean": float(result["halftone"].mean().item()),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(f"rendered {image_path.name} -> {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
