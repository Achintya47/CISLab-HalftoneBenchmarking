"""Dataset fetch + stratified sampling for the 5 content families.

Network reality check: the datasets named in the spec (Table under Sec. 4,
and the Kaggle links in Sec. 8) live on kaggle.com, which is not reachable
from every execution environment (in particular, this project's own sandbox
only allow-lists pypi/npm/github-style hosts). So this module is layered:

  1. `local_datasets_root/<key>/...`  -- if the user has already downloaded
     a dataset (e.g. via `kagglehub`, or the existing `datasets/kodak` used
     by the rest of this repo), it is used directly. This is the path used
     in a normal researcher's environment with `kaggle.json` configured.
  2. `kagglehub.dataset_download(...)` -- attempted best-effort if (1) finds
     nothing and the `kagglehub` package + credentials are available.
  3. A deterministic synthetic generator -- guarantees the whole pipeline
     (including CI/sandboxes with no internet) can always run end-to-end.
     Synthetic images are clearly tagged `synthetic=True` in the manifest
     and in `run.json` provenance so nobody mistakes a smoke run for a real
     benchmark result.

Swap step (1)/(2) for your own loader by registering a new key in
`DATASET_FETCHERS` -- see EXTENDING.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image

from .spec import CONTENT_FAMILIES, DEFAULT_IMAGES_PER_FAMILY

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetImage:
    family: str
    dataset_key: str
    image_id: str
    path: Path | None
    synthetic: bool


# --- Kaggle dataset slugs referenced by the spec (Sec. 8) -------------------
KAGGLE_SLUGS = {
    "financial_data": "mehaksingal/personal-financial-dataset-for-india",
    "rvl_cdip": "pdavpoojan/the-rvlcdip-dataset-test",
    "intel": "puneet6060/intel-image-classification",
    "kodak": "sherylmehta/kodak-dataset",
    "kth_tips": "ag3ntsp1d3rx/kth-tips-2",
    "dtd": "jmexpert/describable-textures-dataset-dtd",
    "fairface": "aibloy/fairface",
    "flowers": "alxmamaev/flowers-recognition",
    "bsds500": "balraj98/berkeley-segmentation-dataset-500-bsds500",
}


def _list_local_images(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _try_kagglehub_download(dataset_key: str) -> Path | None:
    slug = KAGGLE_SLUGS.get(dataset_key)
    if slug is None:
        return None
    try:
        import kagglehub  # type: ignore

        path = kagglehub.dataset_download(slug)
        return Path(path)
    except Exception:
        return None


def _synthetic_image(family: str, index: int, size: int, seed: int) -> np.ndarray:
    """A deterministic, content-family-flavored synthetic stand-in.

    Not a substitute for the real datasets, but keeps every downstream
    stress-variant / metric code path exercised without network access.
    """
    rng = np.random.default_rng(seed + index * 7919)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size

    if family == "edge_text":
        base = (xx > 0.5).astype(np.float64)
        base += 0.15 * np.sin(2 * np.pi * (xx * 6 + index * 0.13))
    elif family == "scenery_gradient":
        base = 0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy) * (0.6 + 0.1 * index))
    elif family == "texture":
        base = rng.random((size, size))
        base = 0.5 + 0.15 * (base - base.mean())
        base = base + 0.1 * np.sin(2 * np.pi * xx * (8 + index))
    elif family == "color_skin":
        base = 0.5 + 0.3 * np.cos(2 * np.pi * (xx - yy) * (1 + 0.2 * index))
    elif family == "pattern":
        period = 6 + (index % 4) * 3
        base = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * xx * size / period))
    else:
        base = 0.5 * np.ones((size, size))

    gray = np.clip(base, 0.0, 1.0)
    r = np.clip(gray + 0.05 * np.sin(2 * np.pi * yy * (2 + index % 3)), 0.0, 1.0)
    g = gray
    b = np.clip(gray + 0.05 * np.cos(2 * np.pi * xx * (2 + index % 3)), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def sample_family_images(
    family: str,
    count: int,
    *,
    local_datasets_root: Path,
    synthetic_size: int = 128,
    seed: int = 0,
    allow_kagglehub: bool = True,
) -> List[DatasetImage]:
    if family not in CONTENT_FAMILIES:
        raise ValueError(f"Unknown content family: {family}")
    dataset_keys = CONTENT_FAMILIES[family]["dataset_keys"]

    candidates: List[Path] = []
    used_key = None
    for dataset_key in dataset_keys:
        local_root = local_datasets_root / dataset_key
        found = _list_local_images(local_root)
        if found:
            candidates = found
            used_key = dataset_key
            break

    if not candidates and allow_kagglehub:
        for dataset_key in dataset_keys:
            downloaded = _try_kagglehub_download(dataset_key)
            if downloaded is not None:
                found = _list_local_images(downloaded)
                if found:
                    candidates = found
                    used_key = dataset_key
                    break

    if candidates:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        chosen = [candidates[int(i)] for i in sorted(indices)]
        return [
            DatasetImage(family=family, dataset_key=used_key or dataset_keys[0], image_id=path.stem, path=path, synthetic=False)
            for path in chosen
        ]

    return [
        DatasetImage(family=family, dataset_key=f"synthetic:{dataset_keys[0]}", image_id=f"{family}-synth-{i:03d}", path=None, synthetic=True)
        for i in range(count)
    ]


def load_image_rgb(item: DatasetImage, *, synthetic_size: int = 128, seed: int = 0) -> np.ndarray:
    if item.synthetic or item.path is None:
        index = int(item.image_id.rsplit("-", 1)[-1])
        return _synthetic_image(item.family, index, synthetic_size, seed)
    with Image.open(item.path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array


def build_manifest(
    families: Dict[str, int] | None = None,
    *,
    local_datasets_root: Path,
    seed: int = 0,
    synthetic_size: int = 128,
) -> List[DatasetImage]:
    families = families or {name: DEFAULT_IMAGES_PER_FAMILY for name in CONTENT_FAMILIES}
    manifest: List[DatasetImage] = []
    for family, count in families.items():
        manifest.extend(
            sample_family_images(family, count, local_datasets_root=local_datasets_root, synthetic_size=synthetic_size, seed=seed)
        )
    return manifest


def manifest_to_json(manifest: List[DatasetImage]) -> list[dict]:
    return [
        {"family": item.family, "dataset_key": item.dataset_key, "image_id": item.image_id, "path": str(item.path) if item.path else None, "synthetic": item.synthetic}
        for item in manifest
    ]


def save_manifest(manifest: List[DatasetImage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_to_json(manifest), indent=2), encoding="utf-8")
