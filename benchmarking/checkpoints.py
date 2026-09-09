from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "checkpoints"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checkpoint(name: str, destination: Path | None = None) -> Path:
    metadata_path = METADATA_DIR / f"{name}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "released":
        raise RuntimeError(f"Checkpoint {name!r} is not released yet; status={metadata.get('status')!r}")
    url, expected = metadata["url"], metadata["sha256"]
    destination = destination or METADATA_DIR / metadata["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
