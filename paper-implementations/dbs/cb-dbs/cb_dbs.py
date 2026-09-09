from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


@dataclass(frozen=True)
class CBDBSConfig:
    hvs_kernel_size: int = 11
    hvs_scale_factor: float = 2000.0
    hvs_luminance: float = 100.0
    alpha: float = 1.0
    beta: float = 1.0
    mono_window_size: int = 3
    cm_window_size: int = 7
    mono_passes: int = 10
    cm_passes: int = 10
    tolerance: float = 1e-8
    random_seed: int | None = 0


@dataclass(frozen=True)
class CBDBSResult:
    source_rgb: np.ndarray
    c_plane: np.ndarray
    m_plane: np.ndarray
    y_plane: np.ndarray
    cm_total: np.ndarray
    cm_density: np.ndarray
    blue_mask: np.ndarray
    rendered_rgb: np.ndarray
    preview_rgb: np.ndarray
    viewed_rgb: np.ndarray
    metrics: Dict[str, float]


@dataclass
class ObjectiveState:
    filtered_error: np.ndarray
    energy: float


@dataclass(frozen=True)
class LocalUpdate:
    top: int
    bottom: int
    left: int
    right: int
    delta_field: np.ndarray
    delta_energy: float


_NEIGHBOR_OFFSETS_CACHE: Dict[int, Tuple[Tuple[int, int], ...]] = {}


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array


def save_rgb_image(path: str | Path, rgb: np.ndarray) -> None:
    clipped = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)


def save_gray_image(path: str | Path, image: np.ndarray) -> None:
    clipped = np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 array")
    if rgb.max() > 1.0 or rgb.min() < 0.0:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    return rgb


def nasanen_kernel(size: int, scale_factor: float, luminance: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("hvs_kernel_size must be odd")
    axis = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    c = 0.525
    d = 3.91
    luminance = max(float(luminance), 1e-6)
    scale_factor = max(float(scale_factor), 1e-6)
    k = (np.pi * scale_factor) / (180.0 * (c * np.log(luminance) + d))
    spatial_radius = (2.0 * np.pi / scale_factor) * radius
    kernel = np.power(k * k + spatial_radius * spatial_radius, -1.5, dtype=np.float32)
    kernel /= np.sum(kernel, dtype=np.float64)
    return kernel.astype(np.float32)


def convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(image, kernel, mode="same").astype(np.float32)


def build_objective_state(binary_map: np.ndarray, target_density: np.ndarray, kernel: np.ndarray) -> ObjectiveState:
    filtered_error = convolve_same(binary_map - target_density, kernel)
    return ObjectiveState(filtered_error=filtered_error, energy=float(np.sum(filtered_error * filtered_error)))


def apply_local_update(objective_state: ObjectiveState, update: LocalUpdate) -> None:
    objective_state.filtered_error[update.top : update.bottom, update.left : update.right] += update.delta_field
    objective_state.energy += update.delta_energy


def evaluate_local_update(
    filtered_error: np.ndarray,
    kernel: np.ndarray,
    updates: Sequence[Tuple[int, int, float]],
    image_shape: Tuple[int, int],
) -> LocalUpdate:
    coords_i = np.asarray([item[0] for item in updates], dtype=np.int32)
    coords_j = np.asarray([item[1] for item in updates], dtype=np.int32)
    deltas = np.asarray([item[2] for item in updates], dtype=np.float32)
    top, bottom, left, right, delta_field, delta_energy = _evaluate_local_update_impl(
        filtered_error,
        kernel,
        coords_i,
        coords_j,
        deltas,
        image_shape[0],
        image_shape[1],
    )
    return LocalUpdate(
        top=int(top),
        bottom=int(bottom),
        left=int(left),
        right=int(right),
        delta_field=delta_field,
        delta_energy=float(delta_energy),
    )


@njit(cache=True)
def _evaluate_local_update_impl(
    filtered_error: np.ndarray,
    kernel: np.ndarray,
    coords_i: np.ndarray,
    coords_j: np.ndarray,
    deltas: np.ndarray,
    image_height: int,
    image_width: int,
) -> Tuple[int, int, int, int, np.ndarray, float]:
    kernel_height, kernel_width = kernel.shape
    radius_y = kernel_height // 2
    radius_x = kernel_width // 2
    top = image_height
    bottom = 0
    left = image_width
    right = 0

    for index in range(coords_i.shape[0]):
        i = coords_i[index]
        j = coords_j[index]
        patch_top = max(0, i - radius_y)
        patch_bottom = min(image_height, i + radius_y + 1)
        patch_left = max(0, j - radius_x)
        patch_right = min(image_width, j + radius_x + 1)
        if patch_top < top:
            top = patch_top
        if patch_bottom > bottom:
            bottom = patch_bottom
        if patch_left < left:
            left = patch_left
        if patch_right > right:
            right = patch_right

    delta_field = np.zeros((bottom - top, right - left), dtype=np.float32)
    for index in range(coords_i.shape[0]):
        i = coords_i[index]
        j = coords_j[index]
        delta = deltas[index]

        patch_top = max(0, i - radius_y)
        patch_bottom = min(image_height, i + radius_y + 1)
        patch_left = max(0, j - radius_x)
        patch_right = min(image_width, j + radius_x + 1)

        kernel_top = patch_top - (i - radius_y)
        kernel_bottom = kernel_top + (patch_bottom - patch_top)
        kernel_left = patch_left - (j - radius_x)
        kernel_right = kernel_left + (patch_right - patch_left)

        for row in range(patch_top, patch_bottom):
            delta_row = row - top
            kernel_row = kernel_top + (row - patch_top)
            for col in range(patch_left, patch_right):
                delta_col = col - left
                kernel_col = kernel_left + (col - patch_left)
                delta_field[delta_row, delta_col] += delta * kernel[kernel_row, kernel_col]

    delta_energy = 0.0
    for row in range(top, bottom):
        delta_row = row - top
        for col in range(left, right):
            delta_col = col - left
            delta_value = delta_field[delta_row, delta_col]
            old_value = filtered_error[row, col]
            delta_energy += 2.0 * old_value * delta_value + delta_value * delta_value

    return top, bottom, left, right, delta_field, float(delta_energy)


def get_neighbor_offsets(window_size: int) -> Tuple[Tuple[int, int], ...]:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    cached = _NEIGHBOR_OFFSETS_CACHE.get(window_size)
    if cached is not None:
        return cached
    radius = window_size // 2
    offsets = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            offsets.append((di, dj))
    result = tuple(offsets)
    _NEIGHBOR_OFFSETS_CACHE[window_size] = result
    return result


def iter_valid_neighbors(i: int, j: int, shape: Tuple[int, int], offsets: Iterable[Tuple[int, int]]):
    height, width = shape
    for di, dj in offsets:
        ni = i + di
        nj = j + dj
        if 0 <= ni < height and 0 <= nj < width:
            yield ni, nj


def initialize_binary_map(target_density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random(target_density.shape, dtype=np.float32)
    return (draws < target_density).astype(np.float32)


def monochrome_dbs(
    target_density: np.ndarray,
    kernel: np.ndarray,
    window_size: int,
    passes: int,
    tolerance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    state = initialize_binary_map(target_density, rng)
    objective_state = build_objective_state(state, target_density, kernel)
    positions = tuple(map(tuple, np.argwhere(np.ones_like(state, dtype=bool))))
    neighbor_offsets = get_neighbor_offsets(window_size)
    image_shape = state.shape

    for _ in range(passes):
        changed = False
        for i, j in positions:
            best_update: tuple[str, int | None, int | None, LocalUpdate] | None = None
            best_delta = 0.0

            toggle_delta = float(1.0 - 2.0 * state[i, j])
            toggle_update = evaluate_local_update(
                objective_state.filtered_error,
                kernel,
                [(i, j, toggle_delta)],
                image_shape,
            )
            if toggle_update.delta_energy + tolerance < best_delta:
                best_delta = toggle_update.delta_energy
                best_update = ("toggle", None, None, toggle_update)

            for ni, nj in iter_valid_neighbors(i, j, image_shape, neighbor_offsets):
                if state[i, j] == state[ni, nj]:
                    continue
                swap_delta = float(state[ni, nj] - state[i, j])
                swap_update = evaluate_local_update(
                    objective_state.filtered_error,
                    kernel,
                    [(i, j, swap_delta), (ni, nj, -swap_delta)],
                    image_shape,
                )
                if swap_update.delta_energy + tolerance < best_delta:
                    best_delta = swap_update.delta_energy
                    best_update = ("swap", ni, nj, swap_update)

            if best_update is None:
                continue

            mode, ni, nj, update = best_update
            if mode == "toggle":
                state[i, j] = 1.0 - state[i, j]
            else:
                assert ni is not None and nj is not None
                state[i, j], state[ni, nj] = state[ni, nj], state[i, j]
            apply_local_update(objective_state, update)
            changed = True
        if not changed:
            break
    return state.astype(np.float32)


def preprocess_cmy(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    c = 1.0 - rgb[:, :, 0]
    m = 1.0 - rgb[:, :, 1]
    y = 1.0 - rgb[:, :, 2]
    combined = c + m
    c_prime = np.where(combined <= 1.0, c, 1.0 - m).astype(np.float32)
    m_prime = np.where(combined <= 1.0, m, 1.0 - c).astype(np.float32)
    cm_total = np.clip(c_prime + m_prime, 0.0, 1.0).astype(np.float32)
    return {
        "c": c.astype(np.float32),
        "m": m.astype(np.float32),
        "y": y.astype(np.float32),
        "c_prime": c_prime,
        "m_prime": m_prime,
        "cm_total": cm_total,
        "combined_cm": combined.astype(np.float32),
    }


def initialize_cm_assignment(
    g_cm: np.ndarray,
    c_prime: np.ndarray,
    m_prime: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    total = np.clip(c_prime + m_prime, 1e-8, None)
    c_probability = np.clip(c_prime / total, 0.0, 1.0)
    draws = rng.random(g_cm.shape, dtype=np.float32)
    assign_c = (g_cm == 1.0) & (draws < c_probability)
    return assign_c.astype(np.float32)


def optimize_cm_swaps(
    g_cm: np.ndarray,
    c_prime: np.ndarray,
    m_prime: np.ndarray,
    kernel: np.ndarray,
    config: CBDBSConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    c_map = initialize_cm_assignment(g_cm, c_prime, m_prime, rng)
    active_mask = g_cm == 1.0
    c_map = np.where(active_mask, c_map, 0.0).astype(np.float32)
    m_map = np.where(active_mask, 1.0 - c_map, 0.0).astype(np.float32)

    c_state = build_objective_state(c_map, c_prime, kernel)
    m_state = build_objective_state(m_map, m_prime, kernel)
    positions = tuple(map(tuple, np.argwhere(active_mask)))
    neighbor_offsets = get_neighbor_offsets(config.cm_window_size)
    image_shape = g_cm.shape

    for _ in range(config.cm_passes):
        changed = False
        for i, j in positions:
            best_neighbor: tuple[int, int, LocalUpdate, LocalUpdate] | None = None
            best_delta = 0.0
            for ni, nj in iter_valid_neighbors(i, j, image_shape, neighbor_offsets):
                if not active_mask[ni, nj]:
                    continue
                if c_map[i, j] == c_map[ni, nj]:
                    continue
                delta_here = float(c_map[ni, nj] - c_map[i, j])
                swap_c = evaluate_local_update(
                    c_state.filtered_error,
                    kernel,
                    [(i, j, delta_here), (ni, nj, -delta_here)],
                    image_shape,
                )
                swap_m = evaluate_local_update(
                    m_state.filtered_error,
                    kernel,
                    [(i, j, -delta_here), (ni, nj, delta_here)],
                    image_shape,
                )
                delta_energy = config.alpha * swap_c.delta_energy + config.beta * swap_m.delta_energy
                if delta_energy + config.tolerance < best_delta:
                    best_delta = delta_energy
                    best_neighbor = (ni, nj, swap_c, swap_m)
            if best_neighbor is None:
                continue
            ni, nj, swap_c, swap_m = best_neighbor
            c_map[i, j], c_map[ni, nj] = c_map[ni, nj], c_map[i, j]
            m_map[i, j], m_map[ni, nj] = m_map[ni, nj], m_map[i, j]
            apply_local_update(c_state, swap_c)
            apply_local_update(m_state, swap_m)
            changed = True
        if not changed:
            break
    return c_map.astype(np.float32), m_map.astype(np.float32)


def render_subtractive_rgb(c_plane: np.ndarray, m_plane: np.ndarray, y_plane: np.ndarray) -> np.ndarray:
    rgb = np.stack(
        [
            1.0 - c_plane,
            1.0 - m_plane,
            1.0 - y_plane,
        ],
        axis=2,
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def render_preview_rgb(c_plane: np.ndarray, m_plane: np.ndarray, y_plane: np.ndarray) -> np.ndarray:
    c = c_plane.astype(bool)
    m = m_plane.astype(bool)
    y = y_plane.astype(bool)
    preview = np.ones(c_plane.shape + (3,), dtype=np.float32)
    preview[c & ~m & ~y] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    preview[~c & m & ~y] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    preview[~c & ~m & y] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    preview[c & m & ~y] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    preview[c & ~m & y] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    preview[~c & m & y] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    preview[c & m & y] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return preview


def apply_viewing_blur(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    channels = [convolve_same(rgb[:, :, channel], kernel) for channel in range(3)]
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0).astype(np.float32)


def run_cb_dbs(rgb: np.ndarray, config: CBDBSConfig | None = None) -> CBDBSResult:
    cfg = config or CBDBSConfig()
    rng = np.random.default_rng(cfg.random_seed)
    source = normalize_rgb(rgb)
    terms = preprocess_cmy(source)
    kernel = nasanen_kernel(cfg.hvs_kernel_size, cfg.hvs_scale_factor, cfg.hvs_luminance)

    cm_total = monochrome_dbs(
        target_density=terms["cm_total"],
        kernel=kernel,
        window_size=cfg.mono_window_size,
        passes=cfg.mono_passes,
        tolerance=cfg.tolerance,
        rng=rng,
    )

    c_prime_map, m_prime_map = optimize_cm_swaps(
        g_cm=cm_total,
        c_prime=terms["c_prime"],
        m_prime=terms["m_prime"],
        kernel=kernel,
        config=cfg,
        rng=rng,
    )

    blue_mask = ((cm_total == 0.0) & (terms["combined_cm"] >= 1.0)).astype(np.float32)
    c_plane = np.clip(c_prime_map + blue_mask, 0.0, 1.0).astype(np.float32)
    m_plane = np.clip(m_prime_map + blue_mask, 0.0, 1.0).astype(np.float32)
    y_plane = monochrome_dbs(
        target_density=terms["y"],
        kernel=kernel,
        window_size=cfg.mono_window_size,
        passes=cfg.mono_passes,
        tolerance=cfg.tolerance,
        rng=rng,
    )

    rendered = render_subtractive_rgb(c_plane, m_plane, y_plane)
    preview = render_preview_rgb(c_plane, m_plane, y_plane)
    viewed = apply_viewing_blur(rendered, kernel)
    metrics = {
        "mean_c": float(c_plane.mean()),
        "mean_m": float(m_plane.mean()),
        "mean_y": float(y_plane.mean()),
        "mean_blue": float(blue_mask.mean()),
        "mean_cm_total": float(cm_total.mean()),
    }
    return CBDBSResult(
        source_rgb=source,
        c_plane=c_plane,
        m_plane=m_plane,
        y_plane=y_plane,
        cm_total=cm_total,
        cm_density=terms["cm_total"],
        blue_mask=blue_mask,
        rendered_rgb=rendered,
        preview_rgb=preview,
        viewed_rgb=viewed,
        metrics=metrics,
    )
