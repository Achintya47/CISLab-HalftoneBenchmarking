from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional accelerator
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

DOT_COLORS: Tuple[str, ...] = ("K", "B", "R", "G", "C", "M", "Y")
DEFAULT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("K", "B"),
    ("R", "M"),
    ("G", "C"),
    ("Y",),
)
PREVIEW_COLORS: Mapping[str, Tuple[float, float, float]] = {
    "K": (0.0, 0.0, 0.0),
    "B": (0.0, 0.0, 1.0),
    "R": (1.0, 0.0, 0.0),
    "G": (0.0, 1.0, 0.0),
    "C": (0.0, 1.0, 1.0),
    "M": (1.0, 0.0, 1.0),
    "Y": (1.0, 1.0, 0.0),
}
_CONVOLUTION_PLAN_CACHE: Dict[Tuple[Tuple[int, int], bytes], "ConvolutionPlan"] = {}
_NEIGHBOR_OFFSETS_CACHE: Dict[int, Tuple[Tuple[int, int], ...]] = {}


@dataclass(frozen=True)
class HCBDBSConfig:
    kernel_size: int = 7
    sigma: float = 1.5
    window_size: int = 7
    monochrome_passes: int = 3
    color_passes: int = 3
    viewing_blur_size: int = 5
    viewing_blur_sigma: float = 0.8
    tolerance: float = 1e-8
    random_seed: int | None = None


@dataclass(frozen=True)
class HCBDBSResult:
    densities: Dict[str, np.ndarray]
    groups: Tuple[Tuple[str, ...], ...]
    final_maps: Dict[str, np.ndarray]
    preview_rgb: np.ndarray
    rendered_rgb: np.ndarray
    viewed_rgb: np.ndarray
    cmy_coverage: Dict[str, np.ndarray]


@dataclass(frozen=True)
class ConvolutionPlan:
    full_shape: Tuple[int, int]
    fft_kernel: np.ndarray
    row_start: int
    row_end: int
    col_start: int
    col_end: int


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


def run_hcb_dbs(
    rgb: np.ndarray,
    config: HCBDBSConfig | None = None,
    groups: Sequence[Sequence[str]] | None = None,
) -> HCBDBSResult:
    cfg = config or HCBDBSConfig()
    rgb = _normalize_rgb(rgb)
    kernel = gaussian_kernel(cfg.kernel_size, cfg.sigma)
    rng = np.random.default_rng(cfg.random_seed)
    densities = mbvcd_exact(rgb)
    hierarchy = tuple(tuple(group) for group in (groups or DEFAULT_GROUPS))
    _validate_groups(hierarchy)

    previous_map = np.zeros(rgb.shape[:2], dtype=np.float32)
    previous_density = np.zeros_like(previous_map)
    final_maps = {color: np.zeros_like(previous_map) for color in DOT_COLORS}

    for subset in hierarchy:
        subset_density = sum(densities[color] for color in subset)
        target_density = np.clip(previous_density + subset_density, 0.0, 1.0)
        cumulative_map = constrained_monochrome_dbs(
            target_density=target_density,
            previous_map=previous_map,
            previous_density=previous_density,
            kernel=kernel,
            window_size=cfg.window_size,
            passes=cfg.monochrome_passes,
            tolerance=cfg.tolerance,
            rng=rng,
        )
        op_map = np.clip(cumulative_map - previous_map, 0.0, 1.0)
        subset_maps = assign_subset_recursive(
            subset=subset,
            op_map=op_map,
            densities=densities,
            kernel=kernel,
            window_size=cfg.window_size,
            passes=cfg.color_passes,
            tolerance=cfg.tolerance,
            rng=rng,
        )
        for color, mask in subset_maps.items():
            final_maps[color] += mask
        previous_map = cumulative_map
        previous_density = target_density

    _validate_final_maps(final_maps)
    preview = render_preview(final_maps)
    cmy_coverage = dot_colors_to_cmy_coverage(final_maps)
    rendered = render_subtractive_rgb(cmy_coverage)
    viewed = apply_viewing_blur(
        rendered,
        kernel_size=cfg.viewing_blur_size,
        sigma=cfg.viewing_blur_sigma,
    )
    return HCBDBSResult(
        densities=densities,
        groups=hierarchy,
        final_maps=final_maps,
        preview_rgb=preview,
        rendered_rgb=rendered,
        viewed_rgb=viewed,
        cmy_coverage=cmy_coverage,
    )


def mbvcd_exact(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    c = 1.0 - r
    m = 1.0 - g
    y = 1.0 - b

    k = np.clip(c + m + y - 2.0, 0.0, 1.0)
    c1 = c - k
    m1 = m - k
    y1 = y - k

    total_rgb = np.clip(c1 + m1 + y1 - (1.0 - k), 0.0, 1.0)
    b_col = total_rgb - np.minimum(y1, total_rgb)

    c2 = c1 - b_col
    m2 = m1 - b_col
    y2 = y1

    rg_overlap = total_rgb - b_col
    r_col = rg_overlap - np.minimum(c2, rg_overlap)
    g_col = rg_overlap - r_col

    c_final = np.clip(c2 - g_col, 0.0, 1.0)
    m_final = np.clip(m2 - r_col, 0.0, 1.0)
    y_final = np.clip(y2 - r_col - g_col, 0.0, 1.0)

    return {
        "K": k.astype(np.float32),
        "B": b_col.astype(np.float32),
        "R": r_col.astype(np.float32),
        "G": g_col.astype(np.float32),
        "C": c_final.astype(np.float32),
        "M": m_final.astype(np.float32),
        "Y": y_final.astype(np.float32),
    }


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    axis = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def constrained_monochrome_dbs(
    target_density: np.ndarray,
    previous_map: np.ndarray,
    previous_density: np.ndarray,
    kernel: np.ndarray,
    window_size: int,
    passes: int,
    tolerance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    current = initialize_cumulative_map(target_density, previous_density, previous_map, rng)
    current = optimize_binary_map(
        current=current,
        target_density=target_density,
        kernel=kernel,
        window_size=window_size,
        passes=passes,
        tolerance=tolerance,
        mutable_mask=previous_map == 0.0,
    )
    current[previous_map == 1.0] = 1.0
    return current.astype(np.float32)


def initialize_cumulative_map(
    target_density: np.ndarray,
    previous_density: np.ndarray,
    previous_map: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    residual = np.clip(
        (target_density - previous_density) / np.clip(1.0 - previous_density, 1e-8, None),
        0.0,
        1.0,
    )
    current = previous_map.copy().astype(np.float32)
    draws = rng.random(target_density.shape, dtype=np.float32)
    fill_mask = (previous_map == 0.0) & (draws < residual)
    current[fill_mask] = 1.0
    return current


def assign_subset_recursive(
    subset: Sequence[str],
    op_map: np.ndarray,
    densities: Mapping[str, np.ndarray],
    kernel: np.ndarray,
    window_size: int,
    passes: int,
    tolerance: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if len(subset) == 1:
        return {subset[0]: op_map.astype(np.float32)}

    split_index = len(subset) // 2
    left = tuple(subset[:split_index])
    right = tuple(subset[split_index:])
    left_target = sum(densities[color] for color in left)
    right_target = sum(densities[color] for color in right)

    left_map, right_map = initialize_binary_split(op_map, left_target, right_target, rng)
    left_map, right_map = optimize_binary_split(
        left_map=left_map,
        right_map=right_map,
        op_map=op_map,
        left_target=left_target,
        right_target=right_target,
        kernel=kernel,
        window_size=window_size,
        passes=passes,
        tolerance=tolerance,
    )

    left_maps = assign_subset_recursive(
        subset=left,
        op_map=left_map,
        densities=densities,
        kernel=kernel,
        window_size=window_size,
        passes=passes,
        tolerance=tolerance,
        rng=rng,
    )
    right_maps = assign_subset_recursive(
        subset=right,
        op_map=right_map,
        densities=densities,
        kernel=kernel,
        window_size=window_size,
        passes=passes,
        tolerance=tolerance,
        rng=rng,
    )
    return {**left_maps, **right_maps}


def initialize_binary_split(
    op_map: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    total = np.clip(left_target + right_target, 1e-8, None)
    left_prob = np.clip(left_target / total, 0.0, 1.0)
    draws = rng.random(op_map.shape, dtype=np.float32)
    left_map = ((op_map == 1.0) & (draws < left_prob)).astype(np.float32)
    right_map = np.clip(op_map - left_map, 0.0, 1.0).astype(np.float32)
    return left_map, right_map


def optimize_binary_split(
    left_map: np.ndarray,
    right_map: np.ndarray,
    op_map: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    kernel: np.ndarray,
    window_size: int,
    passes: int,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    current_left = left_map.copy()
    current_right = right_map.copy()
    left_state = build_objective_state(current_left, left_target, kernel)
    right_state = build_objective_state(current_right, right_target, kernel)
    active_mask = op_map == 1.0
    active_positions = tuple(map(tuple, np.argwhere(active_mask)))
    neighbor_offsets = get_neighbor_offsets(window_size)

    for _ in range(passes):
        changed = False
        frontier = deque(active_positions)
        queued = set(active_positions)
        while frontier:
            i, j = frontier.popleft()
            queued.discard((i, j))

            """
                The DBS's search here accepted the first improving toggle instead of evaluating
                all swaps and picking the best, as implemented in cb_dbs.py's monochrome_dbs. Thus
                exact DBS logic was missing here. Thus if the toggle improves energy at all, even the slightest,
                it is accepted immediately.
            """

            best_update = None
            best_delta = 0.0

            delta_left = current_right[i, j] - current_left[i, j]
            if delta_left != 0.0:
                toggle_left = evaluate_local_update(
                    left_state.filtered_error,
                    kernel,
                    [(i, j, float(delta_left))],
                    current_left.shape,
                )
                toggle_right = evaluate_local_update(
                    right_state.filtered_error,
                    kernel,
                    [(i, j, float(-delta_left))],
                    current_right.shape,
                )
                toggle_delta = toggle_left.delta_energy + toggle_right.delta_energy
                if toggle_delta + tolerance < best_delta:
                    best_delta = toggle_delta
                    best_update = ("toggle", None, None, toggle_left, toggle_right)

            for ni, nj in iter_valid_neighbors(i, j, active_mask, neighbor_offsets):
                if current_left[i, j] == current_left[ni, nj]:
                    continue
                delta_here = float(current_left[ni, nj] - current_left[i, j])
                swap_left = evaluate_local_update(
                    left_state.filtered_error,
                    kernel,
                    [(i, j, delta_here), (ni, nj, -delta_here)],
                    current_left.shape,
                )
                swap_right = evaluate_local_update(
                    right_state.filtered_error,
                    kernel,
                    [(i, j, -delta_here), (ni, nj, delta_here)],
                    current_right.shape,
                )
                swap_delta = swap_left.delta_energy + swap_right.delta_energy
                if swap_delta + tolerance < best_delta:
                    best_delta = swap_delta
                    best_update = ("swap", ni, nj, swap_left, swap_right)

            if best_update is None:
                continue

            mode, ni, nj, update_left, update_right = best_update
            if mode == "toggle":
                current_left[i, j], current_right[i, j] = current_right[i, j], current_left[i, j]
            else:
                current_left[i, j], current_left[ni, nj] = current_left[ni, nj], current_left[i, j]
                current_right[i, j], current_right[ni, nj] = current_right[ni, nj], current_right[i, j]
                enqueue_neighbors(frontier, queued, ni, nj, active_mask, neighbor_offsets)
            apply_local_update(left_state, update_left)
            apply_local_update(right_state, update_right)
            changed = True
            enqueue_neighbors(frontier, queued, i, j, active_mask, neighbor_offsets)
        if not changed:
            break

    return current_left.astype(np.float32), current_right.astype(np.float32)


def optimize_binary_map(
    current: np.ndarray,
    target_density: np.ndarray,
    kernel: np.ndarray,
    window_size: int,
    passes: int,
    tolerance: float,
    mutable_mask: np.ndarray,
) -> np.ndarray:
    state = current.copy()
    objective_state = build_objective_state(state, target_density, kernel)
    active_mask = mutable_mask.astype(bool, copy=False)
    active_positions = tuple(map(tuple, np.argwhere(active_mask)))
    neighbor_offsets = get_neighbor_offsets(window_size)

    for _ in range(passes):
        changed = False
        frontier = deque(active_positions)
        queued = set(active_positions)
        while frontier:
            i, j = frontier.popleft()
            queued.discard((i, j))

            """
                Same algorithmic bug here, doesn't evaluate all swap candidates, rather
                the first best candidate is choosen, thus changed logic from "First Improvement" to
                "Best Improvement".
            """

            best_update = None
            best_delta = 0.0

            delta = float(1.0 - 2.0 * state[i, j])
            toggle_update = evaluate_local_update(
                objective_state.filtered_error,
                kernel,
                [(i, j, delta)],
                state.shape,
            )
            if toggle_update.delta_energy + tolerance < best_delta:
                best_delta = toggle_update.delta_energy
                best_update = ("toggle", None, None, toggle_update)

            for ni, nj in iter_valid_neighbors(i, j, active_mask, neighbor_offsets):
                if state[i, j] == state[ni, nj]:
                    continue
                swap_delta = float(state[ni, nj] - state[i, j])
                swap_update = evaluate_local_update(
                    objective_state.filtered_error,
                    kernel,
                    [(i, j, swap_delta), (ni, nj, -swap_delta)],
                    state.shape,
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
                state[i, j], state[ni, nj] = state[ni, nj], state[i, j]
                enqueue_neighbors(frontier, queued, ni, nj, active_mask, neighbor_offsets)
            apply_local_update(objective_state, update)
            changed = True
            enqueue_neighbors(frontier, queued, i, j, active_mask, neighbor_offsets)
        if not changed:
            break

    return state.astype(np.float32)


def perceptual_energy(binary_map: np.ndarray, target_density: np.ndarray, kernel: np.ndarray) -> float:
    filtered_error = convolve_same(binary_map - target_density, kernel)
    return float(np.sum(filtered_error ** 2))


def split_energy(
    left_map: np.ndarray,
    right_map: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    kernel: np.ndarray,
) -> float:
    return perceptual_energy(left_map, left_target, kernel) + perceptual_energy(
        right_map, right_target, kernel
    )


def build_objective_state(
    binary_map: np.ndarray, target_density: np.ndarray, kernel: np.ndarray
) -> ObjectiveState:
    filtered_error = convolve_same(binary_map - target_density, kernel)
    return ObjectiveState(
        filtered_error=filtered_error,
        energy=float(np.sum(filtered_error ** 2)),
    )


def evaluate_local_update(
    filtered_error: np.ndarray,
    kernel: np.ndarray,
    updates: Sequence[Tuple[int, int, float]],
    image_shape: Tuple[int, int],
) -> LocalUpdate:
    coords_i = np.asarray([update[0] for update in updates], dtype=np.int32)
    coords_j = np.asarray([update[1] for update in updates], dtype=np.int32)
    deltas = np.asarray([update[2] for update in updates], dtype=np.float32)
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


def apply_local_update(objective_state: ObjectiveState, update: LocalUpdate) -> None:
    objective_state.filtered_error[
        update.top : update.bottom,
        update.left : update.right,
    ] += update.delta_field
    objective_state.energy += update.delta_energy


def compute_union_bounds(
    updates: Sequence[Tuple[int, int, float]],
    image_shape: Tuple[int, int],
    kernel_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    bounds = [
        compute_kernel_patch_bounds(i, j, image_shape, kernel_shape)[:4]
        for i, j, _ in updates
    ]
    top = min(bound[0] for bound in bounds)
    bottom = max(bound[1] for bound in bounds)
    left = min(bound[2] for bound in bounds)
    right = max(bound[3] for bound in bounds)
    return top, bottom, left, right


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


def compute_kernel_patch_bounds(
    i: int,
    j: int,
    image_shape: Tuple[int, int],
    kernel_shape: Tuple[int, int],
) -> Tuple[int, int, int, int, int, int, int, int]:
    height, width = image_shape
    kernel_height, kernel_width = kernel_shape
    radius_y = kernel_height // 2
    radius_x = kernel_width // 2

    top = max(0, i - radius_y)
    bottom = min(height, i + radius_y + 1)
    left = max(0, j - radius_x)
    right = min(width, j + radius_x + 1)

    kernel_top = top - (i - radius_y)
    kernel_bottom = kernel_top + (bottom - top)
    kernel_left = left - (j - radius_x)
    kernel_right = kernel_left + (right - left)
    return top, bottom, left, right, kernel_top, kernel_bottom, kernel_left, kernel_right


def convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    plan = get_convolution_plan(image.shape, kernel)
    fft_image = np.fft.rfftn(image, plan.full_shape, axes=(0, 1))
    full = np.fft.irfftn(fft_image * plan.fft_kernel, plan.full_shape, axes=(0, 1))
    return full[
        plan.row_start : plan.row_end,
        plan.col_start : plan.col_end,
    ].astype(np.float32)


def get_convolution_plan(image_shape: Tuple[int, int], kernel: np.ndarray) -> ConvolutionPlan:
    cache_key = (image_shape, kernel.tobytes())
    plan = _CONVOLUTION_PLAN_CACHE.get(cache_key)
    if plan is not None:
        return plan
    full_shape = (
        image_shape[0] + kernel.shape[0] - 1,
        image_shape[1] + kernel.shape[1] - 1,
    )
    row_start = kernel.shape[0] // 2
    col_start = kernel.shape[1] // 2
    plan = ConvolutionPlan(
        full_shape=full_shape,
        fft_kernel=np.fft.rfftn(kernel, full_shape, axes=(0, 1)),
        row_start=row_start,
        row_end=row_start + image_shape[0],
        col_start=col_start,
        col_end=col_start + image_shape[1],
    )
    _CONVOLUTION_PLAN_CACHE[cache_key] = plan
    return plan


def render_preview(final_maps: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = next(iter(final_maps.values())).shape
    preview = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    for color, mask in final_maps.items():
        preview += mask[:, :, None] * np.asarray(PREVIEW_COLORS[color], dtype=np.float32)
    return np.clip(preview, 0.0, 1.0)


def dot_colors_to_cmy_coverage(final_maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cyan = final_maps["K"] + final_maps["B"] + final_maps["G"] + final_maps["C"]
    magenta = final_maps["K"] + final_maps["B"] + final_maps["R"] + final_maps["M"]
    yellow = final_maps["K"] + final_maps["R"] + final_maps["G"] + final_maps["Y"]
    return {
        "C": np.clip(cyan, 0.0, 1.0).astype(np.float32),
        "M": np.clip(magenta, 0.0, 1.0).astype(np.float32),
        "Y": np.clip(yellow, 0.0, 1.0).astype(np.float32),
    }


def render_subtractive_rgb(cmy_coverage: Mapping[str, np.ndarray]) -> np.ndarray:
    cyan = np.clip(cmy_coverage["C"], 0.0, 1.0)
    magenta = np.clip(cmy_coverage["M"], 0.0, 1.0)
    yellow = np.clip(cmy_coverage["Y"], 0.0, 1.0)
    return np.stack(
        (
            1.0 - cyan,
            1.0 - magenta,
            1.0 - yellow,
        ),
        axis=-1,
    ).astype(np.float32)


def apply_viewing_blur(rgb: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size <= 1 or sigma <= 0:
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)
    kernel = gaussian_kernel(kernel_size, sigma)
    channels = [
        convolve_same(rgb[:, :, channel], kernel)
        for channel in range(rgb.shape[2])
    ]
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0).astype(np.float32)


def inside(shape: Tuple[int, int], i: int, j: int) -> bool:
    return 0 <= i < shape[0] and 0 <= j < shape[1]


def get_neighbor_offsets(window_size: int) -> Tuple[Tuple[int, int], ...]:
    offsets = _NEIGHBOR_OFFSETS_CACHE.get(window_size)
    if offsets is not None:
        return offsets
    radius = window_size // 2
    offsets = tuple(
        (di, dj)
        for di in range(-radius, radius + 1)
        for dj in range(-radius, radius + 1)
        if not (di == 0 and dj == 0)
    )
    _NEIGHBOR_OFFSETS_CACHE[window_size] = offsets
    return offsets


def iter_valid_neighbors(
    i: int,
    j: int,
    active_mask: np.ndarray,
    neighbor_offsets: Sequence[Tuple[int, int]],
) -> Sequence[Tuple[int, int]]:
    for di, dj in neighbor_offsets:
        ni = i + di
        nj = j + dj
        if inside(active_mask.shape, ni, nj) and active_mask[ni, nj]:
            yield ni, nj


def enqueue_neighbors(
    frontier: deque[Tuple[int, int]],
    queued: set[Tuple[int, int]],
    i: int,
    j: int,
    active_mask: np.ndarray,
    neighbor_offsets: Sequence[Tuple[int, int]],
) -> None:
    if (i, j) not in queued:
        frontier.append((i, j))
        queued.add((i, j))
    for ni, nj in iter_valid_neighbors(i, j, active_mask, neighbor_offsets):
        if (ni, nj) not in queued:
            frontier.append((ni, nj))
            queued.add((ni, nj))


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    return np.clip(arr, 0.0, 1.0)


def _validate_groups(groups: Sequence[Sequence[str]]) -> None:
    """
        Can cause RecursionError, when the group is empty, a performance bug 
        worth fixing.
    """
    if any(len(group) == 0 for group in groups):
        raise ValueError("groups must not contain empty subsets")
    flattened = [color for group in groups for color in group]
    if tuple(sorted(flattened)) != tuple(sorted(DOT_COLORS)):
        raise ValueError("groups must cover each dot color exactly once")
    if len(flattened) != len(set(flattened)):
        raise ValueError("groups must not repeat colors")


def _validate_final_maps(final_maps: Mapping[str, np.ndarray]) -> None:
    occupancy = sum(final_maps[color] for color in DOT_COLORS)
    if float(np.max(occupancy)) > 1.0 + 1e-6:
        raise ValueError("final maps overlap outside the allowed hierarchy")