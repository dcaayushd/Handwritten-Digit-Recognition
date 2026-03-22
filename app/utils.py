from __future__ import annotations

import io
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from app.config import Settings, build_versioned_model_path


class InputValidationError(ValueError):
    """Raised when uploaded content cannot be processed as a digit image."""


class UnsupportedFileTypeError(InputValidationError):
    """Raised when the upload is not a supported image format."""


@dataclass(frozen=True)
class ConnectedComponent:
    area: int
    top: int
    left: int
    bottom: int
    right: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class ComponentCluster:
    components: tuple[ConnectedComponent, ...]
    score: float
    top: int
    left: int
    bottom: int
    right: int


@dataclass(frozen=True)
class InferenceCandidate:
    tensor: torch.Tensor
    source: str
    hole_count: int
    hole_vertical_bias: float
    aspect_ratio: float


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path(settings: Settings, *, version: str | None = None) -> Path:
    if version is None:
        return settings.model_path
    return build_versioned_model_path(settings.models_dir, version)


def validate_upload(filename: str | None, content_type: str | None, settings: Settings) -> None:
    if not filename:
        raise InputValidationError("Uploaded file must include a filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise UnsupportedFileTypeError(f"Unsupported file extension '{suffix}'. Allowed: {allowed}.")

    if content_type and content_type.lower() not in settings.allowed_content_types:
        allowed = ", ".join(settings.allowed_content_types)
        raise UnsupportedFileTypeError(f"Unsupported content type '{content_type}'. Allowed: {allowed}.")


def _find_connected_components(mask: np.ndarray) -> list[ConnectedComponent]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[ConnectedComponent] = []
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            area = 0
            top = bottom = row
            left = right = col

            while stack:
                current_row, current_col = stack.pop()
                area += 1
                top = min(top, current_row)
                bottom = max(bottom, current_row)
                left = min(left, current_col)
                right = max(right, current_col)

                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if mask[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))

            components.append(
                ConnectedComponent(
                    area=area,
                    top=top,
                    left=left,
                    bottom=bottom + 1,
                    right=right + 1,
                )
            )

    return sorted(components, key=lambda component: component.area, reverse=True)


def _remove_page_lines(image_array: np.ndarray) -> np.ndarray:
    cleaned = image_array.copy()
    height, width = cleaned.shape
    if height == 0 or width == 0:
        return cleaned

    line_threshold = max(24.0, float(cleaned.max()) * 0.35)
    line_mask = cleaned > line_threshold
    if not line_mask.any():
        return cleaned

    dense_rows = np.where(line_mask.mean(axis=1) > 0.45)[0]
    for row in dense_rows:
        row_start = max(0, row - 1)
        row_end = min(height, row + 2)
        cleaned[row_start:row_end, :] = 0.0

    dense_cols = np.where(line_mask.mean(axis=0) > 0.72)[0]
    for col in dense_cols:
        col_start = max(0, col - 1)
        col_end = min(width, col + 2)
        cleaned[:, col_start:col_end] = 0.0

    return cleaned


def _normalized_center_distance(
    bounds: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> float:
    image_height, image_width = image_shape
    top, left, bottom, right = bounds
    center_row = (top + bottom) / 2
    center_col = (left + right) / 2
    normalized_row = (center_row - (image_height / 2)) / max(image_height / 2, 1)
    normalized_col = (center_col - (image_width / 2)) / max(image_width / 2, 1)
    return float(math.sqrt((normalized_row**2) + (normalized_col**2)))


def _component_score(component: ConnectedComponent, image_shape: tuple[int, int]) -> float:
    image_height, image_width = image_shape
    bounding_area = max(component.width * component.height, 1)
    fill_ratio = component.area / bounding_area
    aspect_ratio = component.width / max(component.height, 1)
    center_distance = _normalized_center_distance(
        (component.top, component.left, component.bottom, component.right),
        image_shape,
    )
    touches_border = (
        component.top <= 1
        or component.left <= 1
        or component.bottom >= image_height - 1
        or component.right >= image_width - 1
    )

    score = float(component.area) * (0.60 + min(fill_ratio * 1.1, 0.55))
    score *= max(0.55, 1.28 - (0.32 * min(center_distance, 1.6)))

    if touches_border:
        score *= 0.18
    if aspect_ratio > 6.0 or aspect_ratio < 0.05:
        score *= 0.15
    elif aspect_ratio > 4.6 or aspect_ratio < 0.10:
        score *= 0.45
    if component.width >= int(image_width * 0.75) and component.height <= max(6, int(image_height * 0.08)):
        score *= 0.05
    if (
        touches_border
        and component.height >= int(image_height * 0.75)
        and component.width <= max(6, int(image_width * 0.08))
    ):
        score *= 0.05

    return score


def _boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    first_top, first_left, first_bottom, first_right = first
    second_top, second_left, second_bottom, second_right = second
    return not (
        first_right <= second_left
        or second_right <= first_left
        or first_bottom <= second_top
        or second_bottom <= first_top
    )


def _expanded_bounds(
    component: ConnectedComponent,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape
    padding = max(4, min(14, int(max(component.width, component.height) * 0.22)))
    return (
        max(0, component.top - padding),
        max(0, component.left - padding),
        min(image_height, component.bottom + padding),
        min(image_width, component.right + padding),
    )


def _cluster_components(
    scored_components: list[tuple[ConnectedComponent, float]],
    image_shape: tuple[int, int],
) -> list[ComponentCluster]:
    groups: list[list[tuple[ConnectedComponent, float]]] = []

    for component, score in scored_components:
        candidate_bounds = _expanded_bounds(component, image_shape)
        matching_indices = [
            index
            for index, group in enumerate(groups)
            if any(
                _boxes_overlap(candidate_bounds, _expanded_bounds(existing_component, image_shape))
                for existing_component, _ in group
            )
        ]

        if not matching_indices:
            groups.append([(component, score)])
            continue

        merged_group = [(component, score)]
        for index in reversed(matching_indices):
            merged_group.extend(groups.pop(index))
        groups.append(merged_group)

    clusters: list[ComponentCluster] = []
    for group in groups:
        components = tuple(component for component, _ in group)
        total_area = sum(component.area for component in components)
        top = min(component.top for component in components)
        left = min(component.left for component in components)
        bottom = max(component.bottom for component in components)
        right = max(component.right for component in components)
        bounding_area = max((bottom - top) * (right - left), 1)
        fill_ratio = total_area / bounding_area
        center_distance = _normalized_center_distance((top, left, bottom, right), image_shape)
        cluster_score = float(sum(score for _, score in group))
        cluster_score *= max(0.60, 1.35 - (0.38 * min(center_distance, 1.6)))
        cluster_score *= 0.70 + min(fill_ratio * 0.70, 0.35)
        clusters.append(
            ComponentCluster(
                components=components,
                score=cluster_score,
                top=top,
                left=left,
                bottom=bottom,
                right=right,
            )
        )

    return sorted(clusters, key=lambda cluster: cluster.score, reverse=True)


def _center_on_canvas(
    image_array: np.ndarray,
    *,
    input_size: int,
    inner_padding: int = 8,
) -> np.ndarray:
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="L")
    width, height = image.size
    target_inner_size = max(8, input_size - inner_padding)
    scale = min(target_inner_size / max(width, 1), target_inner_size / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = np.zeros((input_size, input_size), dtype=np.float32)
    top = (input_size - resized_height) // 2
    left = (input_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(resized, dtype=np.float32)

    total_mass = canvas.sum()
    if total_mass <= 0:
        return canvas

    row_positions = np.arange(input_size, dtype=np.float32)
    col_positions = np.arange(input_size, dtype=np.float32)
    center_row = float((canvas.sum(axis=1) * row_positions).sum() / total_mass)
    center_col = float((canvas.sum(axis=0) * col_positions).sum() / total_mass)

    target_center = (input_size - 1) / 2
    shift_row = int(round(target_center - center_row))
    shift_col = int(round(target_center - center_col))

    shifted = np.zeros_like(canvas)
    source_row_start = max(0, -shift_row)
    source_row_end = input_size - max(0, shift_row)
    source_col_start = max(0, -shift_col)
    source_col_end = input_size - max(0, shift_col)
    target_row_start = max(0, shift_row)
    target_row_end = target_row_start + (source_row_end - source_row_start)
    target_col_start = max(0, shift_col)
    target_col_end = target_col_start + (source_col_end - source_col_start)

    shifted[target_row_start:target_row_end, target_col_start:target_col_end] = canvas[
        source_row_start:source_row_end,
        source_col_start:source_col_end,
    ]
    return shifted


def _otsu_threshold(image_array: np.ndarray) -> float:
    clipped = np.clip(image_array, 0, 255).astype(np.uint8)
    histogram = np.bincount(clipped.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0

    probabilities = histogram / total
    cumulative_probabilities = np.cumsum(probabilities)
    cumulative_means = np.cumsum(probabilities * np.arange(256, dtype=np.float64))
    global_mean = cumulative_means[-1]
    denominator = cumulative_probabilities * (1.0 - cumulative_probabilities)

    between_class_variance = np.full(256, np.nan, dtype=np.float64)
    valid = denominator > 0
    between_class_variance[valid] = (
        (global_mean * cumulative_probabilities[valid] - cumulative_means[valid]) ** 2
    ) / denominator[valid]
    if np.isnan(between_class_variance).all():
        return float(np.mean(clipped))
    return float(np.nanargmax(between_class_variance))


def _unique_centered_views(views: list[np.ndarray]) -> list[np.ndarray]:
    unique_views: list[np.ndarray] = []
    seen: set[bytes] = set()
    for view in views:
        encoded = np.clip(view, 0, 255).astype(np.uint8).tobytes()
        if encoded in seen:
            continue
        seen.add(encoded)
        unique_views.append(view)
    return unique_views


def _extract_digit_crop(image_array: np.ndarray) -> np.ndarray:
    image_array = _remove_page_lines(image_array)
    threshold = max(20.0, float(image_array.max()) * 0.28)
    foreground_mask = image_array > threshold
    components = _find_connected_components(foreground_mask)
    if not components:
        raise InputValidationError("Image does not contain a visible digit.")

    scored_components = [
        (component, _component_score(component, image_array.shape))
        for component in components
        if component.area >= 10
    ]
    if not scored_components:
        raise InputValidationError("Image does not contain a clear handwritten digit.")

    component_clusters = _cluster_components(scored_components, image_array.shape)
    best_cluster = component_clusters[0]
    significant_clusters = [
        cluster
        for cluster in component_clusters
        if cluster.score >= best_cluster.score * 0.55
    ]
    if len(significant_clusters) > 1:
        raise InputValidationError("Please upload one centered digit at a time.")

    return image_array[
        best_cluster.top : best_cluster.bottom,
        best_cluster.left : best_cluster.right,
    ]


def _build_candidate_ink_maps(image: Image.Image) -> list[np.ndarray]:
    base_image = ImageOps.autocontrast(image, cutoff=2).filter(ImageFilter.MedianFilter(size=3))
    base_gray = np.asarray(base_image, dtype=np.float32)
    local_background_small = np.asarray(base_image.filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32)
    local_background_large = np.asarray(base_image.filter(ImageFilter.GaussianBlur(radius=18)), dtype=np.float32)

    candidates = [255.0 - base_gray]
    candidates.append(np.clip(local_background_small - base_gray, 0.0, 255.0))
    candidates.append(np.clip(local_background_large - base_gray, 0.0, 255.0))
    return candidates


def _analyze_binary_digit(binary_image: np.ndarray) -> tuple[int, float]:
    mask = binary_image > 127
    if not mask.any():
        return 0, 0.0

    background = ~mask
    height, width = background.shape
    visited = np.zeros_like(background, dtype=bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack: list[tuple[int, int]] = []

    for row in range(height):
        for col in (0, width - 1):
            if background[row, col] and not visited[row, col]:
                visited[row, col] = True
                stack.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if background[row, col] and not visited[row, col]:
                visited[row, col] = True
                stack.append((row, col))

    while stack:
        current_row, current_col = stack.pop()
        for delta_row, delta_col in neighbors:
            next_row = current_row + delta_row
            next_col = current_col + delta_col
            if 0 <= next_row < height and 0 <= next_col < width:
                if background[next_row, next_col] and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

    hole_centers: list[float] = []
    for row in range(height):
        for col in range(width):
            if not background[row, col] or visited[row, col]:
                continue

            area = 0
            row_sum = 0.0
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                area += 1
                row_sum += current_row
                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if background[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))

            if area >= 3:
                hole_centers.append((row_sum / area) / max(height - 1, 1))

    if not hole_centers:
        return 0, 0.0

    vertical_bias = float((sum(hole_centers) / len(hole_centers)) - 0.5)
    return len(hole_centers), vertical_bias


def _resize_for_processing(image: Image.Image, *, max_side: int = 1280) -> Image.Image:
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image

    scale = max_side / largest_side
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def build_inference_candidates(
    image_bytes: bytes,
    *,
    mean: float,
    std: float,
    input_size: int,
) -> list[InferenceCandidate]:
    if not image_bytes:
        raise InputValidationError("Uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("L")
        image = _resize_for_processing(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise InputValidationError("Uploaded file is not a valid PNG or JPEG image.") from exc

    candidates: list[InferenceCandidate] = []
    last_error: InputValidationError | None = None

    for source_index, candidate_ink_map in enumerate(_build_candidate_ink_maps(image), start=1):
        try:
            cropped = _extract_digit_crop(candidate_ink_map)
        except InputValidationError as exc:
            last_error = exc
            continue

        centered_views: list[np.ndarray] = []
        binarized_threshold = max(16.0, _otsu_threshold(cropped))
        binary_source = (cropped >= binarized_threshold).astype(np.float32) * 255.0
        analysis_binary = _center_on_canvas(
            binary_source,
            input_size=input_size,
            inner_padding=8,
        )
        hole_count, hole_vertical_bias = _analyze_binary_digit(analysis_binary)
        aspect_ratio = cropped.shape[1] / max(cropped.shape[0], 1)

        for inner_padding in (10, 6):
            centered = _center_on_canvas(cropped, input_size=input_size, inner_padding=inner_padding)
            centered_image = Image.fromarray(np.clip(centered, 0, 255).astype(np.uint8), mode="L")
            thickened = np.asarray(centered_image.filter(ImageFilter.MaxFilter(size=3)), dtype=np.float32)
            binarized = _center_on_canvas(
                binary_source,
                input_size=input_size,
                inner_padding=inner_padding,
            )

            centered_views.extend([centered, thickened, binarized])

        unique_views = _unique_centered_views(centered_views)
        tensors = [torch.from_numpy(view / 255.0).unsqueeze(0) for view in unique_views]
        batch = torch.stack(tensors, dim=0)
        candidates.append(
            InferenceCandidate(
                tensor=(batch - mean) / std,
                source=f"ink_map_{source_index}",
                hole_count=hole_count,
                hole_vertical_bias=hole_vertical_bias,
                aspect_ratio=float(aspect_ratio),
            )
        )

    if not candidates:
        if last_error is not None:
            raise last_error
        raise InputValidationError("Image does not contain a clear handwritten digit.")

    return candidates


def build_inference_batch(
    image_bytes: bytes,
    *,
    mean: float,
    std: float,
    input_size: int,
) -> torch.Tensor:
    candidates = build_inference_candidates(
        image_bytes,
        mean=mean,
        std=std,
        input_size=input_size,
    )
    return candidates[0].tensor


def preprocess_image_bytes(
    image_bytes: bytes,
    *,
    mean: float,
    std: float,
    input_size: int,
) -> torch.Tensor:
    batch = build_inference_batch(
        image_bytes,
        mean=mean,
        std=std,
        input_size=input_size,
    )
    return batch[:1]
