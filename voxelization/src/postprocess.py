from __future__ import annotations

import numpy as np
from scipy import ndimage


def select_implant_region(
    defect_points: np.ndarray,
    implant_mask: np.ndarray,
    *,
    method: str = "bbox",
    bbox_margin_voxels: int = 12,
    radius_scale: float = 1.1,
    fallback: str = "largest_component",
) -> np.ndarray:
    implant_mask = np.asarray(implant_mask, dtype=bool)
    if method == "none" or not implant_mask.any():
        return implant_mask.astype(np.float32)

    defect_points = np.asarray(defect_points, dtype=np.float32)
    if defect_points.size == 0:
        return apply_component_fallback(implant_mask, fallback).astype(np.float32)

    region_mask = build_defect_region_mask(
        defect_points,
        implant_mask.shape,
        method=method,
        bbox_margin_voxels=bbox_margin_voxels,
        radius_scale=radius_scale,
    )
    filtered = keep_components_touching_region(implant_mask, region_mask)
    if filtered.any():
        return filtered.astype(np.float32)
    return apply_component_fallback(implant_mask, fallback).astype(np.float32)


def build_complete_mask(
    complete_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    return np.asarray(complete_prob >= threshold, dtype=np.float32)


def apply_symmetry_prior(
    complete_prob: np.ndarray,
    defect_region_mask: np.ndarray,
    *,
    axis: int = 0,
    symmetry_weight: float = 0.35,
    defect_only: bool = True,
) -> np.ndarray:
    complete_prob = np.asarray(complete_prob, dtype=np.float32)
    defect_region_mask = np.asarray(defect_region_mask, dtype=bool)
    if symmetry_weight <= 0:
        return complete_prob

    mirrored = np.flip(complete_prob, axis=axis)
    blended = (1.0 - symmetry_weight) * complete_prob + symmetry_weight * mirrored
    if defect_only:
        refined = complete_prob.copy()
        refined[defect_region_mask] = blended[defect_region_mask]
        return refined
    return blended


def build_refined_complete_mask(
    complete_prob: np.ndarray,
    defect_points: np.ndarray,
    volume_shape: tuple[int, int, int],
    *,
    threshold: float = 0.5,
    symmetry_weight: float = 0.0,
    symmetry_axis: int = 0,
    symmetry_defect_only: bool = True,
    defect_region_method: str = "bbox",
    bbox_margin_voxels: int = 12,
    radius_scale: float = 1.1,
) -> np.ndarray:
    complete_prob = np.asarray(complete_prob, dtype=np.float32)
    defect_points = np.asarray(defect_points, dtype=np.float32)
    if defect_points.size == 0:
        return build_complete_mask(complete_prob, threshold=threshold)

    defect_region_mask = build_defect_region_mask(
        defect_points,
        volume_shape,
        method=defect_region_method,
        bbox_margin_voxels=bbox_margin_voxels,
        radius_scale=radius_scale,
    )
    refined_prob = apply_symmetry_prior(
        complete_prob,
        defect_region_mask,
        axis=symmetry_axis,
        symmetry_weight=symmetry_weight,
        defect_only=symmetry_defect_only,
    )
    return build_complete_mask(refined_prob, threshold=threshold)


def build_defect_region_mask(
    defect_points: np.ndarray,
    volume_shape: tuple[int, int, int],
    *,
    method: str,
    bbox_margin_voxels: int,
    radius_scale: float,
) -> np.ndarray:
    coords = np.clip(np.rint(defect_points).astype(int), 0, np.array(volume_shape) - 1)

    if method == "bbox":
        lo = np.clip(coords.min(axis=0) - bbox_margin_voxels, 0, np.array(volume_shape) - 1)
        hi = np.clip(coords.max(axis=0) + bbox_margin_voxels + 1, 0, np.array(volume_shape))
        mask = np.zeros(volume_shape, dtype=bool)
        mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
        return mask

    if method == "radius":
        center = coords.mean(axis=0)
        radius = np.linalg.norm(coords - center, axis=1).max() * radius_scale
        grid = np.indices(volume_shape, dtype=np.float32)
        distances = np.sqrt(((grid - center[:, None, None, None]) ** 2).sum(axis=0))
        return distances <= radius

    raise ValueError(f"Unsupported implant postprocess method: {method}")


def keep_components_touching_region(implant_mask: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(implant_mask)
    if count == 0:
        return np.zeros_like(implant_mask, dtype=bool)

    kept = np.zeros_like(implant_mask, dtype=bool)
    for label in range(1, count + 1):
        component = labels == label
        if np.logical_and(component, region_mask).any():
            kept |= component
    return kept


def apply_component_fallback(implant_mask: np.ndarray, fallback: str) -> np.ndarray:
    if fallback == "none":
        return implant_mask
    if fallback != "largest_component":
        raise ValueError(f"Unsupported implant postprocess fallback: {fallback}")

    labels, count = ndimage.label(implant_mask)
    if count == 0:
        return np.zeros_like(implant_mask, dtype=bool)

    component_sizes = ndimage.sum(implant_mask, labels, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(component_sizes)) + 1
    return labels == largest_label
