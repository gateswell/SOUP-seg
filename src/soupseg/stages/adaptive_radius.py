"""
Stage 2 Extension: Adaptive Radius Integration for SoupSeg v1.1.0.

This module provides the integration layer between the adaptive radius
computation (models/adaptive_radius.py) and the nuclei detection pipeline
(stages/nuclei.py).

When `use_adaptive_radius=True` in the config, the fixed expansion_radius_um
is replaced by per-cell adaptive radii computed from local cell density,
image intensity, and transcript density.

Usage:
    from soupseg.stages.adaptive_radius import adaptive_detect_nuclei

    nuclei_result, cell_mask, cell_props = adaptive_detect_nuclei(
        preprocessed_image,
        transcript_coords=transcript_df[['x', 'y']].values,
        config=AdaptiveRadiusStageConfig(use_adaptive_radius=True),
    )
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import warnings

from ..models.adaptive_radius import (
    AdaptiveRadiusConfig,
    compute_adaptive_radius_map,
    adaptive_expand_nuclei,
    compute_cell_density_map,
    compute_transcript_density_map,
)
from .nuclei import (
    detect_nuclei_otsu,
    detect_nuclei_adaptive,
    NucleiDetectionResult,
)


@dataclass
class AdaptiveRadiusStageConfig:
    """Configuration for adaptive radius nuclei detection."""

    # Detection method
    nuclei_method: str = "otsu"
    min_area_um2: float = 50.0
    max_area_um2: float = 2000.0
    pixel_size_um: float = 0.5

    # Adaptive radius settings
    use_adaptive_radius: bool = False

    # Base (fixed) radius — used when adaptive is disabled
    expansion_radius_um: float = 6.0

    # Adaptive radius config (forwarded to AdaptiveRadiusConfig)
    adaptive_base_radius_um: float = 6.0
    adaptive_min_radius_um: float = 3.0
    adaptive_max_radius_um: float = 12.0
    adaptive_density_sigma_um: float = 30.0
    adaptive_density_influence: float = 0.5
    adaptive_intensity_influence: float = 0.3
    adaptive_transcript_influence: float = 0.2
    adaptive_smooth_sigma_px: float = 3.0

    # Fallback to fixed if adaptive fails
    fallback_to_fixed: bool = True

    verbose: bool = False


def _build_adaptive_config(stage_config: AdaptiveRadiusStageConfig) -> AdaptiveRadiusConfig:
    """Build AdaptiveRadiusConfig from stage config."""
    return AdaptiveRadiusConfig(
        base_radius_um=stage_config.adaptive_base_radius_um,
        min_radius_um=stage_config.adaptive_min_radius_um,
        max_radius_um=stage_config.adaptive_max_radius_um,
        density_sigma_um=stage_config.adaptive_density_sigma_um,
        density_influence=stage_config.adaptive_density_influence,
        intensity_influence=stage_config.adaptive_intensity_influence,
        transcript_influence=stage_config.adaptive_transcript_influence,
        smooth_sigma_px=stage_config.adaptive_smooth_sigma_px,
        pixel_size_um=stage_config.pixel_size_um,
        verbose=stage_config.verbose,
    )


def adaptive_detect_nuclei(
    image: np.ndarray,
    transcript_coords: Optional[np.ndarray] = None,
    config: Optional[AdaptiveRadiusStageConfig] = None,
    **kwargs,
) -> Tuple[NucleiDetectionResult, np.ndarray, List[Dict]]:
    """
    Detect nuclei and expand to cells using adaptive radius.

    This is a drop-in replacement for nuclei.detect_nuclei() that supports
    adaptive per-cell expansion radius.

    Args:
        image: Preprocessed ssDNA image (H, W), float32 [0, 1].
        transcript_coords: Optional Nx2 array of transcript (x, y) in pixels.
        config: AdaptiveRadiusStageConfig. Uses defaults if None.
        **kwargs: Additional arguments passed to Otsu/adaptive detector.

    Returns:
        Tuple of (NucleiDetectionResult, cell_mask, cell_properties)
    """
    if config is None:
        config = AdaptiveRadiusStageConfig()

    pixel_size_um = config.pixel_size_um
    min_area_px = int(config.min_area_um2 / (pixel_size_um ** 2))
    max_area_px = int(config.max_area_um2 / (pixel_size_um ** 2))

    # --- Step 1: Detect nuclei ---
    if config.nuclei_method == "otsu":
        nuclei_result = detect_nuclei_otsu(
            image,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            pixel_size_um=pixel_size_um,
        )
    elif config.nuclei_method == "adaptive":
        nuclei_result = detect_nuclei_adaptive(
            image,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            pixel_size_um=pixel_size_um,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown nuclei method: {config.nuclei_method}")

    nuclei_mask = nuclei_result.nuclei_mask
    nuclei_labels = nuclei_result.nuclei_labels

    if nuclei_result.n_nuclei == 0:
        if config.verbose:
            warnings.warn("No nuclei detected.")
        return nuclei_result, np.zeros_like(nuclei_mask), []

    # --- Step 2: Expand nuclei ---
    if config.use_adaptive_radius:
        if config.verbose:
            print(f"[AdaptiveRadius] Expanding {nuclei_result.n_nuclei} nuclei with adaptive radius")

        adaptive_cfg = _build_adaptive_config(config)

        try:
            expanded_mask, expand_info = adaptive_expand_nuclei(
                image=image,
                nuclei_mask=nuclei_mask,
                nuclei_labels=nuclei_labels,
                transcript_coords=transcript_coords,
                config=adaptive_cfg,
            )

            if config.verbose:
                print(f"[AdaptiveRadius] mean_radius={expand_info['mean_radius_um']:.2f} um, "
                      f"std={expand_info['std_radius_um']:.2f} um")

        except Exception as e:
            if config.fallback_to_fixed:
                if config.verbose:
                    warnings.warn(f"Adaptive radius failed ({e}), falling back to fixed radius")
                return _fixed_expansion(
                    nuclei_mask, nuclei_labels, nuclei_result,
                    config.expansion_radius_um, pixel_size_um
                )
            else:
                raise

    else:
        # Fixed radius expansion
        expanded_mask = _expand_fixed(
            nuclei_mask, nuclei_labels,
            config.expansion_radius_um, pixel_size_um
        )

    # --- Step 3: Build cell properties ---
    from skimage.measure import regionprops, label

    labeled_expanded = label(expanded_mask > 0)
    props = regionprops(labeled_expanded)

    cell_properties = []
    for prop in props:
        # Find corresponding nucleus label
        # (unused in current impl — centroid maps to the correct cell automatically)
        _nucleus_label = expanded_mask[int(prop.centroid[0]), int(prop.centroid[1])]
        
        cell_id = f"cell_{prop.label:05d}"
        cell_prop = {
            "cell_id": cell_id,
            "label": prop.label,
            "area_um2": prop.area * (pixel_size_um ** 2),
            "centroid": prop.centroid,
            "bbox": prop.bbox,
            "eccentricity": getattr(prop, "eccentricity", 0.5),
            "extent": getattr(prop, "extent", 0.0),
        }
        cell_properties.append(cell_prop)

    if config.verbose:
        print(f"[AdaptiveRadius] Detected {len(cell_properties)} cells")

    return nuclei_result, expanded_mask, cell_properties


def _expand_fixed(
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    radius_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Expand nuclei with fixed radius (fallback from nuclei.py)."""
    try:
        import cv2
    except ImportError:
        from scipy.ndimage import grey_dilation
        pass

    radius_px = int(round(radius_um / pixel_size_um))
    cell_mask = np.zeros_like(nuclei_labels)

    for label_id in np.unique(nuclei_labels):
        if label_id == 0:
            continue
        nucleus = (nuclei_labels == label_id).astype(np.uint8)

        if cv2 is not None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1)
            )
            expanded = cv2.dilate(nucleus, kernel, iterations=1)
        else:
            expanded = grey_dilation(nucleus, size=radius_px * 2 + 1)

        cell_mask[expanded > 0] = label_id

    return cell_mask


def _fixed_expansion(
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    nuclei_result: NucleiDetectionResult,
    radius_um: float,
    pixel_size_um: float,
) -> Tuple[NucleiDetectionResult, np.ndarray, List[Dict]]:
    """Fallback to fixed-radius expansion."""
    expanded = _expand_fixed(nuclei_mask, nuclei_labels, radius_um, pixel_size_um)
    return nuclei_result, expanded, []


# Alias
segment_adaptive = adaptive_detect_nuclei
