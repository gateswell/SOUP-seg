"""
Adaptive Dilation Radius Module for SoupSeg v1.1.0.

Instead of using a fixed expansion/dilation radius for all cells, this module
adapts the radius based on:

1. **Local cell density** — In dense regions, cells should expand less to avoid
   overlap; in sparse regions, they can expand more.
2. **Image intensity features** — ssDNA signal strength around each nucleus
   informs how far a cell boundary is likely to extend.
3. **Transcript density** — Regions with high transcript density suggest
   well-defined cellular territories; low density may need larger expansion.

The adaptive radius is computed per-cell (or per-region) and applied during
the nuclei expansion step in Stage 2.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import warnings

from scipy.ndimage import (
    distance_transform_edt,
    label as ndlabel,
    uniform_filter,
    gaussian_filter,
)
from skimage.measure import regionprops, label

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveRadiusConfig:
    """Configuration for adaptive dilation radius computation."""

    # Base radius (microns) — fallback when no adaptation is possible
    base_radius_um: float = 6.0

    # Radius bounds
    min_radius_um: float = 3.0
    max_radius_um: float = 12.0

    # Cell density adaptation
    density_sigma_um: float = 30.0  # Gaussian kernel for density estimation
    density_influence: float = 0.5  # How strongly density affects radius
    # High density → smaller radius; low density → larger radius

    # Image intensity adaptation
    intensity_percentile_low: float = 25.0
    intensity_percentile_high: float = 75.0
    intensity_influence: float = 0.3  # How strongly intensity affects radius
    # Strong signal → can expand further; weak signal → stay close

    # Transcript density adaptation
    transcript_bin_size_um: float = 5.0  # Bin size for transcript density
    transcript_influence: float = 0.2  # How strongly transcript density affects radius

    # Smoothing
    smooth_sigma_px: float = 3.0  # Smooth the radius map to avoid sharp jumps

    # Pixel size
    pixel_size_um: float = 0.5

    verbose: bool = False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_cell_density_map(
    nuclei_mask: np.ndarray,
    pixel_size_um: float = 0.5,
    sigma_um: float = 30.0,
) -> np.ndarray:
    """
    Compute a smooth cell density map from a binary nuclei mask.

    Each nucleus contributes a Gaussian-weighted density. The result is a
    smooth map where high values indicate many nearby nuclei.

    Args:
        nuclei_mask: Binary mask of detected nuclei (H, W).
        pixel_size_um: Pixel size in microns.
        sigma_um: Gaussian kernel sigma in microns.

    Returns:
        Density map (H, W), float64. Higher = more crowded.
    """
    sigma_px = sigma_um / pixel_size_um
    # Convert binary mask to float and smooth
    density = gaussian_filter(nuclei_mask.astype(np.float64), sigma=sigma_px)

    # Normalize to [0, 1]
    d_max = density.max()
    if d_max > 0:
        density /= d_max

    return density


def compute_local_intensity_features(
    image: np.ndarray,
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    pixel_size_um: float = 0.5,
    percentile_low: float = 25.0,
    percentile_high: float = 75.0,
) -> Dict[int, Dict[str, float]]:
    """
    Compute local image intensity features around each nucleus.

    For each nucleus, measure the ssDNA signal intensity in a ring around it
    to estimate how far the cell boundary might extend.

    Args:
        image: Preprocessed ssDNA image (H, W), float32 [0, 1].
        nuclei_mask: Binary nuclei mask (H, W).
        nuclei_labels: Labeled nuclei mask (H, W).
        pixel_size_um: Pixel size in microns.
        percentile_low: Low percentile for intensity range.
        percentile_high: High percentile for intensity range.

    Returns:
        Dict mapping label_id -> {mean_intensity, ring_intensity, spread}
    """
    props = regionprops(nuclei_labels, intensity_image=image)
    features = {}

    for prop in props:
        # Mean intensity inside the nucleus
        mean_inside = prop.mean_intensity

        # Measure intensity in a ring around the nucleus
        # Ring = dilated nucleus minus nucleus
        cell_binary = (nuclei_labels == prop.label)
        from scipy.ndimage import binary_dilation
        structure = np.ones((5, 5))  # ~2.5 um ring at 0.5 um/px
        dilated = binary_dilation(cell_binary, structure=structure)
        ring = dilated & ~cell_binary

        if np.any(ring):
            ring_intensity = image[ring].mean()
        else:
            ring_intensity = 0.0

        # Intensity spread: how much signal extends beyond the nucleus
        # High spread → cell cytoplasm is bright → can expand further
        if mean_inside > 0:
            spread = ring_intensity / (mean_inside + 1e-8)
        else:
            spread = 0.0

        features[prop.label] = {
            "mean_intensity": float(mean_inside),
            "ring_intensity": float(ring_intensity),
            "spread": float(spread),
        }

    return features


def compute_transcript_density_map(
    transcript_coords: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    pixel_size_um: float = 0.5,
    bin_size_um: float = 5.0,
) -> np.ndarray:
    """
    Compute a smooth transcript density map.

    Args:
        transcript_coords: Nx2 array of (x, y) coordinates in pixels, or None.
        image_shape: (H, W) of the image.
        pixel_size_um: Pixel size in microns.
        bin_size_um: Bin size for density estimation.

    Returns:
        Density map (H, W), float64, normalized to [0, 1].
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float64)

    if transcript_coords is None or len(transcript_coords) == 0:
        return density

    bin_px = max(1, int(bin_size_um / pixel_size_um))

    # Histogram-based density
    tx_x = np.clip(transcript_coords[:, 0].astype(int), 0, w - 1)
    tx_y = np.clip(transcript_coords[:, 1].astype(int), 0, h - 1)

    for x, y in zip(tx_x, tx_y):
        density[y, x] += 1.0

    # Smooth with Gaussian
    sigma = bin_px
    density = gaussian_filter(density, sigma=sigma)

    d_max = density.max()
    if d_max > 0:
        density /= d_max

    return density


def compute_adaptive_radius_map(
    image: np.ndarray,
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    transcript_coords: Optional[np.ndarray] = None,
    config: Optional[AdaptiveRadiusConfig] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Compute a per-pixel adaptive dilation radius map.

    The radius at each location is determined by the weighted combination of:
    - Cell density (inverse relationship: dense → small radius)
    - Image intensity (direct relationship: bright → large radius)
    - Transcript density (moderate influence)

    Args:
        image: Preprocessed ssDNA image (H, W), float32 [0, 1].
        nuclei_mask: Binary nuclei mask (H, W).
        nuclei_labels: Labeled nuclei mask (H, W).
        transcript_coords: Optional Nx2 array of transcript coordinates.
        config: AdaptiveRadiusConfig. Uses defaults if None.

    Returns:
        Tuple of:
        - radius_map (H, W): Per-pixel dilation radius in microns.
        - per_cell_radius: Dict mapping label_id -> adaptive radius (um).
    """
    if config is None:
        config = AdaptiveRadiusConfig()

    h, w = image.shape
    pixel_size_um = config.pixel_size_um

    # --- Factor 1: Cell density ---
    density_map = compute_cell_density_map(
        nuclei_mask, pixel_size_um, config.density_sigma_um
    )
    # High density → small radius; density_map ∈ [0, 1]
    density_factor = 1.0 - config.density_influence * density_map
    # density_factor ∈ [1 - density_influence, 1]

    # --- Factor 2: Image intensity ---
    # Normalize image intensity to [0, 1]
    img_norm = image.copy().astype(np.float64)
    i_min = np.percentile(img_norm, config.intensity_percentile_low)
    i_max = np.percentile(img_norm, config.intensity_percentile_high)
    if i_max > i_min:
        img_norm = (img_norm - i_min) / (i_max - i_min)
    else:
        img_norm = np.zeros_like(img_norm)
    img_norm = np.clip(img_norm, 0, 1)

    # Bright signal → can expand further
    intensity_factor = 1.0 - config.intensity_influence + config.intensity_influence * img_norm
    # intensity_factor ∈ [1 - intensity_influence, 1]

    # --- Factor 3: Transcript density ---
    tx_density = compute_transcript_density_map(
        transcript_coords, (h, w), pixel_size_um, config.transcript_bin_size_um
    )
    # Moderate transcript density → moderate radius; very high/low → adjust
    # Use a bell-curve: peak at tx_density = 0.5
    tx_factor_curve = 1.0 - 2.0 * np.abs(tx_density - 0.5)
    tx_factor = 1.0 - config.transcript_influence + config.transcript_influence * tx_factor_curve
    tx_factor = np.clip(tx_factor, 1.0 - config.transcript_influence, 1.0)

    # --- Combine ---
    combined = density_factor * intensity_factor * tx_factor
    # Scale to [min_radius, max_radius]
    combined_min = combined.min()
    combined_max = combined.max()
    if combined_max > combined_min:
        combined_norm = (combined - combined_min) / (combined_max - combined_min)
    else:
        combined_norm = np.ones_like(combined) * 0.5

    radius_map = (
        config.min_radius_um
        + (config.max_radius_um - config.min_radius_um) * combined_norm
    )

    # Smooth the radius map to avoid sharp transitions
    if config.smooth_sigma_px > 0:
        radius_map = gaussian_filter(radius_map, sigma=config.smooth_sigma_px)
        radius_map = np.clip(radius_map, config.min_radius_um, config.max_radius_um)

    # --- Per-cell radius ---
    intensity_features = compute_local_intensity_features(
        image, nuclei_mask, nuclei_labels, pixel_size_um,
        config.intensity_percentile_low, config.intensity_percentile_high,
    )

    props = regionprops(nuclei_labels)
    per_cell_radius = {}

    for prop in props:
        label_id = prop.label

        # Average radius over nucleus region
        cell_pixels = (nuclei_labels == label_id)
        avg_radius = radius_map[cell_pixels].mean()

        # Adjust by intensity spread
        feat = intensity_features.get(label_id, {})
        spread = feat.get("spread", 0.5)
        # spread > 1 means ring is bright → expand more
        spread_multiplier = 0.8 + 0.4 * min(spread, 1.5)  # Range: [0.8, 1.4]

        adjusted = avg_radius * spread_multiplier
        adjusted = np.clip(adjusted, config.min_radius_um, config.max_radius_um)

        per_cell_radius[label_id] = float(adjusted)

    if config.verbose:
        radii = list(per_cell_radius.values())
        print(f"  Adaptive radius: mean={np.mean(radii):.2f} um, "
              f"std={np.std(radii):.2f} um, "
              f"range=[{np.min(radii):.2f}, {np.max(radii):.2f}] um")

    return radius_map, per_cell_radius


def apply_adaptive_dilation(
    nuclei_labels: np.ndarray,
    radius_map: np.ndarray,
    per_cell_radius: Dict[int, float],
    pixel_size_um: float = 0.5,
) -> np.ndarray:
    """
    Apply per-cell adaptive dilation to expand nuclei into cell territories.

    Each nucleus is dilated by its adaptive radius, then overlaps are resolved
    by assigning contested pixels to the nearest nucleus.

    Args:
        nuclei_labels: Labeled nuclei mask (H, W).
        radius_map: Per-pixel radius map in microns (H, W).
        per_cell_radius: Dict mapping label_id -> adaptive radius (um).
        pixel_size_um: Pixel size in microns.

    Returns:
        Expanded cell mask (H, W) with cell labels.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    h, w = nuclei_labels.shape
    expanded = np.zeros((h, w), dtype=np.int32)

    # Sort cells by radius (largest first) to handle overlaps
    sorted_cells = sorted(per_cell_radius.items(), key=lambda x: -x[1])

    # Distance transform from all nuclei
    nuclei_binary = nuclei_labels > 0
    if not np.any(nuclei_binary):
        return nuclei_labels.copy()

    # For each cell, dilate by its adaptive radius
    dilated_masks = {}
    for label_id, radius_um in sorted_cells:
        radius_px = max(1, int(radius_um / pixel_size_um))

        cell_binary = (nuclei_labels == label_id)

        # Create a disk structuring element
        y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
        disk = (x**2 + y**2) <= radius_px**2
        disk = disk.astype(np.uint8)

        dilated = binary_dilation(cell_binary, structure=disk)
        dilated_masks[label_id] = dilated

    # Resolve overlaps using distance to nucleus centroid
    # Compute distance from each pixel to each nucleus
    distance_maps = {}
    props = regionprops(nuclei_labels)

    for prop in props:
        label_id = prop.label
        cell_binary = (nuclei_labels == label_id)
        dist = distance_transform_edt(~cell_binary)
        distance_maps[label_id] = dist

    # Assign each pixel
    for y in range(h):
        for x in range(w):
            if nuclei_labels[y, x] > 0:
                expanded[y, x] = nuclei_labels[y, x]
                continue

            # Find which dilated cells cover this pixel
            covering = [lid for lid, mask in dilated_masks.items() if mask[y, x]]
            if len(covering) == 0:
                continue
            elif len(covering) == 1:
                expanded[y, x] = covering[0]
            else:
                # Assign to nearest nucleus
                best_label = covering[0]
                best_dist = float('inf')
                for lid in covering:
                    d = distance_maps.get(lid, np.full((h, w), float('inf')))[y, x]
                    if d < best_dist:
                        best_dist = d
                        best_label = lid
                expanded[y, x] = best_label

    return expanded


def adaptive_expand_nuclei(
    image: np.ndarray,
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    transcript_coords: Optional[np.ndarray] = None,
    config: Optional[AdaptiveRadiusConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    High-level function: adaptively expand nuclei to cell territories.

    This replaces the fixed-radius expansion in the original SoupSeg v1.0.

    Args:
        image: Preprocessed ssDNA image (H, W), float32 [0, 1].
        nuclei_mask: Binary nuclei mask (H, W).
        nuclei_labels: Labeled nuclei mask (H, W).
        transcript_coords: Optional Nx2 array of transcript coordinates.
        config: AdaptiveRadiusConfig. Uses defaults if None.

    Returns:
        Tuple of:
        - expanded_mask: Cell territory mask with labels (H, W).
        - info: Dict with adaptive radius details.
    """
    if config is None:
        config = AdaptiveRadiusConfig()

    if config.verbose:
        print("=== Adaptive Radius Expansion ===")

    # Compute adaptive radius
    radius_map, per_cell_radius = compute_adaptive_radius_map(
        image, nuclei_mask, nuclei_labels, transcript_coords, config
    )

    # Apply dilation
    expanded = apply_adaptive_dilation(
        nuclei_labels, radius_map, per_cell_radius, config.pixel_size_um
    )

    info = {
        "method": "adaptive",
        "radius_map_shape": radius_map.shape,
        "per_cell_radius": per_cell_radius,
        "mean_radius_um": float(np.mean(list(per_cell_radius.values()))),
        "std_radius_um": float(np.std(list(per_cell_radius.values()))),
        "n_cells": len(per_cell_radius),
    }

    if config.verbose:
        print(f"  Expanded {info['n_cells']} cells with adaptive radius")

    return expanded, info


# ---------------------------------------------------------------------------
# PyTorch-based adaptive radius (optional, for GPU-accelerated computation)
# ---------------------------------------------------------------------------

def compute_adaptive_radius_map_torch(
    image: np.ndarray,
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    transcript_coords: Optional[np.ndarray] = None,
    config: Optional[AdaptiveRadiusConfig] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    GPU-accelerated adaptive radius computation using PyTorch.

    Same interface as compute_adaptive_radius_map but uses torch for
    Gaussian filtering and tensor operations when CUDA is available.

    Falls back to scipy version if PyTorch is not available.
    """
    if not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available, falling back to scipy implementation.")
        return compute_adaptive_radius_map(
            image, nuclei_mask, nuclei_labels, transcript_coords, config
        )

    if config is None:
        config = AdaptiveRadiusConfig()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pixel_size_um = config.pixel_size_um
    h, w = image.shape

    # --- Cell density (using torch) ---
    sigma_px = config.density_sigma_um / pixel_size_um
    nuclei_tensor = torch.from_numpy(nuclei_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    # Gaussian blur via torch
    ksize = int(6 * sigma_px + 1)
    if ksize % 2 == 0:
        ksize += 1
    x_grid = torch.arange(ksize, device=device, dtype=torch.float32) - ksize // 2
    gauss_1d = torch.exp(-x_grid**2 / (2 * sigma_px**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d).unsqueeze(0).unsqueeze(0)

    density_tensor = torch.nn.functional.conv2d(
        nuclei_tensor, gauss_2d, padding=ksize // 2
    )
    density_np = density_tensor.squeeze().cpu().numpy()
    d_max = density_np.max()
    if d_max > 0:
        density_np /= d_max

    # --- Use scipy for the rest (intensity features, transcript density) ---
    # since they don't benefit much from GPU
    density_factor = 1.0 - config.density_influence * density_np

    # Image intensity (same as CPU version)
    img_norm = image.copy().astype(np.float64)
    i_min = np.percentile(img_norm, config.intensity_percentile_low)
    i_max = np.percentile(img_norm, config.intensity_percentile_high)
    if i_max > i_min:
        img_norm = (img_norm - i_min) / (i_max - i_min)
    else:
        img_norm = np.zeros_like(img_norm)
    img_norm = np.clip(img_norm, 0, 1)
    intensity_factor = 1.0 - config.intensity_influence + config.intensity_influence * img_norm

    # Transcript density
    tx_density = compute_transcript_density_map(
        transcript_coords, (h, w), pixel_size_um, config.transcript_bin_size_um
    )
    tx_factor_curve = 1.0 - 2.0 * np.abs(tx_density - 0.5)
    tx_factor = 1.0 - config.transcript_influence + config.transcript_influence * tx_factor_curve
    tx_factor = np.clip(tx_factor, 1.0 - config.transcript_influence, 1.0)

    # Combine
    combined = density_factor * intensity_factor * tx_factor
    combined_min = combined.min()
    combined_max = combined.max()
    if combined_max > combined_min:
        combined_norm = (combined - combined_min) / (combined_max - combined_min)
    else:
        combined_norm = np.ones_like(combined) * 0.5

    radius_map = (
        config.min_radius_um
        + (config.max_radius_um - config.min_radius_um) * combined_norm
    )

    if config.smooth_sigma_px > 0:
        radius_map = gaussian_filter(radius_map, sigma=config.smooth_sigma_px)
        radius_map = np.clip(radius_map, config.min_radius_um, config.max_radius_um)

    # Per-cell radius
    intensity_features = compute_local_intensity_features(
        image, nuclei_mask, nuclei_labels, pixel_size_um,
        config.intensity_percentile_low, config.intensity_percentile_high,
    )

    props = regionprops(nuclei_labels)
    per_cell_radius = {}

    for prop in props:
        label_id = prop.label
        cell_pixels = (nuclei_labels == label_id)
        avg_radius = radius_map[cell_pixels].mean()
        feat = intensity_features.get(label_id, {})
        spread = feat.get("spread", 0.5)
        spread_multiplier = 0.8 + 0.4 * min(spread, 1.5)
        adjusted = np.clip(avg_radius * spread_multiplier, config.min_radius_um, config.max_radius_um)
        per_cell_radius[label_id] = float(adjusted)

    return radius_map, per_cell_radius
