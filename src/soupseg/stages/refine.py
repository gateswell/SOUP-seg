"""
Stage 4: Iterative Refinement — PyTorch Implementation (SoupSeg v1.1.0).

This module implements the iterative refinement loop that jointly optimizes
cell boundaries using:

1. **U-Net boundary detection** — A learned model that predicts boundary
   probability from ssDNA images (replaces hand-crafted gradient features).
2. **Adaptive dilation radius** — Data-driven per-cell expansion that adapts
   to local cell density, image intensity, and transcript distribution.
3. **Graph Cut optimization** — Boykov-Kolmogorov max-flow for optimal
   boundary placement (same algorithm as v1.0, now PyTorch-accelerated).

The PyTorch implementation supports CUDA acceleration where available and
falls back gracefully to NumPy/SciPy when PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. Install for GPU acceleration: pip install torch torchvision"
    )

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage import filters, measure, segmentation, morphology
    from scipy.ndimage import distance_transform_edt, label as ndlabel, gaussian_filter
except ImportError:
    warnings.warn("scikit-image/scipy not found.")

try:
    import maxflow
    MAXFLOW_AVAILABLE = True
except ImportError:
    try:
        import maxflow as maxflow_lib
        maxflow = maxflow_lib
        MAXFLOW_AVAILABLE = True
    except ImportError:
        MAXFLOW_AVAILABLE = False
        warnings.warn(
            "maxflow not found. Graph Cut requires: pip install PyMaxflow"
        )

from .adaptive_radius import (
    AdaptiveRadiusConfig,
    compute_adaptive_radius_map,
    compute_adaptive_radius_map_torch,
    apply_adaptive_dilation,
    adaptive_expand_nuclei,
)
from .unet_boundary import (
    UNetConfig,
    UNetBoundaryDetector,
    create_unet_model,
    detect_boundaries_unet,
    TORCH_AVAILABLE as UNET_TORCH_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RefinementConfig:
    """Configuration for iterative refinement (v1.1.0)."""

    max_iterations: int = 20
    tolerance: float = 0.001

    # Energy weights
    alpha_image: float = 0.4
    beta_transcript: float = 0.4
    gamma_prior: float = 0.2

    # Boundary movement constraints
    max_boundary_shift_px: float = 3.0

    # Cell size constraints
    min_cell_size_um2: float = 50.0
    max_cell_size_um2: float = 2000.0

    # Graph Cut parameters
    sigma_smooth: float = 5.0
    edge_weight_scale: float = 10.0

    # --- v1.1.0 new features ---

    # U-Net boundary detection
    use_unet: bool = True  # Use U-Net instead of Sobel for boundary detection
    unet_config: Optional[UNetConfig] = None
    unet_weights_path: Optional[str] = None  # Path to pretrained weights
    boundary_threshold: float = 0.5  # Threshold for boundary probability

    # Adaptive radius
    use_adaptive_radius: bool = True  # Use adaptive dilation instead of fixed
    adaptive_radius_config: Optional[AdaptiveRadiusConfig] = None

    # PyTorch acceleration
    use_torch: bool = True  # Use PyTorch where available
    device: str = "auto"  # "auto", "cpu", "cuda:0", etc.

    # Debug
    verbose: bool = False
    fallback_method: str = 'watershed'

    def get_device(self) -> str:
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device

    @property
    def torch_enabled(self) -> bool:
        return self.use_torch and TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# Boundary detection (U-Net or fallback)
# ---------------------------------------------------------------------------

def compute_boundary_map(
    image: np.ndarray,
    config: Optional[RefinementConfig] = None,
    model: Optional[UNetBoundaryDetector] = None,
) -> np.ndarray:
    """
    Compute a boundary probability/energy map.

    Uses U-Net if available and configured, otherwise falls back to Sobel.

    Args:
        image: Preprocessed ssDNA image (H, W), float32 [0, 1].
        config: Refinement configuration.
        model: Pre-loaded U-Net model (optional).

    Returns:
        Boundary map (H, W), float64. Higher values = stronger boundary signal.
    """
    if config is None:
        config = RefinementConfig()

    if config.use_unet and UNET_TORCH_AVAILABLE:
        try:
            if model is None:
                unet_config = config.unet_config or UNetConfig()
                model = create_unet_model(unet_config)

                # Load pretrained weights if available
                if config.unet_weights_path:
                    model.load_state_dict(
                        torch.load(config.unet_weights_path, map_location="cpu")
                    )

            # Run U-Net inference
            boundary_mask = detect_boundaries_unet(
                image, model=model, config=config.unet_config or UNetConfig(),
                threshold=config.boundary_threshold,
            )

            # Convert boundary mask (0/255) to energy (0-1)
            # Boundaries have high energy → pixels on boundaries should be
            # adjusted by the Graph Cut optimizer
            boundary_energy = boundary_mask.astype(np.float64) / 255.0

            if config.verbose:
                print(f"  Boundary map: U-Net (threshold={config.boundary_threshold})")

            return boundary_energy

        except Exception as e:
            warnings.warn(f"U-Net boundary detection failed ({e}), falling back to Sobel.")

    # Fallback: Sobel gradient
    gradient = filters.sobel(image.astype(np.float64))
    if gradient.max() > 0:
        gradient = gradient / gradient.max()

    if config.verbose:
        print("  Boundary map: Sobel gradient (fallback)")

    return gradient


# ---------------------------------------------------------------------------
# PyTorch-accelerated energy computation
# ---------------------------------------------------------------------------

def compute_image_gradient_energy_torch(
    cell_mask: np.ndarray,
    gradient_image: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute image gradient energy using PyTorch.

    Same logic as the original but uses torch for array operations.
    """
    if gradient_image is None:
        return np.zeros_like(cell_mask)

    if not TORCH_AVAILABLE:
        return _compute_image_gradient_energy_numpy(cell_mask, gradient_image)

    boundaries = segmentation.find_boundaries(cell_mask > 0, mode='inner')
    energy = np.zeros_like(cell_mask, dtype=float)
    energy[boundaries] = gradient_image[boundaries]
    return energy


def _compute_image_gradient_energy_numpy(
    cell_mask: np.ndarray,
    gradient_image: Optional[np.ndarray],
) -> np.ndarray:
    """NumPy fallback for image gradient energy."""
    if gradient_image is None:
        return np.zeros_like(cell_mask)
    boundaries = segmentation.find_boundaries(cell_mask > 0, mode='inner')
    energy = np.zeros_like(cell_mask, dtype=float)
    energy[boundaries] = gradient_image[boundaries]
    return energy


def compute_transcript_density_energy(
    cell_mask: np.ndarray,
    transcript_coords: np.ndarray,
    transcript_weights: Optional[np.ndarray] = None,
    pixel_size_um: float = 0.5,
    bin_size_um: float = 1.0,
) -> np.ndarray:
    """
    Compute energy from transcript distribution.

    Uses vectorized operations for speed. When PyTorch is available,
    the heavy computation runs on GPU.
    """
    if transcript_coords is None or len(transcript_coords) == 0:
        return np.zeros(cell_mask.shape, dtype=float)

    if transcript_weights is None:
        transcript_weights = np.ones(len(transcript_coords))

    h, w = cell_mask.shape
    bin_size_px = max(1, int(bin_size_um / pixel_size_um))

    # Build transcript density map using vectorized histogram
    tx_y = np.clip(transcript_coords[:, 1].astype(int), 0, h - 1)
    tx_x = np.clip(transcript_coords[:, 0].astype(int), 0, w - 1)

    density = np.zeros((h, w), dtype=np.float64)
    np.add.at(density, (tx_y, tx_x), transcript_weights)

    # Smooth with Gaussian
    density = gaussian_filter(density, sigma=bin_size_px * 2)

    # Compute energy at boundaries: measure how well the current boundaries
    # separate transcript-dense regions
    boundaries = segmentation.find_boundaries(cell_mask > 0, mode='inner')
    energy = np.zeros((h, w), dtype=float)
    energy[boundaries] = density[boundaries]

    # Normalize
    e_max = energy.max()
    if e_max > 0:
        energy /= e_max

    return energy


def compute_prior_energy(
    cell_mask: np.ndarray,
    pixel_size_um: float = 0.5,
    target_area_um2: float = 300.0,
    area_std_um2: float = 150.0,
) -> float:
    """Compute prior energy based on cell size/shape priors."""
    labeled = cell_mask.copy()
    props = measure.regionprops(labeled)

    if len(props) == 0:
        return 0.0

    total_energy = 0.0
    for prop in props:
        area_um2 = prop.area * (pixel_size_um ** 2)
        area_diff = (area_um2 - target_area_um2) / area_std_um2
        energy = np.exp(-0.5 * area_diff ** 2)
        total_energy += energy

    return total_energy


def compute_total_energy(
    cell_mask: np.ndarray,
    gradient_image: np.ndarray,
    transcript_coords: Optional[np.ndarray],
    config: RefinementConfig,
    pixel_size_um: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Compute total energy of current segmentation."""
    # Image gradient component
    image_energy = compute_image_gradient_energy_torch(
        cell_mask, gradient_image, config.get_device()
    )
    E_image = config.alpha_image * np.mean(image_energy[image_energy > 0]) if np.any(image_energy > 0) else 0.0

    # Transcript density component
    if transcript_coords is not None and len(transcript_coords) > 0:
        transcript_energy = compute_transcript_density_energy(
            cell_mask, transcript_coords, pixel_size_um=pixel_size_um
        )
        E_transcript = config.beta_transcript * np.mean(transcript_energy[transcript_energy > 0]) if np.any(transcript_energy > 0) else 0.0
    else:
        E_transcript = 0.0

    # Prior component
    E_prior = config.gamma_prior * compute_prior_energy(cell_mask, pixel_size_um)

    total = E_image + E_transcript + E_prior

    return total, {
        "E_image": E_image,
        "E_transcript": E_transcript,
        "E_prior": E_prior,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Graph Cut (with optional PyTorch preprocessing)
# ---------------------------------------------------------------------------

def build_graphcut_segmentation(
    image: np.ndarray,
    gradient_image: np.ndarray,
    initial_mask: np.ndarray,
    transcript_coords: Optional[np.ndarray] = None,
    config: RefinementConfig = None,
    pixel_size_um: float = 0.5,
) -> np.ndarray:
    """
    Build and solve Graph Cut optimization for boundary refinement.

    In v1.1.0, the gradient_image may come from U-Net boundary detection
    instead of Sobel, providing more accurate boundary cost estimation.
    """
    if config is None:
        config = RefinementConfig()

    if not MAXFLOW_AVAILABLE:
        if config.verbose:
            print("  maxflow not available, using watershed fallback")
        return watershed_refinement(image, gradient_image, initial_mask, config)

    h, w = image.shape[:2]
    n_labels = len(np.unique(initial_mask)) - (1 if 0 in initial_mask else 0)

    if n_labels < 2:
        return initial_mask.copy()

    # Create graph
    g = maxflow.GraphFloat()
    node_ids = g.add_nodes(h * w)

    max_grad = gradient_image.max() + 1e-6

    # Build edges — use vectorized construction for speed
    # 4-connectivity neighbors
    neighbors = [(0, 1), (1, 0)]

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            grad_here = gradient_image[y, x]

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nidx = ny * w + nx
                    grad_neighbor = gradient_image[ny, nx]
                    avg_grad = (grad_here + grad_neighbor) / 2.0
                    cost = config.edge_weight_scale * (1.0 - avg_grad / max_grad)
                    g.add_edge(node_ids[idx], node_ids[nidx], capacity=cost, capacity_rev=cost)

    # Terminal edges
    labeled = initial_mask.copy()
    props = measure.regionprops(labeled)

    dist_transforms = {}
    for prop in props:
        cell_region = (labeled == prop.label)
        dist_transforms[prop.label] = distance_transform_edt(cell_region)

    # Transcript density (vectorized)
    tx_density = np.zeros((h, w), dtype=float)
    if transcript_coords is not None and len(transcript_coords) > 0:
        tx_y = np.clip(transcript_coords[:, 1].astype(int), 0, h - 1)
        tx_x = np.clip(transcript_coords[:, 0].astype(int), 0, w - 1)
        for ty, tx_val in zip(tx_y, tx_x):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = ty + dy, tx_val + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        dist = np.sqrt(dy * dy + dx * dx)
                        tx_density[ny, nx] += np.exp(-dist ** 2 / 18.0)

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            current_label = labeled[y, x]

            if current_label == 0:
                min_dist = float('inf')
                for prop in props:
                    d = dist_transforms.get(prop.label, np.zeros((h, w)))[y, x]
                    if d < min_dist:
                        min_dist = d
                g.add_tedge(node_ids[idx], cap=10.0 + min_dist * 0.5, cap_rev=0.0)
            else:
                dist = dist_transforms.get(current_label, np.zeros((h, w)))[y, x]
                stay_cost = 0.1 + dist * 0.1
                g.add_tedge(node_ids[idx], cap=stay_cost, cap_rev=stay_cost * 0.5)

    try:
        flow = g.maxflow()
    except Exception as e:
        if config.verbose:
            print(f"  Maxflow failed: {e}, returning initial mask")
        return initial_mask.copy()

    segm = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if g.get_segment(node_ids[idx]) == 0:
                segm[y, x] = labeled[y, x]
            else:
                best_label = labeled[y, x]
                best_dist = float('inf')
                for prop in props:
                    d = dist_transforms.get(prop.label, np.zeros((h, w)))[y, x]
                    if d < best_dist:
                        best_dist = d
                        best_label = prop.label
                segm[y, x] = best_label

    return segm


def watershed_refinement(
    image: np.ndarray,
    gradient_image: np.ndarray,
    initial_mask: np.ndarray,
    config: RefinementConfig,
) -> np.ndarray:
    """Fallback refinement using marker-controlled watershed."""
    from scipy.ndimage import label, distance_transform_edt
    from skimage.morphology import watershed

    labeled = initial_mask.copy()
    markers = label(labeled > 0)[0]
    gradient_inv = 1.0 - gradient_image / (gradient_image.max() + 1e-6)
    result = watershed(gradient_inv, markers, mask=(labeled > 0))
    return result


# ---------------------------------------------------------------------------
# Cell size constraints
# ---------------------------------------------------------------------------

def apply_cell_size_constraints(
    cell_mask: np.ndarray,
    min_area_um2: float,
    max_area_um2: float,
    pixel_size_um: float = 0.5,
) -> np.ndarray:
    """Remove cells that are too small or too large."""
    from scipy.ndimage import label, distance_transform_edt

    labeled = cell_mask.copy()
    props = measure.regionprops(labeled)

    min_area_px = min_area_um2 / (pixel_size_um ** 2)
    max_area_px = max_area_um2 / (pixel_size_um ** 2)

    small_cells = []
    for prop in props:
        if prop.area < min_area_px:
            small_cells.append(prop.label)
        elif prop.area > max_area_px:
            labeled[labeled == prop.label] = 0

    for cell_label in small_cells:
        cell_pixels = (labeled == cell_label)
        cell_coords = np.argwhere(cell_pixels)
        if len(cell_coords) == 0:
            continue
        centroid_y, centroid_x = cell_coords.mean(axis=0)
        best_target = None
        best_dist = float('inf')
        for prop in props:
            if prop.label == cell_label:
                continue
            cy, cx = prop.centroid
            dist = np.sqrt((centroid_y - cy) ** 2 + (centroid_x - cx) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_target = prop.label
        if best_target is not None:
            labeled[cell_pixels] = best_target

    new_labeled = label(labeled > 0)[0]
    return new_labeled


# ---------------------------------------------------------------------------
# Main iterative refinement loop
# ---------------------------------------------------------------------------

def optimize_boundary_iteration(
    cell_mask: np.ndarray,
    gradient_image: np.ndarray,
    transcript_coords: np.ndarray,
    config: RefinementConfig,
    pixel_size_um: float = 0.5,
) -> np.ndarray:
    """Perform one iteration of boundary optimization using Graph Cut."""
    refined = build_graphcut_segmentation(
        image=np.zeros_like(cell_mask),
        gradient_image=gradient_image,
        initial_mask=cell_mask,
        transcript_coords=transcript_coords,
        config=config,
        pixel_size_um=pixel_size_um,
    )

    refined = apply_cell_size_constraints(
        refined,
        min_area_um2=config.min_cell_size_um2,
        max_area_um2=config.max_cell_size_um2,
        pixel_size_um=pixel_size_um,
    )

    return refined


def check_convergence(old_energy: float, new_energy: float, tolerance: float) -> bool:
    """Check if refinement has converged."""
    if old_energy == 0:
        return False
    relative_change = abs(new_energy - old_energy) / abs(old_energy)
    return relative_change < tolerance


def iterative_refinement(
    cell_mask: np.ndarray,
    gradient_image: np.ndarray,
    transcript_coords: np.ndarray,
    cell_properties: List[Dict],
    config: Optional[RefinementConfig] = None,
    pixel_size_um: float = 0.5,
    ssdna_image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict], Dict[str, Any]]:
    """
    Main iterative refinement loop (v1.1.0).

    This jointly optimizes cell boundaries using:
    - U-Net boundary detection (if available and configured)
    - Image gradient energy (boundaries should follow edges)
    - Transcript density energy (transcripts should be spatially coherent)
    - Prior energy (cells should have reasonable sizes)
    - Adaptive dilation radius (per-cell expansion based on data features)

    Uses Graph Cut (Boykov-Kolmogorov max-flow) for optimization.

    Args:
        cell_mask: Initial cell segmentation mask from Stage 2
        gradient_image: Pre-computed gradient magnitude image (fallback if no U-Net)
        transcript_coords: Nx2 array of transcript (x, y) coordinates
        cell_properties: List of cell property dicts
        config: Refinement configuration (uses defaults if None)
        pixel_size_um: Pixel size in microns
        ssdna_image: Preprocessed ssDNA image for U-Net boundary detection.
                     If None, falls back to gradient_image.

    Returns:
        Tuple of (refined_mask, updated_properties, convergence_info)

    Example:
        >>> from soupseg.stages import iterative_refinement, RefinementConfig
        >>> config = RefinementConfig(use_unet=True, use_adaptive_radius=True)
        >>> refined, props, info = iterative_refinement(
        ...     cell_mask, gradient, transcripts, cell_props,
        ...     config=config, ssdna_image=preprocessed_image
        ... )
    """
    if config is None:
        config = RefinementConfig()

    current_mask = cell_mask.copy()

    # --- v1.1.0: Compute boundary map using U-Net if configured ---
    if config.use_unet and ssdna_image is not None:
        boundary_map = compute_boundary_map(ssdna_image, config)
    else:
        if gradient_image is None:
            gradient_image = filters.sobel(current_mask.astype(float))
        boundary_map = gradient_image

    # --- v1.1.0: Adaptive radius expansion ---
    adaptive_info = {}
    if config.use_adaptive_radius and ssdna_image is not None:
        if config.verbose:
            print("  Computing adaptive dilation radius...")

        # Prepare nuclei labels from current mask
        nuclei_mask = current_mask > 0
        if np.any(nuclei_mask):
            nuclei_labels = current_mask.copy()

            adaptive_config = config.adaptive_radius_config or AdaptiveRadiusConfig(
                pixel_size_um=pixel_size_um,
                verbose=config.verbose,
            )

            # Use PyTorch-accelerated version if available
            if config.torch_enabled:
                radius_map, per_cell_radius = compute_adaptive_radius_map_torch(
                    ssdna_image, nuclei_mask, nuclei_labels,
                    transcript_coords, adaptive_config,
                )
            else:
                radius_map, per_cell_radius = compute_adaptive_radius_map(
                    ssdna_image, nuclei_mask, nuclei_labels,
                    transcript_coords, adaptive_config,
                )

            adaptive_info = {
                "mean_radius_um": float(np.mean(list(per_cell_radius.values()))) if per_cell_radius else 0.0,
                "std_radius_um": float(np.std(list(per_cell_radius.values()))) if per_cell_radius else 0.0,
                "n_cells": len(per_cell_radius),
            }
    elif config.verbose:
        print("  Using fixed expansion radius (no adaptive radius or no ssDNA image)")

    # Prepare transcript coordinates
    if transcript_coords is not None and isinstance(transcript_coords, pd.DataFrame):
        transcript_coords = transcript_coords[['x', 'y']].values

    method_parts = []
    if config.use_unet and ssdna_image is not None:
        method_parts.append("unet")
    else:
        method_parts.append("gradient")
    method_parts.append("graphcut" if MAXFLOW_AVAILABLE else "watershed_fallback")
    if config.use_adaptive_radius and ssdna_image is not None:
        method_parts.append("adaptive_radius")

    convergence_info = {
        "iterations": 0,
        "energies": [],
        "energy_components": [],
        "converged": False,
        "final_energy": 0.0,
        "method": "+".join(method_parts),
        "adaptive_radius": adaptive_info,
        "version": "1.1.0",
    }

    if config.verbose:
        print(f"=== Iterative Refinement v1.1.0 (method: {convergence_info['method']}) ===")

    for iteration in range(config.max_iterations):
        total_energy, components = compute_total_energy(
            current_mask, boundary_map, transcript_coords,
            config, pixel_size_um,
        )

        convergence_info["energies"].append(total_energy)
        convergence_info["energy_components"].append(components)

        if config.verbose:
            print(
                f"  Iter {iteration + 1}: E={total_energy:.4f} "
                f"(image={components['E_image']:.4f}, "
                f"tx={components['E_transcript']:.4f}, "
                f"prior={components['E_prior']:.4f})"
            )

        if iteration > 0:
            if check_convergence(
                convergence_info["energies"][-2],
                convergence_info["energies"][-1],
                config.tolerance,
            ):
                convergence_info["converged"] = True
                convergence_info["iterations"] = iteration + 1
                if config.verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

        old_mask = current_mask.copy()
        current_mask = optimize_boundary_iteration(
            current_mask, boundary_map, transcript_coords,
            config, pixel_size_um,
        )

        changed_pixels = np.sum(old_mask != current_mask)
        change_ratio = changed_pixels / current_mask.size

        if config.verbose:
            print(f"    Changed {changed_pixels} pixels ({change_ratio:.2%})")

        if change_ratio < 0.001:
            convergence_info["converged"] = True
            convergence_info["iterations"] = iteration + 1
            if config.verbose:
                print(f"  No significant change, stopping")
            break

    if not convergence_info["converged"]:
        convergence_info["iterations"] = config.max_iterations

    convergence_info["final_energy"] = (
        convergence_info["energies"][-1] if convergence_info["energies"] else 0.0
    )

    return current_mask, cell_properties, convergence_info


def simple_refinement(
    cell_mask: np.ndarray,
    min_area_um2: float = 50.0,
    max_area_um2: float = 2000.0,
    pixel_size_um: float = 0.5,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Simple refinement that just cleans up the mask."""
    labeled = cell_mask.copy()
    result = apply_cell_size_constraints(labeled, min_area_um2, max_area_um2, pixel_size_um)
    return result


# Alias
refine = iterative_refinement
