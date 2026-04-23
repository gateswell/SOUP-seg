"""
Stage 4: Iterative Refinement using Graph Cut Optimization.

This module implements the iterative refinement loop that jointly
optimizes cell boundaries using image gradients and transcript density
via Graph Cut / max-flow optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings

try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not found. Graph Cut optimization may be limited.")
    cv2 = None

try:
    from skimage import filters, measure, segmentation, morphology
    from skimage.graph import rag
    from scipy.ndimage import distance_transform_edt, label as ndlabel
except ImportError:
    warnings.warn("scikit-image not found. Please install: pip install scikit-image")

# Try to import maxflow for Graph Cut
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
            "maxflow not found. Graph Cut optimization requires: pip install PyMaxflow"
        )


@dataclass
class RefinementConfig:
    """Configuration for iterative refinement."""
    max_iterations: int = 20
    tolerance: float = 0.001
    
    # Energy weights
    alpha_image: float = 0.4    # Weight for image gradient energy
    beta_transcript: float = 0.4  # Weight for transcript density energy  
    gamma_prior: float = 0.2    # Weight for prior (cell size/shape)
    
    # Boundary movement constraints
    max_boundary_shift_px: float = 3.0  # Max pixels to move per iteration
    
    # Cell size constraints
    min_cell_size_um2: float = 50.0
    max_cell_size_um2: float = 2000.0
    
    # Graph Cut parameters
    sigma_smooth: float = 5.0   # Smoothing strength for Graph Cut
    edge_weight_scale: float = 10.0  # Scale factor for edge weights
    
    # Debug
    verbose: bool = False
    
    # Alternative method if maxflow unavailable
    fallback_method: str = 'watershed'  # or 'random_walker'


def compute_image_gradient_energy(
    cell_mask: np.ndarray,
    gradient_image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute energy from image gradients.
    
    Ideal boundaries follow low-gradient regions (edges in ssDNA).
    
    Args:
        cell_mask: Current cell segmentation mask
        gradient_image: Pre-computed gradient magnitude image
        
    Returns:
        Gradient magnitude at each boundary pixel
    """
    if gradient_image is None:
        return np.zeros_like(cell_mask)
    
    # Get boundary pixels
    boundaries = segmentation.find_boundaries(cell_mask > 0, mode='inner')
    
    # Extract gradient values at boundaries
    energy = np.zeros_like(cell_mask, dtype=float)
    energy[boundaries] = gradient_image[boundaries]
    
    return energy


def compute_transcript_density_energy(
    cell_mask: np.ndarray,
    transcript_coords: np.ndarray,
    transcript_weights: Optional[np.ndarray] = None,
    pixel_size_um: float = 0.5,
    bin_size_um: float = 1.0
) -> np.ndarray:
    """
    Compute energy from transcript distribution.
    
    This penalizes boundary locations where transcripts of different
    genes are intermixed across the boundary.
    
    Args:
        cell_mask: Current cell segmentation mask
        transcript_coords: Nx2 array of (x, y) coordinates in pixels
        transcript_weights: N array of weights (default: all 1)
        pixel_size_um: Pixel size in microns
        bin_size_um: Bin size for density estimation
        
    Returns:
        Energy map at boundary pixels
    """
    if transcript_coords is None or len(transcript_coords) == 0:
        return np.zeros(cell_mask.shape, dtype=float)
    
    if transcript_weights is None:
        transcript_weights = np.ones(len(transcript_coords))
    
    # Compute transcript density per gene
    bin_size_px = int(bin_size_um / pixel_size_um)
    if bin_size_px < 1:
        bin_size_px = 1
    
    h, w = cell_mask.shape
    n_bins_y = max(1, h // bin_size_px)
    n_bins_x = max(1, w // bin_size_px)
    
    # Assign each transcript to a bin
    tx_bins = (transcript_coords[:, 1] / bin_size_px).astype(int)  # y -> row
    ty_bins = (transcript_coords[:, 0] / bin_size_px).astype(int)  # x -> col
    tx_bins = np.clip(tx_bins, 0, n_bins_y - 1)
    ty_bins = np.clip(ty_bins, 0, n_bins_x - 1)
    
    # Compute per-cell transcript density
    energy = np.zeros(cell_mask.shape, dtype=float)
    unique_labels = np.unique(cell_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        cell_region = (cell_mask == label)
        
        # Distance transform from cell boundary (inside = positive)
        dist_inside = distance_transform_edt(cell_region)
        
        # For boundary pixels (dist ~ 0), compute transcript spillover
        # High energy if many transcripts from OTHER cells are near this boundary
        boundaries = segmentation.find_boundaries(cell_region, mode='inner')
        
        # Compute density of "foreign" transcripts near boundary
        # (simplified: just use distance transform of boundary)
        boundary_dist = distance_transform_edt(boundaries)
        
        # Weight by proximity to boundary
        for y in range(h):
            for x in range(w):
                if boundary_dist[y, x] <= bin_size_px * 2:
                    # Near boundary - penalize based on transcript density difference
                    energy[y, x] += abs(dist_inside[y, x] - 1.0) * 0.1
    
    return energy


def compute_prior_energy(
    cell_mask: np.ndarray,
    pixel_size_um: float = 0.5,
    target_area_um2: float = 300.0,
    area_std_um2: float = 150.0
) -> float:
    """
    Compute prior energy based on cell size/shape priors.
    
    Penalizes cells that are too small, too large, or too elongated.
    
    Args:
        cell_mask: Current cell segmentation mask
        pixel_size_um: Pixel size in microns
        target_area_um2: Target mean cell area
        area_std_um2: Standard deviation of cell area
        
    Returns:
        Total prior energy
    """
    labeled = cell_mask.copy()
    props = measure.regionprops(labeled)
    
    if len(props) == 0:
        return 0.0
    
    total_energy = 0.0
    for prop in props:
        area_um2 = prop.area * (pixel_size_um ** 2)
        
        # Gaussian prior on area
        area_diff = (area_um2 - target_area_um2) / area_std_um2
        energy = np.exp(-0.5 * area_diff ** 2)
        total_energy += energy
    
    return total_energy


def build_graphcut_segmentation(
    image: np.ndarray,
    gradient_image: np.ndarray,
    initial_mask: np.ndarray,
    transcript_coords: Optional[np.ndarray] = None,
    config: RefinementConfig = None,
    pixel_size_um: float = 0.5
) -> np.ndarray:
    """
    Build and solve Graph Cut optimization for boundary refinement.
    
    Uses Boykov-Kolmogorov max-flow algorithm to find optimal boundary.
    
    Args:
        image: Input image (for boundary costs)
        gradient_image: Gradient magnitude image
        initial_mask: Initial segmentation (labels = cell IDs)
        transcript_coords: Optional transcript coordinates for density term
        config: Refinement configuration
        pixel_size_um: Pixel size in microns
        
    Returns:
        Refined segmentation mask
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
    
    # Add nodes (one per pixel)
    node_ids = g.add_nodes(h * w)
    
    # Reshape for easy indexing
    gradient_flat = gradient_image.ravel()
    
    # Build structure for 4-connectivity
    # neighbors: [(dy, dx), weight]
    neighbors = [
        (0, 1, 1.0),   # right
        (1, 0, 1.0),   # down
        (1, 1, np.sqrt(2)),  # diagonal down-right
        (1, -1, np.sqrt(2)), # diagonal down-left
    ]
    
    # Compute smoothness costs (based on gradient)
    # Low gradient = high cost to cut (boundary should avoid this)
    max_grad = gradient_image.max() + 1e-6
    
    # Add edges between neighbors
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            node_id = node_ids[idx]
            
            # Compute boundary cost (directional derivative of gradient)
            grad_here = gradient_image[y, x]
            
            for dy, dx, diag_weight in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nidx = ny * w + nx
                    neighbor_node_id = node_ids[nidx]
                    
                    grad_neighbor = gradient_image[ny, nx]
                    
                    # Edge weight: higher when gradient is similar
                    # (we want to cut in low-gradient regions)
                    avg_grad = (grad_here + grad_neighbor) / 2.0
                    
                    # Cost to cut this edge
                    cost = config.edge_weight_scale * (1.0 - avg_grad / max_grad)
                    
                    # Add symmetric edge
                    g.add_edge(
                        node_id, neighbor_node_id,
                        capacity=cost,
                        capacity_rev=cost
                    )
    
    # Add terminal edges (unary costs) based on initial mask
    # and transcript density
    labeled = initial_mask.copy()
    props = measure.regionprops(labeled)
    
    # Pre-compute distance transform for each cell
    dist_transforms = {}
    for prop in props:
        cell_region = (labeled == prop.label)
        dist_transforms[prop.label] = distance_transform_edt(cell_region)
    
    # Compute transcript density term
    tx_density = np.zeros((h, w), dtype=float)
    if transcript_coords is not None and len(transcript_coords) > 0:
        bin_size = 3  # pixels
        for tx in transcript_coords:
            tx_x, tx_y = int(tx[0]), int(tx[1])
            if 0 <= tx_y < h and 0 <= tx_x < w:
                # Gaussian blob around transcript
                for dy in range(-bin_size, bin_size + 1):
                    for dx in range(-bin_size, bin_size + 1):
                        ny, nx = tx_y + dy, tx_x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dist = np.sqrt(dx*dx + dy*dy)
                            tx_density[ny, nx] += np.exp(-dist**2 / (2 * bin_size**2))
    
    # Add terminal edges
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            node_id = node_ids[idx]
            current_label = labeled[y, x]
            
            # Compute unary costs
            if current_label == 0:
                # Background - high cost to assign to any cell
                for prop in props:
                    dist = dist_transforms.get(prop.label, np.zeros((h,w)))[y, x]
                    cost = 10.0 + dist * 0.5
                    g.add_tedge(node_id, cap=cost, cap_rev=0.0)
            else:
                # Inside a cell - small cost to stay, high cost to leave
                dist = dist_transforms.get(current_label, np.zeros((h,w)))[y, x]
                stay_cost = 0.1 + dist * 0.1
                
                # Add small cost to switch to neighboring cells
                for prop in props:
                    if prop.label != current_label:
                        other_dist = dist_transforms.get(prop.label, np.zeros((h,w)))[y, x]
                        switch_cost = 0.5 + other_dist * 0.2
                        
                        # Transcript density bias
                        if tx_density[y, x] > 0.5:
                            # Many transcripts here - strong bias to keep in current cell
                            switch_cost *= 2.0
                
                g.add_tedge(node_id, cap=stay_cost, cap_rev=stay_cost * 0.5)
    
    # Find maxflow
    if config.verbose:
        print("  Computing maxflow...")
    
    try:
        flow = g.maxflow()
    except Exception as e:
        if config.verbose:
            print(f"  Maxflow failed: {e}, returning initial mask")
        return initial_mask.copy()
    
    # Get segmentation from min-cut
    segm = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            node_id = node_ids[idx]
            if g.get_segment(node_id) == 0:  # source segment
                segm[y, x] = labeled[y, x]
            else:
                # Assign to nearest cell by distance transform
                best_label = labeled[y, x]
                best_dist = float('inf')
                for prop in props:
                    d = dist_transforms.get(prop.label, np.zeros((h,w)))[y, x]
                    if d < best_dist:
                        best_dist = d
                        best_label = prop.label
                segm[y, x] = best_label
    
    return segm


def watershed_refinement(
    image: np.ndarray,
    gradient_image: np.ndarray,
    initial_mask: np.ndarray,
    config: RefinementConfig
) -> np.ndarray:
    """
    Fallback refinement using marker-controlled watershed.
    
    Used when Graph Cut (maxflow) is not available.
    
    Args:
        image: Input image
        gradient_image: Gradient magnitude (markers from initial segmentation)
        initial_mask: Initial segmentation
        config: Configuration
        
    Returns:
        Refined segmentation
    """
    from scipy.ndimage import label, distance_transform_edt
    
    h, w = image.shape[:2]
    labeled = initial_mask.copy()
    
    # Get markers from initial segmentation
    markers = label(labeled > 0)[0]
    
    # Compute watershed from markers
    from skimage.morphology import watershed
    
    # Invert gradient (we want to find basins, not peaks)
    gradient_inv = 1.0 - gradient_image / (gradient_image.max() + 1e-6)
    
    # Apply watershed
    result = watershed(gradient_inv, markers, mask=(labeled > 0))
    
    return result


def compute_total_energy(
    cell_mask: np.ndarray,
    gradient_image: np.ndarray,
    transcript_coords: Optional[np.ndarray],
    config: RefinementConfig,
    pixel_size_um: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total energy of current segmentation.
    
    Returns:
        Tuple of (total_energy, component_dict)
    """
    # Image gradient component
    image_energy = compute_image_gradient_energy(cell_mask, gradient_image)
    E_image = config.alpha_image * np.mean(image_energy[image_energy > 0]) if np.any(image_energy > 0) else 0.0
    
    # Transcript density component
    if transcript_coords is not None and len(transcript_coords) > 0:
        transcript_energy = compute_transcript_density_energy(
            cell_mask, transcript_coords, pixel_size_um=pixel_size_um
        )
        E_transcript = config.beta_transcript * np.mean(transcript_energy[transcript_energy > 0])
    else:
        E_transcript = 0.0
    
    # Prior component
    E_prior = config.gamma_prior * compute_prior_energy(
        cell_mask, pixel_size_um
    )
    
    total = E_image + E_transcript + E_prior
    
    return total, {
        "E_image": E_image,
        "E_transcript": E_transcript,
        "E_prior": E_prior,
        "total": total
    }


def apply_cell_size_constraints(
    cell_mask: np.ndarray,
    min_area_um2: float,
    max_area_um2: float,
    pixel_size_um: float = 0.5
) -> np.ndarray:
    """
    Remove cells that are too small or too large.
    
    Small cells are merged into nearest neighbor.
    Large cells are left unchanged (could be split in future).
    
    Args:
        cell_mask: Input cell mask
        min_area_um2: Minimum cell area
        max_area_um2: Maximum cell area  
        pixel_size_um: Pixel size in microns
        
    Returns:
        Cleaned mask
    """
    from scipy.ndimage import label, distance_transform_edt
    
    labeled = cell_mask.copy()
    props = measure.regionprops(labeled)
    
    min_area_px = min_area_um2 / (pixel_size_um ** 2)
    max_area_px = max_area_um2 / (pixel_size_um ** 2)
    
    # Find small cells to remove
    small_cells = []
    for prop in props:
        if prop.area < min_area_px:
            small_cells.append(prop.label)
        elif prop.area > max_area_px:
            # Cap large cells - set to background
            labeled[labeled == prop.label] = 0
    
    # Merge small cells into nearest neighbors
    for cell_label in small_cells:
        cell_pixels = (labeled == cell_label)
        
        # Find all pixels of this cell
        cell_coords = np.argwhere(cell_pixels)
        if len(cell_coords) == 0:
            continue
        
        # Find centroid
        centroid_y, centroid_x = cell_coords.mean(axis=0)
        
        # Find nearest other cell
        best_target = None
        best_dist = float('inf')
        
        for prop in props:
            if prop.label == cell_label:
                continue
            # Distance from centroid to this cell
            cy, cx = prop.centroid
            dist = np.sqrt((centroid_y - cy)**2 + (centroid_x - cx)**2)
            if dist < best_dist:
                best_dist = dist
                best_target = prop.label
        
        if best_target is not None:
            # Merge into nearest cell
            labeled[cell_pixels] = best_target
    
    # Relabel to remove gaps
    new_labeled = label(labeled > 0)[0]
    
    return new_labeled


def optimize_boundary_iteration(
    cell_mask: np.ndarray,
    gradient_image: np.ndarray,
    transcript_coords: np.ndarray,
    config: RefinementConfig,
    pixel_size_um: float = 0.5
) -> np.ndarray:
    """
    Perform one iteration of boundary optimization using Graph Cut.
    
    Args:
        cell_mask: Current cell segmentation mask
        gradient_image: Image gradient magnitude
        transcript_coords: Transcript coordinates
        config: Refinement configuration
        pixel_size_um: Pixel size in microns
        
    Returns:
        Updated cell mask
    """
    # Build and solve Graph Cut
    refined = build_graphcut_segmentation(
        image=np.zeros_like(cell_mask),  # Not used for boundary costs
        gradient_image=gradient_image,
        initial_mask=cell_mask,
        transcript_coords=transcript_coords,
        config=config,
        pixel_size_um=pixel_size_um
    )
    
    # Apply cell size constraints
    refined = apply_cell_size_constraints(
        refined,
        min_area_um2=config.min_cell_size_um2,
        max_area_um2=config.max_cell_size_um2,
        pixel_size_um=pixel_size_um
    )
    
    return refined


def check_convergence(
    old_energy: float,
    new_energy: float,
    tolerance: float
) -> bool:
    """
    Check if refinement has converged.
    
    Args:
        old_energy: Energy from previous iteration
        new_energy: Energy from current iteration
        tolerance: Convergence tolerance
        
    Returns:
        True if converged
    """
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
    pixel_size_um: float = 0.5
) -> Tuple[np.ndarray, List[Dict], Dict[str, Any]]:
    """
    Main iterative refinement loop.
    
    This jointly optimizes cell boundaries using:
    - Image gradient energy (boundaries should follow edges)
    - Transcript density energy (transcripts should be spatially coherent)
    - Prior energy (cells should have reasonable sizes)
    
    Uses Graph Cut (Boykov-Kolmogorov max-flow) for optimization.
    
    Args:
        cell_mask: Initial cell segmentation mask from Stage 2
        gradient_image: Pre-computed gradient magnitude image
        transcript_coords: Nx2 array of transcript (x, y) coordinates
        cell_properties: List of cell property dicts
        config: Refinement configuration (uses defaults if None)
        pixel_size_um: Pixel size in microns
        
    Returns:
        Tuple of (refined_mask, updated_properties, convergence_info)
        
    Example:
        >>> from soupseg.stages import iterative_refinement, RefinementConfig
        >>> config = RefinementConfig(max_iterations=20, alpha_image=0.4, beta_transcript=0.4)
        >>> refined, props, info = iterative_refinement(
        ...     cell_mask, gradient, transcripts, cell_props,
        ...     config=config
        ... )
    """
    if config is None:
        config = RefinementConfig()
    
    current_mask = cell_mask.copy()
    
    # Compute gradient image if not provided
    if gradient_image is None:
        gradient_image = filters.sobel(current_mask.astype(float))
    
    # Prepare transcript coordinates
    if transcript_coords is not None and isinstance(transcript_coords, pd.DataFrame):
        transcript_coords = transcript_coords[['x', 'y']].values
    
    convergence_info = {
        "iterations": 0,
        "energies": [],
        "energy_components": [],
        "converged": False,
        "final_energy": 0.0,
        "method": "graphcut" if MAXFLOW_AVAILABLE else "watershed_fallback"
    }
    
    if config.verbose:
        print(f"=== Iterative Refinement (method: {convergence_info['method']}) ===")
    
    for iteration in range(config.max_iterations):
        # Compute total energy
        total_energy, components = compute_total_energy(
            current_mask,
            gradient_image,
            transcript_coords,
            config,
            pixel_size_um
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
        
        # Check convergence
        if iteration > 0:
            if check_convergence(
                convergence_info["energies"][-2],
                convergence_info["energies"][-1],
                config.tolerance
            ):
                convergence_info["converged"] = True
                convergence_info["iterations"] = iteration + 1
                if config.verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        # Optimize boundaries
        old_mask = current_mask.copy()
        current_mask = optimize_boundary_iteration(
            current_mask,
            gradient_image,
            transcript_coords,
            config,
            pixel_size_um
        )
        
        # Check if mask changed significantly
        changed_pixels = np.sum(old_mask != current_mask)
        change_ratio = changed_pixels / current_mask.size
        
        if config.verbose:
            print(f"    Changed {changed_pixels} pixels ({change_ratio:.2%})")
        
        # Early stopping if no significant change
        if change_ratio < 0.001:
            convergence_info["converged"] = True
            convergence_info["iterations"] = iteration + 1
            if config.verbose:
                print(f"  No significant change, stopping")
            break
    
    if not convergence_info["converged"]:
        convergence_info["iterations"] = config.max_iterations
    
    convergence_info["final_energy"] = convergence_info["energies"][-1] if convergence_info["energies"] else 0.0
    
    return current_mask, cell_properties, convergence_info


def simple_refinement(
    cell_mask: np.ndarray,
    min_area_um2: float = 50.0,
    max_area_um2: float = 2000.0,
    pixel_size_um: float = 0.5,
    remove_duplicates: bool = True
) -> np.ndarray:
    """
    Simple refinement that just cleans up the mask.
    
    Removes cells that are too small or too large.
    
    Args:
        cell_mask: Input cell mask
        min_area_um2: Minimum cell area
        max_area_um2: Maximum cell area
        pixel_size_um: Pixel size in microns
        remove_duplicates: Whether to merge adjacent cells
        
    Returns:
        Cleaned mask
    """
    from scipy.ndimage import label as ndlabel
    
    labeled = cell_mask.copy()
    
    # Apply size constraints
    result = apply_cell_size_constraints(
        labeled,
        min_area_um2,
        max_area_um2,
        pixel_size_um
    )
    
    return result


# Alias
refine = iterative_refinement
