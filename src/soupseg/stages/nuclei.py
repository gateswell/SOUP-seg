"""
Stage 2: Nuclei Detection and Cell Initialization.

This module handles nuclei detection from preprocessed ssDNA images
and initializes cell boundaries via nuclear expansion.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import warnings

try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not found. Some functions may not work.")

try:
    from skimage import io, filters, morphology, measure, segmentation
    from skimage.filters import threshold_otsu, threshold_local
    from skimage.morphology import opening, closing, disk, watershed
    from skimage.measure import regionprops, label
except ImportError:
    warnings.warn("scikit-image not found. Please install: pip install scikit-image")

try:
    from shapely.geometry import Polygon, MultiPolygon, Point
    from shapely.ops import unary_union
except ImportError:
    warnings.warn("shapely not found. Please install: pip install shapely")

# Default parameters for Stereo-seq v1.3 (0.5 um/pixel)
DEFAULT_PIXEL_SIZE_UM = 0.5


@dataclass
class NucleiDetectionResult:
    """Container for nuclei detection results."""
    nuclei_mask: np.ndarray  # Binary mask of nuclei
    nuclei_labels: np.ndarray  # Labeled regions (integer labels)
    n_nuclei: int  # Number of nuclei detected
    nuclei_properties: List[Dict]  # Properties of each nuclei (area, centroid, etc.)


def detect_nuclei_otsu(
    image: np.ndarray,
    min_area_px: int = 200,
    max_area_px: int = 80000,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM
) -> NucleiDetectionResult:
    """
    Detect nuclei using Otsu's automatic thresholding.
    
    Args:
        image: Preprocessed image (2D array, 0-1 range)
        min_area_px: Minimum nuclei area in pixels
        max_area_px: Maximum nuclei area in pixels
        pixel_size_um: Pixel size in microns
        
    Returns:
        NucleiDetectionResult with masks and properties
    """
    # Otsu thresholding
    thresh = threshold_otsu(image)
    binary = image > thresh
    
    # Clean up with morphology
    binary = opening(binary, disk(2))
    binary = closing(binary, disk(2))
    
    # Label connected components
    labeled = label(binary)
    props = regionprops(labeled, image)
    
    # Filter by area
    filtered_labels = []
    filtered_props = []
    for prop in props:
        area_px = prop.area
        area_um2 = area_px * (pixel_size_um ** 2)
        
        # Convert area thresholds to pixels
        min_area_um2 = min_area_px * (pixel_size_um ** 2)
        max_area_um2 = max_area_px * (pixel_size_um ** 2)
        
        if min_area_um2 <= area_um2 <= max_area_um2:
            filtered_labels.append(prop.label)
            filtered_props.append({
                "label": prop.label,
                "area_px": area_px,
                "area_um2": area_um2,
                "centroid": prop.centroid,  # (row, col) = (y, x)
                "bbox": prop.bbox,
            })
    
    # Create filtered mask
    filtered_mask = np.isin(labeled, filtered_labels)
    filtered_labeled = labeled * filtered_mask
    
    return NucleiDetectionResult(
        nuclei_mask=filtered_mask.astype(np.uint8),
        nuclei_labels=filtered_labeled,
        n_nuclei=len(filtered_props),
        nuclei_properties=filtered_props
    )


def detect_nuclei_adaptive(
    image: np.ndarray,
    block_size: int = 35,
    offset: float = 0.01,
    min_area_px: int = 200,
    max_area_px: int = 80000,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM
) -> NucleiDetectionResult:
    """
    Detect nuclei using adaptive thresholding.
    
    Better for images with uneven illumination.
    
    Args:
        image: Preprocessed image
        block_size: Size of a pixel neighborhood for adaptive threshold
        offset: Constant subtracted from weighted mean
        min_area_px: Minimum nuclei area in pixels
        max_area_px: Maximum nuclei area in pixels
        pixel_size_um: Pixel size in microns
        
    Returns:
        NucleiDetectionResult
    """
    # Adaptive thresholding
    binary = threshold_local(image, block_size, offset=offset)
    
    # Clean up with morphology
    binary = opening(binary, disk(2))
    binary = closing(binary, disk(2))
    
    # Label connected components
    labeled = label(binary)
    props = regionprops(labeled, image)
    
    # Filter by area
    filtered_labels = []
    filtered_props = []
    for prop in props:
        area_px = prop.area
        area_um2 = area_px * (pixel_size_um ** 2)
        min_area_um2 = min_area_px * (pixel_size_um ** 2)
        max_area_um2 = max_area_px * (pixel_size_um ** 2)
        
        if min_area_um2 <= area_um2 <= max_area_um2:
            filtered_labels.append(prop.label)
            filtered_props.append({
                "label": prop.label,
                "area_px": area_px,
                "area_um2": area_um2,
                "centroid": prop.centroid,
                "bbox": prop.bbox,
            })
    
    # Create filtered mask
    filtered_mask = np.isin(labeled, filtered_labels)
    filtered_labeled = labeled * filtered_mask
    
    return NucleiDetectionResult(
        nuclei_mask=filtered_mask.astype(np.uint8),
        nuclei_labels=filtered_labeled,
        n_nuclei=len(filtered_props),
        nuclei_properties=filtered_props
    )


def expand_nuclei_to_cells(
    nuclei_mask: np.ndarray,
    nuclei_labels: np.ndarray,
    nuclei_properties: List[Dict],
    expansion_radius_um: float = 6.0,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    method: str = 'binary_dilation'
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Expand nuclei to approximate cell boundaries.
    
    This is a key step that converts nuclear boundaries to cell boundaries.
    
    Args:
        nuclei_mask: Binary mask of nuclei
        nuclei_labels: Labeled nuclei image
        nuclei_properties: Properties of each nucleus
        expansion_radius_um: How much to expand each nucleus (in microns)
        pixel_size_um: Pixel size in microns
        method: 'binary_dilation' or 'convex_hull'
        
    Returns:
        Tuple of (cell_mask, cell_properties)
    """
    from scipy import ndimage
    
    # Convert expansion radius to pixels
    expansion_radius_px = int(round(expansion_radius_um / pixel_size_um))
    
    if method == 'binary_dilation':
        # Dilate each labeled nucleus
        cell_mask = np.zeros_like(nuclei_labels)
        
        cell_properties = []
        for i, prop in enumerate(nuclei_properties):
            label_id = prop["label"]
            
            # Get individual nucleus
            nucleus = (nuclei_labels == label_id).astype(np.uint8)
            
            # Dilate
            if cv2 is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                    (expansion_radius_px * 2 + 1,) * 2)
                expanded = cv2.dilate(nucleus, kernel, iterations=1)
            else:
                from scipy.ndimage import grey_dilation
                expanded = grey_dilation(nucleus, size=expansion_radius_px * 2 + 1)
            
            # Add to cell mask
            cell_mask[expanded > 0] = label_id
            
            # Update properties
            new_prop = prop.copy()
            new_prop["cell_id"] = f"cell_{i:05d}"
            cell_properties.append(new_prop)
    
    elif method == 'convex_hull':
        # Use convex hull of each nucleus (smoother boundaries)
        cell_mask = np.zeros_like(nuclei_labels)
        cell_properties = []
        
        for i, prop in enumerate(nuclei_properties):
            label_id = prop["label"]
            
            # Get nucleus pixels
            nucleus_pixels = np.argwhere(nuclei_labels == label_id)
            
            if len(nucleus_pixels) < 3:
                # Not enough points for hull
                continue
            
            # Compute convex hull polygon
            from scipy.spatial import ConvexHull
            points = nucleus_pixels[:, ::-1]  # Convert (row, col) to (x, y)
            
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                # Expand by radius using offset
                expanded_hull = expand_polygon(hull_points, expansion_radius_px)
                
                # Fill in the mask
                from skimage.draw import polygon
                rr, cc = polygon(expanded_hull[:, 1], expanded_hull[:, 0], 
                                 shape=cell_mask.shape)
                cell_mask[rr, cc] = label_id
                
                new_prop = prop.copy()
                new_prop["cell_id"] = f"cell_{i:05d}"
                cell_properties.append(new_prop)
                
            except Exception:
                continue
    
    else:
        raise ValueError(f"Unknown expansion method: {method}")
    
    return cell_mask, cell_properties


def expand_polygon(points: np.ndarray, offset: float) -> np.ndarray:
    """
    Expand a polygon outward by a given offset.
    
    Uses shapely for robust polygon offsetting.
    
    Args:
        points: Nx2 array of polygon vertices (x, y)
        offset: Offset distance in pixels
        
    Returns:
        Expanded polygon vertices
    """
    poly = Polygon(points)
    expanded = poly.buffer(offset)
    
    if expanded.is_empty or not expanded.is_valid:
        return points
    
    # Extract exterior coordinates
    if hasattr(expanded, 'exterior'):
        return np.array(expanded.exterior.coords)
    elif hasattr(expanded, 'geoms'):
        # MultiPolygon - take the largest
        largest = max(expanded.geoms, key=lambda p: p.area)
        return np.array(largest.exterior.coords)
    else:
        return points


def detect_nuclei(
    image: np.ndarray,
    method: str = 'otsu',
    expansion_radius_um: float = 6.0,
    min_area_um2: float = 50.0,
    max_area_um2: float = 2000.0,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    **kwargs
) -> Tuple[NucleiDetectionResult, np.ndarray, List[Dict]]:
    """
    Complete nuclei detection and cell initialization pipeline.
    
    This combines nuclei detection with expansion to cell boundaries.
    
    Args:
        image: Preprocessed ssDNA image
        method: 'otsu' or 'adaptive'
        expansion_radius_um: Radius to expand nuclei to cell boundaries
        min_area_um2: Minimum cell area in um^2
        max_area_um2: Maximum cell area in um^2
        pixel_size_um: Pixel size in microns
        **kwargs: Additional arguments for specific detection methods
        
    Returns:
        Tuple of (NucleiDetectionResult, cell_mask, cell_properties)
        
    Example:
        >>> from soupseg.stages import load_image, preprocess_image, detect_nuclei
        >>> img = load_image('ssdna.tiff')
        >>> preprocessed = preprocess_image(img)
        >>> nuclei_result, cell_mask, cells = detect_nuclei(preprocessed)
    """
    # Convert area thresholds to pixels
    min_area_px = int(min_area_um2 / (pixel_size_um ** 2))
    max_area_px = int(max_area_um2 / (pixel_size_um ** 2))
    
    # Detect nuclei
    if method == 'otsu':
        nuclei_result = detect_nuclei_otsu(
            image, min_area_px, max_area_px, pixel_size_um
        )
    elif method == 'adaptive':
        nuclei_result = detect_nuclei_adaptive(
            image, 
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            pixel_size_um=pixel_size_um,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Expand nuclei to cells
    cell_mask, cell_properties = expand_nuclei_to_cells(
        nuclei_result.nuclei_mask,
        nuclei_result.nuclei_labels,
        nuclei_result.nuclei_properties,
        expansion_radius_um=expansion_radius_um,
        pixel_size_um=pixel_size_um,
        method='binary_dilation'
    )
    
    return nuclei_result, cell_mask, cell_properties


def initialize_cells(
    preprocessed_image: np.ndarray,
    nuclei_result: NucleiDetectionResult,
    cell_mask: np.ndarray,
    cell_properties: List[Dict],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM
) -> List[Dict]:
    """
    Convert detection results to Cell objects.
    
    Args:
        preprocessed_image: The preprocessed image
        nuclei_result: Result from detect_nuclei
        cell_mask: Cell segmentation mask
        cell_properties: Properties of each cell
        
    Returns:
        List of cell dictionaries ready for Cell objects
    """
    from skimage.measure import regionprops, label
    
    # Label the cell mask
    labeled_cells = label(cell_mask > 0)
    props = regionprops(labeled_cells)
    
    cells = []
    for prop in props:
        # Get centroid (row, col) -> (x, y)
        cy, cx = prop.centroid
        
        cell_dict = {
            "cell_id": f"cell_{prop.label:05d}",
            "centroid": (float(cx * pixel_size_um), float(cy * pixel_size_um)),
            "area_um2": prop.area * (pixel_size_um ** 2),
            "bbox": prop.bbox,
            "eccentricity": prop.eccentricity,
            "extent": prop.extent,
        }
        cells.append(cell_dict)
    
    return cells


# Alias for convenience
segment = detect_nuclei
