"""
Stage 3: Transcript Assignment.

This module handles assignment of transcripts to cells based on
spatial proximity and gene co-expression patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import Voronoi, KDTree, cKDTree
from scipy.ndimage import distance_transform_edt
import warnings

try:
    from shapely.geometry import Point, Polygon
except ImportError:
    warnings.warn("shapely not found. Please install: pip install shapely")


def voronoi_assignment(
    transcripts_df: pd.DataFrame,
    cell_polygons: List[Tuple[str, Polygon]],
    pixel_size_um: float = 0.5
) -> pd.DataFrame:
    """
    Assign transcripts to cells using Voronoi diagram.
    
    This is a fast initial assignment method that assigns each transcript
    to the nearest cell centroid.
    
    Args:
        transcripts_df: DataFrame with columns ['x', 'y', 'gene']
        cell_polygons: List of (cell_id, shapely Polygon)
        pixel_size_um: Pixel size in microns
        
    Returns:
        DataFrame with added 'cell_id' column
    """
    df = transcripts_df.copy()
    
    # Extract cell centroids
    centroids = np.array([
        [poly.centroid.x, poly.centroid.y] 
        for _, poly in cell_polygons
    ])
    cell_ids = [cell_id for cell_id, _ in cell_polygons]
    
    # Build KD-tree for fast nearest neighbor lookup
    tree = KDTree(centroids)
    
    # Find nearest cell for each transcript
    coords = df[['x', 'y']].values * pixel_size_um  # Convert to microns
    distances, indices = tree.query(coords, k=1)
    
    # Assign to nearest cell
    df['cell_id'] = [cell_ids[i] for i in indices]
    df['distance_to_cell'] = distances
    
    return df


def polygon_assignment(
    transcripts_df: pd.DataFrame,
    cell_polygons: List[Tuple[str, Polygon]],
    pixel_size_um: float = 0.5
) -> pd.DataFrame:
    """
    Assign transcripts to cells based on polygon containment.
    
    This is more accurate than Voronoi but slower.
    Each transcript is assigned to the cell whose polygon contains it.
    
    Args:
        transcripts_df: DataFrame with columns ['x', 'y', 'gene']
        cell_polygons: List of (cell_id, shapely Polygon)
        pixel_size_um: Pixel size in microns
        
    Returns:
        DataFrame with added 'cell_id' column
    """
    df = transcripts_df.copy()
    
    # Convert coordinates
    coords = df[['x', 'y']].values * pixel_size_um
    
    # Assign each transcript to containing polygon
    assignments = []
    for x, y in coords:
        point = Point(x, y)
        assigned = False
        for cell_id, polygon in cell_polygons:
            if polygon.contains(point):
                assignments.append(cell_id)
                assigned = True
                break
        if not assigned:
            assignments.append(None)
    
    df['cell_id'] = assignments
    
    return df


def distance_based_assignment(
    transcripts_df: pd.DataFrame,
    cell_polygons: List[Tuple[str, Polygon]],
    max_distance_um: float = 10.0,
    pixel_size_um: float = 0.5
) -> pd.DataFrame:
    """
    Assign transcripts based on distance to cell boundaries.
    
    A transcript is assigned to a cell if:
    1. It falls inside the cell polygon, OR
    2. It is within max_distance_um of the cell boundary AND
       it is closer to this cell than any other cell
    
    Args:
        transcripts_df: DataFrame with columns ['x', 'y', 'gene']
        cell_polygons: List of (cell_id, shapely Polygon)
        max_distance_um: Maximum distance from boundary for assignment
        pixel_size_um: Pixel size in microns
        
    Returns:
        DataFrame with added 'cell_id' column
    """
    df = transcripts_df.copy()
    coords = df[['x', 'y']].values * pixel_size_um
    
    # Pre-compute centroids
    centroids = np.array([
        [poly.centroid.x, poly.centroid.y] 
        for _, poly in cell_polygons
    ])
    cell_ids = [cell_id for cell_id, _ in cell_polygons]
    
    # Build KD-tree
    tree = KDTree(centroids)
    
    # For each transcript, find nearest cells
    distances, indices = tree.query(coords, k=3)  # Check 3 nearest
    
    assignments = []
    for i, (x, y) in enumerate(coords):
        point = Point(x, y)
        
        # First, check direct containment
        for cell_id, polygon in cell_polygons:
            if polygon.contains(point):
                assignments.append(cell_id)
                break
        else:
            # Not contained - use distance to boundary
            best_cell = None
            best_dist = float('inf')
            
            for j in range(3):  # Check 3 nearest cells
                cell_id = cell_ids[indices[i][j]]
                cell_poly = cell_polygons[[c for c, _ in cell_polygons].index(cell_id)][1]
                
                # Distance to polygon boundary
                dist_to_boundary = cell_poly.exterior.distance(point)
                
                if dist_to_boundary <= max_distance_um and dist_to_boundary < best_dist:
                    best_dist = dist_to_boundary
                    best_cell = cell_id
            
            assignments.append(best_cell)
    
    df['cell_id'] = assignments
    return df


def gene_coexpression_correction(
    transcripts_df: pd.DataFrame,
    cell_polygons: List[Tuple[str, Polygon]],
    correction_threshold: float = 0.7,
    pixel_size_um: float = 0.5
) -> pd.DataFrame:
    """
    Correct transcript assignments using gene co-expression patterns.
    
    The idea: if two transcripts of the SAME gene are assigned to DIFFERENT cells,
    but one of them is much closer to the boundary, it might be a diffusion artifact.
    
    This is a simplified version - full implementation would be more sophisticated.
    
    Args:
        transcripts_df: DataFrame with columns ['x', 'y', 'gene', 'cell_id']
        cell_polygons: List of (cell_id, shapely Polygon)
        correction_threshold: Fraction of transcripts of a gene that should
                             be in the "dominant" cell
        pixel_size_um: Pixel size in microns
        
    Returns:
        DataFrame with corrected 'cell_id' column
    """
    df = transcripts_df.copy()
    
    # For each gene, check if assignment is balanced
    for gene in df['gene'].unique():
        gene_mask = df['gene'] == gene
        gene_transcripts = df[gene_mask]
        
        # Count transcripts per cell for this gene
        cell_counts = gene_transcripts['cell_id'].value_counts()
        
        if len(cell_counts) <= 1:
            continue
        
        # Check if one cell dominates
        dominant_cell = cell_counts.index[0]
        dominant_ratio = cell_counts.iloc[0] / len(gene_transcripts)
        
        if dominant_ratio < correction_threshold:
            # Gene is shared across multiple cells - this is expected for
            # some genes (e.g., housekeeping genes)
            continue
        
        # For non-dominant cells with this gene, check if they're "correct"
        # Simplified: just mark them, actual implementation would be more complex
        for cell_id in cell_counts.index[1:]:
            cell_transcripts = gene_transcripts[gene_transcripts['cell_id'] == cell_id]
            
            # Could add more sophisticated correction here
            pass
    
    return df


def compute_transcript_density(
    transcripts_df: pd.DataFrame,
    image_shape: Tuple[int, int],
    pixel_size_um: float = 0.5,
    bin_size_um: float = 1.0
) -> np.ndarray:
    """
    Compute transcript density across the tissue.
    
    Args:
        transcripts_df: DataFrame with ['x', 'y', 'gene']
        image_shape: Shape of the image (height, width)
        pixel_size_um: Pixel size in microns
        bin_size_um: Size of bins for density calculation
        
    Returns:
        2D array of transcript density
    """
    bin_size_px = int(bin_size_um / pixel_size_um)
    density = np.zeros((
        image_shape[0] // bin_size_px,
        image_shape[1] // bin_size_px
    ))
    
    coords = (transcripts_df[['y', 'x']].values * pixel_size_um / bin_size_um).astype(int)
    
    for y, x in coords:
        if 0 <= y < density.shape[0] and 0 <= x < density.shape[1]:
            density[y, x] += 1
    
    return density


def assign_transcripts(
    transcripts_df: pd.DataFrame,
    cell_polygons: List[Tuple[str, Polygon]],
    method: str = 'voronoi',
    max_distance_um: Optional[float] = None,
    pixel_size_um: float = 0.5,
    use_coexpression: bool = False,
    image_shape: Optional[Tuple[int, int]] = None,
    diffusion_correction: bool = True
) -> pd.DataFrame:
    """
    Main transcript assignment function.
    
    Args:
        transcripts_df: DataFrame with columns ['x', 'y', 'gene'] or 
                       ['transcript_id', 'x', 'y', 'gene']
        cell_polygons: List of (cell_id, shapely Polygon)
        method: 'voronoi', 'polygon', or 'distance'
        max_distance_um: Maximum distance for 'distance' method
        pixel_size_um: Pixel size in microns
        use_coexpression: Whether to use gene co-expression correction
        image_shape: Shape of image for density calculation
        diffusion_correction: Apply simple diffusion correction
        
    Returns:
        DataFrame with 'cell_id' column added
        
    Example:
        >>> from soupseg.stages import assign_transcripts
        >>> df = pd.read_csv('transcripts.csv')
        >>> assigned = assign_transcripts(df, cell_polygons, method='voronoi')
    """
    df = transcripts_df.copy()
    
    # Ensure required columns exist
    required_cols = ['x', 'y', 'gene']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add transcript_id if not present
    if 'transcript_id' not in df.columns:
        df['transcript_id'] = [f"tx_{i:08d}" for i in range(len(df))]
    
    # Add cell_id column
    if method == 'voronoi':
        df = voronoi_assignment(df, cell_polygons, pixel_size_um)
    elif method == 'polygon':
        df = polygon_assignment(df, cell_polygons, pixel_size_um)
    elif method == 'distance':
        if max_distance_um is None:
            max_distance_um = 10.0
        df = distance_based_assignment(
            df, cell_polygons, max_distance_um, pixel_size_um
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Optional co-expression correction
    if use_coexpression:
        df = gene_coexpression_correction(df, cell_polygons, pixel_size_um)
    
    # Optional diffusion correction
    if diffusion_correction and method != 'polygon':
        # Simple correction: for each cell, compute expected vs actual transcript count
        # and flag cells with unusually low counts (potential boundary issues)
        pass  # Placeholder for more sophisticated correction
    
    return df


# Alias
voronoi_assignment_enhanced = assign_transcripts
