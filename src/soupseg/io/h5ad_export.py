"""
H5AD export for SoupSeg results.

Provides conversion of SoupSeg segmentation results to AnnData format,
GeoJSON polygon export, and TIFF mask export.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Optional dependency guards
try:
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

from ..models.cell import CellCollection
from ..models.transcript import TranscriptCollection


def _check_scanpy():
    """Raise if scanpy/anndata is not available."""
    if not SCANPY_AVAILABLE:
        raise ImportError(
            "anndata is required for h5ad export. "
            "Install it with: pip install anndata scanpy"
        )


def _check_tifffile():
    """Raise if tifffile is not available."""
    if not TIFFFILE_AVAILABLE:
        raise ImportError(
            "tifffile is required for mask export. "
            "Install it with: pip install tifffile"
        )


def cells_to_anndata(
    cells: CellCollection,
    transcripts: TranscriptCollection,
    obsm_spatial_name: str = "spatial",
) -> "ad.AnnData":
    """
    Convert SoupSeg results to AnnData format.

    The returned AnnData contains:
    - X: cells × genes expression matrix (float32)
    - obs: cell_id, area_um2, n_transcripts, n_genes, centroid_x, centroid_y
    - obsm[obsm_spatial_name]: spatial coordinates (centroids)
    - uns: pipeline metadata

    Args:
        cells: CellCollection with cell information.
        transcripts: TranscriptCollection with transcript assignments.
        obsm_spatial_name: Key for spatial coordinates in obsm.

    Returns:
        AnnData object.

    Raises:
        ImportError: If anndata is not installed.
    """
    _check_scanpy()

    # Get assigned transcripts
    assigned = transcripts.get_assigned_transcripts()

    # Build expression matrix (genes × cells, then transpose)
    cell_ids = [c.cell_id for c in cells]
    all_genes = sorted(set(t.gene for t in assigned))

    expr_matrix = np.zeros((len(all_genes), len(cells)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    for t in assigned:
        if t.cell_id in cell_to_idx:
            gi = gene_to_idx[t.gene]
            ci = cell_to_idx[t.cell_id]
            expr_matrix[gi, ci] += t.count

    # Create AnnData (cells × genes)
    adata = ad.AnnData(X=expr_matrix.T)
    adata.var_names = all_genes
    adata.obs_names = cell_ids

    # obs: cell metadata
    adata.obs["cell_id"] = [c.cell_id for c in cells]
    adata.obs["area_um2"] = [c.area_um2 for c in cells]
    adata.obs["n_transcripts"] = [c.n_transcripts for c in cells]
    adata.obs["n_genes"] = [c.n_genes for c in cells]
    adata.obs["centroid_x"] = [c.centroid[0] for c in cells]
    adata.obs["centroid_y"] = [c.centroid[1] for c in cells]

    # obsm: spatial coordinates
    spatial_coords = np.array(
        [[c.centroid[0], c.centroid[1]] for c in cells]
    )
    adata.obsm[obsm_spatial_name] = spatial_coords

    # uns: pipeline metadata
    adata.uns["soupseg_version"] = "1.1.0"
    adata.uns["pipeline"] = "SoupSeg"

    return adata


def save_polygons_geojson(
    cells: CellCollection,
    output_path: str,
):
    """
    Save cell polygons as GeoJSON for spatial tools compatibility.

    Each cell becomes a Feature with:
    - id: cell_id
    - properties: cell_id, area_um2, n_transcripts, n_genes
    - geometry: Polygon (in micron coordinates)

    Args:
        cells: CellCollection with cell polygons.
        output_path: Path to save the GeoJSON file.
    """
    features = []
    for cell in cells:
        coords = list(cell.polygon.exterior.coords)
        feature = {
            "type": "Feature",
            "id": cell.cell_id,
            "properties": {
                "cell_id": cell.cell_id,
                "area_um2": cell.area_um2,
                "n_transcripts": cell.n_transcripts,
                "n_genes": cell.n_genes,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


def save_cell_mask(
    cells: CellCollection,
    image_shape: Tuple[int, int],
    output_path: str,
    pixel_size_um: float = 0.5,
):
    """
    Save segmentation mask as a 32-bit TIFF.

    Each pixel is labeled with the integer cell ID (0 = background).
    Cell label indices are extracted from cell_id strings (e.g. "cell_00042" → 42).

    Args:
        cells: CellCollection with cell polygons.
        image_shape: (height, width) of the original image in pixels.
        output_path: Path to save the TIFF file.
        pixel_size_um: Pixel size in microns for coordinate conversion.

    Raises:
        ImportError: If tifffile or scikit-image is not installed.
    """
    _check_tifffile()

    from skimage import draw

    mask = np.zeros(image_shape, dtype=np.uint32)

    for cell in cells:
        # Extract integer label from cell_id (e.g. "cell_00042" → 42)
        try:
            label_idx = int(cell.cell_id.split("_")[1])
        except (IndexError, ValueError):
            # Fallback: use a hash-based index
            label_idx = hash(cell.cell_id) % (2**16) + 1

        # Get polygon, fix if invalid
        poly = cell.polygon
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Convert micron coordinates to pixel coordinates
        coords = np.array(poly.exterior.coords) / pixel_size_um
        coords = coords.astype(int)

        # Fill polygon in the mask
        rr, cc = draw.polygon(coords[:, 1], coords[:, 0], shape=image_shape)
        mask[rr, cc] = label_idx

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, mask)


def add_polygons_to_anndata(
    adata: "ad.AnnData",
    cells: CellCollection,
    key: str = "polygons",
) -> "ad.AnnData":
    """
    Add polygon coordinate arrays to AnnData as obsm.

    Each cell's polygon exterior coordinates are stored as a variable-length
    NumPy array under adata.obsm[key]. Only call this when polygon data
    is actually needed, as it increases file size significantly.

    Args:
        adata: AnnData object to modify.
        cells: CellCollection with cell polygons.
        key: Key for polygon data in obsm.

    Returns:
        The modified AnnData (modified in-place).

    Raises:
        ImportError: If anndata is not installed.
    """
    _check_scanpy()

    # Store polygon coordinates as a list of arrays
    # (variable-length polygons cannot be stored as a single 2D array)
    polygons = [
        np.array(list(cell.polygon.exterior.coords))
        for cell in cells
    ]
    adata.obsm[key] = np.array(polygons, dtype=object)

    return adata
