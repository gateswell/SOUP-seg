"""
H5AD export for SoupSeg results - v1.0.0 compatible.
"""

import numpy as np
from typing import Tuple, Optional
import json


def cells_to_anndata(cells, transcripts, obsm_spatial_name="spatial"):
    """
    Convert CellCollection + TranscriptCollection to AnnData.

    Args:
        cells: CellCollection with Cell objects.
        transcripts: TranscriptCollection with Transcript objects.
        obsm_spatial_name: Key for spatial coordinates in adata.obsm.

    Returns:
        AnnData object with gene × cell expression matrix, cell metadata,
        and spatial coordinates.
    """
    import anndata as ad

    # Aggregate gene × cell expression matrix
    assigned = transcripts.get_assigned_transcripts()
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

    adata = ad.AnnData(X=expr_matrix.T)
    adata.var_names = all_genes
    adata.obs_names = cell_ids

    # Cell metadata
    adata.obs["cell_id"] = cell_ids
    adata.obs["area_um2"] = [c.area_um2 for c in cells]
    adata.obs["n_transcripts"] = [c.n_transcripts for c in cells]
    adata.obs["n_genes"] = [c.n_genes for c in cells]
    adata.obs["centroid_x"] = [c.centroid[0] for c in cells]
    adata.obs["centroid_y"] = [c.centroid[1] for c in cells]

    # Spatial coordinates in obsm
    spatial = np.array([[c.centroid[0], c.centroid[1]] for c in cells])
    adata.obsm[obsm_spatial_name] = spatial

    return adata


def save_polygons_geojson(cells, output_path):
    """
    Save cell polygons as GeoJSON.

    Args:
        cells: CellCollection or list of Cell objects.
        output_path: Path to write the GeoJSON file.
    """
    cell_list = cells.cells if hasattr(cells, 'cells') else cells
    features = []
    for cell in cell_list:
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
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        }
        features.append(feature)

    with open(output_path, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)


def save_cell_mask(cells, image_shape, output_path, pixel_size_um=0.5):
    """
    Save 32-bit labeled mask as TIFF.

    Args:
        cells: CellCollection or list of Cell objects.
        image_shape: (height, width) of the output mask.
        output_path: Path to write the TIFF file.
        pixel_size_um: Pixel size in microns for coordinate scaling.
    """
    from skimage import draw
    import tifffile

    cell_list = cells.cells if hasattr(cells, 'cells') else cells
    mask = np.zeros(image_shape, dtype=np.uint32)
    for cell in cell_list:
        label_idx = int(cell.cell_id.split("_")[1])
        poly = cell.polygon
        if not poly.is_valid:
            poly = poly.buffer(0)
        coords = np.array(poly.exterior.coords) / pixel_size_um
        coords = coords.astype(int)
        rr, cc = draw.polygon(coords[:, 1], coords[:, 0], shape=image_shape)
        mask[rr, cc] = label_idx

    tifffile.imwrite(output_path, mask)


def add_polygons_to_anndata(adata, cells, key="polygons"):
    """
    Add polygon coordinates to AnnData obsm (optional).

    Args:
        adata: AnnData object to augment.
        cells: CellCollection or list of Cell objects.
        key: Key for adata.obsm.

    Returns:
        The augmented AnnData object.
    """
    cell_list = cells.cells if hasattr(cells, 'cells') else cells
    polygons = [cell.polygon for cell in cell_list]
    adata.obsm[key] = polygons
    return adata
