"""
I/O utilities for SoupSeg.

Provides export functions for h5ad, GeoJSON, and TIFF mask formats.
"""

from .h5ad_export import (
    cells_to_anndata,
    save_polygons_geojson,
    save_cell_mask,
    add_polygons_to_anndata,
    SCANPY_AVAILABLE,
    TIFFFILE_AVAILABLE,
)

__all__ = [
    "cells_to_anndata",
    "save_polygons_geojson",
    "save_cell_mask",
    "add_polygons_to_anndata",
    "SCANPY_AVAILABLE",
    "TIFFFILE_AVAILABLE",
]
