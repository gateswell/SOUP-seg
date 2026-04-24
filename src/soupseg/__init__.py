"""
SoupSeg: ssDNA-Informed Iterative Cell Segmentation for Stereo-seq

A cell segmentation tool that combines ssDNA image-based nuclear detection
with transcriptomic constraints for Stereo-seq v1.3 data.
"""

__version__ = "1.1.0"
__author__ = "SoupSeg Team"

from .pipeline import SoupSeg
from .models.cell import Cell, CellCollection
from .models.transcript import Transcript, TranscriptCollection

# v1.1.0: U-Net boundary detection and adaptive radius
# These handle torch absence gracefully (raise ImportError on use)
from .models.unet_boundary import (
    UNetConfig,
    create_unet_model,
    detect_boundaries_unet,
    TORCH_AVAILABLE as UNET_TORCH_AVAILABLE,
)
from .models.adaptive_radius import (
    AdaptiveRadiusConfig,
    compute_adaptive_radius_map,
    adaptive_expand_nuclei,
)

# Conditionally available (stubs when torch not installed)
try:
    from .models.unet_boundary import UNetBoundaryDetector, BoundaryLoss
except (ImportError, AttributeError):
    pass

# v1.1.0: I/O export utilities
from .io.h5ad_export import (
    cells_to_anndata,
    save_polygons_geojson,
    save_cell_mask,
    add_polygons_to_anndata,
    SCANPY_AVAILABLE,
    TIFFFILE_AVAILABLE,
)

__all__ = [
    "SoupSeg",
    "Cell",
    "CellCollection",
    "Transcript",
    "TranscriptCollection",
    # v1.1.0
    "UNetConfig",
    "create_unet_model",
    "detect_boundaries_unet",
    "AdaptiveRadiusConfig",
    "compute_adaptive_radius_map",
    "adaptive_expand_nuclei",
    # v1.1.0: I/O export
    "cells_to_anndata",
    "save_polygons_geojson",
    "save_cell_mask",
    "add_polygons_to_anndata",
    "SCANPY_AVAILABLE",
    "TIFFFILE_AVAILABLE",
]
