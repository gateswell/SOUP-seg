"""
Cell and Transcript data models for SoupSeg.
"""

from .cell import Cell, CellCollection
from .transcript import Transcript, TranscriptCollection

# v1.1.0: U-Net boundary detection and adaptive radius
from .unet_boundary import (
    UNetConfig,
    UNetBoundaryDetector,
    create_unet_model,
    detect_boundaries_unet,
    train_unet,
    BoundaryLoss,
)
from .adaptive_radius import (
    AdaptiveRadiusConfig,
    compute_adaptive_radius_map,
    compute_adaptive_radius_map_torch,
    apply_adaptive_dilation,
    adaptive_expand_nuclei,
)

__all__ = [
    # Cell / Transcript
    "Cell",
    "CellCollection",
    "Transcript",
    "TranscriptCollection",
    # U-Net boundary detection (v1.1.0)
    "UNetConfig",
    "UNetBoundaryDetector",
    "create_unet_model",
    "detect_boundaries_unet",
    "train_unet",
    "BoundaryLoss",
    # Adaptive radius (v1.1.0)
    "AdaptiveRadiusConfig",
    "compute_adaptive_radius_map",
    "compute_adaptive_radius_map_torch",
    "apply_adaptive_dilation",
    "adaptive_expand_nuclei",
]
