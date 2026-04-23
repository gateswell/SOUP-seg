"""
Cell and Transcript data models for SoupSeg.
"""

from .cell import Cell, CellCollection
from .transcript import Transcript, TranscriptCollection

# v1.1.0: U-Net boundary detection and adaptive radius
# These are always importable (they handle torch absence internally)
from .unet_boundary import (
    UNetConfig,
    create_unet_model,
    detect_boundaries_unet,
    train_unet,
    TORCH_AVAILABLE as UNET_TORCH_AVAILABLE,
)
from .adaptive_radius import (
    AdaptiveRadiusConfig,
    compute_adaptive_radius_map,
    compute_adaptive_radius_map_torch,
    apply_adaptive_dilation,
    adaptive_expand_nuclei,
)

# Conditionally import torch-dependent classes
try:
    from .unet_boundary import UNetBoundaryDetector, BoundaryLoss
except ImportError:
    pass

__all__ = [
    # Cell / Transcript
    "Cell",
    "CellCollection",
    "Transcript",
    "TranscriptCollection",
    # U-Net boundary detection (v1.1.0)
    "UNetConfig",
    "create_unet_model",
    "detect_boundaries_unet",
    "train_unet",
    # Adaptive radius (v1.1.0)
    "AdaptiveRadiusConfig",
    "compute_adaptive_radius_map",
    "compute_adaptive_radius_map_torch",
    "apply_adaptive_dilation",
    "adaptive_expand_nuclei",
]
