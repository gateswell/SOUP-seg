"""
SoupSeg processing stages.
"""

from .preprocess import preprocess_image, load_image, gaussian_denoise, clahe_enhance
from .nuclei import detect_nuclei, initialize_cells, detect_nuclei_otsu
from .assignment import assign_transcripts, voronoi_assignment, polygon_assignment
from .refine import (
    iterative_refinement,
    RefinementConfig,
    build_graphcut_segmentation,
    simple_refinement,
    apply_cell_size_constraints,
    # v1.1.0 new exports
    compute_boundary_map,
)

# v1.1.0: adaptive radius (from models, re-exported for convenience)
try:
    from .refine import adaptive_expand_nuclei
except ImportError:
    from ..models.adaptive_radius import adaptive_expand_nuclei

__all__ = [
    # Preprocessing
    "preprocess_image",
    "load_image",
    "gaussian_denoise",
    "clahe_enhance",
    # Nuclei detection
    "detect_nuclei",
    "initialize_cells",
    "detect_nuclei_otsu",
    # Transcript assignment
    "assign_transcripts",
    "voronoi_assignment",
    "polygon_assignment",
    # Refinement
    "iterative_refinement",
    "RefinementConfig",
    "build_graphcut_segmentation",
    "simple_refinement",
    "apply_cell_size_constraints",
    # v1.1.0
    "compute_boundary_map",
    "adaptive_expand_nuclei",
]
