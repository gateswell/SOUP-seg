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
)

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
]
