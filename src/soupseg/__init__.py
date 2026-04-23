"""
SoupSeg: ssDNA-Informed Iterative Cell Segmentation for Stereo-seq

A cell segmentation tool that combines ssDNA image-based nuclear detection
with transcriptomic constraints for Stereo-seq v1.3 data.
"""

__version__ = "0.1.0"
__author__ = "SoupSeg Team"

from .pipeline import SoupSeg
from .models.cell import Cell, CellCollection
from .models.transcript import Transcript, TranscriptCollection

__all__ = [
    "SoupSeg",
    "Cell",
    "CellCollection", 
    "Transcript",
    "TranscriptCollection",
]
