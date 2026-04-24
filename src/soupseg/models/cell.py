"""
Cell data model for SoupSeg.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon, Point
import numpy as np


@dataclass
class Cell:
    """
    Represents a single cell with its boundary and transcript information.
    
    Attributes:
        cell_id: Unique identifier for the cell
        polygon: Shapely Polygon representing the cell boundary
        centroid: (x, y) coordinates of the cell centroid
        area_um2: Area of the cell in square microns
        n_transcripts: Number of transcripts assigned to this cell
        n_genes: Number of unique genes expressed
        top_genes: List of top expressed genes
        gene_counts: Dictionary mapping gene names to transcript counts
    """
    
    cell_id: str
    polygon: Polygon
    centroid: Tuple[float, float] = field(default=(0.0, 0.0))
    area_um2: float = 0.0
    n_transcripts: int = 0
    n_genes: int = 0
    top_genes: List[str] = field(default_factory=list)
    gene_counts: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute derived fields after initialization."""
        if self.area_um2 == 0.0 and self.polygon.is_valid:
            # Assuming 0.5 um per pixel for Stereo-seq v1.3
            self.area_um2 = self.polygon.area * (0.5 ** 2)
        
        if self.centroid == (0.0, 0.0) and self.polygon.is_valid:
            cx, cy = self.polygon.centroid.xy
            self.centroid = (cx[0], cy[0])
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) is inside this cell."""
        return self.polygon.contains(Point(x, y))
    
    def add_transcript(self, gene: str):
        """Add a transcript to this cell."""
        self.n_transcripts += 1
        if gene not in self.gene_counts:
            self.gene_counts[gene] = 0
            self.n_genes += 1
        self.gene_counts[gene] += 1
        self._update_top_genes()
    
    def _update_top_genes(self, top_n: int = 10):
        """Update the list of top expressed genes."""
        sorted_genes = sorted(self.gene_counts.items(), key=lambda x: x[1], reverse=True)
        self.top_genes = [g[0] for g in sorted_genes[:top_n]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cell_id": self.cell_id,
            "polygon": list(self.polygon.exterior.coords),
            "centroid": list(self.centroid),
            "area_um2": round(self.area_um2, 2),
            "n_transcripts": self.n_transcripts,
            "n_genes": self.n_genes,
            "top_genes": self.top_genes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Cell:
        """Create a Cell from a dictionary."""
        coords = data["polygon"]
        polygon = Polygon(coords)
        cell = cls(
            cell_id=data["cell_id"],
            polygon=polygon,
            centroid=tuple(data["centroid"]),
            area_um2=data.get("area_um2", 0.0),
            n_transcripts=data.get("n_transcripts", 0),
            n_genes=data.get("n_genes", 0),
            top_genes=data.get("top_genes", []),
        )
        # Restore gene counts if available
        if "gene_counts" in data:
            cell.gene_counts = data["gene_counts"]
        return cell


class CellCollection:
    """
    A collection of Cell objects with utility methods.
    """
    
    def __init__(self, cells: Optional[List[Cell]] = None):
        self.cells: List[Cell] = cells or []
    
    def add_cell(self, cell: Cell):
        """Add a cell to the collection."""
        self.cells.append(cell)
    
    def __len__(self) -> int:
        return len(self.cells)
    
    def __iter__(self):
        return iter(self.cells)
    
    def __getitem__(self, idx: int) -> Cell:
        return self.cells[idx]
    
    def get_cell_by_id(self, cell_id: str) -> Optional[Cell]:
        """Get a cell by its ID."""
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire collection to a dictionary."""
        return {
            "n_cells": len(self.cells),
            "cells": [cell.to_dict() for cell in self.cells],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CellCollection:
        """Create a CellCollection from a dictionary."""
        cells = [Cell.from_dict(c) for c in data["cells"]]
        return cls(cells)
    
    def save_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> CellCollection:
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_anndata(self, transcripts):
        """Convert to AnnData with transcript expression data."""
        from ..io.h5ad_export import cells_to_anndata
        return cells_to_anndata(self, transcripts)

    def save_polygons(self, output_path):
        """Save cell polygons as GeoJSON."""
        from ..io.h5ad_export import save_polygons_geojson
        save_polygons_geojson(self.cells, output_path)

    def save_mask(self, image_shape, output_path, pixel_size_um=0.5):
        """Save 32-bit labeled mask as TIFF."""
        from ..io.h5ad_export import save_cell_mask
        save_cell_mask(self.cells, image_shape, output_path, pixel_size_um)

    def summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.cells:
            return {}
        
        areas = [c.area_um2 for c in self.cells]
        n_transcripts = [c.n_transcripts for c in self.cells]
        n_genes = [c.n_genes for c in self.cells]
        
        return {
            "n_cells": len(self.cells),
            "mean_area_um2": np.mean(areas),
            "median_area_um2": np.median(areas),
            "std_area_um2": np.std(areas),
            "mean_transcripts_per_cell": np.mean(n_transcripts),
            "median_transcripts_per_cell": np.median(n_transcripts),
            "mean_genes_per_cell": np.mean(n_genes),
            "median_genes_per_cell": np.median(n_genes),
        }
