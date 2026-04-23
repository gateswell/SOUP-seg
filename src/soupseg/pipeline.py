"""
Main Pipeline for SoupSeg.

This module provides the high-level interface for running the complete
ssDNA-informed cell segmentation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import warnings

from .stages.preprocess import load_image, preprocess_image
from .stages.nuclei import detect_nuclei, NucleiDetectionResult
from .stages.assignment import assign_transcripts
from .stages.refine import iterative_refinement, simple_refinement, RefinementConfig
from .models.cell import Cell, CellCollection
from .models.transcript import Transcript, TranscriptCollection


class SoupSegResult:
    """Container for SoupSeg results."""
    
    def __init__(
        self,
        cells: CellCollection,
        transcripts: TranscriptCollection,
        metadata: Dict[str, Any],
        convergence_info: Optional[Dict] = None
    ):
        self.cells = cells
        self.transcripts = transcripts
        self.metadata = metadata
        self.convergence_info = convergence_info or {}
    
    def save(self, output_dir: str):
        """Save results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cells
        self.cells.save_json(str(output_dir / "cells.json"))
        
        # Save transcripts
        self.transcripts.save_json(str(output_dir / "transcripts.json"))
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save convergence info if present
        if self.convergence_info:
            with open(output_dir / "convergence.json", 'w') as f:
                json.dump(self.convergence_info, f, indent=2)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "cells": self.cells.summary_stats(),
            "transcripts": self.transcripts.summary_stats(),
            "metadata": self.metadata,
            "convergence": self.convergence_info
        }


class SoupSeg:
    """
    Main class for SoupSeg pipeline.
    
    Example:
        >>> from soupseg import SoupSeg
        >>> seg = SoupSeg()
        >>> result = seg.run(
        ...     ssdna_image='data/ssdna.tiff',
        ...     transcript_file='data/transcripts.csv',
        ...     output_dir='output/'
        ... )
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pixel_size_um: float = 0.5
    ):
        """
        Initialize SoupSeg.
        
        Args:
            config: Configuration dictionary. If None, uses defaults.
            pixel_size_um: Pixel size in microns (0.5 for Stereo-seq v1.3)
        """
        self.config = config or self._default_config()
        self.pixel_size_um = pixel_size_um
        self._unet_model = None  # Lazy-loaded U-Net model
        self._gnn_model = None   # Lazy-loaded GNN model
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # Preprocessing
            "denoise": True,
            "denoise_sigma": 1.5,
            "clahe_clip_limit": 0.03,
            "clahe_kernel_size": (8, 8),
            "remove_background": True,
            "normalize": True,
            
            # Nuclei detection
            "nuclei_method": "otsu",
            "min_area_um2": 50.0,
            "max_area_um2": 2000.0,
            "expansion_radius_um": 6.0,
            
            # --- v1.1.0: Adaptive Radius ---
            "use_adaptive_radius": False,
            "adaptive_radius_config": {
                "adaptive_base_radius_um": 6.0,
                "adaptive_min_radius_um": 3.0,
                "adaptive_max_radius_um": 12.0,
                "adaptive_density_sigma_um": 30.0,
                "adaptive_density_influence": 0.5,
                "adaptive_intensity_influence": 0.3,
                "adaptive_transcript_influence": 0.2,
                "adaptive_smooth_sigma_px": 3.0,
            },
            
            # --- v1.1.0: U-Net Boundary Detection ---
            "use_unet_boundary": False,
            "unet_boundary_config": {
                "base_features": 32,
                "depth": 4,
                "device": "auto",
                "tile_size": 512,
                "tile_overlap": 64,
            },
            "unet_boundary_threshold": 0.5,
            
            # --- v1.1.0: GNN Boundary Refinement ---
            "use_gnn_boundary": False,
            "gnn_boundary_config": {
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "distance_threshold_um": 50.0,
                "max_neighbors": 10,
                "boundary_threshold": 0.5,
            },
            
            # Transcript assignment
            "assignment_method": "voronoi",
            "max_distance_um": 10.0,
            "use_coexpression": False,
            "diffusion_correction": True,
            
            # Refinement
            "do_refinement": True,
            "refinement_config": {
                "max_iterations": 20,
                "tolerance": 0.001,
                "alpha_image": 0.4,
                "beta_transcript": 0.4,
                "gamma_prior": 0.2,
                "max_boundary_shift_px": 3.0,
                "sigma_smooth": 5.0,
                "edge_weight_scale": 10.0,
            }
        }
    
    def run(
        self,
        ssdna_image: str,
        transcript_file: str,
        output_dir: str,
        registration_matrix: Optional[str] = None,
        return_result: bool = True,
        save_visualization: bool = False
    ) -> SoupSegResult:
        """
        Run the complete SoupSeg pipeline.
        
        Args:
            ssdna_image: Path to ssDNA image file (TIFF/PNG)
            transcript_file: Path to transcript CSV file
            output_dir: Directory to save results
            registration_matrix: Optional path to registration matrix
            return_result: Whether to return SoupSegResult object
            save_visualization: Whether to save visualization images
            
        Returns:
            SoupSegResult object containing cells, transcripts, and metadata
        """
        print(f"=== SoupSeg Pipeline ===")
        print(f"Image: {ssdna_image}")
        print(f"Transcripts: {transcript_file}")
        print()
        
        # Stage 0: Load data
        print("Stage 0: Loading data...")
        image = load_image(ssdna_image)
        transcripts_df = pd.read_csv(transcript_file)
        print(f"  Image shape: {image.shape}")
        print(f"  Transcript count: {len(transcripts_df)}")
        print()
        
        # Stage 1: Preprocessing
        print("Stage 1: Preprocessing...")
        preprocessed = preprocess_image(
            image,
            denoise=self.config.get("denoise", True),
            denoise_sigma=self.config.get("denoise_sigma", 1.5),
            clip_limit=self.config.get("clahe_clip_limit", 0.03),
            remove_background=self.config.get("remove_background", True),
            normalize=self.config.get("normalize", True)
        )
        print(f"  Preprocessing complete.")
        print()
        
        # Stage 2: Nuclei detection and cell initialization
        print("Stage 2: Nuclei detection and cell initialization...")
        
        if self.config.get("use_adaptive_radius", False):
            # Use adaptive radius expansion (v1.1.0)
            from .stages.adaptive_radius import (
                adaptive_detect_nuclei,
                AdaptiveRadiusStageConfig,
            )
            adaptive_cfg_dict = self.config.get("adaptive_radius_config", {})
            stage_config = AdaptiveRadiusStageConfig(
                nuclei_method=self.config.get("nuclei_method", "otsu"),
                min_area_um2=self.config.get("min_area_um2", 50.0),
                max_area_um2=self.config.get("max_area_um2", 2000.0),
                pixel_size_um=self.pixel_size_um,
                use_adaptive_radius=True,
                expansion_radius_um=self.config.get("expansion_radius_um", 6.0),
                adaptive_base_radius_um=adaptive_cfg_dict.get("adaptive_base_radius_um", 6.0),
                adaptive_min_radius_um=adaptive_cfg_dict.get("adaptive_min_radius_um", 3.0),
                adaptive_max_radius_um=adaptive_cfg_dict.get("adaptive_max_radius_um", 12.0),
                adaptive_density_sigma_um=adaptive_cfg_dict.get("adaptive_density_sigma_um", 30.0),
                adaptive_density_influence=adaptive_cfg_dict.get("adaptive_density_influence", 0.5),
                adaptive_intensity_influence=adaptive_cfg_dict.get("adaptive_intensity_influence", 0.3),
                adaptive_transcript_influence=adaptive_cfg_dict.get("adaptive_transcript_influence", 0.2),
                adaptive_smooth_sigma_px=adaptive_cfg_dict.get("adaptive_smooth_sigma_px", 3.0),
                verbose=False,
            )
            # Build transcript coords
            transcript_coords = transcripts_df[['x', 'y']].values if 'x' in transcripts_df.columns else None
            nuclei_result, cell_mask, cell_properties = adaptive_detect_nuclei(
                preprocessed,
                transcript_coords=transcript_coords,
                config=stage_config,
            )
            print(f"  [Adaptive Radius v1.1.0] Detected {nuclei_result.n_nuclei} nuclei")
            print(f"  Initialized {len(cell_properties)} cells with adaptive expansion")
        else:
            nuclei_result, cell_mask, cell_properties = detect_nuclei(
                preprocessed,
                method=self.config.get("nuclei_method", "otsu"),
                expansion_radius_um=self.config.get("expansion_radius_um", 6.0),
                min_area_um2=self.config.get("min_area_um2", 50.0),
                max_area_um2=self.config.get("max_area_um2", 2000.0),
                pixel_size_um=self.pixel_size_um
            )
            print(f"  Detected {nuclei_result.n_nuclei} nuclei")
            print(f"  Initialized {len(cell_properties)} cells")
        print()
        
        # Build cell polygons
        cell_polygons = self._build_cell_polygons(cell_mask, cell_properties)
        
        # Stage 3: Transcript assignment
        print("Stage 3: Transcript assignment...")
        assigned_df = assign_transcripts(
            transcripts_df,
            cell_polygons,
            method=self.config.get("assignment_method", "voronoi"),
            max_distance_um=self.config.get("max_distance_um", 10.0),
            pixel_size_um=self.pixel_size_um,
            use_coexpression=self.config.get("use_coexpression", False),
            diffusion_correction=self.config.get("diffusion_correction", True)
        )
        assignment_rate = assigned_df['cell_id'].notna().mean()
        print(f"  Assignment rate: {assignment_rate:.1%}")
        print()
        
        # Stage 4: Iterative refinement (placeholder - full implementation needed)
        convergence_info = {}
        if self.config.get("do_refinement", True):
            print("Stage 4: Iterative refinement...")
            
            # Build transcript coordinates
            transcript_coords = assigned_df[['x', 'y']].values
            
            # Compute gradient image
            from skimage import filters
            gradient_image = filters.sobel(preprocessed.astype(float))
            
            # Get refinement config
            refine_config_dict = self.config.get("refinement_config", {})
            refine_config = RefinementConfig(**refine_config_dict)
            
            # Run refinement
            refined_mask, updated_props, convergence_info = iterative_refinement(
                cell_mask,
                gradient_image,
                transcript_coords,
                cell_properties,
                config=refine_config,
                pixel_size_um=self.pixel_size_um
            )
            
            print(f"  Refinement {'converged' if convergence_info.get('converged') else 'max iterations'}")
            print(f"  Final energy: {convergence_info.get('final_energy', 0):.4f}")
            print()
        else:
            # Simple cleanup
            refined_mask = simple_refinement(
                cell_mask,
                min_area_um2=self.config.get("min_area_um2", 50.0),
                max_area_um2=self.config.get("max_area_um2", 2000.0),
                pixel_size_um=self.pixel_size_um
            )
        
        # Build final results
        print("Building final results...")
        
        # Build Cell objects
        cells = self._build_cell_collection(refined_mask, assigned_df)
        
        # Build Transcript objects
        from .models.transcript import TranscriptCollection
        transcripts = TranscriptCollection.from_dataframe(assigned_df)
        
        # Build metadata
        metadata = {
            "pipeline": "SoupSeg",
            "version": "1.1.0",
            "pixel_size_um": self.pixel_size_um,
            "image_shape": list(image.shape),
            "n_cells_initial": len(cell_properties),
            "n_cells_final": len(cells),
            "n_transcripts": len(transcripts_df),
            "assignment_rate": float(assignment_rate),
            "config": self.config
        }
        
        result = SoupSegResult(
            cells=cells,
            transcripts=transcripts,
            metadata=metadata,
            convergence_info=convergence_info
        )
        
        # Save results
        print(f"Saving results to {output_dir}...")
        result.save(output_dir)
        
        # Save visualization if requested
        if save_visualization:
            self._save_visualization(
                result, preprocessed, refined_mask, output_dir
            )
        
        print("Done!")
        print()
        print("Summary:")
        print(f"  Cells: {len(cells)}")
        print(f"  Transcripts assigned: {len(transcripts.get_assigned_transcripts())}")
        print(f"  Assignment rate: {assignment_rate:.1%}")
        
        if return_result:
            return result
    
    def _build_cell_polygons(
        self,
        cell_mask: np.ndarray,
        cell_properties: List[Dict]
    ) -> List[Tuple[str, Any]]:
        """Build list of (cell_id, Polygon) tuples from cell mask."""
        from shapely.geometry import Polygon
        from skimage.measure import find_contours, label
        
        labeled = label(cell_mask > 0)
        polygons = []
        
        for prop in cell_properties:
            cell_id = prop.get("cell_id", f"cell_{prop['label']:05d}")
            label_id = prop.get("label", 0)
            
            if label_id == 0:
                continue
            
            # Find contours for this cell
            cell_binary = (labeled == label_id).astype(np.uint8)
            contours = find_contours(cell_binary, 0.5)
            
            if len(contours) > 0:
                # Use the largest contour
                largest = max(contours, key=len)
                # Convert to (x, y) format and scale
                coords = np.array([
                    (pt[1] * self.pixel_size_um, pt[0] * self.pixel_size_um)
                    for pt in largest
                ])
                
                if len(coords) >= 3:
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and not poly.is_empty:
                            polygons.append((cell_id, poly))
                    except Exception:
                        continue
        
        return polygons
    
    def _build_cell_collection(
        self,
        cell_mask: np.ndarray,
        assigned_df: pd.DataFrame
    ) -> CellCollection:
        """Build CellCollection from mask and transcript assignments."""
        from skimage.measure import regionprops, label
        
        labeled = label(cell_mask > 0)
        props = regionprops(labeled)
        
        cells_list = []
        for prop in props:
            # Get transcripts for this cell
            cell_transcripts = assigned_df[assigned_df['cell_id'] == f"cell_{prop.label:05d}"]
            
            cell = Cell(
                cell_id=f"cell_{prop.label:05d}",
                polygon=Polygon(),  # Placeholder - full impl would extract actual polygon
                centroid=(prop.centroid[1] * self.pixel_size_um, 
                          prop.centroid[0] * self.pixel_size_um),
                area_um2=prop.area * (self.pixel_size_um ** 2),
                n_transcripts=len(cell_transcripts),
                n_genes=cell_transcripts['gene'].nunique() if len(cell_transcripts) > 0 else 0
            )
            
            if len(cell_transcripts) > 0:
                gene_counts = cell_transcripts.groupby('gene').size().to_dict()
                cell.gene_counts = gene_counts
                cell.top_genes = sorted(
                    gene_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
                cell.top_genes = [g[0] for g in cell.top_genes]
            
            cells_list.append(cell)
        
        return CellCollection(cells_list)
    
    def _save_visualization(
        self,
        result: SoupSegResult,
        preprocessed_image: np.ndarray,
        cell_mask: np.ndarray,
        output_dir: str
    ):
        """Save visualization of segmentation results."""
        import matplotlib.pyplot as plt
        from skimage import color
        
        output_dir = Path(output_dir)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original ssDNA with cell overlay
        ax = axes[0]
        ax.imshow(preprocessed_image, cmap='gray')
        
        # Overlay cell boundaries
        labeled = label(cell_mask > 0)
        ax.imshow(labeled, cmap='random', alpha=0.3)
        ax.set_title(f"Cell Segmentation\n({len(result.cells)} cells)")
        ax.axis('off')
        
        # Transcript density
        ax = axes[1]
        ax.imshow(preprocessed_image, cmap='gray')
        
        # Plot transcripts colored by cell
        assigned = result.transcripts.get_assigned_transcripts()
        if len(assigned) > 0:
            x_coords = [t.x * self.pixel_size_um for t in assigned[:10000]]
            y_coords = [t.y * self.pixel_size_um for t in assigned[:10000]]
            ax.scatter(x_coords, y_coords, s=0.1, alpha=0.5)
        
        ax.set_title(f"Transcript Distribution\n({len(assigned)} assigned)")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "segmentation_overview.png", dpi=300)
        plt.close()


def run_pipeline(
    ssdna_image: str,
    transcript_file: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SoupSegResult:
    """
    Convenience function to run the pipeline.
    
    Example:
        >>> from soupseg.pipeline import run_pipeline
        >>> result = run_pipeline(
        ...     ssdna_image='ssdna.tiff',
        ...     transcript_file='transcripts.csv',
        ...     output_dir='output/'
        ... )
    """
    seg = SoupSeg(config=config, **kwargs)
    return seg.run(ssdna_image, transcript_file, output_dir)
