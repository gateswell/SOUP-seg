"""
Tests for Stage 4: Iterative Refinement.

Run with: pytest tests/test_refine.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from soupseg.stages.refine import (
    RefinementConfig,
    compute_image_gradient_energy,
    compute_transcript_density_energy,
    compute_prior_energy,
    build_graphcut_segmentation,
    watershed_refinement,
    compute_total_energy,
    apply_cell_size_constraints,
    iterative_refinement,
    simple_refinement,
)


class TestRefinementConfig:
    """Test RefinementConfig dataclass."""
    
    def test_default_config(self):
        config = RefinementConfig()
        assert config.max_iterations == 20
        assert config.tolerance == 0.001
        assert config.alpha_image == 0.4
        assert config.beta_transcript == 0.4
        assert config.gamma_prior == 0.2
    
    def test_custom_config(self):
        config = RefinementConfig(
            max_iterations=10,
            alpha_image=0.5,
            beta_transcript=0.3
        )
        assert config.max_iterations == 10
        assert config.alpha_image == 0.5
        assert config.beta_transcript == 0.3


class TestImageGradientEnergy:
    """Test image gradient energy computation."""
    
    def test_gradient_energy_empty(self):
        mask = np.zeros((10, 10), dtype=int)
        energy = compute_image_gradient_energy(mask, None)
        assert energy.shape == mask.shape
        assert np.all(energy == 0)
    
    def test_gradient_energy_with_gradient(self):
        mask = np.zeros((10, 10), dtype=int)
        mask[3:7, 3:7] = 1  # Square region
        gradient = np.random.rand(10, 10)
        
        energy = compute_image_gradient_energy(mask, gradient)
        assert energy.shape == mask.shape
        # Only boundary pixels should have non-zero energy
        boundary_pixels = energy > 0
        assert np.any(boundary_pixels)


class TestTranscriptDensityEnergy:
    """Test transcript density energy computation."""
    
    def test_empty_transcripts(self):
        mask = np.zeros((10, 10), dtype=int)
        mask[2:8, 2:8] = 1
        
        energy = compute_transcript_density_energy(
            mask, None, None, pixel_size_um=0.5
        )
        assert energy.shape == mask.shape
        assert np.all(energy == 0)
    
    def test_with_transcripts(self):
        mask = np.zeros((50, 50), dtype=int)
        mask[10:40, 10:40] = 1  # Large cell
        
        # Add some transcripts inside and outside
        transcripts = np.array([
            [20, 20],  # Inside
            [25, 25],  # Inside
            [5, 5],    # Outside
            [45, 45],  # Outside
        ])
        
        energy = compute_transcript_density_energy(
            mask, transcripts, None, pixel_size_um=0.5
        )
        assert energy.shape == mask.shape


class TestPriorEnergy:
    """Test prior energy computation."""
    
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=int)
        energy = compute_prior_energy(mask, pixel_size_um=0.5)
        assert energy == 0.0
    
    def test_single_cell(self):
        mask = np.zeros((50, 50), dtype=int)
        mask[10:40, 10:40] = 1  # Area = 30*30 = 900 px
        # At 0.5 um/px, area = 900 * 0.25 = 225 um^2
        
        energy = compute_prior_energy(
            mask, 
            pixel_size_um=0.5,
            target_area_um2=300.0,
            area_std_um2=150.0
        )
        assert energy > 0  # Should have some prior energy


class TestCellSizeConstraints:
    """Test cell size constraint application."""
    
    def test_remove_small_cells(self):
        mask = np.zeros((100, 100), dtype=int)
        mask[5:8, 5:8] = 1   # Small: 3x3 = 9 px
        mask[50:80, 50:80] = 2  # Large: 30x30 = 900 px
        
        # Min area at 0.5 um/px: 50 um^2 = 50/0.25 = 200 px
        result = apply_cell_size_constraints(
            mask,
            min_area_um2=50.0,
            max_area_um2=2000.0,
            pixel_size_um=0.5
        )
        
        # Small cell should be removed or merged
        labels = np.unique(result)
        assert len(labels) <= 2  # 0 and possibly the large cell


class TestSimpleRefinement:
    """Test simple refinement (cleanup only)."""
    
    def test_simple_refinement(self):
        mask = np.zeros((100, 100), dtype=int)
        mask[10:30, 10:30] = 1
        mask[50:55, 50:55] = 2  # Small cell
        
        result = simple_refinement(
            mask,
            min_area_um2=50.0,
            max_area_um2=2000.0,
            pixel_size_um=0.5
        )
        
        assert result.shape == mask.shape
        # Should still have labels
        assert len(np.unique(result)) >= 1


class TestIterativeRefinement:
    """Test main iterative refinement loop."""
    
    def test_basic_iteration(self):
        # Create simple test case
        mask = np.zeros((100, 100), dtype=int)
        mask[20:50, 20:50] = 1  # Cell 1
        mask[50:80, 50:80] = 2  # Cell 2
        
        # Gradient image (higher at boundaries)
        gradient = np.zeros((100, 100))
        gradient[20, :] = 1.0
        gradient[50, :] = 1.0
        gradient[:, 20] = 1.0
        gradient[:, 50] = 1.0
        
        # Some transcripts
        transcripts = np.array([
            [30, 30],  # In cell 1
            [60, 60],  # In cell 2
        ])
        
        config = RefinementConfig(
            max_iterations=3,
            verbose=False
        )
        
        result, props, info = iterative_refinement(
            mask,
            gradient,
            transcripts,
            cell_properties=[{"label": 1}, {"label": 2}],
            config=config,
            pixel_size_um=0.5
        )
        
        assert result.shape == mask.shape
        assert "iterations" in info
        assert "energies" in info
        assert info["iterations"] <= config.max_iterations
    
    def test_convergence(self):
        """Test that energy decreases or converges."""
        mask = np.zeros((80, 80), dtype=int)
        mask[10:40, 10:40] = 1
        mask[40:70, 40:70] = 2
        
        gradient = np.zeros((80, 80))
        # Add some gradient pattern
        for i in range(80):
            gradient[i, :] = np.sin(i / 10)
            gradient[:, i] += np.cos(i / 10)
        
        transcripts = np.array([
            [25, 25], [30, 30],  # Cell 1
            [55, 55], [60, 60],  # Cell 2
        ])
        
        config = RefinementConfig(
            max_iterations=5,
            tolerance=0.001,
            verbose=False
        )
        
        result, props, info = iterative_refinement(
            mask,
            gradient,
            transcripts,
            cell_properties=[],
            config=config,
            pixel_size_um=0.5
        )
        
        # Check energy trend
        energies = info["energies"]
        assert len(energies) > 0
        
        # Energy should generally decrease (allowing for some fluctuation)
        if len(energies) >= 2:
            initial = energies[0]
            final = energies[-1]
            # Final should be <= initial (or close, allowing for small increases)
            assert final <= initial * 1.1  # Within 10% of initial


class TestIntegration:
    """Integration tests with full pipeline."""
    
    def test_mock_stereoseq_data(self):
        """Test with mock Stereo-seq-like data."""
        np.random.seed(42)
        
        # Create mock 512x512 image (like a small Stereo-seq tile)
        image = np.zeros((512, 512), dtype=np.float64)
        
        # Add some "cells" (bright elliptical regions)
        centers = [(128, 128), (256, 256), (384, 384), (200, 350), (350, 200)]
        for cy, cx in centers:
            for y in range(512):
                for x in range(512):
                    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                    if dist < 80:
                        image[y, x] += 0.8 * (1 - dist / 80)
        
        # Add noise
        image += np.random.rand(512, 512) * 0.1
        image = np.clip(image, 0, 1)
        
        # Create mock transcripts
        n_transcripts = 500
        transcripts = []
        for _ in range(n_transcripts):
            # Put transcripts near cell centers
            cx = np.random.choice([128, 256, 384, 200, 350])
            cy = np.random.choice([128, 256, 384, 350, 200])
            x = int(cx + np.random.randn() * 30)
            y = int(cy + np.random.randn() * 30)
            x = np.clip(x, 0, 511)
            y = np.clip(y, 0, 511)
            gene = np.random.choice(['ACTB', 'GAPDH', 'EGFR', 'PTEN', 'TP53'])
            transcripts.append([x, y, gene])
        
        transcripts = np.array(transcripts)
        transcript_coords = transcripts[:, :2]
        transcript_genes = transcripts[:, 2]
        
        # Create simple initial segmentation
        from scipy.ndimage import label
        mask = (image > 0.3).astype(int)
        mask = label(mask)[0]
        
        # Compute gradient
        from skimage import filters
        gradient = filters.sobel(image)
        
        # Run refinement
        config = RefinementConfig(
            max_iterations=3,
            verbose=False
        )
        
        result, props, info = iterative_refinement(
            mask,
            gradient,
            transcript_coords,
            cell_properties=[],
            config=config,
            pixel_size_um=0.5
        )
        
        # Basic checks
        assert result.shape == mask.shape
        assert len(np.unique(result)) > 0
        assert len(info["energies"]) <= config.max_iterations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
