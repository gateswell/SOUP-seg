"""
Tests for Stage 4: Iterative Refinement (v1.1.0 PyTorch).

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
    compute_boundary_map,
    compute_image_gradient_energy_torch,
    compute_transcript_density_energy,
    compute_prior_energy,
    compute_total_energy,
    build_graphcut_segmentation,
    watershed_refinement,
    apply_cell_size_constraints,
    iterative_refinement,
    simple_refinement,
)

from soupseg.models.unet_boundary import (
    UNetConfig,
    create_unet_model,
    detect_boundaries_unet,
    TORCH_AVAILABLE as UNET_TORCH_AVAILABLE,
)

from soupseg.models.adaptive_radius import (
    AdaptiveRadiusConfig,
    compute_cell_density_map,
    compute_local_intensity_features,
    compute_transcript_density_map,
    compute_adaptive_radius_map,
    apply_adaptive_dilation,
    adaptive_expand_nuclei,
)


# ---------------------------------------------------------------------------
# RefinementConfig tests
# ---------------------------------------------------------------------------

class TestRefinementConfig:
    def test_default_config(self):
        config = RefinementConfig()
        assert config.max_iterations == 20
        assert config.tolerance == 0.001
        assert config.alpha_image == 0.4
        assert config.beta_transcript == 0.4
        assert config.gamma_prior == 0.2
        # v1.1.0 new fields
        assert config.use_unet is True
        assert config.use_adaptive_radius is True
        assert config.use_torch is True

    def test_custom_config(self):
        config = RefinementConfig(
            max_iterations=10,
            alpha_image=0.5,
            beta_transcript=0.3,
            use_unet=False,
            use_adaptive_radius=False,
        )
        assert config.max_iterations == 10
        assert config.alpha_image == 0.5
        assert config.beta_transcript == 0.3
        assert config.use_unet is False
        assert config.use_adaptive_radius is False

    def test_device_auto(self):
        config = RefinementConfig()
        device = config.get_device()
        assert device in ("cpu", "cuda")

    def test_torch_enabled_property(self):
        config = RefinementConfig(use_torch=False)
        assert config.torch_enabled is False


# ---------------------------------------------------------------------------
# Boundary detection tests
# ---------------------------------------------------------------------------

class TestBoundaryMap:
    def test_sobel_fallback(self):
        """When use_unet=False, should use Sobel gradient."""
        image = np.random.rand(64, 64).astype(np.float32)
        config = RefinementConfig(use_unet=False)
        boundary = compute_boundary_map(image, config)
        assert boundary.shape == image.shape
        assert boundary.dtype in (np.float32, np.float64)
        assert boundary.max() <= 1.0

    @pytest.mark.skipif(not UNET_TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_unet_boundary(self):
        """When use_unet=True and PyTorch available, should run U-Net."""
        image = np.random.rand(64, 64).astype(np.float32)
        config = RefinementConfig(use_unet=True, boundary_threshold=0.5)
        boundary = compute_boundary_map(image, config)
        assert boundary.shape == image.shape
        assert boundary.max() <= 1.0


# ---------------------------------------------------------------------------
# U-Net model tests
# ---------------------------------------------------------------------------

class TestUNetModel:
    @pytest.mark.skipif(not UNET_TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_create_model(self):
        config = UNetConfig(base_features=16, depth=3)
        model = create_unet_model(config)
        assert model is not None
        assert isinstance(model.base_features, int)

    @pytest.mark.skipif(not UNET_TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_forward_pass(self):
        import torch
        config = UNetConfig(base_features=16, depth=3)
        model = create_unet_model(config)
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    @pytest.mark.skipif(not UNET_TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_boundary(self):
        image = np.random.rand(128, 128).astype(np.float32)
        config = UNetConfig(base_features=16, depth=3, tile_size=128)
        model = create_unet_model(config)
        result = model.predict_boundary(image, device='cpu', tile_size=128)
        assert result.shape == (128, 128)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.skipif(not UNET_TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_detect_boundaries_unet(self):
        image = np.random.rand(64, 64).astype(np.float32)
        boundary_mask = detect_boundaries_unet(image, threshold=0.5)
        assert boundary_mask.shape == (64, 64)
        assert set(np.unique(boundary_mask)).issubset({0, 255})


# ---------------------------------------------------------------------------
# Adaptive radius tests
# ---------------------------------------------------------------------------

class TestAdaptiveRadius:
    def test_cell_density_map(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:30, 20:30] = 1
        mask[60:70, 60:70] = 1
        density = compute_cell_density_map(mask, pixel_size_um=0.5, sigma_um=10.0)
        assert density.shape == mask.shape
        assert density.max() <= 1.0
        assert density.min() >= 0.0

    def test_transcript_density_map(self):
        coords = np.array([[25, 25], [65, 65], [10, 10]], dtype=np.float64)
        density = compute_transcript_density_map(coords, (100, 100), pixel_size_um=0.5)
        assert density.shape == (100, 100)
        assert density.max() <= 1.0

    def test_transcript_density_map_empty(self):
        density = compute_transcript_density_map(None, (100, 100))
        assert np.all(density == 0)

    def test_compute_adaptive_radius_map(self):
        image = np.random.rand(100, 100).astype(np.float32)
        nuclei_mask = np.zeros((100, 100), dtype=np.uint8)
        nuclei_mask[20:30, 20:30] = 1
        nuclei_mask[60:70, 60:70] = 1
        nuclei_labels = nuclei_mask.copy()
        nuclei_labels[60:70, 60:70] = 2

        config = AdaptiveRadiusConfig(
            min_radius_um=3.0,
            max_radius_um=12.0,
            verbose=True,
        )

        radius_map, per_cell_radius = compute_adaptive_radius_map(
            image, nuclei_mask, nuclei_labels, config=config
        )

        assert radius_map.shape == (100, 100)
        assert radius_map.min() >= config.min_radius_um
        assert radius_map.max() <= config.max_radius_um
        assert len(per_cell_radius) == 2
        for r in per_cell_radius.values():
            assert config.min_radius_um <= r <= config.max_radius_um

    def test_apply_adaptive_dilation(self):
        nuclei_labels = np.zeros((100, 100), dtype=np.int32)
        nuclei_labels[20:30, 20:30] = 1
        nuclei_labels[60:70, 60:70] = 2

        radius_map = np.full((100, 100), 6.0)  # Uniform 6 um radius
        per_cell_radius = {1: 6.0, 2: 6.0}

        expanded = apply_adaptive_dilation(
            nuclei_labels, radius_map, per_cell_radius, pixel_size_um=0.5
        )

        assert expanded.shape == (100, 100)
        # Should have more non-zero pixels than input
        assert np.sum(expanded > 0) > np.sum(nuclei_labels > 0)

    def test_adaptive_expand_nuclei(self):
        image = np.random.rand(100, 100).astype(np.float32)
        nuclei_mask = np.zeros((100, 100), dtype=np.uint8)
        nuclei_mask[20:30, 20:30] = 1
        nuclei_mask[60:70, 60:70] = 1
        nuclei_labels = nuclei_mask.copy()
        nuclei_labels[60:70, 60:70] = 2

        expanded, info = adaptive_expand_nuclei(
            image, nuclei_mask, nuclei_labels,
            config=AdaptiveRadiusConfig(pixel_size_um=0.5),
        )

        assert expanded.shape == (100, 100)
        assert "method" in info
        assert info["method"] == "adaptive"
        assert info["n_cells"] == 2


# ---------------------------------------------------------------------------
# Refinement integration tests
# ---------------------------------------------------------------------------

class TestImageGradientEnergy:
    def test_gradient_energy_empty(self):
        mask = np.zeros((10, 10), dtype=int)
        energy = compute_image_gradient_energy_torch(mask, None)
        assert energy.shape == mask.shape
        assert np.all(energy == 0)

    def test_gradient_energy_with_gradient(self):
        mask = np.zeros((10, 10), dtype=int)
        mask[3:7, 3:7] = 1
        gradient = np.random.rand(10, 10)
        energy = compute_image_gradient_energy_torch(mask, gradient)
        assert energy.shape == mask.shape
        assert np.any(energy > 0)


class TestTranscriptDensityEnergy:
    def test_empty_transcripts(self):
        mask = np.zeros((10, 10), dtype=int)
        mask[2:8, 2:8] = 1
        energy = compute_transcript_density_energy(mask, None, pixel_size_um=0.5)
        assert np.all(energy == 0)

    def test_with_transcripts(self):
        mask = np.zeros((50, 50), dtype=int)
        mask[10:40, 10:40] = 1
        transcripts = np.array([[20, 20], [25, 25], [5, 5], [45, 45]])
        energy = compute_transcript_density_energy(mask, transcripts, pixel_size_um=0.5)
        assert energy.shape == mask.shape


class TestPriorEnergy:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=int)
        energy = compute_prior_energy(mask, pixel_size_um=0.5)
        assert energy == 0.0

    def test_single_cell(self):
        mask = np.zeros((50, 50), dtype=int)
        mask[10:40, 10:40] = 1
        energy = compute_prior_energy(mask, pixel_size_um=0.5, target_area_um2=300.0, area_std_um2=150.0)
        assert energy > 0


class TestCellSizeConstraints:
    def test_remove_small_cells(self):
        mask = np.zeros((100, 100), dtype=int)
        mask[5:8, 5:8] = 1   # Small: 9 px
        mask[50:80, 50:80] = 2  # Large: 900 px
        result = apply_cell_size_constraints(mask, min_area_um2=50.0, max_area_um2=2000.0, pixel_size_um=0.5)
        labels = np.unique(result)
        assert len(labels) <= 3  # 0, possibly merged small cell, and large cell


class TestSimpleRefinement:
    def test_simple_refinement(self):
        mask = np.zeros((100, 100), dtype=int)
        mask[10:30, 10:30] = 1
        mask[50:55, 50:55] = 2
        result = simple_refinement(mask, min_area_um2=50.0, max_area_um2=2000.0, pixel_size_um=0.5)
        assert result.shape == mask.shape
        assert len(np.unique(result)) >= 1


class TestIterativeRefinement:
    def test_basic_iteration(self):
        mask = np.zeros((100, 100), dtype=int)
        mask[20:50, 20:50] = 1
        mask[50:80, 50:80] = 2
        gradient = np.zeros((100, 100))
        gradient[20, :] = 1.0
        gradient[50, :] = 1.0
        transcripts = np.array([[30, 30], [60, 60]])
        config = RefinementConfig(max_iterations=3, verbose=False, use_unet=False)
        result, props, info = iterative_refinement(
            mask, gradient, transcripts,
            cell_properties=[{"label": 1}, {"label": 2}],
            config=config, pixel_size_um=0.5,
        )
        assert result.shape == mask.shape
        assert "iterations" in info
        assert info["iterations"] <= config.max_iterations
        assert info["version"] == "1.1.0"

    def test_convergence(self):
        mask = np.zeros((80, 80), dtype=int)
        mask[10:40, 10:40] = 1
        mask[40:70, 40:70] = 2
        gradient = np.zeros((80, 80))
        for i in range(80):
            gradient[i, :] = np.sin(i / 10)
            gradient[:, i] += np.cos(i / 10)
        transcripts = np.array([[25, 25], [30, 30], [55, 55], [60, 60]])
        config = RefinementConfig(max_iterations=5, tolerance=0.001, verbose=False, use_unet=False)
        result, props, info = iterative_refinement(
            mask, gradient, transcripts,
            cell_properties=[], config=config, pixel_size_um=0.5,
        )
        energies = info["energies"]
        assert len(energies) > 0
        if len(energies) >= 2:
            assert energies[-1] <= energies[0] * 1.1


class TestIntegration:
    def test_mock_stereoseq_data(self):
        np.random.seed(42)
        image = np.zeros((128, 128), dtype=np.float64)
        centers = [(32, 32), (96, 96)]
        for cy, cx in centers:
            for y in range(128):
                for x in range(128):
                    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                    if dist < 40:
                        image[y, x] += 0.8 * (1 - dist / 40)
        image += np.random.rand(128, 128) * 0.1
        image = np.clip(image, 0, 1).astype(np.float32)

        n_transcripts = 200
        transcripts = []
        for _ in range(n_transcripts):
            cx = np.random.choice([32, 96])
            cy = np.random.choice([32, 96])
            x = int(cx + np.random.randn() * 15)
            y = int(cy + np.random.randn() * 15)
            x = np.clip(x, 0, 127)
            y = np.clip(y, 0, 127)
            transcripts.append([x, y])
        transcript_coords = np.array(transcripts)

        from scipy.ndimage import label
        mask = (image > 0.3).astype(int)
        mask = label(mask)[0]

        from skimage import filters
        gradient = filters.sobel(image)

        config = RefinementConfig(max_iterations=2, verbose=False, use_unet=False)
        result, props, info = iterative_refinement(
            mask, gradient, transcript_coords,
            cell_properties=[], config=config, pixel_size_um=0.5,
        )
        assert result.shape == mask.shape
        assert len(np.unique(result)) > 0
        assert info["version"] == "1.1.0"

    def test_with_adaptive_radius(self):
        """Test iterative refinement with adaptive radius enabled."""
        np.random.seed(42)
        image = np.random.rand(100, 100).astype(np.float32) * 0.5
        image[20:40, 20:40] = 0.9
        image[60:80, 60:80] = 0.9

        mask = np.zeros((100, 100), dtype=int)
        mask[25:35, 25:35] = 1
        mask[65:75, 65:75] = 2

        gradient = np.zeros((100, 100))
        gradient[25, :] = 1.0
        gradient[35, :] = 1.0
        gradient[65, :] = 1.0
        gradient[75, :] = 1.0

        config = RefinementConfig(
            max_iterations=2,
            verbose=False,
            use_unet=False,
            use_adaptive_radius=True,
        )

        result, props, info = iterative_refinement(
            mask, gradient, None,
            cell_properties=[], config=config,
            pixel_size_um=0.5,
            ssdna_image=image,
        )

        assert result.shape == mask.shape
        assert "adaptive_radius" in info
        assert info["version"] == "1.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
