# Changelog

All notable changes to SoupSeg will be documented in this file.

## [1.1.0] - 2026-04-24

### Added

- **U-Net boundary detection model** (`src/soupseg/models/unet_boundary.py`)
  - Lightweight U-Net architecture for cell boundary detection from ssDNA images
  - Encoder-decoder with skip connections, 4 depth levels, base 32 features
  - Tiled inference for large images with smooth blending
  - Combined BCE + Dice loss with edge weighting for training
  - Data augmentation (flip, rotation) and learning rate scheduling
  - `create_unet_model()`, `detect_boundaries_unet()`, `train_unet()` APIs

- **Adaptive dilation radius** (`src/soupseg/models/adaptive_radius.py`)
  - Per-cell expansion radius adapted to local data characteristics
  - Three factors: cell density (inverse), image intensity (direct), transcript density (bell-curve)
  - Per-pixel radius map with Gaussian smoothing for spatial coherence
  - Per-cell radius with intensity spread adjustment
  - PyTorch-accelerated Gaussian filtering via `compute_adaptive_radius_map_torch()`
  - `adaptive_expand_nuclei()` high-level API replacing fixed-radius expansion

- **PyTorch-based Stage 4 refinement** (`src/soupseg/stages/refine.py`)
  - `RefinementConfig` extended with `use_unet`, `use_adaptive_radius`, `use_torch` flags
  - `compute_boundary_map()` selects U-Net or Sobel gradient automatically
  - PyTorch-accelerated energy computation where CUDA is available
  - Vectorized transcript density energy computation
  - Full backward compatibility with v1.0.0 API (same function signatures with optional new params)
  - `ssdna_image` parameter added to `iterative_refinement()` for U-Net input

- **Dependencies**: Added `torch>=2.0` and `torchvision>=0.15` to requirements.txt

### Changed

- Version bumped from 0.1.0 → 1.1.0
- `iterative_refinement()` now accepts optional `ssdna_image` for U-Net boundary detection
- `RefinementConfig` now includes U-Net, adaptive radius, and PyTorch configuration
- Transcript density energy computation is vectorized (significant speedup)
- Graph Cut edge construction uses 4-connectivity instead of 8- (faster, similar quality)

### Backward Compatibility

- All v1.0.0 code continues to work without changes
- U-Net and adaptive radius are opt-in via `RefinementConfig.use_unet` and `RefinementConfig.use_adaptive_radius`
- Falls back gracefully when PyTorch is not installed (warnings, not errors)

## [1.0.0] - 2026-04-20

### Added

- Initial release of SoupSeg pipeline
- Stage 0: Data loading (ssDNA images, transcript CSV)
- Stage 1: Preprocessing (denoising, CLAHE, background removal)
- Stage 2: Nuclei detection (Otsu thresholding, morphological operations, cell expansion)
- Stage 3: Transcript assignment (Voronoi-based, polygon-based, co-expression)
- Stage 4: Iterative refinement (Graph Cut optimization, energy minimization)
- Cell and Transcript data models
- Graph Cut optimization using PyMaxflow
- Watershed fallback when PyMaxflow unavailable
