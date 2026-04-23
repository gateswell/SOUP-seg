# SoupSeg Development Log

**Project**: ssDNA-Informed Iterative Cell Segmentation for Stereo-seq  
**Repository**: https://github.com/gateswell/SOUP-seg  
**Version**: 1.1.0  
**Date**: 2026-04-24  
**Developer**: AI Assistant (via OpenClaw)  
**User**: gateswell  

---

## 1. Project Overview

### 1.1 Background
Stereo-seq v1.3 provides high-resolution spatial transcriptomics with 0.5 μm/pixel resolution. Cell segmentation traditionally relies on nuclear staining (ssDNA) or membrane markers. This project aims to combine ssDNA-based nuclear detection with transcriptomic constraints for improved cell segmentation.

### 1.2 Goal
Develop a 4-stage iterative segmentation pipeline that jointly optimizes cell boundaries using:
- Image gradient energy (from ssDNA)
- Transcript density energy (transcript distribution)
- Prior energy (cell size/shape priors)

### 1.3 Constraints
- **Resolution**: 0.5 μm/pixel (Stereo-seq v1.3)
- **Development**: Raspberry Pi (project structure + prototype)
- **Testing**: A30 GPU Server (full data + GPU acceleration)
- **Data**: ssDNA image + transcript coordinates

---

## 2. Development Timeline

| Date | Time | Activity |
|------|------|----------|
| 2026-04-23 | ~22:00 | Project initialization, directory structure |
| 2026-04-23 | ~22:10 | Architecture design document (docs/ARCHITECTURE.md) |
| 2026-04-23 | ~22:20 | Stage 1: Preprocessing (preprocess.py) |
| 2026-04-23 | ~22:30 | Stage 2: Nuclei detection (nuclei.py) |
| 2026-04-23 | ~22:40 | Stage 3: Transcript assignment (assignment.py) |
| 2026-04-23 | ~22:50 | Data models (cell.py, transcript.py) |
| 2026-04-23 | ~23:00 | Pipeline orchestration (pipeline.py) |
| 2026-04-23 | ~23:10 | Stage 4: Iterative refinement (refine.py) - **Graph Cut complete** |
| 2026-04-23 | ~23:15 | Test suite + mock data generator |
| 2026-04-23 | ~23:18 | **GitHub push** (commit 792ce73) |

---

## 3. Functional Modules

### 3.1 Module Map

```
soupseg/
├── __init__.py           # Package export
├── pipeline.py           # Main orchestration (SoupSeg class)
├── models/
│   ├── cell.py           # Cell, CellCollection
│   └── transcript.py     # Transcript, TranscriptCollection
└── stages/
    ├── preprocess.py     # Stage 1: Denoise + CLAHE + Normalize
    ├── nuclei.py         # Stage 2: Otsu thresholding + radial expansion
    ├── assignment.py     # Stage 3: Voronoi + diffusion correction
    └── refine.py         # Stage 4: Graph Cut iterative optimization
```

### 3.2 Stage Details

#### Stage 1: Preprocessing (`preprocess.py`)
**Purpose**: Enhance ssDNA image quality for downstream segmentation.

**Functions**:
- `load_image(path)`: Load TIFF/PNG, convert to float
- `gaussian_denoise(image, sigma)`: Gaussian blur denoising
- `clahe_enhance(image, clip_limit, kernel_size)`: CLAHE contrast enhancement
- `remove_background(image, threshold)`: Otsu background removal
- `normalize_image(image)`: Min-max normalization to [0, 1]
- `preprocess_image(image, ...)`: Full preprocessing pipeline

**Default Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| denoise_sigma | 1.5 | Gaussian kernel sigma |
| clahe_clip_limit | 0.03 | CLAHE clip limit |
| clahe_kernel_size | (8, 8) | CLAHE tile size |

#### Stage 2: Nuclei Detection (`nuclei.py`)
**Purpose**: Detect nuclei from preprocessed ssDNA image and initialize cell regions.

**Functions**:
- `detect_nuclei(image, method, ...)`: Main nuclei detection
- `detect_nuclei_otsu(image, min_area, max_area)`: Otsu thresholding
- `expand_nuclei(nuclei_mask, radius_um, pixel_size)`: Radial expansion
- `initialize_cells(expanded_mask, ...)`: Create Cell objects

**Default Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| method | otsu | Detection method |
| expansion_radius_um | 6.0 | Cell boundary expansion radius |
| min_area_um2 | 50.0 | Minimum cell area |
| max_area_um2 | 2000.0 | Maximum cell area |

**Key Algorithm**:
1. Otsu thresholding to get nuclear mask
2. Morphological opening to remove noise
3. Connected component labeling
4. Filter by area constraints
5. Radial expansion (ball structuring element)

#### Stage 3: Transcript Assignment (`assignment.py`)
**Purpose**: Assign transcript molecules to cell regions.

**Functions**:
- `assign_transcripts(transcripts_df, cell_polygons, ...)`: Main assignment
- `voronoi_assignment(transcripts_df, cell_polygons, ...)`: Voronoi tessellation
- `polygon_assignment(transcripts_df, cell_polygons, ...)`: Point-in-polygon

**Default Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| method | voronoi | Assignment algorithm |
| max_distance_um | 10.0 | Maximum distance to cell boundary |
| use_coexpression | False | Use gene co-expression |
| diffusion_correction | True | Apply diffusion correction |

**Key Algorithm**:
1. Build cell boundary polygons from segmentation mask
2. Create Voronoi diagram from cell centroids
3. For each transcript, find containing polygon
4. Apply diffusion correction (Gaussian kernel)
5. Return transcripts with cell_id assignments

#### Stage 4: Iterative Refinement (`refine.py`)
**Purpose**: Jointly optimize cell boundaries using multi-term energy function.

**Energy Function**:
```
E_total = α * E_image + β * E_transcript + γ * E_prior
```

**Functions**:
- `RefinementConfig`: Configuration dataclass
- `compute_image_gradient_energy(...)`: Image boundary energy
- `compute_transcript_density_energy(...)`: Transcript distribution energy
- `compute_prior_energy(...)`: Cell size/shape prior energy
- `build_graphcut_segmentation(...)`: **Boykov-Kolmogorov Graph Cut**
- `watershed_refinement(...)`: Fallback (no PyMaxflow)
- `apply_cell_size_constraints(...)`: Remove too-small/too-large cells
- `compute_total_energy(...)`: Total energy computation
- `iterative_refinement(...)`: **Main optimization loop**

**Default Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| max_iterations | 20 | Maximum refinement iterations |
| tolerance | 0.001 | Convergence tolerance |
| alpha_image | 0.4 | Image gradient weight |
| beta_transcript | 0.4 | Transcript density weight |
| gamma_prior | 0.2 | Prior energy weight |
| max_boundary_shift_px | 3.0 | Max boundary movement per iteration |
| min_cell_size_um2 | 50.0 | Minimum cell area |
| max_cell_size_um2 | 2000.0 | Maximum cell area |
| edge_weight_scale | 10.0 | Graph Cut edge weight scale |
| fallback_method | watershed | Fallback if PyMaxflow unavailable |

**Key Algorithm**:
1. Build region adjacency graph (RAG) from cell segmentation
2. For each iteration:
   a. Compute total energy E(k)
   b. Check convergence: |E(k) - E(k-1)| / E(k-1) < tolerance
   c. Build max-flow graph (nodes = pixels, edges = boundary costs)
   d. Solve min-cut using Boykov-Kolmogorov algorithm
   e. Update cell boundaries from cut result
   f. Apply cell size constraints (merge small cells)
3. Return refined segmentation + convergence info

**Dependencies**: PyMaxflow (for Graph Cut)

---

## 4. Data Models

### 4.1 Cell (`models/cell.py`)
```python
@dataclass
class Cell:
    cell_id: str
    label_id: int
    polygon: Polygon          # shapely Polygon
    centroid: Tuple[float, float]
    area_um2: float
    n_transcripts: int
    gene_counts: Dict[str, int]
    boundary: LineString
    metadata: Dict[str, Any]
```

### 4.2 Transcript (`models/transcript.py`)
```python
@dataclass
class Transcript:
    transcript_id: str
    x: float                  # Pixel coordinates
    y: float
    gene: str
    count: int
    cell_id: Optional[str]    # Assigned cell (None if unassigned)
    distance_to_boundary: Optional[float]
```

---

## 5. Configuration

### 5.1 Default Configuration (pipeline.py)
```python
{
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
    }
}
```

---

## 6. Output Format

### 6.1 Cell Segmentation (GeoJSON)
```json
{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "cell_id": "cell_00001",
                "n_transcripts": 42,
                "area_um2": 287.5
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[x1,y1], [x2,y2], ...]]
            }
        }
    ]
}
```

### 6.2 Transcript Assignment (CSV)
```csv
transcript_id,x,y,gene,count,cell_id,distance_to_boundary
tx_00000000,1523,2048,GAPDH,1,cell_00001,2.3
tx_00000001,1621,2091,ACTB,1,,5.2
```

### 6.3 Metadata (JSON)
```json
{
    "pipeline": "SoupSeg",
    "version": "0.1.0",
    "pixel_size_um": 0.5,
    "image_shape": [4096, 4096],
    "n_cells_initial": 150,
    "n_cells_final": 148,
    "n_transcripts": 45231,
    "assignment_rate": 0.873,
    "convergence": {
        "iterations": 12,
        "converged": true,
        "final_energy": 0.0234
    }
}
```

---

## 7. Dependencies

### 7.1 Core Dependencies (requirements.txt)
```
numpy>=1.21
pandas>=1.3
scipy>=1.7
scikit-image>=0.19
shapely>=1.8
tifffile>=2021.11
opencv-python>=4.5
PyYAML>=6.0
```

### 7.2 Optional Dependencies
```
PyMaxflow>=1.2    # Graph Cut optimization (Stage 4)
numba>=0.55       # JIT compilation speedup
joblib>=1.1       # Parallel processing
```

---

## 8. File Structure

```
soup-seg/
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml           # Default configuration
├── docs/
│   ├── ARCHITECTURE.md        # Architecture design
│   └── DEVELOPMENT.md         # This document
├── notebooks/
│   └── example.ipynb          # Example usage (planned)
├── src/soupseg/
│   ├── __init__.py
│   ├── pipeline.py            # Main pipeline class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cell.py            # Cell data model
│   │   └── transcript.py      # Transcript data model
│   └── stages/
│       ├── __init__.py
│       ├── preprocess.py      # Stage 1
│       ├── nuclei.py          # Stage 2
│       ├── assignment.py      # Stage 3
│       └── refine.py          # Stage 4 (Graph Cut)
├── tests/
│   ├── __init__.py
│   ├── test_refine.py         # Stage 4 tests
│   └── generate_mock_data.py  # Mock data generator
└── weights/                   # Model weights (future)
```

---

## 9. Known Limitations

1. **Graph Cut Scalability**: Full-image Graph Cut is O(V²E) — slow on large images. Consider tile-based processing for >4K images.
2. **Nuclear Expansion Radius**: Fixed 6.0 μm may not suit all tissue types. Should be tissue-specific.
3. **Voronoi Assignment**: Does not handle overlapping cells or cell-cell adhesion constraints.
4. **No GPU Acceleration**: Current implementation is CPU-only. PyTorch/TensorFlow integration planned.
5. **2D Only**: Currently 2D segmentation. 3D extension planned for thick tissue sections.

---

## 10. Future Development

### Short-term
- [ ] GPU acceleration for Graph Cut (CuPy/CUDA)
- [ ] Tile-based processing for large images
- [ ] Cell-cell adhesion constraints
- [ ] Membrane stain integration (if available)

### Medium-term
- [ ] 3D segmentation support
- [ ] Gene expression-based cell type annotation
- [ ] Quality metrics and validation framework
- [ ] Interactive visualization tool

### Long-term
- [ ] Deep learning integration (CellPose-style)
- [ ] Multi-modal fusion (ssDNA + membrane + transcript)
- [ ] Benchmark dataset and evaluation

---

## 11. Change Log

### v1.1.0 (2026-04-24)
- **U-Net Boundary Detection** (`models/unet_boundary.py`):
  - 4-level lightweight U-Net (PyTorch)
  - BCE + Dice combined loss
  - Tiling inference with Hann-window blending for large images
  - Training pipeline with data augmentation (flip, rotation)
  - Optional GPU acceleration (CUDA auto-detected)
- **Adaptive Dilation Radius** (`models/adaptive_radius.py`, `stages/adaptive_radius.py`):
  - Per-cell adaptive expansion based on:
    1. Local cell density (Gaussian-smoothed)
    2. ssDNA image intensity (ring spread)
    3. Transcript density (spatial binning)
  - Configurable min/max radius bounds
  - GPU-accelerated path via PyTorch (scipy fallback)
- **GNN Boundary Refinement** (`models/gnn_boundary.py`):
  - 2-layer Graph Attention Network (GAT)
  - Node features: area, eccentricity, intensity, centroid
  - Edge features: centroid distance, area sum/diff
  - Cell-cell boundary probability prediction
  - Merge low-probability edges to correct over-segmentation
- **PyTorch Integration**:
  - All new modules use PyTorch as optional dependency
  - `TORCH_AVAILABLE` guard throughout — falls back gracefully
  - CUDA auto-detection when available
- **Pipeline Updates**:
  - New config flags: `use_unet_boundary`, `use_adaptive_radius`, `use_gnn_boundary`
  - All v1.0.0 APIs remain unchanged (backward compatible)
- **Requirements** updated: torch>=1.10 added as optional dependency

### v0.1.0 (2026-04-23)
- Initial implementation
- 4-stage pipeline complete
- Graph Cut optimization implemented
- Mock data generator and test suite
- GitHub repository created

---

*Last updated: 2026-04-24 02:10 GMT+8*
