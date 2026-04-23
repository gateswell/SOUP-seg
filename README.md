# SoupSeg

**ssDNA-Informed Iterative Cell Segmentation for Stereo-seq**

A cell segmentation tool that combines ssDNA image-based nuclear detection with transcriptomic constraints for Stereo-seq v1.3 data.

## Features

- **Native Stereo-seq v1.3 support** (0.5 μm resolution)
- **Multi-stage pipeline**: Image preprocessing → Nuclei detection → Transcript assignment → Iterative refinement
- **Modular design**: Use each stage independently
- **Configurable**: YAML-based configuration
- **Open source**: MIT License

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/soup-seg.git
cd soup-seg

# Install dependencies
pip install -r requirements.txt

# Install SoupSeg
pip install -e .
```

## Quick Start

```python
from soupseg import SoupSeg

# Initialize
seg = SoupSeg()

# Run pipeline
result = seg.run(
    ssdna_image='path/to/ssdna.tiff',
    transcript_file='path/to/transcripts.csv',
    output_dir='output/'
)

# View results
print(f"Segmented {len(result.cells)} cells")
print(f"Assignment rate: {result.transcripts.summary_stats()['assignment_rate']:.1%}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and algorithm details
- [API Reference](docs/API.md) - Detailed API documentation
- [Tutorial](docs/TUTORIAL.md) - Step-by-step usage guide

## Development Status

**⚠️ This is a prototype framework.** The code structure and interfaces are in place, but some advanced features (especially Stage 4 iterative refinement) are still placeholder implementations.

### Implemented
- ✅ Image preprocessing (CLAHE, denoising, background removal)
- ✅ Nuclei detection (Otsu, adaptive thresholding)
- ✅ Nuclear expansion to cell boundaries
- ✅ Voronoi-based transcript assignment
- ✅ Basic cell and transcript data models
- ✅ Result saving (JSON format)

### To be implemented
- 🔄 Full iterative refinement (Graph Cut / Random Walk optimization)
- 🔄 Diffusion correction module
- 🔄 Gene co-expression correction
- 🔄 GPU acceleration
- 🔄 Web interface (napari plugin)

## License

MIT License

## Citation

If you use SoupSeg in your research, please cite:

```
SoupSeg: ssDNA-Informed Iterative Cell Segmentation for Stereo-seq
```

## Contact

For questions and issues, please open an issue on GitHub.
