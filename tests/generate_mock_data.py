"""
Generate mock Stereo-seq data for testing SoupSeg.

This script creates synthetic ssDNA images and transcript coordinates
for testing the segmentation pipeline.

Usage:
    python tests/generate_mock_data.py --output-dir tests/data/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def generate_ellipse_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radii: Tuple[int, int],
    angle: float = 0
) -> np.ndarray:
    """
    Generate an elliptical mask.
    
    Args:
        shape: (height, width)
        center: (y, x) center coordinates
        radii: (ry, rx) semi-axes lengths
        angle: Rotation angle in degrees
        
    Returns:
        Binary mask with ellipse = 1
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    
    # Center coordinates
    cy, cx = center
    ry, rx = radii
    
    # Rotation matrix
    theta = np.radians(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    
    # Ellipse equation (rotated)
    dx = (x - cx) / rx
    dy = (y - cy) / ry
    
    if angle != 0:
        dx_new = cos_a * dx + sin_a * dy
        dy_new = -sin_a * dx + cos_a * dy
        dx, dy = dx_new, dy_new
    
    mask = (dx**2 + dy**2) <= 1.0
    return mask.astype(np.uint8)


def generate_mock_ssdna_image(
    shape: Tuple[int, int] = (1024, 1024),
    n_cells: int = 20,
    seed: int = 42
) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate a mock ssDNA image with nuclei.
    
    Args:
        shape: Image shape (height, width)
        n_cells: Number of cells to generate
        seed: Random seed
        
    Returns:
        Tuple of (image, cell_info_list)
    """
    np.random.seed(seed)
    
    h, w = shape
    image = np.zeros(shape, dtype=np.float64)
    
    cell_info = []
    
    # Generate random cell positions (avoiding edges)
    margin = 50
    n_attempts = n_cells * 10
    positions = []
    
    for _ in range(n_attempts):
        if len(positions) >= n_cells:
            break
        
        cy = np.random.randint(margin, h - margin)
        cx = np.random.randint(margin, w - margin)
        
        # Check distance from other cells
        min_dist = 60
        too_close = False
        for py, px in positions:
            dist = np.sqrt((cy - py)**2 + (cx - px)**2)
            if dist < min_dist:
                too_close = True
                break
        
        if not too_close:
            positions.append((cy, cx))
    
    # Generate cells
    for i, (cy, cx) in enumerate(positions):
        # Random ellipse parameters
        ry = np.random.randint(15, 35)
        rx = np.random.randint(15, 35)
        angle = np.random.uniform(0, 180)
        
        # Intensity (nuclei are bright)
        intensity = np.random.uniform(0.6, 1.0)
        
        # Add noise
        noise_level = np.random.uniform(0.05, 0.15)
        
        # Generate mask
        mask = generate_ellipse_mask(shape, (cy, cx), (ry, rx), angle)
        
        # Add to image with smooth edges
        for y in range(max(0, cy - ry - 5), min(h, cy + ry + 5)):
            for x in range(max(0, cx - rx - 5), min(w, cx + rx + 5)):
                mx = (x - cx) / rx
                my = (y - cy) / ry
                if angle != 0:
                    cos_a, sin_a = np.cos(-np.radians(angle)), np.sin(-np.radians(angle))
                    mx_new = cos_a * mx + sin_a * my
                    my_new = -sin_a * mx + cos_a * my
                    mx, my = mx_new, my_new
                
                dist = mx**2 + my**2
                if dist <= 1.2:  # Slightly larger for smooth falloff
                    alpha = max(0, 1 - dist / 1.2)
                    image[y, x] += intensity * alpha
        
        cell_info.append({
            "cell_id": f"cell_{i:05d}",
            "center_y": cy,
            "center_x": cx,
            "radius_y": ry,
            "radius_x": rx,
            "angle": angle,
            "intensity": intensity
        })
    
    # Add uniform background noise
    image += np.random.rand(h, w) * noise_level
    image = np.clip(image, 0, 1)
    
    return image, cell_info


def generate_mock_transcripts(
    cell_info: List[dict],
    shape: Tuple[int, int],
    transcripts_per_cell: int = 50,
    n_background: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate mock transcript coordinates.
    
    Args:
        cell_info: List of cell info dicts from generate_mock_ssdna_image
        shape: Image shape (height, width)
        transcripts_per_cell: Average transcripts per cell
        n_background: Number of background (unassigned) transcripts
        seed: Random seed
        
    Returns:
        DataFrame with columns [x, y, gene, cell_id]
    """
    np.random.seed(seed)
    
    h, w = shape
    transcripts = []
    transcript_id = 0
    
    # Gene list (common genes)
    genes = [
        'ACTB', 'GAPDH', 'EGFR', 'PTEN', 'TP53', 
        'BRCA1', 'MYC', 'AKT1', 'KRAS', 'BRAF',
        'ERBB2', 'ALK', 'ROS1', 'PDGFRA', 'FGFR1'
    ]
    
    # Generate transcripts for each cell
    for cell in cell_info:
        n_tx = int(np.random.normal(transcripts_per_cell, 15))
        n_tx = max(10, min(n_tx, 150))  # Clamp
        
        cy, cx = cell["center_y"], cell["center_x"]
        ry, rx = cell["radius_y"], cell["radius_x"]
        
        for _ in range(n_tx):
            # Random position within ellipse (biased toward center)
            r = np.sqrt(np.random.rand())  # sqrt for uniform distribution
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Elliptical coordinates
            dy = r * ry * np.sin(theta)
            dx = r * rx * np.cos(theta)
            
            y = int(cy + dy)
            x = int(cx + dx)
            
            if 0 <= y < h and 0 <= x < w:
                gene = np.random.choice(genes)
                transcripts.append({
                    "transcript_id": f"tx_{transcript_id:08d}",
                    "x": x,
                    "y": y,
                    "gene": gene,
                    "count": 1,
                    "cell_id": cell["cell_id"]
                })
                transcript_id += 1
    
    # Add background transcripts (outside cells)
    for _ in range(n_background):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        gene = np.random.choice(genes)
        transcripts.append({
            "transcript_id": f"tx_{transcript_id:08d}",
            "x": x,
            "y": y,
            "gene": gene,
            "count": 1,
            "cell_id": None  # Unassigned
        })
        transcript_id += 1
    
    return pd.DataFrame(transcripts)


def save_mock_tiff(filepath: str, image: np.ndarray):
    """Save image as TIFF."""
    try:
        import tifffile
        tifffile.imwrite(filepath, (image * 255).astype(np.uint8))
    except ImportError:
        # Fallback to PIL
        from PIL import Image
        img = Image.fromarray((image * 255).astype(np.uint8))
        img.save(filepath)


def main():
    parser = argparse.ArgumentParser(description="Generate mock Stereo-seq data")
    parser.add_argument("--output-dir", default="tests/data", help="Output directory")
    parser.add_argument("--image-size", type=int, default=1024, help="Image size (NxN)")
    parser.add_argument("--n-cells", type=int, default=20, help="Number of cells")
    parser.add_argument("--tx-per-cell", type=int, default=50, help="Transcripts per cell")
    parser.add_argument("--n-background", type=int, default=200, help="Background transcripts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating mock data...")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Number of cells: {args.n_cells}")
    print(f"  Transcripts per cell: {args.tx_per_cell}")
    print(f"  Background transcripts: {args.n_background}")
    print()
    
    # Generate image
    image, cell_info = generate_mock_ssdna_image(
        shape=(args.image_size, args.image_size),
        n_cells=args.n_cells,
        seed=args.seed
    )
    
    # Save image
    image_path = output_dir / "mock_ssdna.tiff"
    save_mock_tiff(str(image_path), image)
    print(f"Saved: {image_path}")
    
    # Save cell ground truth
    import json
    gt_path = output_dir / "cell_ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(cell_info, f, indent=2)
    print(f"Saved: {gt_path}")
    
    # Generate and save transcripts
    transcripts = generate_mock_transcripts(
        cell_info=cell_info,
        shape=(args.image_size, args.image_size),
        transcripts_per_cell=args.tx_per_cell,
        n_background=args.n_background,
        seed=args.seed
    )
    
    tx_path = output_dir / "mock_transcripts.csv"
    transcripts.to_csv(tx_path, index=False)
    print(f"Saved: {tx_path}")
    
    print()
    print(f"Summary:")
    print(f"  Total cells: {len(cell_info)}")
    print(f"  Total transcripts: {len(transcripts)}")
    print(f"  Assigned transcripts: {transcripts['cell_id'].notna().sum()}")
    print(f"  Unassigned transcripts: {transcripts['cell_id'].isna().sum()}")
    
    return cell_info, transcripts


if __name__ == "__main__":
    main()
