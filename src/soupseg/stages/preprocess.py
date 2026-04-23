"""
Stage 1: Image Preprocessing for ssDNA images.

This module handles loading and preprocessing of ssDNA images
for nuclei detection in Stereo-seq data.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import cv2
except ImportError:
    warnings.warn("OpenCV (cv2) not found. Using scipy for basic operations.")
    cv2 = None

try:
    from skimage import io, filters, restoration
    from skimage.exposure import equalize_adapthist
    from skimage.morphology import white_tophat, disk
except ImportError:
    warnings.warn("scikit-image not found. Please install: pip install scikit-image")


# Default parameters for Stereo-seq v1.3 (0.5 um/pixel)
DEFAULT_PIXEL_SIZE_UM = 0.5


def load_image(
    filepath: str,
    as_gray: bool = True
) -> np.ndarray:
    """
    Load an image file (TIFF, PNG) as a numpy array.
    
    Args:
        filepath: Path to the image file
        as_gray: If True, convert to grayscale
        
    Returns:
        Image as 2D numpy array (float64, 0-1 range)
    """
    try:
        from skimage import io
        img = io.imread(filepath)
    except Exception:
        # Fallback to PIL
        from PIL import Image
        img = np.array(Image.open(filepath))
    
    # Convert to grayscale if needed
    if as_gray and len(img.shape) == 3:
        # Handle RGB or RGBA
        if img.shape[2] >= 3:
            img = np.mean(img[:, :, :3], axis=2)
    
    # Normalize to 0-1 range
    if img.max() > 1.0:
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)
    
    return img


def gaussian_denoise(
    image: np.ndarray,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Apply Gaussian smoothing for noise reduction.
    
    Args:
        image: Input image (2D array)
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        Denoised image
    """
    if cv2 is not None:
        # OpenCV is faster
        denoised = cv2.GaussianBlur(
            image.astype(np.float32),
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )
        return denoised.astype(np.float64)
    else:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=sigma)


def clahe_enhance(
    image: np.ndarray,
    clip_limit: float = 0.03,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    This enhances local contrast which is crucial for detecting
    nuclei boundaries in ssDNA images.
    
    Args:
        image: Input image (2D array, 0-1 range)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    if cv2 is not None:
        # Convert to 8-bit for OpenCV CLAHE
        img_8bit = (image * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit * 255,
            tileGridSize=tile_grid_size
        )
        enhanced = clahe.apply(img_8bit)
        return enhanced.astype(np.float64) / 255.0
    else:
        # Fallback to skimage
        from skimage.exposure import equalize_adapthist
        enhanced = equalize_adapthist(image, clip_limit=clip_limit)
        return enhanced


def estimate_background(
    image: np.ndarray,
    method: str = 'gaussian'
) -> np.ndarray:
    """
    Estimate and subtract background from the image.
    
    Args:
        image: Input image
        method: 'gaussian' or 'rolling_ball'
        
    Returns:
        Background-removed image
    """
    if method == 'gaussian':
        # Estimate background using large Gaussian blur
        if cv2 is not None:
            background = cv2.GaussianBlur(
                image.astype(np.float32),
                ksize=(101, 101),
                sigmaX=30
            )
        else:
            from scipy.ndimage import gaussian_filter
            background = gaussian_filter(image, sigma=30)
    elif method == 'rolling_ball':
        # Rolling ball background estimation (more accurate but slower)
        # Simplified: use morphological opening
        from skimage.morphology import opening, disk
        footprint = disk(30)
        background = opening(image, footprint)
    else:
        raise ValueError(f"Unknown background estimation method: {method}")
    
    # Subtract background and clip to valid range
    result = image - background
    result = np.clip(result, 0, 1)
    
    return result


def normalize_image(
    image: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize image to 0-1 range.
    
    Args:
        image: Input image
        method: 'minmax' (linear stretch) or 'zscore' (standardize)
        
    Returns:
        Normalized image
    """
    if method == 'minmax':
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image
    elif method == 'zscore':
        mean = image.mean()
        std = image.std()
        if std > 0:
            normalized = (image - mean) / std
            return np.clip(normalized, -3, 3)  # Typical z-score clipping
        return image - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_image(
    image: np.ndarray,
    denoise: bool = True,
    denoise_sigma: float = 1.5,
    enhance_contrast: bool = True,
    clip_limit: float = 0.03,
    remove_background: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Complete preprocessing pipeline for ssDNA images.
    
    This applies:
    1. Gaussian denoising
    2. CLAHE contrast enhancement
    3. Background estimation and subtraction
    4. Min-max normalization
    
    Args:
        image: Raw ssDNA image (2D array)
        denoise: Whether to apply Gaussian denoising
        denoise_sigma: Sigma for Gaussian denoising
        enhance_contrast: Whether to apply CLAHE
        clip_limit: CLAHE clip limit (0.01-0.05 recommended)
        remove_background: Whether to estimate and subtract background
        normalize: Whether to normalize to 0-1 range
        
    Returns:
        Preprocessed image ready for nuclei detection
        
    Example:
        >>> from soupseg.stages import load_image, preprocess_image
        >>> img = load_image('ssdna.tiff')
        >>> preprocessed = preprocess_image(img, denoise_sigma=2.0)
    """
    result = image.copy()
    
    # Step 1: Denoise
    if denoise:
        result = gaussian_denoise(result, sigma=denoise_sigma)
    
    # Step 2: Contrast enhancement
    if enhance_contrast:
        result = clahe_enhance(result, clip_limit=clip_limit)
    
    # Step 3: Background removal
    if remove_background:
        result = estimate_background(result, method='gaussian')
    
    # Step 4: Normalize
    if normalize:
        result = normalize_image(result, method='minmax')
    
    return result


def preprocess_batch(
    image_paths: list,
    output_dir: str,
    **kwargs
) -> dict:
    """
    Preprocess multiple images.
    
    Args:
        image_paths: List of paths to images
        output_dir: Directory to save preprocessed images
        **kwargs: Arguments to pass to preprocess_image
        
    Returns:
        Dict mapping input paths to output paths
    """
    import os
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for path in image_paths:
        img = load_image(path)
        preprocessed = preprocess_image(img, **kwargs)
        
        # Save preprocessed image
        import cv2
        output_path = output_dir / f"{Path(path).stem}_preprocessed.tiff"
        cv2.imwrite(str(output_path), (preprocessed * 255).astype(np.uint8))
        results[str(path)] = str(output_path)
    
    return results


# Convenience function alias
apply = preprocess_image
