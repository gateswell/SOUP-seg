"""
U-Net Boundary Detection Model for SoupSeg v1.1.0.

PyTorch implementation of a lightweight U-Net for detecting cell boundaries
from ssDNA images. The model outputs a probability map where high values
indicate likely boundary locations.

Architecture:
- Encoder: 4 downsampling blocks (Conv + BN + ReLU + MaxPool)
- Bottleneck: 2 conv layers at lowest resolution
- Decoder: 4 upsampling blocks (ConvTranspose + skip connections + Conv + BN + ReLU)
- Output: 1x1 Conv + Sigmoid for boundary probability

The model is designed to be lightweight enough for inference on CPU while
still capturing multi-scale boundary features.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. U-Net boundary detection requires: pip install torch torchvision"
    )


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Double convolution block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x)

    class EncoderBlock(nn.Module):
        """Encoder block: ConvBlock + MaxPool2d."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = ConvBlock(in_channels, out_channels)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            """Returns (pooled_output, skip_connection)."""
            skip = self.conv(x)
            return self.pool(skip), skip

    class DecoderBlock(nn.Module):
        """Decoder block: ConvTranspose2d + skip concat + ConvBlock."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            self.conv = ConvBlock(out_channels * 2, out_channels)

        def forward(self, x: Tensor, skip: Tensor) -> Tensor:
            x = self.up(x)
            # Handle size mismatch due to rounding
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                           diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class UNetBoundaryDetector(nn.Module):
        """
        Lightweight U-Net for cell boundary detection.

        Input: 1-channel ssDNA image (float32, normalized 0-1)
        Output: 1-channel boundary probability map (float32, 0-1)

        The model can be used with pretrained weights or trained from scratch
        on labeled boundary masks.
        """

        def __init__(
            self,
            in_channels: int = 1,
            base_features: int = 32,
            depth: int = 4,
        ):
            """
            Args:
                in_channels: Number of input channels (1 for grayscale ssDNA).
                base_features: Number of feature maps in the first encoder layer.
                               Doubled at each depth level (32 -> 64 -> 128 -> 256).
                depth: Number of encoder/decoder levels (default 4).
            """
            super().__init__()
            self.in_channels = in_channels
            self.base_features = base_features
            self.depth = depth

            # Build encoder
            self.encoders = nn.ModuleList()
            ch_in = in_channels
            for i in range(depth):
                ch_out = base_features * (2 ** i)
                self.encoders.append(EncoderBlock(ch_in, ch_out))
                ch_in = ch_out

            # Bottleneck
            bottleneck_ch = base_features * (2 ** depth)
            self.bottleneck = ConvBlock(ch_in, bottleneck_ch)

            # Build decoder
            self.decoders = nn.ModuleList()
            ch_in = bottleneck_ch
            for i in range(depth - 1, -1, -1):
                ch_out = base_features * (2 ** i)
                self.decoders.append(DecoderBlock(ch_in, ch_out))
                ch_in = ch_out

            # Output head
            self.output_conv = nn.Conv2d(base_features, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize convolutional weights with Kaiming normal."""
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (B, C, H, W)

            Returns:
                Boundary probability map of shape (B, 1, H, W)
            """
            # Encoder path with skip connections
            skips = []
            for encoder in self.encoders:
                x, skip = encoder(x)
                skips.append(skip)

            # Bottleneck
            x = self.bottleneck(x)

            # Decoder path
            for decoder, skip in zip(self.decoders, reversed(skips)):
                x = decoder(x, skip)

            return self.sigmoid(self.output_conv(x))

        def predict_boundary(
            self,
            image: np.ndarray,
            device: Optional[str] = None,
            tile_size: int = 512,
            overlap: int = 64,
        ) -> np.ndarray:
            """
            Run inference on a large image with tiling.

            Args:
                image: 2D numpy array (H, W), float32 in [0, 1].
                device: Torch device string (e.g. 'cpu', 'cuda:0').
                        Auto-detected if None.
                tile_size: Tile size for large images.
                overlap: Overlap between tiles to reduce boundary artifacts.

            Returns:
                Boundary probability map (H, W), float32 in [0, 1].
            """
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.to(device)
            self.eval()

            h, w = image.shape
            if image.ndim == 2:
                image = image[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
            elif image.ndim == 3:
                image = image[np.newaxis, :, :, :]  # (1, C, H, W)

            # Small image: process directly
            if h <= tile_size and w <= tile_size:
                with torch.no_grad():
                    tensor = torch.from_numpy(image).float().to(device)
                    output = self(tensor)
                    return output.squeeze().cpu().numpy()

            # Large image: tile-based inference
            output = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32)

            # Create smooth blending weight (Hann window)
            blend_h = np.hanning(tile_size)
            blend_w = np.hanning(tile_size)
            blend_2d = np.outer(blend_h, blend_w)

            for y in range(0, h, tile_size - overlap):
                for x in range(0, w, tile_size - overlap):
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    y_start = y_end - min(tile_size, y_end - y)
                    x_start = x_end - min(tile_size, x_end - x)

                    tile = image[:, :, y_start:y_end, x_start:x_end]

                    # Pad if needed
                    th, tw = tile.shape[2], tile.shape[3]
                    if th < tile_size or tw < tile_size:
                        padded = np.zeros((1, 1, tile_size, tile_size), dtype=np.float32)
                        padded[:, :, :th, :tw] = tile
                        tile = padded

                    with torch.no_grad():
                        tensor = torch.from_numpy(tile).float().to(device)
                        pred = self(tensor).squeeze().cpu().numpy()

                    # Blend
                    bh, bw = min(tile_size, y_end - y_start), min(tile_size, x_end - x_start)
                    w_slice = blend_2d[:bh, :bw]
                    output[y_start:y_end, x_start:x_end] += pred[:bh, :bw] * w_slice
                    weight[y_start:y_end, x_start:x_start] += w_slice

            weight = np.maximum(weight, 1e-8)
            return output / weight

    class BoundaryLoss(nn.Module):
        """
        Combined loss for boundary detection training.

        Combines:
        - BCE loss for pixel-wise classification
        - Dice loss for handling class imbalance (boundaries are sparse)
        - Optional edge-weighted BCE (higher weight on boundary pixels)
        """

        def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                     edge_weight: float = 1.5):
            super().__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.edge_weight = edge_weight

        def dice_loss(self, pred: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return 1.0 - dice.mean()

        def forward(self, pred: Tensor, target: Tensor) -> Tensor:
            # Edge-weighted BCE
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            weight_map = torch.ones_like(target)
            weight_map[target > 0.5] = self.edge_weight
            bce = (bce * weight_map).mean()

            # Dice
            dice = self.dice_loss(pred, target)

            return self.bce_weight * bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Factory / convenience functions
# ---------------------------------------------------------------------------

@dataclass
class UNetConfig:
    """Configuration for U-Net boundary detector."""
    in_channels: int = 1
    base_features: int = 32
    depth: int = 4
    device: str = "auto"
    tile_size: int = 512
    tile_overlap: int = 64

    def get_device(self) -> str:
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device


def create_unet_model(config: Optional[UNetConfig] = None) -> "UNetBoundaryDetector":
    """
    Create a U-Net boundary detection model.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        UNetBoundaryDetector instance

    Raises:
        ImportError: If PyTorch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for U-Net boundary detection. "
            "Install with: pip install torch torchvision"
        )

    if config is None:
        config = UNetConfig()

    model = UNetBoundaryDetector(
        in_channels=config.in_channels,
        base_features=config.base_features,
        depth=config.depth,
    )

    device = config.get_device()
    model.to(device)

    return model


def detect_boundaries_unet(
    image: np.ndarray,
    model: Optional["UNetBoundaryDetector"] = None,
    config: Optional[UNetConfig] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Detect cell boundaries using U-Net.

    Args:
        image: 2D numpy array (H, W), float32 normalized to [0, 1].
        model: Pre-trained UNetBoundaryDetector. Created from config if None.
        config: Model configuration. Uses defaults if both model and config are None.
        threshold: Probability threshold for boundary classification.

    Returns:
        Binary boundary mask (H, W), uint8, values 0 or 255.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for U-Net boundary detection.")

    if config is None:
        config = UNetConfig()

    if model is None:
        model = create_unet_model(config)

    device = config.get_device()

    # Ensure image is float32 [0, 1]
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    # Run inference
    prob_map = model.predict_boundary(
        image,
        device=device,
        tile_size=config.tile_size,
        overlap=config.tile_overlap,
    )

    # Threshold
    boundary_mask = (prob_map > threshold).astype(np.uint8) * 255

    return boundary_mask


def train_unet(
    train_images: np.ndarray,
    train_boundaries: np.ndarray,
    val_images: Optional[np.ndarray] = None,
    val_boundaries: Optional[np.ndarray] = None,
    config: Optional[UNetConfig] = None,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    patch_size: int = 256,
    augment: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the U-Net boundary detection model.

    Args:
        train_images: Training images, (N, H, W) float32 [0, 1].
        train_boundaries: Training boundary masks, (N, H, W) float32 [0, 1].
        val_images: Optional validation images.
        val_boundaries: Optional validation boundaries.
        config: Model configuration.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Initial learning rate.
        patch_size: Random crop size for training patches.
        augment: Whether to apply data augmentation.
        save_path: Path to save best model weights.
        verbose: Whether to print training progress.

    Returns:
        Dictionary with training history and final metrics.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")

    if config is None:
        config = UNetConfig()

    device = config.get_device()
    model = create_unet_model(config)
    criterion = BoundaryLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    n_train = len(train_images)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        indices = np.random.permutation(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_imgs = []
            batch_bnds = []

            for idx in batch_idx:
                img = train_images[idx]
                bnd = train_boundaries[idx]

                # Random crop
                h, w = img.shape
                if h > patch_size and w > patch_size:
                    y = np.random.randint(0, h - patch_size)
                    x = np.random.randint(0, w - patch_size)
                    img = img[y:y+patch_size, x:x+patch_size]
                    bnd = bnd[y:y+patch_size, x:x+patch_size]
                elif h < patch_size or w < patch_size:
                    # Pad
                    pad_h = max(0, patch_size - h)
                    pad_w = max(0, patch_size - w)
                    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
                    bnd = np.pad(bnd, ((0, pad_h), (0, pad_w)), mode='reflect')

                # Data augmentation
                if augment:
                    if np.random.rand() > 0.5:
                        img = np.flip(img, axis=1).copy()
                        bnd = np.flip(bnd, axis=1).copy()
                    if np.random.rand() > 0.5:
                        img = np.flip(img, axis=0).copy()
                        bnd = np.flip(bnd, axis=0).copy()
                    # Random rotation 90/180/270
                    k = np.random.randint(0, 4)
                    if k > 0:
                        img = np.rot90(img, k).copy()
                        bnd = np.rot90(bnd, k).copy()

                batch_imgs.append(img)
                batch_bnds.append(bnd)

            imgs_tensor = torch.from_numpy(
                np.stack(batch_imgs)[:, np.newaxis, :, :]
            ).float().to(device)
            bnds_tensor = torch.from_numpy(
                np.stack(batch_bnds)[:, np.newaxis, :, :]
            ).float().to(device)

            optimizer.zero_grad()
            pred = model(imgs_tensor)
            loss = criterion(pred, bnds_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss_val = 0.0
        if val_images is not None and val_boundaries is not None:
            model.eval()
            with torch.no_grad():
                for j in range(0, len(val_images), batch_size):
                    v_imgs = torch.from_numpy(
                        val_images[j:j+batch_size][:, np.newaxis, :, :]
                    ).float().to(device)
                    v_bnds = torch.from_numpy(
                        val_boundaries[j:j+batch_size][:, np.newaxis, :, :]
                    ).float().to(device)
                    v_pred = model(v_imgs)
                    v_loss = criterion(v_pred, v_bnds)
                    val_loss_val += v_loss.item()

            val_loss_val /= max(len(val_images) // batch_size, 1)
            history["val_loss"].append(val_loss_val)
            scheduler.step(val_loss_val)

            # Save best model
            if save_path and val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                torch.save(model.state_dict(), save_path)
        else:
            scheduler.step(avg_train_loss)
            if save_path and avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                torch.save(model.state_dict(), save_path)

        if verbose and (epoch + 1) % 5 == 0:
            msg = f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}"
            if val_images is not None:
                msg += f" - val_loss: {val_loss_val:.4f}"
            print(msg)

    return {
        "history": history,
        "best_val_loss": best_val_loss if val_images is not None else best_val_loss,
        "model": model,
    }
