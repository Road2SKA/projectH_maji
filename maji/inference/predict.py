"""Prediction utilities for flood mapping inference.

This module provides functions for:
- Tile normalization using ML4Floods statistics
- Padding for UNet compatibility
- Single-tile and tiled inference
- Discrete classification from probability maps
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator

import numpy as np
import torch
import torch.nn.functional as F

from maji.constants import (
    CLASS_CLOUD,
    CLASS_INVALID,
    CLASS_LAND,
    CLASS_WATER,
    DEFAULT_CLOUD_THRESHOLD,
    DEFAULT_MODEL_BANDS,
    DEFAULT_WATER_THRESHOLD,
    get_normalization_arrays,
)

if TYPE_CHECKING:
    import torch.nn as nn


def normalize_tile(
    tile: np.ndarray,
    bands: list[str] | None = None,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> np.ndarray:
    """Apply per-band normalization to a tile.

    Normalizes using (value - mean) / std for each band.

    Parameters
    ----------
    tile : numpy.ndarray
        Input tile with shape (channels, height, width).
    bands : list[str] or None, optional
        Band names for auto-loading normalization constants from
        NORMALIZATION. If None and means/stds not provided, uses
        DEFAULT_MODEL_BANDS.
    means : numpy.ndarray or None, optional
        Array of mean values, shape (channels,). If provided along with
        stds, these override the bands parameter.
    stds : numpy.ndarray or None, optional
        Array of std values, shape (channels,). Must be provided if
        means is provided.

    Returns
    -------
    numpy.ndarray
        Normalized tile with same shape as input, dtype float32.

    Raises
    ------
    ValueError
        If means is provided but stds is not (or vice versa).
    KeyError
        If a band name is not found in NORMALIZATION.

    Examples
    --------
    >>> tile = np.random.rand(6, 256, 256).astype(np.float32) * 10000
    >>> normalized = normalize_tile(tile, bands=["B02", "B03", "B04", "B08", "B11", "B12"])
    >>> normalized.shape
    (6, 256, 256)
    """
    if (means is None) != (stds is None):
        raise ValueError("Both means and stds must be provided, or neither")

    if means is None:
        if bands is None:
            bands = DEFAULT_MODEL_BANDS
        means, stds = get_normalization_arrays(bands)

    # Reshape for broadcasting: (C,) -> (C, 1, 1)
    means_reshaped = means[:, np.newaxis, np.newaxis]
    stds_reshaped = stds[:, np.newaxis, np.newaxis]

    return ((tile - means_reshaped) / stds_reshaped).astype(np.float32)


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 16,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad tensor height and width to be divisible by multiple.

    UNet requires input dimensions to be divisible by 2^num_levels (16 for
    4-level UNet) to avoid shape mismatches in skip connections.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (..., height, width).
    multiple : int, optional
        Value that height and width should be divisible by (default: 16).

    Returns
    -------
    padded : torch.Tensor
        Padded tensor with height and width divisible by multiple.
    original_shape : tuple[int, int]
        Original (height, width) for cropping output back.

    Examples
    --------
    >>> x = torch.randn(1, 6, 100, 100)
    >>> padded, orig = pad_to_multiple(x, 16)
    >>> padded.shape
    torch.Size([1, 6, 112, 112])
    >>> orig
    (100, 100)
    """
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # F.pad takes (left, right, top, bottom) for last two dims
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    return x, (h, w)


@torch.no_grad()
def run_inference(
    model: "nn.Module",
    tile: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single normalized tile.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded UNet model in eval mode.
    tile : numpy.ndarray
        Normalized tile with shape (channels, height, width).
    device : str or torch.device, optional
        Device for inference (default: "cpu").

    Returns
    -------
    water_prob : numpy.ndarray
        Water probability map, shape (height, width), values in [0, 1].
    cloud_prob : numpy.ndarray
        Cloud probability map, shape (height, width), values in [0, 1].

    Notes
    -----
    The model outputs 2 channels: [cloud_prob, water_prob]. This function
    returns them as (water_prob, cloud_prob) for intuitive usage.
    """
    # Add batch dimension and move to device
    x = torch.from_numpy(tile).unsqueeze(0).to(device)

    # Pad for UNet compatibility
    x_padded, (orig_h, orig_w) = pad_to_multiple(x, multiple=16)

    # Forward pass
    logits = model(x_padded)
    probs = torch.sigmoid(logits)

    # Remove batch dim and crop to original size
    probs = probs[0, :, :orig_h, :orig_w].cpu().numpy()

    # Model outputs [cloud, water] - return as (water, cloud)
    return probs[1], probs[0]


def run_inference_tiled(
    model: "nn.Module",
    image: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 64,
    device: str | torch.device = "cpu",
    bands: list[str] | None = None,
) -> Generator[dict, None, None]:
    """Run inference on a large image using sliding window tiles.

    Processes image in overlapping tiles and yields results for each tile.
    The caller is responsible for stitching tiles together.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded UNet model in eval mode.
    image : numpy.ndarray
        Full image with shape (channels, height, width). Should be raw
        (unnormalized) values.
    tile_size : int, optional
        Size of square tiles to process (default: 1024).
    overlap : int, optional
        Overlap between adjacent tiles (default: 64).
    device : str or torch.device, optional
        Device for inference (default: "cpu").
    bands : list[str] or None, optional
        Band names for normalization. If None, uses DEFAULT_MODEL_BANDS.

    Yields
    ------
    dict
        Dictionary with keys:
        - "row": starting row index
        - "col": starting column index
        - "water_prob": water probability array for tile
        - "cloud_prob": cloud probability array for tile
        - "tile_size": actual tile size (may be smaller at edges)

    Examples
    --------
    >>> for result in run_inference_tiled(model, image, tile_size=512):
    ...     print(f"Tile at ({result['row']}, {result['col']})")
    """
    if bands is None:
        bands = DEFAULT_MODEL_BANDS

    _, height, width = image.shape
    stride = tile_size - overlap

    for row in range(0, height, stride):
        for col in range(0, width, stride):
            # Extract tile (handle edge cases)
            row_end = min(row + tile_size, height)
            col_end = min(col + tile_size, width)
            tile = image[:, row:row_end, col:col_end]

            # Normalize and run inference
            tile_norm = normalize_tile(tile, bands=bands)
            water_prob, cloud_prob = run_inference(model, tile_norm, device)

            yield {
                "row": row,
                "col": col,
                "water_prob": water_prob,
                "cloud_prob": cloud_prob,
                "tile_size": (row_end - row, col_end - col),
            }


def classify_prediction(
    water_prob: np.ndarray,
    cloud_prob: np.ndarray,
    invalid_mask: np.ndarray | None = None,
    th_water: float = DEFAULT_WATER_THRESHOLD,
    th_cloud: float = DEFAULT_CLOUD_THRESHOLD,
) -> np.ndarray:
    """Convert probability maps to discrete class labels.

    Classification logic:
    1. Start with all pixels as LAND
    2. Pixels with water_prob > th_water become WATER
    3. Pixels with cloud_prob > th_cloud become CLOUD (overrides water)
    4. Pixels in invalid_mask become INVALID

    Parameters
    ----------
    water_prob : numpy.ndarray
        Water probability map, shape (height, width), values in [0, 1].
    cloud_prob : numpy.ndarray
        Cloud probability map, shape (height, width), values in [0, 1].
    invalid_mask : numpy.ndarray or None, optional
        Boolean mask where True indicates invalid pixels (e.g., nodata).
    th_water : float, optional
        Threshold for water classification (default: 0.5).
    th_cloud : float, optional
        Threshold for cloud classification (default: 0.5).

    Returns
    -------
    numpy.ndarray
        Classification map with shape (height, width), dtype uint8.
        Values: 0=invalid, 1=land, 2=water, 3=cloud.

    Examples
    --------
    >>> water = np.array([[0.1, 0.9], [0.3, 0.7]])
    >>> cloud = np.array([[0.1, 0.1], [0.9, 0.1]])
    >>> classify_prediction(water, cloud)
    array([[1, 2],
           [3, 2]], dtype=uint8)
    """
    # Start with land
    pred = np.full(water_prob.shape, CLASS_LAND, dtype=np.uint8)

    # Apply water threshold
    pred[water_prob > th_water] = CLASS_WATER

    # Apply cloud threshold (clouds override water)
    pred[cloud_prob > th_cloud] = CLASS_CLOUD

    # Apply invalid mask last
    if invalid_mask is not None:
        pred[invalid_mask] = CLASS_INVALID

    return pred
