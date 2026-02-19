"""Visualization utilities for flood mapping.

This module provides:
- Classification colormaps and color definitions
- RGB composite creation for Sentinel-2 imagery
- Legend generation utilities
"""

from __future__ import annotations

import numpy as np

from maji.constants import (
    CLASS_CLOUD,
    CLASS_INVALID,
    CLASS_LAND,
    CLASS_NAMES,
    CLASS_WATER,
)

# ---------------------------------------------------------------------------
# Classification Colors
# ---------------------------------------------------------------------------

# RGBA colors for each class (values 0-1)
CLASSIFICATION_COLORS: dict[int, tuple[float, float, float, float]] = {
    CLASS_INVALID: (0.0, 0.0, 0.0, 1.0),       # Black
    CLASS_LAND: (0.76, 0.70, 0.50, 1.0),       # Tan/brown
    CLASS_WATER: (0.0, 0.3, 0.8, 1.0),         # Blue
    CLASS_CLOUD: (0.9, 0.9, 0.9, 1.0),         # Light gray
}

# List of colors in class order for matplotlib ListedColormap
_CMAP_COLORS = [
    CLASSIFICATION_COLORS[CLASS_INVALID],
    CLASSIFICATION_COLORS[CLASS_LAND],
    CLASSIFICATION_COLORS[CLASS_WATER],
    CLASSIFICATION_COLORS[CLASS_CLOUD],
]


def get_classification_cmap():
    """Create a matplotlib ListedColormap for classification visualization.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap with 4 colors for Invalid, Land, Water, Cloud classes.

    Examples
    --------
    >>> cmap = get_classification_cmap()
    >>> plt.imshow(prediction, cmap=cmap, vmin=0, vmax=3)
    """
    from matplotlib.colors import ListedColormap
    return ListedColormap(_CMAP_COLORS)


# Create colormap on import for convenience (lazy import matplotlib)
try:
    from matplotlib.colors import ListedColormap
    CLASSIFICATION_CMAP = ListedColormap(_CMAP_COLORS)
except ImportError:
    CLASSIFICATION_CMAP = None  # type: ignore[assignment]


def get_class_legend() -> dict[int, str]:
    """Get dictionary mapping class values to names.

    Returns
    -------
    dict[int, str]
        Mapping from class integer to human-readable name.

    Examples
    --------
    >>> legend = get_class_legend()
    >>> legend[2]
    'Water'
    """
    return dict(CLASS_NAMES)


# ---------------------------------------------------------------------------
# RGB Composite Creation
# ---------------------------------------------------------------------------

def create_rgb_composite(
    tile: np.ndarray,
    bands: tuple[int, int, int] = (4, 3, 1),
    percentile_stretch: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """Create a false color composite for visualization.

    Default band combination (4, 3, 1) creates SWIR-NIR-Green composite
    which highlights water (dark) vs vegetation (bright green).

    Parameters
    ----------
    tile : numpy.ndarray
        Input tile with shape (C, H, W) where channels follow the model
        band order: [B02, B03, B04, B08, B11, B12].
    bands : tuple[int, int, int], optional
        Channel indices for (Red, Green, Blue) in output. Default (4, 3, 1)
        maps to B11 (SWIR), B08 (NIR), B03 (Green).
    percentile_stretch : tuple[float, float], optional
        Percentiles for contrast stretching (default: (2, 98)).

    Returns
    -------
    numpy.ndarray
        RGB composite with shape (H, W, 3), values in [0, 1], dtype float32.

    Notes
    -----
    Common band combinations for DEFAULT_MODEL_BANDS [B02, B03, B04, B08, B11, B12]:
    - (4, 3, 1): SWIR-NIR-Green - water dark, vegetation bright green
    - (2, 3, 1): Red-NIR-Green (False color infrared)
    - (2, 1, 0): True color (Red-Green-Blue)

    Examples
    --------
    >>> tile = np.random.rand(6, 256, 256).astype(np.float32) * 10000
    >>> rgb = create_rgb_composite(tile, bands=(4, 3, 1))
    >>> rgb.shape
    (256, 256, 3)
    """
    # Stack selected bands into (H, W, 3)
    rgb = np.stack([tile[b] for b in bands], axis=-1)

    # Per-channel percentile stretch
    result = np.zeros_like(rgb, dtype=np.float32)
    p_low, p_high = percentile_stretch

    for i in range(3):
        lo = np.percentile(rgb[:, :, i], p_low)
        hi = np.percentile(rgb[:, :, i], p_high)
        result[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo + 1e-8), 0, 1)

    return result


def create_classification_overlay(
    rgb: np.ndarray,
    classification: np.ndarray,
    alpha: float = 0.5,
    hide_land: bool = True,
) -> np.ndarray:
    """Overlay classification colors on an RGB image.

    Parameters
    ----------
    rgb : numpy.ndarray
        Background RGB image, shape (H, W, 3), values in [0, 1].
    classification : numpy.ndarray
        Classification map, shape (H, W), values in {0, 1, 2, 3}.
    alpha : float, optional
        Opacity of classification overlay (default: 0.5).
    hide_land : bool, optional
        If True, land pixels show only RGB (no overlay). Default True.

    Returns
    -------
    numpy.ndarray
        Blended image, shape (H, W, 3), values in [0, 1].

    Examples
    --------
    >>> overlay = create_classification_overlay(rgb, prediction, alpha=0.6)
    >>> plt.imshow(overlay)
    """
    # Create color overlay
    overlay = np.zeros_like(rgb)
    for class_val, color in CLASSIFICATION_COLORS.items():
        if hide_land and class_val == CLASS_LAND:
            continue
        mask = classification == class_val
        overlay[mask] = color[:3]  # RGB only, no alpha

    # Blend where overlay is non-zero
    mask = np.any(overlay > 0, axis=-1)
    result = rgb.copy()
    result[mask] = (1 - alpha) * rgb[mask] + alpha * overlay[mask]

    return result
