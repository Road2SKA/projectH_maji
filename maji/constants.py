"""Shared constants for Sentinel-2 band configurations and normalization.

This module provides centralized definitions for:
- Sentinel-2 band names and resolutions
- Normalization statistics from ML4Floods
- Classification thresholds and class labels
- Tile dimensions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Sentinel-2 Band Definitions
# ---------------------------------------------------------------------------

# All Sentinel-2 L2A bands (13 bands total)
ALL_S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# Native resolution (meters) for each band
BAND_RESOLUTION: dict[str, int] = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
    "SCL": 20,
}

# Scene Classification Layer band name
SCL_BAND = "SCL"

# Default bands for current WorldFloods model (bgriswirs configuration)
# Maps to ml4floods channel indices [1, 2, 3, 7, 11, 12] (0-based)
DEFAULT_MODEL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Default bands to download (model bands + cloud mask)
DEFAULT_DOWNLOAD_BANDS = DEFAULT_MODEL_BANDS + [SCL_BAND]

# Legacy aliases for backward compatibility
MODEL_BANDS = DEFAULT_MODEL_BANDS
DOWNLOAD_BANDS = DEFAULT_DOWNLOAD_BANDS

# ---------------------------------------------------------------------------
# Normalization Constants (from ML4Floods SENTINEL2_NORMALIZATION)
# ---------------------------------------------------------------------------

# Format: band_name -> (mean, std) for all 13 Sentinel-2 bands
# Values computed from WorldFloods training data
NORMALIZATION: dict[str, tuple[float, float]] = {
    "B01": (3787.06, 2634.44),
    "B02": (3758.07, 2794.10),
    "B03": (3238.08, 2549.49),
    "B04": (3418.90, 2811.78),
    "B05": (3450.23, 2776.93),
    "B06": (4030.95, 2632.14),
    "B07": (4164.17, 2657.43),
    "B08": (3981.96, 2500.48),
    "B8A": (4226.75, 2589.29),
    "B09": (1868.30, 1820.90),
    "B10": (399.39, 761.36),
    "B11": (2391.66, 1500.03),
    "B12": (1790.32, 1241.98),
}


def get_normalization_arrays(
    bands: list[str] | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Get normalization arrays for specified bands.

    Parameters
    ----------
    bands : list[str] or None, optional
        Band names to get normalization for. If None, uses
        DEFAULT_MODEL_BANDS.

    Returns
    -------
    means : numpy.ndarray
        Array of mean values, shape (len(bands),).
    stds : numpy.ndarray
        Array of std values, shape (len(bands),).

    Raises
    ------
    KeyError
        If a requested band is not in NORMALIZATION.

    Examples
    --------
    >>> means, stds = get_normalization_arrays(["B02", "B03", "B04"])
    >>> means.shape
    (3,)
    """
    import numpy as np

    if bands is None:
        bands = DEFAULT_MODEL_BANDS

    means = np.array([NORMALIZATION[b][0] for b in bands], dtype=np.float32)
    stds = np.array([NORMALIZATION[b][1] for b in bands], dtype=np.float32)
    return means, stds


# ---------------------------------------------------------------------------
# Classification Constants
# ---------------------------------------------------------------------------

# Discrete class values for prediction output
CLASS_INVALID = 0
CLASS_LAND = 1
CLASS_WATER = 2
CLASS_CLOUD = 3

# Human-readable class names
CLASS_NAMES: dict[int, str] = {
    CLASS_INVALID: "Invalid",
    CLASS_LAND: "Land",
    CLASS_WATER: "Water",
    CLASS_CLOUD: "Cloud",
}

# Default probability thresholds for classification
DEFAULT_WATER_THRESHOLD = 0.5
DEFAULT_CLOUD_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Tile Dimensions
# ---------------------------------------------------------------------------

# Full MGRS tile dimensions at 10m resolution
TILE_HEIGHT_10M = 10980
TILE_WIDTH_10M = 10980
