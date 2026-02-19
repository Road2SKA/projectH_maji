"""Tests for maji.constants module."""

from __future__ import annotations

import numpy as np
import pytest

from maji.constants import (
    ALL_S2_BANDS,
    BAND_RESOLUTION,
    CLASS_CLOUD,
    CLASS_INVALID,
    CLASS_LAND,
    CLASS_NAMES,
    CLASS_WATER,
    DEFAULT_CLOUD_THRESHOLD,
    DEFAULT_DOWNLOAD_BANDS,
    DEFAULT_MODEL_BANDS,
    DEFAULT_WATER_THRESHOLD,
    NORMALIZATION,
    SCL_BAND,
    TILE_HEIGHT_10M,
    TILE_WIDTH_10M,
    get_normalization_arrays,
)


class TestBandDefinitions:
    """Tests for band constant definitions."""

    def test_all_s2_bands_count(self):
        """Sentinel-2 has 13 spectral bands."""
        assert len(ALL_S2_BANDS) == 13

    def test_all_s2_bands_unique(self):
        """All band names should be unique."""
        assert len(ALL_S2_BANDS) == len(set(ALL_S2_BANDS))

    def test_default_model_bands_subset_of_all(self):
        """Model bands should be a subset of all S2 bands."""
        for band in DEFAULT_MODEL_BANDS:
            assert band in ALL_S2_BANDS

    def test_default_download_bands_includes_scl(self):
        """Download bands should include SCL for cloud masking."""
        assert SCL_BAND in DEFAULT_DOWNLOAD_BANDS

    def test_default_download_bands_includes_model_bands(self):
        """Download bands should include all model bands."""
        for band in DEFAULT_MODEL_BANDS:
            assert band in DEFAULT_DOWNLOAD_BANDS

    def test_band_resolution_covers_all_bands(self):
        """Band resolution dict should cover all S2 bands plus SCL."""
        for band in ALL_S2_BANDS:
            assert band in BAND_RESOLUTION
        assert SCL_BAND in BAND_RESOLUTION

    def test_band_resolutions_valid_values(self):
        """Band resolutions should be 10, 20, or 60 meters."""
        valid_resolutions = {10, 20, 60}
        for band, res in BAND_RESOLUTION.items():
            assert res in valid_resolutions, f"Invalid resolution for {band}: {res}"


class TestNormalization:
    """Tests for normalization constants."""

    def test_normalization_covers_all_bands(self):
        """Normalization dict should have entries for all 13 S2 bands."""
        for band in ALL_S2_BANDS:
            assert band in NORMALIZATION, f"Missing normalization for {band}"

    def test_normalization_values_positive(self):
        """Mean and std should be positive."""
        for band, (mean, std) in NORMALIZATION.items():
            assert mean > 0, f"Mean for {band} should be positive"
            assert std > 0, f"Std for {band} should be positive"

    def test_normalization_values_reasonable(self):
        """Values should be in a reasonable range for Sentinel-2 L2A data."""
        for band, (mean, std) in NORMALIZATION.items():
            # Sentinel-2 L2A values typically 0-10000 (sometimes higher)
            assert 0 < mean < 10000, f"Mean for {band} out of range"
            assert 0 < std < 5000, f"Std for {band} out of range"

    def test_get_normalization_arrays_default(self):
        """get_normalization_arrays should return arrays for default bands."""
        means, stds = get_normalization_arrays()
        assert len(means) == len(DEFAULT_MODEL_BANDS)
        assert len(stds) == len(DEFAULT_MODEL_BANDS)
        assert means.dtype == np.float32
        assert stds.dtype == np.float32

    def test_get_normalization_arrays_custom_bands(self):
        """get_normalization_arrays should work with custom band list."""
        bands = ["B02", "B03", "B04"]
        means, stds = get_normalization_arrays(bands)
        assert len(means) == 3
        assert len(stds) == 3
        # Check values match NORMALIZATION
        assert means[0] == pytest.approx(NORMALIZATION["B02"][0])
        assert stds[0] == pytest.approx(NORMALIZATION["B02"][1])

    def test_get_normalization_arrays_invalid_band(self):
        """get_normalization_arrays should raise KeyError for unknown band."""
        with pytest.raises(KeyError):
            get_normalization_arrays(["INVALID_BAND"])


class TestClassification:
    """Tests for classification constants."""

    def test_class_values_unique(self):
        """Class values should be unique integers."""
        values = [CLASS_INVALID, CLASS_LAND, CLASS_WATER, CLASS_CLOUD]
        assert len(values) == len(set(values))

    def test_class_values_sequential(self):
        """Class values should be 0-3."""
        assert CLASS_INVALID == 0
        assert CLASS_LAND == 1
        assert CLASS_WATER == 2
        assert CLASS_CLOUD == 3

    def test_class_names_complete(self):
        """CLASS_NAMES should have entry for each class."""
        expected = {CLASS_INVALID, CLASS_LAND, CLASS_WATER, CLASS_CLOUD}
        assert set(CLASS_NAMES.keys()) == expected

    def test_class_names_values(self):
        """CLASS_NAMES should map to expected strings."""
        assert CLASS_NAMES[CLASS_INVALID] == "Invalid"
        assert CLASS_NAMES[CLASS_LAND] == "Land"
        assert CLASS_NAMES[CLASS_WATER] == "Water"
        assert CLASS_NAMES[CLASS_CLOUD] == "Cloud"

    def test_default_thresholds_valid_range(self):
        """Default thresholds should be in [0, 1]."""
        assert 0 <= DEFAULT_WATER_THRESHOLD <= 1
        assert 0 <= DEFAULT_CLOUD_THRESHOLD <= 1


class TestTileDimensions:
    """Tests for tile dimension constants."""

    def test_tile_dimensions_standard(self):
        """Tile dimensions should be standard MGRS at 10m."""
        # MGRS tiles are 109.8 km at 10m resolution
        assert TILE_HEIGHT_10M == 10980
        assert TILE_WIDTH_10M == 10980
