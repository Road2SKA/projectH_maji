"""Tests for maji.viz module."""

from __future__ import annotations

import numpy as np
import pytest

from maji.constants import CLASS_CLOUD, CLASS_INVALID, CLASS_LAND, CLASS_WATER
from maji.viz import (
    CLASSIFICATION_COLORS,
    create_classification_overlay,
    create_rgb_composite,
    get_class_legend,
    get_classification_cmap,
)


class TestClassificationColors:
    """Tests for classification color definitions."""

    def test_colors_for_all_classes(self):
        """CLASSIFICATION_COLORS should have entry for each class."""
        expected_classes = {CLASS_INVALID, CLASS_LAND, CLASS_WATER, CLASS_CLOUD}
        assert set(CLASSIFICATION_COLORS.keys()) == expected_classes

    def test_colors_are_rgba(self):
        """Each color should be RGBA tuple with 4 values."""
        for class_val, color in CLASSIFICATION_COLORS.items():
            assert len(color) == 4, f"Color for class {class_val} should have 4 values"

    def test_colors_in_range(self):
        """Color values should be in [0, 1]."""
        for class_val, color in CLASSIFICATION_COLORS.items():
            for i, val in enumerate(color):
                assert 0 <= val <= 1, f"Color value {i} for class {class_val} out of range"


class TestGetClassificationCmap:
    """Tests for get_classification_cmap function."""

    def test_cmap_type(self):
        """Should return a matplotlib ListedColormap."""
        from matplotlib.colors import ListedColormap
        cmap = get_classification_cmap()
        assert isinstance(cmap, ListedColormap)

    def test_cmap_has_4_colors(self):
        """Colormap should have 4 colors (one per class)."""
        cmap = get_classification_cmap()
        assert cmap.N == 4


class TestGetClassLegend:
    """Tests for get_class_legend function."""

    def test_legend_returns_dict(self):
        """get_class_legend should return a dictionary."""
        legend = get_class_legend()
        assert isinstance(legend, dict)

    def test_legend_has_all_classes(self):
        """Legend should have entry for each class."""
        legend = get_class_legend()
        expected_classes = {CLASS_INVALID, CLASS_LAND, CLASS_WATER, CLASS_CLOUD}
        assert set(legend.keys()) == expected_classes

    def test_legend_values_are_strings(self):
        """Legend values should be strings."""
        legend = get_class_legend()
        for val in legend.values():
            assert isinstance(val, str)


class TestCreateRgbComposite:
    """Tests for create_rgb_composite function."""

    def test_output_shape(self):
        """Output should be (H, W, 3)."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        rgb = create_rgb_composite(tile)
        assert rgb.shape == (64, 64, 3)

    def test_output_dtype(self):
        """Output should be float32."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        rgb = create_rgb_composite(tile)
        assert rgb.dtype == np.float32

    def test_output_range(self):
        """Output values should be in [0, 1]."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        rgb = create_rgb_composite(tile)
        assert rgb.min() >= 0
        assert rgb.max() <= 1

    def test_custom_bands(self):
        """Should work with custom band indices."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        # True color: R=B04(idx 2), G=B03(idx 1), B=B02(idx 0)
        rgb = create_rgb_composite(tile, bands=(2, 1, 0))
        assert rgb.shape == (64, 64, 3)

    def test_custom_percentile_stretch(self):
        """Should work with custom percentile stretch."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        rgb = create_rgb_composite(tile, percentile_stretch=(5, 95))
        assert rgb.min() >= 0
        assert rgb.max() <= 1

    def test_consistent_output(self):
        """Same input should produce same output."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        rgb1 = create_rgb_composite(tile)
        rgb2 = create_rgb_composite(tile)
        np.testing.assert_array_equal(rgb1, rgb2)


class TestCreateClassificationOverlay:
    """Tests for create_classification_overlay function."""

    def test_output_shape(self):
        """Output should match input RGB shape."""
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        classification = np.ones((64, 64), dtype=np.uint8) * CLASS_LAND
        result = create_classification_overlay(rgb, classification)
        assert result.shape == rgb.shape

    def test_output_range(self):
        """Output values should be in [0, 1]."""
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        classification = np.random.choice([0, 1, 2, 3], size=(64, 64)).astype(np.uint8)
        result = create_classification_overlay(rgb, classification)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_hide_land_default(self):
        """By default, land pixels should show original RGB."""
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        classification = np.ones((10, 10), dtype=np.uint8) * CLASS_LAND
        result = create_classification_overlay(rgb, classification, hide_land=True)
        # Land pixels should be unchanged
        np.testing.assert_array_equal(result, rgb)

    def test_show_land(self):
        """When hide_land=False, land pixels should be blended."""
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        classification = np.ones((10, 10), dtype=np.uint8) * CLASS_LAND
        result = create_classification_overlay(rgb, classification, hide_land=False)
        # Land pixels should be different from original
        assert not np.allclose(result, rgb)

    def test_water_overlay(self):
        """Water pixels should have blue tint."""
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        classification = np.ones((10, 10), dtype=np.uint8) * CLASS_WATER
        result = create_classification_overlay(rgb, classification, alpha=1.0)
        # Blue channel should be higher than red
        assert result[:, :, 2].mean() > result[:, :, 0].mean()

    def test_alpha_parameter(self):
        """Alpha should control blend strength."""
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        classification = np.ones((10, 10), dtype=np.uint8) * CLASS_WATER

        result_low = create_classification_overlay(rgb, classification, alpha=0.1)
        result_high = create_classification_overlay(rgb, classification, alpha=0.9)

        # Higher alpha should be more different from original
        diff_low = np.abs(result_low - rgb).mean()
        diff_high = np.abs(result_high - rgb).mean()
        assert diff_high > diff_low
