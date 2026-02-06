"""Tests for maji.inference.predict module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maji.constants import (
    CLASS_CLOUD,
    CLASS_INVALID,
    CLASS_LAND,
    CLASS_WATER,
    DEFAULT_MODEL_BANDS,
)
from maji.inference.model import UNet
from maji.inference.predict import (
    classify_prediction,
    normalize_tile,
    pad_to_multiple,
    run_inference,
    run_inference_tiled,
)


class TestNormalizeTile:
    """Tests for normalize_tile function."""

    def test_normalize_tile_shape_preserved(self):
        """normalize_tile should preserve input shape."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        result = normalize_tile(tile, bands=DEFAULT_MODEL_BANDS)
        assert result.shape == tile.shape

    def test_normalize_tile_dtype(self):
        """normalize_tile should return float32."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 10000
        result = normalize_tile(tile, bands=DEFAULT_MODEL_BANDS)
        assert result.dtype == np.float32

    def test_normalize_tile_centered(self):
        """Normalized values should be roughly centered around 0."""
        # Create tile with values close to typical S2 values
        tile = np.ones((6, 64, 64), dtype=np.float32) * 3500
        result = normalize_tile(tile, bands=DEFAULT_MODEL_BANDS)
        # Should be centered (roughly between -2 and 2 for typical values)
        assert result.mean() < 2
        assert result.mean() > -2

    def test_normalize_tile_custom_stats(self):
        """normalize_tile should use custom means/stds if provided."""
        tile = np.ones((3, 64, 64), dtype=np.float32) * 100
        means = np.array([100, 100, 100], dtype=np.float32)
        stds = np.array([10, 10, 10], dtype=np.float32)
        result = normalize_tile(tile, means=means, stds=stds)
        # (100 - 100) / 10 = 0
        np.testing.assert_allclose(result, 0, atol=1e-6)

    def test_normalize_tile_partial_params_error(self):
        """normalize_tile should raise error if only means or stds provided."""
        tile = np.random.rand(6, 64, 64).astype(np.float32)
        means = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        with pytest.raises(ValueError, match="Both means and stds"):
            normalize_tile(tile, means=means)

    def test_normalize_tile_default_bands(self):
        """normalize_tile should use DEFAULT_MODEL_BANDS if bands not specified."""
        tile = np.random.rand(6, 64, 64).astype(np.float32) * 5000
        # Should not raise - uses default bands
        result = normalize_tile(tile)
        assert result.shape == tile.shape


class TestPadToMultiple:
    """Tests for pad_to_multiple function."""

    def test_pad_already_divisible(self):
        """No padding needed when already divisible."""
        x = torch.randn(1, 6, 128, 128)
        padded, orig = pad_to_multiple(x, 16)
        assert padded.shape == x.shape
        assert orig == (128, 128)

    def test_pad_needed(self):
        """Should pad to next multiple."""
        x = torch.randn(1, 6, 100, 100)
        padded, orig = pad_to_multiple(x, 16)
        assert padded.shape[-2] == 112  # Next multiple of 16
        assert padded.shape[-1] == 112
        assert orig == (100, 100)

    def test_pad_returns_original_size(self):
        """Should return original size for cropping."""
        x = torch.randn(1, 6, 65, 97)
        _, orig = pad_to_multiple(x, 16)
        assert orig == (65, 97)

    def test_pad_different_multiples(self):
        """Should work with different multiple values."""
        x = torch.randn(1, 6, 100, 100)
        padded, _ = pad_to_multiple(x, 32)
        assert padded.shape[-2] % 32 == 0
        assert padded.shape[-1] % 32 == 0


class TestRunInference:
    """Tests for run_inference function."""

    @pytest.fixture
    def model(self):
        """Create a UNet model for testing."""
        model = UNet(in_channels=6, num_classes=2)
        model.eval()
        return model

    def test_run_inference_output_shape(self, model):
        """run_inference should return water and cloud probability maps."""
        tile = np.random.randn(6, 64, 64).astype(np.float32)
        water, cloud = run_inference(model, tile, device="cpu")
        assert water.shape == (64, 64)
        assert cloud.shape == (64, 64)

    def test_run_inference_output_range(self, model):
        """Probabilities should be in [0, 1]."""
        tile = np.random.randn(6, 64, 64).astype(np.float32)
        water, cloud = run_inference(model, tile, device="cpu")
        assert water.min() >= 0
        assert water.max() <= 1
        assert cloud.min() >= 0
        assert cloud.max() <= 1

    def test_run_inference_non_divisible_size(self, model):
        """run_inference should handle non-16-divisible sizes."""
        tile = np.random.randn(6, 100, 100).astype(np.float32)
        water, cloud = run_inference(model, tile, device="cpu")
        # Output should be cropped back to original size
        assert water.shape == (100, 100)
        assert cloud.shape == (100, 100)


class TestRunInferenceTiled:
    """Tests for run_inference_tiled function."""

    @pytest.fixture
    def model(self):
        """Create a UNet model for testing."""
        model = UNet(in_channels=6, num_classes=2)
        model.eval()
        return model

    def test_run_inference_tiled_yields_results(self, model):
        """run_inference_tiled should yield dictionaries."""
        image = np.random.rand(6, 256, 256).astype(np.float32) * 5000
        results = list(run_inference_tiled(
            model, image, tile_size=128, overlap=0, device="cpu"
        ))
        # Should yield 4 tiles for 256x256 with 128 tile size
        assert len(results) == 4

    def test_run_inference_tiled_result_keys(self, model):
        """Each result should have expected keys."""
        image = np.random.rand(6, 128, 128).astype(np.float32) * 5000
        for result in run_inference_tiled(
            model, image, tile_size=128, overlap=0, device="cpu"
        ):
            assert "row" in result
            assert "col" in result
            assert "water_prob" in result
            assert "cloud_prob" in result
            assert "tile_size" in result

    def test_run_inference_tiled_coordinates(self, model):
        """Tile coordinates should cover the image."""
        image = np.random.rand(6, 256, 256).astype(np.float32) * 5000
        results = list(run_inference_tiled(
            model, image, tile_size=128, overlap=0, device="cpu"
        ))
        coords = [(r["row"], r["col"]) for r in results]
        assert (0, 0) in coords
        assert (0, 128) in coords
        assert (128, 0) in coords
        assert (128, 128) in coords


class TestClassifyPrediction:
    """Tests for classify_prediction function."""

    def test_classify_all_land(self):
        """Low probabilities should result in land."""
        water = np.zeros((10, 10))
        cloud = np.zeros((10, 10))
        result = classify_prediction(water, cloud)
        assert np.all(result == CLASS_LAND)

    def test_classify_water(self):
        """High water probability should result in water class."""
        water = np.ones((10, 10)) * 0.8
        cloud = np.zeros((10, 10))
        result = classify_prediction(water, cloud, th_water=0.5)
        assert np.all(result == CLASS_WATER)

    def test_classify_cloud(self):
        """High cloud probability should result in cloud class."""
        water = np.zeros((10, 10))
        cloud = np.ones((10, 10)) * 0.8
        result = classify_prediction(water, cloud, th_cloud=0.5)
        assert np.all(result == CLASS_CLOUD)

    def test_classify_cloud_overrides_water(self):
        """Cloud should override water when both are high."""
        water = np.ones((10, 10)) * 0.8
        cloud = np.ones((10, 10)) * 0.8
        result = classify_prediction(water, cloud, th_water=0.5, th_cloud=0.5)
        assert np.all(result == CLASS_CLOUD)

    def test_classify_invalid_mask(self):
        """Invalid mask should set pixels to invalid."""
        water = np.ones((10, 10)) * 0.8
        cloud = np.zeros((10, 10))
        invalid = np.zeros((10, 10), dtype=bool)
        invalid[0:5, 0:5] = True
        result = classify_prediction(water, cloud, invalid_mask=invalid)
        assert np.all(result[0:5, 0:5] == CLASS_INVALID)
        assert np.all(result[5:, 5:] == CLASS_WATER)

    def test_classify_custom_thresholds(self):
        """Custom thresholds should be respected."""
        water = np.ones((10, 10)) * 0.3  # Below default 0.5
        cloud = np.zeros((10, 10))
        # With lower threshold, should be water
        result = classify_prediction(water, cloud, th_water=0.2)
        assert np.all(result == CLASS_WATER)

    def test_classify_output_dtype(self):
        """Output should be uint8."""
        water = np.random.rand(10, 10)
        cloud = np.random.rand(10, 10)
        result = classify_prediction(water, cloud)
        assert result.dtype == np.uint8

    def test_classify_mixed(self):
        """Test mixed classification scenario."""
        water = np.array([[0.1, 0.9], [0.3, 0.7]])
        cloud = np.array([[0.1, 0.1], [0.9, 0.1]])
        result = classify_prediction(water, cloud, th_water=0.5, th_cloud=0.5)
        assert result[0, 0] == CLASS_LAND
        assert result[0, 1] == CLASS_WATER
        assert result[1, 0] == CLASS_CLOUD
        assert result[1, 1] == CLASS_WATER
