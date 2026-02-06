"""Tests for maji.inference.model module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from maji.inference.model import UNet, load_model, strip_prefix


class TestUNet:
    """Tests for UNet architecture."""

    def test_unet_init_default(self):
        """UNet should initialize with default parameters."""
        model = UNet()
        assert isinstance(model, torch.nn.Module)

    def test_unet_init_custom(self):
        """UNet should accept custom in_channels and num_classes."""
        model = UNet(in_channels=3, num_classes=5)
        # Check input conv has correct in_channels
        assert model.dconv_down1[0].in_channels == 3
        # Check output conv has correct num_classes
        assert model.conv_last.out_channels == 5

    def test_unet_forward_shape(self):
        """UNet forward pass should preserve spatial dimensions."""
        model = UNet(in_channels=6, num_classes=2)
        x = torch.randn(1, 6, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 256, 256)

    def test_unet_forward_batch(self):
        """UNet should handle batch size > 1."""
        model = UNet(in_channels=6, num_classes=2)
        x = torch.randn(4, 6, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2, 128, 128)

    def test_unet_forward_non_power_of_2(self):
        """UNet should handle non-power-of-2 dimensions (with padding)."""
        model = UNet(in_channels=6, num_classes=2)
        # 100 is not divisible by 16, so this tests internal handling
        x = torch.randn(1, 6, 100, 100)
        with torch.no_grad():
            out = model(x)
        # Output may be padded, but should work
        assert out.shape[0] == 1
        assert out.shape[1] == 2

    def test_unet_eval_mode(self):
        """UNet should support eval mode."""
        model = UNet()
        model.eval()
        assert not model.training

    def test_unet_parameter_count(self):
        """UNet should have expected parameter count."""
        model = UNet(in_channels=6, num_classes=2)
        param_count = sum(p.numel() for p in model.parameters())
        # Roughly 7.7M parameters for standard 4-level UNet
        assert param_count > 7_000_000
        assert param_count < 10_000_000


class TestStripPrefix:
    """Tests for strip_prefix utility."""

    def test_strip_prefix_basic(self):
        """strip_prefix should remove prefix from keys."""
        sd = {"network.conv1.weight": torch.randn(3, 3)}
        result = strip_prefix(sd, "network.")
        assert "conv1.weight" in result
        assert "network.conv1.weight" not in result

    def test_strip_prefix_multiple_keys(self):
        """strip_prefix should work on all keys."""
        sd = {
            "network.layer1.weight": torch.randn(3, 3),
            "network.layer2.bias": torch.randn(3),
        }
        result = strip_prefix(sd, "network.")
        assert set(result.keys()) == {"layer1.weight", "layer2.bias"}

    def test_strip_prefix_no_match(self):
        """strip_prefix should preserve keys without prefix."""
        sd = {"conv1.weight": torch.randn(3, 3)}
        result = strip_prefix(sd, "network.")
        assert "conv1.weight" in result

    def test_strip_prefix_empty_dict(self):
        """strip_prefix should handle empty dict."""
        result = strip_prefix({}, "network.")
        assert result == {}

    def test_strip_prefix_preserves_values(self):
        """strip_prefix should not modify tensor values."""
        tensor = torch.randn(3, 3)
        sd = {"network.weight": tensor}
        result = strip_prefix(sd, "network.")
        assert torch.equal(result["weight"], tensor)


class TestLoadModel:
    """Tests for load_model function."""

    @pytest.fixture
    def temp_weights(self):
        """Create temporary model weights file."""
        model = UNet(in_channels=6, num_classes=2)
        # Add network. prefix as ml4floods does
        state_dict = {f"network.{k}": v for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_load_model_basic(self, temp_weights):
        """load_model should load weights successfully."""
        model = load_model(temp_weights)
        assert isinstance(model, UNet)
        assert not model.training  # Should be in eval mode

    def test_load_model_device(self, temp_weights):
        """load_model should move model to specified device."""
        model = load_model(temp_weights, device="cpu")
        # Check a parameter is on CPU
        param = next(model.parameters())
        assert param.device.type == "cpu"

    def test_load_model_custom_channels(self, temp_weights):
        """load_model should accept custom in_channels."""
        # Note: this will fail to load weights if in_channels doesn't match
        # Here we use the same as what was saved (6)
        model = load_model(temp_weights, in_channels=6)
        assert model.dconv_down1[0].in_channels == 6

    def test_load_model_missing_file(self):
        """load_model should raise FileNotFoundError for missing weights."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.pt")

    def test_load_model_inference_works(self, temp_weights):
        """Loaded model should perform inference correctly."""
        model = load_model(temp_weights)
        x = torch.randn(1, 6, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 64, 64)
