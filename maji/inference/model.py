"""UNet architecture and model loading utilities.

This module provides the UNet semantic segmentation model architecture
used for water/cloud classification, along with utilities for loading
pre-trained weights from ML4Floods checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from maji.constants import DEFAULT_MODEL_BANDS

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two consecutive Conv2d-ReLU blocks.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    nn.Sequential
        Sequential module with two conv-relu pairs.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level UNet for semantic segmentation.

    Standard encoder-decoder architecture with skip connections,
    designed for water/cloud classification from Sentinel-2 imagery.

    Architecture:
        Encoder: 4 double-conv blocks with max pooling
        Decoder: 3 upsampling + skip connection + double-conv blocks
        Output: 1x1 conv to num_classes

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default: 6 for bgriswirs bands).
    num_classes : int, optional
        Number of output classes (default: 2 for water/cloud binary heads).

    Examples
    --------
    >>> model = UNet(in_channels=6, num_classes=2)
    >>> x = torch.randn(1, 6, 256, 256)
    >>> out = model(x)
    >>> out.shape
    torch.Size([1, 2, 256, 256])
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        # Encoder
        self.dconv_down1 = _double_conv(in_channels, 64)
        self.dconv_down2 = _double_conv(64, 128)
        self.dconv_down3 = _double_conv(128, 256)
        self.dconv_down4 = _double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # Decoder
        self.dconv_up3 = _double_conv(256 + 512, 256)
        self.dconv_up2 = _double_conv(128 + 256, 128)
        self.dconv_up1 = _double_conv(64 + 128, 64)

        # Output
        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, num_classes, height, width).
        """
        # Encoder
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # Decoder with skip connections
        x = F.interpolate(x, size=conv3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = F.interpolate(x, size=conv2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = F.interpolate(x, size=conv1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        return self.conv_last(x)


def strip_prefix(state_dict: dict[str, Any], prefix: str = "network.") -> dict[str, Any]:
    """Remove prefix from state dict keys.

    The ml4floods checkpoint saves weights as "network.dconv_down1.0.weight"
    but our UNet expects "dconv_down1.0.weight".

    Parameters
    ----------
    state_dict : dict
        PyTorch state dict with prefixed keys.
    prefix : str, optional
        Prefix to remove (default: "network.").

    Returns
    -------
    dict
        State dict with prefix stripped from all keys.

    Examples
    --------
    >>> sd = {"network.conv1.weight": torch.randn(3, 3)}
    >>> stripped = strip_prefix(sd, "network.")
    >>> list(stripped.keys())
    ['conv1.weight']
    """
    return {k.replace(prefix, ""): v for k, v in state_dict.items()}


def load_model(
    weights_path: str | Path,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    in_channels: int | None = None,
) -> UNet:
    """Load a pre-trained UNet model from weights file.

    Parameters
    ----------
    weights_path : str or Path
        Path to the model weights file (.pt).
    config_path : str, Path, or None, optional
        Path to config.json containing model hyperparameters. If provided,
        reads in_channels and num_classes from the config. If None, uses
        defaults.
    device : str or torch.device, optional
        Device to load model onto (default: "cpu").
    in_channels : int or None, optional
        Number of input channels. If None, reads from config or uses
        len(DEFAULT_MODEL_BANDS).

    Returns
    -------
    UNet
        Loaded model in eval mode on the specified device.

    Raises
    ------
    FileNotFoundError
        If weights_path does not exist.

    Examples
    --------
    >>> model = load_model("models/WF2_unet_rbgiswirs/model.pt")
    >>> model.eval()
    UNet(...)
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Read config if provided
    num_classes = 2
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            hyperparams = config.get("model_params", {}).get("hyperparameters", {})
            num_classes = hyperparams.get("num_classes", 2)
            # Could also read channel_configuration here if needed
            logger.debug(
                "Loaded config: num_classes=%d, channel_config=%s",
                num_classes,
                hyperparams.get("channel_configuration"),
            )

    # Determine input channels
    if in_channels is None:
        in_channels = len(DEFAULT_MODEL_BANDS)

    # Create model
    model = UNet(in_channels=in_channels, num_classes=num_classes)

    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = strip_prefix(state_dict, "network.")
    model.load_state_dict(state_dict)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Loaded model with %d parameters on %s", param_count, device)

    return model
