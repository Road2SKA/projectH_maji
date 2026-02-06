"""Inference subpackage for flood mapping.

This subpackage provides:
- UNet model architecture
- Model loading utilities
- Normalization and prediction functions
- Classification utilities
"""

from maji.inference.model import UNet, load_model, strip_prefix
from maji.inference.predict import (
    classify_prediction,
    normalize_tile,
    pad_to_multiple,
    run_inference,
    run_inference_tiled,
)

__all__ = [
    # Model
    "UNet",
    "load_model",
    "strip_prefix",
    # Prediction
    "normalize_tile",
    "pad_to_multiple",
    "run_inference",
    "run_inference_tiled",
    "classify_prediction",
]
