"""Maji — Sentinel-2 flood-mapping data pipeline."""

from maji.config import Settings
from maji.constants import (
    CLASS_NAMES,
    DEFAULT_DOWNLOAD_BANDS,
    DEFAULT_MODEL_BANDS,
    DOWNLOAD_BANDS,
    MODEL_BANDS,
    NORMALIZATION,
)
from maji.download import create_s3_session, download_tile, download_tiles
from maji.search import search_scenes, select_best_scenes

__all__ = [
    # Config
    "Settings",
    # Constants
    "MODEL_BANDS",
    "DOWNLOAD_BANDS",
    "DEFAULT_MODEL_BANDS",
    "DEFAULT_DOWNLOAD_BANDS",
    "NORMALIZATION",
    "CLASS_NAMES",
    # Search
    "search_scenes",
    "select_best_scenes",
    # Download
    "create_s3_session",
    "download_tile",
    "download_tiles",
]

# Optional inference exports (requires torch)
try:
    from maji.inference import (
        UNet,
        classify_prediction,
        load_model,
        normalize_tile,
        run_inference,
        run_inference_tiled,
    )

    __all__.extend([
        "UNet",
        "load_model",
        "normalize_tile",
        "run_inference",
        "run_inference_tiled",
        "classify_prediction",
    ])
except ImportError:
    # torch not installed - inference features unavailable
    pass
