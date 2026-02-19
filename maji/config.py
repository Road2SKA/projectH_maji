"""Configuration via pydantic-settings, reading from .env.

Uses ``pydantic-settings`` to load project configuration from environment
variables and an optional ``.env`` file. See ``env.example`` in the
repository root for a template.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Project settings loaded from environment / ``.env`` file.

    All fields can be set via environment variables (case-insensitive) or
    by placing key-value pairs in a ``.env`` file at the project root.
    See ``env.example`` for a fully-commented template.

    Attributes
    ----------
    cdse_client_id : str
        OAuth2 client ID for the Copernicus Data Space Ecosystem (CDSE).
        Required only for authenticated STAC queries.
    cdse_client_secret : str
        OAuth2 client secret corresponding to ``cdse_client_id``.
    cdse_s3_access_key : str
        S3 access key for the CDSE ``eodata`` bucket. Generate at
        https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials
    cdse_s3_secret_key : str
        S3 secret key corresponding to ``cdse_s3_access_key``.
    data_dir : Path
        Root directory for downloaded satellite data (default: ``data/``).
    output_dir : Path
        Root directory for model outputs (default: ``output/``).
    device : str
        Compute device for inference, e.g. ``"cpu"`` or ``"cuda"``
        (default: ``"cpu"``).
    """

    # OAuth2 credentials (for authenticated STAC queries)
    cdse_client_id: str = ""
    cdse_client_secret: str = ""

    # S3 credentials (for eodata bucket access)
    cdse_s3_access_key: str = ""
    cdse_s3_secret_key: str = ""

    # Processing directories
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")

    # Compute device
    device: str = "cpu"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
