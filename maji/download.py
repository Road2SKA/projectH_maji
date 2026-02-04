"""Download Sentinel-2 bands from CDSE S3 and produce GeoTIFFs.

Reads individual JP2 band files directly from the CDSE eodata bucket
via rasterio's /vsis3/ driver, resamples 20m bands to 10m, and
stacks into a single multi-band Cloud-Optimized GeoTIFF.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.session import AWSSession
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

# --- Band configuration -----------------------------------------------

MODEL_BANDS = ["B03", "B04", "B08", "B8A", "B11", "B12"]
CLOUD_BAND = "SCL"
ALL_BANDS = MODEL_BANDS + [CLOUD_BAND]

BAND_RESOLUTION: dict[str, int] = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20,
    "B11": 20, "B12": 20, "SCL": 20,
    "B01": 60, "B09": 60,
}

# Full MGRS tile dimensions at 10m
TARGET_HEIGHT = 10980
TARGET_WIDTH = 10980

# Retry configuration for transient S3 errors
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubled each retry

# CDSE concurrency limit
_MAX_CDSE_WORKERS = 4


def create_s3_session(
    access_key: str,
    secret_key: str,
    endpoint_url: str = "https://eodata.dataspace.copernicus.eu",
) -> AWSSession:
    """Create a rasterio AWSSession configured for CDSE S3.

    Parameters
    ----------
    access_key : str
        S3 access key for the CDSE ``eodata`` bucket.
    secret_key : str
        Corresponding S3 secret key.
    endpoint_url : str, optional
        S3-compatible endpoint URL (default:
        ``https://eodata.dataspace.copernicus.eu``).

    Returns
    -------
    rasterio.session.AWSSession
        Configured session that can be passed to
        :func:`rasterio.Env` or directly to download helpers.

    Notes
    -----
    The CDSE eodata bucket uses an S3-compatible API but is **not**
    hosted on AWS.  The session sets ``aws_unsigned=False`` so that
    signed requests are sent using the provided credentials.
    """
    return AWSSession(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        aws_unsigned=False,
    )


def _read_band_with_retry(
    href: str,
    target_shape: tuple[int, int],
    resampling: Resampling,
    max_retries: int = MAX_RETRIES,
) -> tuple[np.ndarray, dict]:
    """Read a single band from S3 with retry on transient errors.

    Parameters
    ----------
    href : str
        S3 path to the band file (e.g. ``s3://eodata/.../B03_10m.jp2``).
    target_shape : tuple of int
        ``(height, width)`` to resample to (typically 10 980 x 10 980
        for 10 m resolution).
    resampling : rasterio.enums.Resampling
        Resampling method (``bilinear`` for reflectance bands,
        ``nearest`` for the SCL classification band).
    max_retries : int, optional
        Number of retry attempts (default: :data:`MAX_RETRIES`).

    Returns
    -------
    data : numpy.ndarray
        2-D ``uint16`` array of shape *target_shape*.
    meta : dict
        Contains ``"crs"`` (:class:`rasterio.crs.CRS`),
        ``"transform"`` (:class:`rasterio.transform.Affine`), and
        ``"native_shape"`` (tuple of int) from the source dataset.

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.  The original
        :class:`rasterio.errors.RasterioIOError` is chained as the
        cause.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with rasterio.open(href) as src:
                native_shape = (src.height, src.width)
                meta = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "native_shape": native_shape,
                }

                if native_shape == target_shape:
                    data = src.read(1)
                else:
                    data = src.read(
                        1,
                        out_shape=target_shape,
                        resampling=resampling,
                    )

                return data, meta

        except rasterio.errors.RasterioIOError as e:
            last_error = e
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "S3 read failed for %s (attempt %d/%d), retrying in %.1fs: %s",
                href, attempt + 1, max_retries, wait, e,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to read {href} after {max_retries} attempts"
    ) from last_error


def download_tile(
    scene_assets: dict[str, str],
    mgrs_tile: str,
    scene_date: str,
    data_dir: Path,
    session: AWSSession,
    bands: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Download all bands for one scene and write a multi-band GeoTIFF.

    Bands are written one at a time to avoid holding the full stack in
    memory simultaneously.

    Parameters
    ----------
    scene_assets : dict[str, str]
        ``{band_name: s3_href}`` mapping from search results.
    mgrs_tile : str
        Five-character MGRS tile code, e.g. ``"37MFT"``.
    scene_date : str
        ISO date string, e.g. ``"2026-01-10"`` (used in the output
        filename).
    data_dir : Path
        Root data directory.  A sub-directory named *mgrs_tile* is
        created automatically.
    session : rasterio.session.AWSSession
        Authenticated session from :func:`create_s3_session`.
    bands : list[str] or None, optional
        Bands to download (default: :data:`ALL_BANDS`).
    overwrite : bool, optional
        If ``False`` (default) and the output file already exists,
        the download is skipped.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved GeoTIFF, e.g.
        ``data/37MFT/2026-01-10_S2L2A.tif``.

    Raises
    ------
    KeyError
        If *scene_assets* is missing an href for any requested band.
    RuntimeError
        If a band cannot be read after all retry attempts (propagated
        from :func:`_read_band_with_retry`).

    Notes
    -----
    * Each band is read from S3 individually and written to the output
      file one band at a time, keeping peak memory to a single 10 980 x
      10 980 ``uint16`` array (~230 MB).
    * The output uses a Cloud-Optimized GeoTIFF profile: ``deflate``
      compression, 512 x 512 internal tiles.
    * 20 m bands are resampled to 10 m using bilinear interpolation,
      except for the SCL (Scene Classification Layer) which uses
      nearest-neighbour to preserve class values.
    """
    if bands is None:
        bands = list(ALL_BANDS)

    # Output path
    tile_dir = data_dir / mgrs_tile
    tile_dir.mkdir(parents=True, exist_ok=True)
    out_path = tile_dir / f"{scene_date}_S2L2A.tif"

    if out_path.exists() and not overwrite:
        logger.info("Skipping %s (already exists)", out_path)
        return out_path

    target_shape = (TARGET_HEIGHT, TARGET_WIDTH)
    reference_crs = None
    reference_transform = None

    # First pass: read all bands inside rasterio.Env, write band-by-band
    with rasterio.Env(
        session=session,
        AWS_VIRTUAL_HOSTING=False,
        GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2",
    ):
        # Read first 10m band to get CRS/transform for the output file
        for band_name in bands:
            if BAND_RESOLUTION.get(band_name) == 10:
                href = scene_assets.get(band_name)
                if href is None:
                    continue
                _, meta = _read_band_with_retry(
                    href, target_shape, Resampling.bilinear,
                )
                if meta["native_shape"] == target_shape:
                    reference_crs = meta["crs"]
                    reference_transform = meta["transform"]
                    break

        # Fallback: derive 10m transform from a 20m band
        if reference_crs is None:
            first_href = scene_assets[bands[0]]
            _, meta = _read_band_with_retry(
                first_href, target_shape, Resampling.bilinear,
            )
            reference_crs = meta["crs"]
            t = meta["transform"]
            native_h, native_w = meta["native_shape"]
            scale = native_h / TARGET_HEIGHT  # e.g. 0.5 for 20m→10m
            reference_transform = Affine(
                t.a * scale, t.b, t.c,
                t.d, t.e * scale, t.f,
            )

        profile = {
            "driver": "GTiff",
            "dtype": "uint16",
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "count": len(bands),
            "crs": reference_crs,
            "transform": reference_transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            for band_idx, band_name in enumerate(bands, start=1):
                href = scene_assets.get(band_name)
                if href is None:
                    raise KeyError(
                        f"No S3 href found for band {band_name} in scene assets"
                    )

                resampling = (
                    Resampling.nearest if band_name == "SCL"
                    else Resampling.bilinear
                )

                logger.info(
                    "Reading %s (%dm) → %s",
                    band_name,
                    BAND_RESOLUTION.get(band_name, 0),
                    "native" if BAND_RESOLUTION.get(band_name) == 10 else "resampled to 10m",
                )

                data, _ = _read_band_with_retry(href, target_shape, resampling)
                dst.write(data, band_idx)
                dst.set_band_description(band_idx, band_name)

    size_mb = out_path.stat().st_size / 1e6
    logger.info("Saved %s (%d bands, %.1f MB)", out_path, len(bands), size_mb)
    return out_path


def download_tiles(
    scenes: pd.DataFrame,
    data_dir: Path,
    session: AWSSession,
    bands: list[str] | None = None,
    max_workers: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    """Download multiple scenes sequentially.

    Iterates over rows in *scenes* and calls :func:`download_tile`
    for each one.  Failures are logged but do not halt the loop.

    Parameters
    ----------
    scenes : pandas.DataFrame
        DataFrame from :mod:`maji.search` with at least ``mgrs_tile``,
        ``datetime``, and ``assets`` columns.
    data_dir : Path
        Root data directory.
    session : rasterio.session.AWSSession
        Authenticated session from :func:`create_s3_session`.
    bands : list[str] or None, optional
        Bands to download (default: :data:`ALL_BANDS`).
    max_workers : int, optional
        Reserved for future parallel downloads (default 1).  Values
        above 4 are clamped — see *Warns*.
    overwrite : bool, optional
        Re-download existing files (default ``False``).

    Returns
    -------
    list[pathlib.Path]
        Paths to successfully saved GeoTIFFs.  The list may be
        shorter than *scenes* if individual tiles fail.

    Raises
    ------
    Exception
        Individual tile failures are caught and logged; they do not
        propagate.

    Warns
    -----
    UserWarning
        If ``max_workers`` exceeds the CDSE concurrency limit of 4
        simultaneous connections.

    Notes
    -----
    CDSE enforces a limit of **4 concurrent S3 connections** per
    credential set.  ``max_workers`` is clamped to
    :data:`_MAX_CDSE_WORKERS` (4) and a :class:`UserWarning` is
    emitted when the caller exceeds this value.
    """
    if max_workers > _MAX_CDSE_WORKERS:
        warnings.warn(
            f"max_workers={max_workers} exceeds CDSE limit of "
            f"{_MAX_CDSE_WORKERS} concurrent connections; clamping to "
            f"{_MAX_CDSE_WORKERS}",
            stacklevel=2,
        )
        max_workers = _MAX_CDSE_WORKERS

    paths: list[Path] = []
    for _, row in scenes.iterrows():
        scene_date = row["datetime"].strftime("%Y-%m-%d")
        try:
            path = download_tile(
                scene_assets=row["assets"],
                mgrs_tile=row["mgrs_tile"],
                scene_date=scene_date,
                data_dir=Path(data_dir),
                session=session,
                bands=bands,
                overwrite=overwrite,
            )
            paths.append(path)
        except Exception:
            logger.error(
                "Failed to download %s/%s",
                row["mgrs_tile"], scene_date,
                exc_info=True,
            )

    return paths
