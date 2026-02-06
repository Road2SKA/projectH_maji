"""Download Sentinel-2 bands from CDSE S3 and produce GeoTIFFs.

Downloads individual JP2 band files from the CDSE eodata bucket
via boto3, resamples 20m bands to 10m, and stacks into a single
multi-band Cloud-Optimized GeoTIFF.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.session import AWSSession
from rasterio.transform import Affine
from tqdm.auto import tqdm

from maji.constants import (
    BAND_RESOLUTION,
    DEFAULT_DOWNLOAD_BANDS,
    DEFAULT_MODEL_BANDS,
    SCL_BAND,
    TILE_HEIGHT_10M,
    TILE_WIDTH_10M,
)

logger = logging.getLogger(__name__)

# --- Band configuration (imported from constants) ---------------------

# Backward-compatible aliases
MODEL_BANDS = DEFAULT_MODEL_BANDS
CLOUD_BAND = SCL_BAND
ALL_BANDS = DEFAULT_DOWNLOAD_BANDS

# Full MGRS tile dimensions at 10m
TARGET_HEIGHT = TILE_HEIGHT_10M
TARGET_WIDTH = TILE_WIDTH_10M

# Retry configuration for transient S3 errors
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubled each retry

# CDSE concurrency limit
_MAX_CDSE_WORKERS = 4


def create_s3_session(
    access_key: str,
    secret_key: str,
    endpoint_url: str = "https://eodata.dataspace.copernicus.eu",
    verbose: bool = False,
) -> AWSSession:
    """Create a rasterio AWSSession configured for CDSE S3.

    Also sets environment variables for boto3 access to the same
    endpoint.

    Parameters
    ----------
    access_key : str
        S3 access key for the CDSE ``eodata`` bucket.
    secret_key : str
        Corresponding S3 secret key.
    endpoint_url : str, optional
        S3-compatible endpoint URL (default:
        ``https://eodata.dataspace.copernicus.eu``).
    verbose : bool, optional
        If ``True``, print connection status messages (default: ``False``).

    Returns
    -------
    rasterio.session.AWSSession
        Configured session that can be passed to download helpers.

    Raises
    ------
    ConnectionError
        If the S3 connection test fails. The error message includes
        details about the failure and how to fix it.

    Notes
    -----
    The CDSE eodata bucket uses an S3-compatible API but is **not**
    hosted on AWS.  The session sets ``aws_unsigned=False`` so that
    signed requests are sent using the provided credentials.

    Environment variables ``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, and ``AWS_S3_ENDPOINT`` are set for
    boto3 compatibility.
    """
    from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

    # Set environment variables for boto3 access
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_S3_ENDPOINT"] = endpoint_url

    # Show connection info if verbose
    cred_prefix = access_key[:8] if len(access_key) >= 8 else access_key
    if verbose:
        print("Connecting to CDSE S3...")
        print(f"  endpoint: {endpoint_url}")
        print(f"  credentials: {cred_prefix}...")

    # Test connection by listing objects in the eodata bucket
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        s3_client.list_objects_v2(Bucket="eodata", MaxKeys=1)
        if verbose:
            print("  Connected successfully")
    except NoCredentialsError as e:
        if verbose:
            print("  Connection failed: Missing credentials")
        raise ConnectionError(
            "Missing S3 credentials. "
            "Check that CDSE_ACCESS_KEY and CDSE_SECRET_KEY are set."
        ) from e
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        if error_code in ("403", "AccessDenied"):
            if verbose:
                print(f"  Connection failed: Invalid credentials (403 Forbidden)")
            raise ConnectionError(
                f"Invalid S3 credentials (403 Forbidden). "
                f"Check your CDSE_ACCESS_KEY and CDSE_SECRET_KEY. "
                f"Generate new keys at https://eodata-s3keysmanager.dataspace.copernicus.eu/"
            ) from e
        elif error_code in ("404", "NoSuchBucket"):
            if verbose:
                print(f"  Connection failed: Bucket not found (404)")
            raise ConnectionError(
                f"S3 bucket 'eodata' not found. "
                f"Check that the endpoint URL is correct: {endpoint_url}"
            ) from e
        else:
            if verbose:
                print(f"  Connection failed: {error_code} - {error_msg}")
            raise ConnectionError(
                f"S3 connection failed ({error_code}): {error_msg}"
            ) from e
    except EndpointConnectionError as e:
        if verbose:
            print(f"  Connection failed: Could not reach endpoint")
        raise ConnectionError(
            f"Could not connect to S3 endpoint: {endpoint_url}. "
            f"Check your network connection and endpoint URL."
        ) from e

    return AWSSession(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        aws_unsigned=False,
    )


def _parse_s3_url(href: str) -> tuple[str, str]:
    """Parse an S3 URL into bucket and key.

    Parameters
    ----------
    href : str
        S3 URL in the form ``s3://bucket/key/path``.

    Returns
    -------
    bucket : str
        The bucket name.
    key : str
        The object key (path within the bucket).
    """
    parsed = urlparse(href)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _read_band_with_retry(
    href: str,
    target_shape: tuple[int, int],
    resampling: Resampling,
    s3_client,
    max_retries: int = MAX_RETRIES,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Read a single band from S3 with retry on transient errors.

    Downloads the file via boto3 to a temporary file, then opens it
    locally with rasterio.

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
    s3_client : botocore.client.S3
        Boto3 S3 client configured for CDSE.
    max_retries : int, optional
        Number of retry attempts (default: :data:`MAX_RETRIES`).
    progress_callback : callable, optional
        Called with bytes downloaded during S3 transfer. Used for
        tqdm progress bar integration.

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
        If all retry attempts are exhausted.  The original exception
        is chained as the cause.
    """
    bucket, key = _parse_s3_url(href)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jp2", delete=True) as tmp:
                s3_client.download_file(
                    bucket, key, tmp.name,
                    Callback=progress_callback,
                )

                with rasterio.open(tmp.name) as src:
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

        except Exception as e:
            last_error = e
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "S3 download failed for %s (attempt %d/%d), retrying in %.1fs: %s",
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
    verbose: bool = False,
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
    verbose : bool, optional
        If ``True``, print progress information and show tqdm progress
        bars during download (default: ``False``).

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

    if verbose:
        print(f"Downloading tile {mgrs_tile} ({scene_date})")
        print(f"  output: {out_path}")

    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  Skipping (file exists): {out_path}")
        logger.info("Skipping %s (already exists)", out_path)
        return out_path

    if verbose:
        print(f"  bands:  {len(bands)} ({', '.join(bands)})")

    # Create boto3 S3 client using credentials from environment
    # (set by create_s3_session)
    endpoint_url = os.environ.get(
        "AWS_S3_ENDPOINT", "https://eodata.dataspace.copernicus.eu"
    )
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)

    target_shape = (TARGET_HEIGHT, TARGET_WIDTH)
    reference_crs = None
    reference_transform = None

    # Read first 10m band to get CRS/transform for the output file
    for band_name in bands:
        if BAND_RESOLUTION.get(band_name) == 10:
            href = scene_assets.get(band_name)
            if href is None:
                continue
            _, meta = _read_band_with_retry(
                href, target_shape, Resampling.bilinear, s3_client,
            )
            if meta["native_shape"] == target_shape:
                reference_crs = meta["crs"]
                reference_transform = meta["transform"]
                break

    # Fallback: derive 10m transform from a 20m band
    if reference_crs is None:
        first_href = scene_assets[bands[0]]
        _, meta = _read_band_with_retry(
            first_href, target_shape, Resampling.bilinear, s3_client,
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
        # Create band iterator with optional progress bar
        band_iter = enumerate(bands, start=1)
        if verbose:
            band_iter = tqdm(
                list(band_iter),
                desc="  Bands",
                unit="band",
                leave=True,
            )

        for band_idx, band_name in band_iter:
            href = scene_assets.get(band_name)
            if href is None:
                raise KeyError(
                    f"No S3 href found for band {band_name} in scene assets"
                )

            resampling = (
                Resampling.nearest if band_name == "SCL"
                else Resampling.bilinear
            )
            resolution = BAND_RESOLUTION.get(band_name, 0)

            logger.info(
                "Reading %s (%dm) → %s",
                band_name,
                resolution,
                "native" if resolution == 10 else "resampled to 10m",
            )

            # Update progress bar description if verbose
            if verbose and hasattr(band_iter, "set_postfix"):
                resample_str = "native" if resolution == 10 else f"{resolution}m→10m"
                band_iter.set_postfix_str(f"{band_name} ({resample_str})")

            data, _ = _read_band_with_retry(
                href, target_shape, resampling, s3_client,
            )
            dst.write(data, band_idx)
            dst.set_band_description(band_idx, band_name)

    size_mb = out_path.stat().st_size / 1e6
    if verbose:
        print(f"  Saved {mgrs_tile}/{scene_date}_S2L2A.tif ({size_mb:.1f} MB)")
    logger.info("Saved %s (%d bands, %.1f MB)", out_path, len(bands), size_mb)
    return out_path


def download_tiles(
    scenes: pd.DataFrame,
    data_dir: Path,
    session: AWSSession,
    bands: list[str] | None = None,
    max_workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
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
    verbose : bool, optional
        If ``True``, print progress information and show tqdm progress
        bars during download (default: ``False``).

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

    if bands is None:
        bands = list(ALL_BANDS)

    n_scenes = len(scenes)
    if verbose:
        print(f"Downloading {n_scenes} scene(s) to {data_dir}/")
        print(f"  bands: {', '.join(bands)}")
        print()

    paths: list[Path] = []
    failures: list[str] = []

    for idx, (_, row) in enumerate(scenes.iterrows(), start=1):
        scene_date = row["datetime"].strftime("%Y-%m-%d")
        tile_id = row["mgrs_tile"]

        if verbose:
            print(f"[tile {idx}/{n_scenes}] {tile_id} ({scene_date})")

        try:
            path = download_tile(
                scene_assets=row["assets"],
                mgrs_tile=tile_id,
                scene_date=scene_date,
                data_dir=Path(data_dir),
                session=session,
                bands=bands,
                overwrite=overwrite,
                verbose=verbose,
            )
            paths.append(path)
        except Exception:
            failures.append(f"{tile_id}/{scene_date}")
            logger.error(
                "Failed to download %s/%s",
                tile_id, scene_date,
                exc_info=True,
            )
            if verbose:
                print(f"  FAILED: {tile_id}/{scene_date}")

        if verbose:
            print()

    if verbose:
        total_mb = sum(p.stat().st_size for p in paths) / 1e6
        print(f"Downloaded {len(paths)}/{n_scenes} tiles ({total_mb:.1f} MB total)")
        if failures:
            print(f"  Failed: {', '.join(failures)}")

    return paths
