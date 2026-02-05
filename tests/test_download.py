"""Tests for maji.download — S3 band download + GeoTIFF writing."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

from maji.download import (
    ALL_BANDS,
    BAND_RESOLUTION,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    _parse_s3_url,
    _read_band_with_retry,
    create_s3_session,
    download_tile,
    download_tiles,
)


# --- create_s3_session ---------------------------------------------------


class TestCreateS3Session:
    """Tests for :func:`maji.download.create_s3_session`."""

    @patch("maji.download.AWSSession")
    def test_creates_session(self, mock_aws_session):
        """Default call passes correct credentials and CDSE endpoint."""
        create_s3_session("mykey", "mysecret")
        mock_aws_session.assert_called_once_with(
            aws_access_key_id="mykey",
            aws_secret_access_key="mysecret",
            endpoint_url="https://eodata.dataspace.copernicus.eu",
            aws_unsigned=False,
        )

    @patch("maji.download.AWSSession")
    def test_custom_endpoint(self, mock_aws_session):
        """Custom endpoint_url is forwarded to AWSSession."""
        create_s3_session("k", "s", endpoint_url="https://custom.example.com")
        mock_aws_session.assert_called_once_with(
            aws_access_key_id="k",
            aws_secret_access_key="s",
            endpoint_url="https://custom.example.com",
            aws_unsigned=False,
        )

    @patch("maji.download.AWSSession")
    def test_sets_environment_variables(self, mock_aws_session):
        """Should set environment variables for boto3 access."""
        create_s3_session("mykey", "mysecret", endpoint_url="https://test.example.com")
        assert os.environ["AWS_ACCESS_KEY_ID"] == "mykey"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "mysecret"
        assert os.environ["AWS_S3_ENDPOINT"] == "https://test.example.com"


# --- _parse_s3_url ----------------------------------------------------------


class TestParseS3Url:
    """Tests for :func:`maji.download._parse_s3_url`."""

    def test_parses_standard_url(self):
        """Should parse bucket and key from standard S3 URL."""
        bucket, key = _parse_s3_url("s3://eodata/Sentinel-2/L2A/path/to/file.jp2")
        assert bucket == "eodata"
        assert key == "Sentinel-2/L2A/path/to/file.jp2"

    def test_handles_simple_key(self):
        """Should handle single-component key."""
        bucket, key = _parse_s3_url("s3://mybucket/file.txt")
        assert bucket == "mybucket"
        assert key == "file.txt"


# --- _read_band_with_retry -----------------------------------------------


def _mock_rasterio_open(height, width, crs=None, transform=None, data=None):
    """Create a mock rasterio dataset context manager.

    Parameters
    ----------
    height, width : int
        Dimensions of the fake raster.
    crs : rasterio.crs.CRS or None, optional
        Coordinate reference system (default: EPSG:32637).
    transform : rasterio.transform.Affine or None, optional
        Affine transform (default: 10 m resolution).
    data : numpy.ndarray or None, optional
        Array returned by ``read()`` (default: all-ones ``uint16``).

    Returns
    -------
    unittest.mock.MagicMock
        Context manager that yields a mock dataset.
    """
    if crs is None:
        crs = CRS.from_epsg(32637)
    if transform is None:
        transform = Affine(10.0, 0, 500000, 0, -10.0, 10000000)
    if data is None:
        data = np.ones((height, width), dtype=np.uint16)

    mock_ds = MagicMock()
    mock_ds.height = height
    mock_ds.width = width
    mock_ds.crs = crs
    mock_ds.transform = transform
    mock_ds.read.return_value = data

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_ds)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestReadBandWithRetry:
    """Tests for :func:`maji.download._read_band_with_retry`."""

    @patch("maji.download.rasterio.open")
    @patch("maji.download.tempfile.NamedTemporaryFile")
    def test_10m_band_no_resampling(self, mock_tempfile, mock_open):
        """10m band at target shape — should read without resampling."""
        target = (TARGET_HEIGHT, TARGET_WIDTH)

        # Mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.jp2"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_open.return_value = _mock_rasterio_open(TARGET_HEIGHT, TARGET_WIDTH)

        mock_s3_client = MagicMock()

        data, meta = _read_band_with_retry(
            "s3://eodata/.../B03_10m.jp2", target, Resampling.bilinear,
            s3_client=mock_s3_client,
        )

        # Verify S3 download was called
        mock_s3_client.download_file.assert_called_once_with(
            "eodata", ".../B03_10m.jp2", "/tmp/test.jp2"
        )
        assert data.shape == target
        assert meta["native_shape"] == target
        # read() called with just band index (no out_shape)
        mock_ds = mock_open.return_value.__enter__()
        mock_ds.read.assert_called_once_with(1)

    @patch("maji.download.rasterio.open")
    @patch("maji.download.tempfile.NamedTemporaryFile")
    def test_20m_band_resampled(self, mock_tempfile, mock_open):
        """20m band (5490×5490) — should be resampled to target shape."""
        target = (TARGET_HEIGHT, TARGET_WIDTH)

        # Mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.jp2"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        resampled_data = np.ones(target, dtype=np.uint16)
        mock_ds = MagicMock()
        mock_ds.height = 5490
        mock_ds.width = 5490
        mock_ds.crs = CRS.from_epsg(32637)
        mock_ds.transform = Affine(20.0, 0, 500000, 0, -20.0, 10000000)
        mock_ds.read.return_value = resampled_data

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_ds)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = ctx

        mock_s3_client = MagicMock()

        data, meta = _read_band_with_retry(
            "s3://eodata/.../B8A_20m.jp2", target, Resampling.bilinear,
            s3_client=mock_s3_client,
        )

        mock_ds.read.assert_called_once_with(
            1, out_shape=target, resampling=Resampling.bilinear,
        )
        assert meta["native_shape"] == (5490, 5490)

    @patch("maji.download.rasterio.open")
    @patch("maji.download.tempfile.NamedTemporaryFile")
    def test_scl_uses_nearest(self, mock_tempfile, mock_open):
        """SCL band should use nearest-neighbor resampling."""
        target = (TARGET_HEIGHT, TARGET_WIDTH)

        # Mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.jp2"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_ds = MagicMock()
        mock_ds.height = 5490
        mock_ds.width = 5490
        mock_ds.crs = CRS.from_epsg(32637)
        mock_ds.transform = Affine(20.0, 0, 500000, 0, -20.0, 10000000)
        mock_ds.read.return_value = np.ones(target, dtype=np.uint16)

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_ds)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = ctx

        mock_s3_client = MagicMock()

        _read_band_with_retry(
            "s3://eodata/.../SCL_20m.jp2", target, Resampling.nearest,
            s3_client=mock_s3_client,
        )

        mock_ds.read.assert_called_once_with(
            1, out_shape=target, resampling=Resampling.nearest,
        )

    @patch("maji.download.time.sleep")
    @patch("maji.download.rasterio.open")
    @patch("maji.download.tempfile.NamedTemporaryFile")
    def test_retry_on_transient_error(self, mock_tempfile, mock_open, mock_sleep):
        """Should retry and succeed on the second attempt."""
        target = (TARGET_HEIGHT, TARGET_WIDTH)

        # Mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.jp2"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        # First call fails, second succeeds
        good_ctx = _mock_rasterio_open(TARGET_HEIGHT, TARGET_WIDTH)
        mock_open.side_effect = [
            rasterio.errors.RasterioIOError("Connection reset"),
            good_ctx,
        ]

        mock_s3_client = MagicMock()

        data, meta = _read_band_with_retry(
            "s3://eodata/.../B03_10m.jp2", target, Resampling.bilinear,
            s3_client=mock_s3_client,
        )

        assert mock_open.call_count == 2
        mock_sleep.assert_called_once()  # slept between retries

    @patch("maji.download.time.sleep")
    @patch("maji.download.rasterio.open")
    @patch("maji.download.tempfile.NamedTemporaryFile")
    def test_exhausted_retries(self, mock_tempfile, mock_open, mock_sleep):
        """Should raise RuntimeError after all retries fail."""
        target = (TARGET_HEIGHT, TARGET_WIDTH)

        # Mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.jp2"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        mock_open.side_effect = rasterio.errors.RasterioIOError("timeout")

        mock_s3_client = MagicMock()

        with pytest.raises(RuntimeError, match="Failed to read"):
            _read_band_with_retry(
                "s3://eodata/.../B03_10m.jp2", target, Resampling.bilinear,
                s3_client=mock_s3_client,
                max_retries=3,
            )

        assert mock_open.call_count == 3


# --- download_tile --------------------------------------------------------


class TestDownloadTile:
    """Tests for :func:`maji.download.download_tile`."""

    def _make_assets(self):
        return {band: f"s3://eodata/.../{band}.jp2" for band in ALL_BANDS}

    @patch("maji.download.boto3.client")
    @patch("maji.download._read_band_with_retry")
    @patch("maji.download.rasterio.open")
    def test_writes_geotiff(self, mock_rio_open, mock_read, mock_boto_client, tmp_path):
        """download_tile should create a multi-band GeoTIFF."""
        crs = CRS.from_epsg(32637)
        transform = Affine(10.0, 0, 500000, 0, -10.0, 10000000)

        mock_read.return_value = (
            np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint16),
            {"crs": crs, "transform": transform, "native_shape": (TARGET_HEIGHT, TARGET_WIDTH)},
        )

        # Mock the writer and make __exit__ create the file on disk
        # (simulating what rasterio does when closing)
        out_file = tmp_path / "37MFT" / "2026-01-10_S2L2A.tif"
        mock_writer = MagicMock()
        mock_rio_open.return_value.__enter__ = MagicMock(return_value=mock_writer)

        def _create_file(*args, **kwargs):
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"\x00" * 100)
            return False

        mock_rio_open.return_value.__exit__ = _create_file

        session = MagicMock()
        assets = self._make_assets()

        result = download_tile(
            scene_assets=assets,
            mgrs_tile="37MFT",
            scene_date="2026-01-10",
            data_dir=tmp_path,
            session=session,
            bands=ALL_BANDS,
            overwrite=True,
        )

        assert result == out_file
        # boto3.client should be called to create S3 client
        mock_boto_client.assert_called_once()
        # Should have read 7 bands (first 10m band for ref, then all 7 for writing)
        # Minimum: 1 ref read + 7 band reads = 8
        assert mock_read.call_count >= len(ALL_BANDS)
        # Verify the writer had write() called for each band
        assert mock_writer.write.call_count == len(ALL_BANDS)
        assert mock_writer.set_band_description.call_count == len(ALL_BANDS)

    @patch("maji.download.boto3.client")
    @patch("maji.download._read_band_with_retry")
    @patch("maji.download.rasterio.open")
    def test_skip_existing(self, mock_rio_open, mock_read, mock_boto_client, tmp_path):
        """Should skip download when file already exists and overwrite=False."""
        tile_dir = tmp_path / "37MFT"
        tile_dir.mkdir()
        existing = tile_dir / "2026-01-10_S2L2A.tif"
        existing.write_text("dummy")

        session = MagicMock()
        result = download_tile(
            scene_assets=self._make_assets(),
            mgrs_tile="37MFT",
            scene_date="2026-01-10",
            data_dir=tmp_path,
            session=session,
            overwrite=False,
        )

        assert result == existing
        mock_read.assert_not_called()
        # boto3 client should not be created when skipping
        mock_boto_client.assert_not_called()

    @patch("maji.download.boto3.client")
    @patch("maji.download._read_band_with_retry")
    @patch("maji.download.rasterio.open")
    def test_overwrite_existing(self, mock_rio_open, mock_read, mock_boto_client, tmp_path):
        """Should re-download when overwrite=True even if file exists."""
        tile_dir = tmp_path / "37MFT"
        tile_dir.mkdir()
        existing = tile_dir / "2026-01-10_S2L2A.tif"
        existing.write_text("dummy")

        crs = CRS.from_epsg(32637)
        transform = Affine(10.0, 0, 500000, 0, -10.0, 10000000)
        mock_read.return_value = (
            np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint16),
            {"crs": crs, "transform": transform, "native_shape": (TARGET_HEIGHT, TARGET_WIDTH)},
        )

        mock_writer = MagicMock()
        mock_rio_open.return_value.__enter__ = MagicMock(return_value=mock_writer)

        def _create_file(*args, **kwargs):
            existing.write_bytes(b"\x00" * 100)
            return False

        mock_rio_open.return_value.__exit__ = _create_file

        session = MagicMock()
        result = download_tile(
            scene_assets=self._make_assets(),
            mgrs_tile="37MFT",
            scene_date="2026-01-10",
            data_dir=tmp_path,
            session=session,
            overwrite=True,
        )

        assert result == existing
        assert mock_read.call_count >= len(ALL_BANDS)


# --- download_tiles -------------------------------------------------------


class TestDownloadTiles:
    """Tests for :func:`maji.download.download_tiles`."""

    @patch("maji.download.download_tile")
    def test_calls_download_tile_per_row(self, mock_dl_tile):
        """download_tiles should call download_tile for each row."""
        mock_dl_tile.return_value = Path("data/37MFT/2026-01-10_S2L2A.tif")

        scenes = pd.DataFrame([
            {"mgrs_tile": "37MFT", "datetime": pd.Timestamp("2026-01-10"), "assets": {"B03": "href"}},
            {"mgrs_tile": "36MZE", "datetime": pd.Timestamp("2026-01-11"), "assets": {"B03": "href"}},
        ])

        session = MagicMock()
        paths = download_tiles(scenes, Path("data"), session)

        assert len(paths) == 2
        assert mock_dl_tile.call_count == 2

    @patch("maji.download.download_tile")
    def test_max_workers_clamped(self, mock_dl_tile):
        """max_workers > 4 should emit a warning and be clamped."""
        mock_dl_tile.return_value = Path("data/37MFT/2026-01-10_S2L2A.tif")

        scenes = pd.DataFrame([
            {"mgrs_tile": "37MFT", "datetime": pd.Timestamp("2026-01-10"), "assets": {}},
        ])

        session = MagicMock()
        with pytest.warns(UserWarning, match="exceeds CDSE limit"):
            download_tiles(scenes, Path("data"), session, max_workers=10)
