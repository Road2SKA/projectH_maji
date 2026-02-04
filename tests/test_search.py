"""Tests for maji.search — STAC catalog search."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from shapely.geometry import box

from maji.search import (
    BANDS_OF_INTEREST,
    _extract_band_assets,
    _extract_mgrs_tile,
    search_scenes,
    select_best_scenes,
)


# --- _extract_mgrs_tile ------------------------------------------------


class TestExtractMgrsTile:
    """Tests for :func:`maji.search._extract_mgrs_tile`."""

    def test_grid_code(self):
        """Extract tile from standard grid:code property."""
        props = {"grid:code": "MGRS-37MFT"}
        assert _extract_mgrs_tile(props) == "37MFT"

    def test_fallback_to_title(self):
        """Fall back to parsing the title when grid:code is absent."""
        props = {"title": "S2A_MSIL2A_20260110T073251_N0500_R049_T36MZE_20260110T100000"}
        assert _extract_mgrs_tile(props) == "36MZE"

    def test_fallback_to_id(self):
        """Fall back to parsing the id when grid:code and title are absent."""
        props = {"id": "S2A_MSIL2A_20260110T073251_N0500_R049_T36MZE_20260110T100000"}
        assert _extract_mgrs_tile(props) == "36MZE"

    def test_missing_returns_empty(self):
        """Return empty string when no tile code can be determined."""
        props = {"title": "no-tile-info-here"}
        assert _extract_mgrs_tile(props) == ""


# --- _extract_band_assets -----------------------------------------------


class TestExtractBandAssets:
    """Tests for :func:`maji.search._extract_band_assets`."""

    def test_complete_assets(self, sample_stac_item):
        """All BANDS_OF_INTEREST are extracted from a complete asset dict."""
        result = _extract_band_assets(sample_stac_item.assets)
        assert set(result.keys()) == set(BANDS_OF_INTEREST)
        assert result["B03"] == "s3://eodata/.../B03_10m.jp2"
        assert result["SCL"] == "s3://eodata/.../SCL_20m.jp2"

    def test_b08_does_not_match_b8a(self):
        """B08 key must not accidentally match B8A_20m."""
        assets = {
            "B8A_20m": SimpleNamespace(href="s3://eodata/.../B8A_20m.jp2"),
        }
        result = _extract_band_assets(assets, bands=["B08"])
        assert "B08" not in result

    def test_b8a_matches_correctly(self):
        """B08 and B8A are resolved to the correct assets."""
        assets = {
            "B08_10m": SimpleNamespace(href="s3://eodata/.../B08_10m.jp2"),
            "B8A_20m": SimpleNamespace(href="s3://eodata/.../B8A_20m.jp2"),
        }
        result = _extract_band_assets(assets, bands=["B08", "B8A"])
        assert result["B08"] == "s3://eodata/.../B08_10m.jp2"
        assert result["B8A"] == "s3://eodata/.../B8A_20m.jp2"

    def test_dict_assets(self):
        """Assets can be plain dicts (not just SimpleNamespace objects)."""
        assets = {
            "B03_10m": {"href": "s3://eodata/.../B03_10m.jp2"},
        }
        result = _extract_band_assets(assets, bands=["B03"])
        assert result["B03"] == "s3://eodata/.../B03_10m.jp2"


# --- search_scenes -------------------------------------------------------


def _make_item(item_id, grid_code, cloud_cover, dt_str, geometry=None):
    """Create a mock STAC item for search-layer tests.

    Parameters
    ----------
    item_id : str
        STAC item identifier.
    grid_code : str
        Value for ``properties["grid:code"]`` (e.g. ``"MGRS-37MFT"``).
    cloud_cover : float
        Cloud cover percentage (0–100).
    dt_str : str
        ISO-8601 datetime string.
    geometry : dict or None, optional
        GeoJSON geometry; defaults to a 1x1 degree box.

    Returns
    -------
    types.SimpleNamespace
        Object mimicking a :class:`pystac.Item`.
    """
    if geometry is None:
        geometry = box(36.0, -2.0, 37.0, -1.0).__geo_interface__

    assets = {
        "B03_10m": SimpleNamespace(href="s3://eodata/.../B03_10m.jp2"),
        "B04_10m": SimpleNamespace(href="s3://eodata/.../B04_10m.jp2"),
        "B08_10m": SimpleNamespace(href="s3://eodata/.../B08_10m.jp2"),
        "B8A_20m": SimpleNamespace(href="s3://eodata/.../B8A_20m.jp2"),
        "B11_20m": SimpleNamespace(href="s3://eodata/.../B11_20m.jp2"),
        "B12_20m": SimpleNamespace(href="s3://eodata/.../B12_20m.jp2"),
        "SCL_20m": SimpleNamespace(href="s3://eodata/.../SCL_20m.jp2"),
    }

    return SimpleNamespace(
        id=item_id,
        properties={
            "datetime": dt_str,
            "eo:cloud_cover": cloud_cover,
            "grid:code": grid_code,
        },
        geometry=geometry,
        assets=assets,
    )


class TestSearchScenes:
    """Tests for :func:`maji.search.search_scenes`."""

    @patch("maji.search.Client")
    def test_returns_dataframe(self, mock_client_cls):
        """Successful search returns a sorted GeoDataFrame with expected columns."""
        items = [
            _make_item("item1", "MGRS-37MFT", 10.0, "2026-01-10T07:00:00Z"),
            _make_item("item2", "MGRS-36MZE", 20.0, "2026-01-11T07:00:00Z"),
        ]
        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.item_collection.return_value = items
        mock_catalog.search.return_value = mock_search
        mock_client_cls.open.return_value = mock_catalog

        df = search_scenes(
            bbox=(36.0, -2.0, 37.0, -1.0),
            start="2026-01-01",
            end="2026-01-15",
        )

        assert len(df) == 2
        assert list(df.columns) >= ["scene_id", "mgrs_tile", "datetime", "cloud_cover", "geometry", "assets"]
        assert set(df["mgrs_tile"]) == {"37MFT", "36MZE"}

    @patch("maji.search.Client")
    def test_skips_items_missing_bands(self, mock_client_cls):
        """Items with incomplete band assets should be skipped."""
        good_item = _make_item("good", "MGRS-37MFT", 10.0, "2026-01-10T07:00:00Z")

        # Item missing all assets except B03
        bad_item = SimpleNamespace(
            id="bad",
            properties={
                "datetime": "2026-01-10T07:00:00Z",
                "eo:cloud_cover": 5.0,
                "grid:code": "MGRS-36MZE",
            },
            geometry=box(36.0, -2.0, 37.0, -1.0).__geo_interface__,
            assets={"B03_10m": SimpleNamespace(href="s3://eodata/.../B03_10m.jp2")},
        )

        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.item_collection.return_value = [good_item, bad_item]
        mock_catalog.search.return_value = mock_search
        mock_client_cls.open.return_value = mock_catalog

        df = search_scenes(
            bbox=(36.0, -2.0, 37.0, -1.0),
            start="2026-01-01",
            end="2026-01-15",
        )

        assert len(df) == 1
        assert df.iloc[0]["scene_id"] == "good"

    @patch("maji.search.Client")
    def test_empty_results(self, mock_client_cls):
        """Empty STAC response returns an empty GeoDataFrame with correct schema."""
        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.item_collection.return_value = []
        mock_catalog.search.return_value = mock_search
        mock_client_cls.open.return_value = mock_catalog

        df = search_scenes(
            bbox=(36.0, -2.0, 37.0, -1.0),
            start="2026-01-01",
            end="2026-01-15",
        )

        assert len(df) == 0
        assert "scene_id" in df.columns


# --- select_best_scenes --------------------------------------------------


class TestSelectBestScenes:
    """Tests for :func:`maji.search.select_best_scenes`."""

    def _make_scenes_df(self):
        """Create a sample GeoDataFrame with multiple scenes per tile."""
        import geopandas as gpd

        rows = [
            {"scene_id": "a1", "mgrs_tile": "37MFT", "datetime": pd.Timestamp("2026-01-05"), "cloud_cover": 30.0, "geometry": box(0, 0, 1, 1), "assets": {}},
            {"scene_id": "a2", "mgrs_tile": "37MFT", "datetime": pd.Timestamp("2026-01-10"), "cloud_cover": 10.0, "geometry": box(0, 0, 1, 1), "assets": {}},
            {"scene_id": "b1", "mgrs_tile": "36MZE", "datetime": pd.Timestamp("2026-01-06"), "cloud_cover": 20.0, "geometry": box(0, 0, 1, 1), "assets": {}},
            {"scene_id": "b2", "mgrs_tile": "36MZE", "datetime": pd.Timestamp("2026-01-12"), "cloud_cover": 25.0, "geometry": box(0, 0, 1, 1), "assets": {}},
        ]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def test_least_cloudy(self):
        """least_cloudy strategy picks the lowest cloud_cover per tile."""
        df = self._make_scenes_df()
        best = select_best_scenes(df, strategy="least_cloudy")
        assert len(best) == 2
        tile_scenes = best.set_index("mgrs_tile")
        assert tile_scenes.loc["37MFT", "scene_id"] == "a2"  # cloud_cover=10
        assert tile_scenes.loc["36MZE", "scene_id"] == "b1"  # cloud_cover=20

    def test_most_recent(self):
        """most_recent strategy picks the latest datetime per tile."""
        df = self._make_scenes_df()
        best = select_best_scenes(df, strategy="most_recent")
        assert len(best) == 2
        tile_scenes = best.set_index("mgrs_tile")
        assert tile_scenes.loc["37MFT", "scene_id"] == "a2"  # Jan 10
        assert tile_scenes.loc["36MZE", "scene_id"] == "b2"  # Jan 12

    def test_all(self):
        """'all' strategy returns every row unfiltered."""
        df = self._make_scenes_df()
        result = select_best_scenes(df, strategy="all")
        assert len(result) == 4

    def test_invalid_strategy(self):
        """Unrecognised strategy raises ValueError."""
        df = self._make_scenes_df()
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_best_scenes(df, strategy="invalid")
