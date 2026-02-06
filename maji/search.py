"""CDSE STAC catalog search for Sentinel-2 L2A scenes.

Searches the free CDSE STAC endpoint (no auth needed) and returns
a normalised GeoDataFrame with scene metadata and S3 asset paths.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import geopandas as gpd
import pandas as pd
from pystac_client import Client
from shapely.geometry import shape

from maji.constants import DEFAULT_DOWNLOAD_BANDS

logger = logging.getLogger(__name__)

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
COLLECTION = "sentinel-2-l2a"

# Bands we care about (model bands + cloud mask)
# Imported from constants, with backward-compatible alias
BANDS_OF_INTEREST = DEFAULT_DOWNLOAD_BANDS


def _extract_mgrs_tile(properties: dict) -> str:
    """Extract MGRS tile ID from STAC item properties.

    CDSE uses ``grid:code`` with format ``MGRS-37MFT``.
    Falls back to parsing the product name if grid:code is absent.

    Parameters
    ----------
    properties : dict
        STAC item ``properties`` dictionary.

    Returns
    -------
    str
        Five-character MGRS tile code (e.g. ``"37MFT"``), or an empty
        string if no tile code can be determined.

    Notes
    -----
    The primary extraction path reads the ``grid:code`` property and
    strips the ``MGRS-`` prefix.  When ``grid:code`` is missing (some
    older catalog entries), the function falls back to scanning
    underscore-delimited segments of the ``title`` or ``id`` property
    for a six-character token starting with ``T`` (e.g. ``T36MZE``).
    """
    grid_code = properties.get("grid:code", "")
    if grid_code.startswith("MGRS-"):
        return grid_code[5:]  # Strip "MGRS-" prefix

    # Fallback: parse from product/scene ID
    # S2A_MSIL2A_20260110T..._T36MZE_... → "36MZE"
    scene_id = properties.get("title", "") or properties.get("id", "")
    for part in scene_id.split("_"):
        if part.startswith("T") and len(part) == 6:
            return part[1:]  # Strip leading "T"

    return ""


def _extract_band_assets(
    assets: dict[str, Any],
    bands: list[str] | None = None,
) -> dict[str, str]:
    """Extract S3 hrefs for requested bands from STAC item assets.

    Parameters
    ----------
    assets : dict[str, Any]
        STAC item ``assets`` mapping.  Values may be ``pystac.Asset``
        objects (with an ``href`` attribute) or plain dicts with an
        ``"href"`` key.
    bands : list[str] or None, optional
        Band names to extract (default: :data:`BANDS_OF_INTEREST`).

    Returns
    -------
    dict[str, str]
        ``{band_name: s3_href}`` for every band that was found.

    Notes
    -----
    Asset keys in CDSE follow the pattern ``<band>_<resolution>``
    (e.g. ``B03_10m``, ``B8A_20m``).  Matching uses
    ``key.startswith(band + "_")`` so that ``B08`` does **not**
    accidentally match ``B8A_20m``.  If no key-prefix match is found
    the function falls back to searching ``href`` paths for
    ``_<band>_`` or ``_<band>.`` substrings.
    """
    if bands is None:
        bands = BANDS_OF_INTEREST

    result: dict[str, str] = {}
    for band in bands:
        band_upper = band.upper()
        # Try key-prefix match first (e.g. "B03_10m", "B8A_20m", "SCL_20m")
        for key, asset in assets.items():
            href = getattr(asset, "href", None) or asset.get("href", "")
            key_upper = key.upper()
            # Match "B03_..." but not "B08" matching "B8A_..."
            if key_upper.startswith(band_upper + "_") or key_upper == band_upper:
                result[band] = href
                break
        else:
            # Fallback: match on href path
            for key, asset in assets.items():
                href = getattr(asset, "href", None) or asset.get("href", "")
                if f"_{band}_" in href or f"_{band}." in href:
                    result[band] = href
                    break

    return result


def search_scenes(
    bbox: tuple[float, float, float, float],
    start: str | datetime,
    end: str | datetime,
    max_cloud: float = 40.0,
    max_items: int = 100,
) -> gpd.GeoDataFrame:
    """Search CDSE STAC for Sentinel-2 L2A scenes.

    Queries the free Copernicus Data Space Ecosystem STAC endpoint
    (no authentication required) and returns a normalised
    :class:`~geopandas.GeoDataFrame` with scene metadata and S3
    asset paths.

    Parameters
    ----------
    bbox : tuple of float
        ``(west, south, east, north)`` in EPSG:4326 decimal degrees.
    start, end : str or datetime
        Date range as ISO-8601 strings (``"YYYY-MM-DD"``) or
        :class:`~datetime.datetime` objects.
    max_cloud : float, optional
        Maximum cloud cover percentage, 0–100 (default 40).
    max_items : int, optional
        Maximum number of STAC items to return (default 100).

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``scene_id``, ``mgrs_tile``, ``datetime``,
        ``cloud_cover``, ``geometry``, ``assets``.  Sorted by
        ``(mgrs_tile, datetime)``.  CRS is EPSG:4326.

    Raises
    ------
    pystac_client.exceptions.APIError
        If the STAC endpoint is unreachable or returns a server error.

    Notes
    -----
    * STAC endpoint: ``https://stac.dataspace.copernicus.eu/v1/``
    * Collection: ``sentinel-2-l2a``
    * Pagination is handled internally by ``pystac_client``; the
      ``max_items`` parameter caps the total number of results.
    * Items whose MGRS tile cannot be determined or that are missing
      any of the :data:`BANDS_OF_INTEREST` are silently dropped
      (a warning is logged).

    Examples
    --------
    >>> from maji.search import search_scenes
    >>> df = search_scenes(
    ...     bbox=(36.0, -2.0, 37.0, -1.0),
    ...     start="2026-01-01",
    ...     end="2026-01-31",
    ...     max_cloud=25.0,
    ... )
    >>> df.columns.tolist()
    ['scene_id', 'mgrs_tile', 'datetime', 'cloud_cover', 'geometry', 'assets']
    """
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=max_items,
    )

    items = search.item_collection()
    logger.info("STAC search returned %d items", len(items))

    rows: list[dict] = []
    for item in items:
        props = item.properties
        mgrs_tile = _extract_mgrs_tile(props)
        if not mgrs_tile:
            logger.warning("Could not extract MGRS tile from item %s", item.id)
            continue

        band_assets = _extract_band_assets(item.assets)
        if len(band_assets) < len(BANDS_OF_INTEREST):
            missing = set(BANDS_OF_INTEREST) - set(band_assets.keys())
            logger.warning(
                "Item %s missing bands: %s — skipping", item.id, missing
            )
            continue

        rows.append(
            {
                "scene_id": item.id,
                "mgrs_tile": mgrs_tile,
                "datetime": pd.Timestamp(props.get("datetime")),
                "cloud_cover": props.get("eo:cloud_cover", 100.0),
                "geometry": shape(item.geometry),
                "assets": band_assets,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["scene_id", "mgrs_tile", "datetime", "cloud_cover", "geometry", "assets"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    df = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    df = df.sort_values(["mgrs_tile", "datetime"]).reset_index(drop=True)
    return df


def select_best_scenes(
    scenes: gpd.GeoDataFrame,
    strategy: str = "least_cloudy",
) -> gpd.GeoDataFrame:
    """For each MGRS tile, select the best scene.

    Parameters
    ----------
    scenes : geopandas.GeoDataFrame
        Output of :func:`search_scenes`.
    strategy : str, optional
        Selection strategy (default ``"least_cloudy"``):

        * ``"least_cloudy"`` — scene with the lowest ``cloud_cover``
          per tile.
        * ``"most_recent"`` — scene with the latest ``datetime`` per
          tile.
        * ``"all"`` — no filtering; return every scene.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered DataFrame with one row per MGRS tile (or all rows
        if ``strategy="all"``).  Index is reset.

    Raises
    ------
    ValueError
        If *strategy* is not one of the recognised values.

    Examples
    --------
    >>> best = select_best_scenes(scenes, strategy="least_cloudy")
    >>> best["mgrs_tile"].is_unique
    True
    """
    if strategy == "all":
        return scenes

    if strategy == "least_cloudy":
        idx = scenes.groupby("mgrs_tile")["cloud_cover"].idxmin()
    elif strategy == "most_recent":
        idx = scenes.groupby("mgrs_tile")["datetime"].idxmax()
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return scenes.loc[idx].reset_index(drop=True)
