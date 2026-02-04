"""Shared test fixtures for the maji test suite."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from shapely.geometry import box


def _make_asset(href: str) -> SimpleNamespace:
    """Create a minimal object that looks like a pystac Asset.

    Parameters
    ----------
    href : str
        S3 href to assign to the fake asset.

    Returns
    -------
    types.SimpleNamespace
        Object with a single ``href`` attribute, mirroring the
        interface of :class:`pystac.Asset`.
    """
    return SimpleNamespace(href=href)


@pytest.fixture()
def sample_stac_item():
    """A mock STAC item with 7 band assets and standard properties.

    Returns
    -------
    types.SimpleNamespace
        Fake STAC item with ``id``, ``properties``, ``geometry``, and
        ``assets`` attributes.  ``properties["grid:code"]`` is set to
        ``"MGRS-37MFT"``.
    """
    assets = {
        "B03_10m": _make_asset("s3://eodata/.../B03_10m.jp2"),
        "B04_10m": _make_asset("s3://eodata/.../B04_10m.jp2"),
        "B08_10m": _make_asset("s3://eodata/.../B08_10m.jp2"),
        "B8A_20m": _make_asset("s3://eodata/.../B8A_20m.jp2"),
        "B11_20m": _make_asset("s3://eodata/.../B11_20m.jp2"),
        "B12_20m": _make_asset("s3://eodata/.../B12_20m.jp2"),
        "SCL_20m": _make_asset("s3://eodata/.../SCL_20m.jp2"),
    }

    geometry = box(36.0, -2.0, 37.0, -1.0).__geo_interface__

    item = SimpleNamespace(
        id="S2A_MSIL2A_20260110T073251_N0500_R049_T37MFT_20260110T100000",
        properties={
            "datetime": "2026-01-10T07:32:51Z",
            "eo:cloud_cover": 12.5,
            "grid:code": "MGRS-37MFT",
        },
        geometry=geometry,
        assets=assets,
    )
    return item


@pytest.fixture()
def sample_stac_item_no_grid():
    """A mock STAC item without ``grid:code`` but with tile in product name.

    Returns
    -------
    types.SimpleNamespace
        Fake STAC item whose ``properties`` lack ``grid:code``.  The
        MGRS tile ``36MZE`` can only be extracted from the ``title``
        field via the fallback parser.
    """
    assets = {
        "B03_10m": _make_asset("s3://eodata/.../B03_10m.jp2"),
        "B04_10m": _make_asset("s3://eodata/.../B04_10m.jp2"),
        "B08_10m": _make_asset("s3://eodata/.../B08_10m.jp2"),
        "B8A_20m": _make_asset("s3://eodata/.../B8A_20m.jp2"),
        "B11_20m": _make_asset("s3://eodata/.../B11_20m.jp2"),
        "B12_20m": _make_asset("s3://eodata/.../B12_20m.jp2"),
        "SCL_20m": _make_asset("s3://eodata/.../SCL_20m.jp2"),
    }

    geometry = box(36.0, -2.0, 37.0, -1.0).__geo_interface__

    item = SimpleNamespace(
        id="S2A_MSIL2A_20260110T073251_N0500_R049_T36MZE_20260110T100000",
        properties={
            "datetime": "2026-01-10T07:32:51Z",
            "eo:cloud_cover": 25.0,
            "title": "S2A_MSIL2A_20260110T073251_N0500_R049_T36MZE_20260110T100000",
        },
        geometry=geometry,
        assets=assets,
    )
    return item
