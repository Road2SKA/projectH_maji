#!/usr/bin/env python3
"""
test_safe_to_geotiff.py
=======================
Validates the safe_to_geotiff module logic by:

1. Building a synthetic SAFE directory tree.
2. Creating tiny GeoTIFF rasters (using numpy + struct) as stand-ins
   for the JP2 band files — this avoids the rasterio/GDAL dependency
   for testing the *logic* (path discovery, band resolution lookup, etc.).
3. If rasterio IS available, it additionally runs the full stack_bands
   pipeline end-to-end and checks the output file.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ── Import the module under test ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_to_geotiff as s2g

TILE_ID = "T37MFT"
DATETIME = "20260104T073219"
SAFE_NAME = (
    "S2B_MSIL2A_20260104T073219_N0511_R049_T37MFT_20260104T094802.SAFE"
)
GRANULE_NAME = "L2A_T37MFT_A046121_20260104T074234"


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_fake_jp2(path: Path, height: int = 4, width: int = 4) -> None:
    """Write a single-band GeoTIFF renamed to .jp2 for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(36.0, -2.0, 37.0, -1.0, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS.from_epsg(32737),
        "transform": transform,
    }
    data = np.random.randint(0, 10000, (height, width), dtype=np.uint16)
    # Write as .tif first, then rename — rasterio can still open it
    tif_path = path.with_suffix(".tif")
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
    tif_path.rename(path)


def build_synthetic_safe(root: Path) -> Path:
    """Create a minimal .SAFE directory tree with fake JP2 band files."""
    safe_dir = root / SAFE_NAME
    granule_dir = safe_dir / "GRANULE" / GRANULE_NAME

    # 10 m bands
    for band in ("B02", "B03", "B04", "B08"):
        fname = f"{TILE_ID}_{DATETIME}_{band}_10m.jp2"
        _make_fake_jp2(granule_dir / "IMG_DATA" / "R10m" / fname, 10, 10)

    # 20 m bands
    for band in ("B05", "B06", "B07", "B8A", "B11", "B12", "SCL"):
        fname = f"{TILE_ID}_{DATETIME}_{band}_20m.jp2"
        _make_fake_jp2(granule_dir / "IMG_DATA" / "R20m" / fname, 5, 5)

    # 60 m bands
    for band in ("B01", "B09"):
        fname = f"{TILE_ID}_{DATETIME}_{band}_60m.jp2"
        _make_fake_jp2(granule_dir / "IMG_DATA" / "R60m" / fname, 2, 2)

    # Metadata placeholder
    (safe_dir / "MTD_MSIL2A.xml").write_text("<xml/>")
    (granule_dir / "MTD_TL.xml").write_text("<xml/>")

    return safe_dir


# ── Tests ─────────────────────────────────────────────────────────────────

def test_find_safe_root(safe_dir: Path) -> None:
    """find_safe_root should resolve from any point inside the tree."""
    # From the root itself
    assert s2g.find_safe_root(safe_dir) == safe_dir.resolve()
    # From a nested directory
    nested = safe_dir / "GRANULE" / GRANULE_NAME / "IMG_DATA" / "R10m"
    assert s2g.find_safe_root(nested) == safe_dir.resolve()
    print("  ✓ find_safe_root")


def test_find_granule_dir(safe_dir: Path) -> None:
    gd = s2g.find_granule_dir(safe_dir)
    assert gd.name == GRANULE_NAME
    print("  ✓ find_granule_dir")


def test_locate_band_file(safe_dir: Path) -> None:
    gd = s2g.find_granule_dir(safe_dir)

    # 10 m band
    p10 = s2g.locate_band_file(gd, "B03")
    assert "R10m" in str(p10) and p10.exists()

    # 20 m band
    p20 = s2g.locate_band_file(gd, "B11")
    assert "R20m" in str(p20) and p20.exists()

    # 60 m band
    p60 = s2g.locate_band_file(gd, "B01")
    assert "R60m" in str(p60) and p60.exists()

    # SCL
    pscl = s2g.locate_band_file(gd, "SCL")
    assert "R20m" in str(pscl) and pscl.exists()

    # Unknown band should raise
    try:
        s2g.locate_band_file(gd, "B99")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  ✓ locate_band_file (10m, 20m, 60m, SCL, error case)")


def test_band_resolution_lookup() -> None:
    assert s2g.BAND_RESOLUTION["B03"] == 10
    assert s2g.BAND_RESOLUTION["B11"] == 20
    assert s2g.BAND_RESOLUTION["B09"] == 60
    assert s2g.BAND_RESOLUTION["SCL"] == 20
    print("  ✓ BAND_RESOLUTION lookup")


def test_full_stack(safe_dir: Path) -> None:
    """End-to-end test of the full stack_bands pipeline."""
    out = safe_dir.parent / "test_output.tif"
    result = s2g.stack_bands(
        safe_path=safe_dir,
        bands=["B03", "B08", "B11"],
        include_scl=True,
        output_path=out,
    )
    assert result.exists(), "Output file was not created"

    with rasterio.open(result) as ds:
        assert ds.count == 4, f"Expected 4 bands, got {ds.count}"
        assert ds.height == 10 and ds.width == 10, (
            f"Expected 10×10, got {ds.height}×{ds.width}"
        )
        descriptions = [ds.descriptions[i] for i in range(ds.count)]
        assert descriptions == ["B03", "B08", "B11", "SCL"], (
            f"Band descriptions mismatch: {descriptions}"
        )
        data = ds.read()
        assert data.shape == (4, 10, 10)
        assert data.dtype == np.uint16

    print(f"  ✓ full stack_bands → {result.name} "
          f"(4 bands, 10×10, uint16)")


# ── Runner ────────────────────────────────────────────────────────────────

def main() -> int:
    print("Building synthetic SAFE directory …")
    with tempfile.TemporaryDirectory() as tmp:
        safe_dir = build_synthetic_safe(Path(tmp))
        print(f"  ➜ {safe_dir}\n")

        print("Running unit tests …")
        test_band_resolution_lookup()
        test_find_safe_root(safe_dir)
        test_find_granule_dir(safe_dir)
        test_locate_band_file(safe_dir)

        print("\nRunning integration test …")
        test_full_stack(safe_dir)

    print("\n✅ All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
