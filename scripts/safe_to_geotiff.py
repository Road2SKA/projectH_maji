#!/usr/bin/env python3
"""
safe_to_geotiff.py
==================
Extract specific Sentinel-2 L2A bands from a .SAFE product directory
and produce a single, multi-band Cloud-Optimised GeoTIFF (COG) suitable
for ML workflows.

Features
--------
* Selects user-specified bands (default: flood-mapping set B03, B08, B11).
* Resamples coarser bands (20 m / 60 m) to the 10 m grid so every layer
  aligns pixel-for-pixel.
* Optionally extracts the Scene Classification Layer (SCL) as an extra
  band for cloud / water masking.
* Writes a single multi-band COG with band descriptions in metadata.

Usage
-----
    python safe_to_geotiff.py <path_to.SAFE> [--bands B03 B08 B11] \
                              [--include-scl] [--output stacked.tif]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

# ---------------------------------------------------------------------------
# Band resolution lookup (native metres)
# ---------------------------------------------------------------------------
BAND_RESOLUTION: dict[str, int] = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20,
    "B06": 20, "B07": 20, "B08": 10, "B8A": 20, "B09": 60,
    "B10": 60, "B11": 20, "B12": 20, "SCL": 20,
}

# Default band set for flood mapping (Green, NIR, SWIR-1)
DEFAULT_BANDS = ["B03", "B08", "B11"]
TARGET_RES = 10  # metres – resample everything to 10 m


def find_safe_root(path: Path) -> Path:
    """Accept either a .SAFE dir or any file/dir inside it."""
    p = path.resolve()
    while p != p.parent:
        if p.suffix == ".SAFE" and p.is_dir():
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not locate a .SAFE directory from {path}")


def find_granule_dir(safe_root: Path) -> Path:
    """Return the single granule directory inside GRANULE/."""
    granules = [p for p in (safe_root / "GRANULE").iterdir() if p.is_dir()]
    if len(granules) != 1:
        raise RuntimeError(
            f"Expected 1 granule directory, found {len(granules)}"
        )
    return granules[0]


def locate_band_file(granule_dir: Path, band: str) -> Path:
    """Find the JP2 file for a given band at its native resolution."""
    native_res = BAND_RESOLUTION.get(band)
    if native_res is None:
        raise ValueError(f"Unknown band: {band}")

    res_dir = granule_dir / "IMG_DATA" / f"R{native_res}m"
    pattern = f"*_{band}_{native_res}m.jp2"
    matches = list(res_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern} in {res_dir}"
        )
    return matches[0]


def read_band_resampled(
    band_path: Path,
    target_height: int,
    target_width: int,
    target_transform: "Affine",
    target_crs: "rasterio.crs.CRS",
) -> np.ndarray:
    """Read a single band and resample to the target 10 m grid."""
    with rasterio.open(band_path) as src:
        if src.height == target_height and src.width == target_width:
            return src.read(1)

        data = src.read(
            1,
            out_shape=(target_height, target_width),
            resampling=Resampling.bilinear,
        )
        return data


def stack_bands(
    safe_path: str | Path,
    bands: list[str],
    include_scl: bool = False,
    output_path: str | Path | None = None,
) -> Path:
    """Main entry-point: stack bands into a COG.

    Parameters
    ----------
    safe_path : path to the .SAFE directory (or anything inside it).
    bands : list of band identifiers, e.g. ["B03", "B08", "B11"].
    include_scl : if True, append the SCL band as the last layer.
    output_path : destination GeoTIFF.  Defaults to
                  ``<safe_name>_stacked.tif`` next to the .SAFE dir.

    Returns
    -------
    Path to the written GeoTIFF.
    """
    safe_root = find_safe_root(Path(safe_path))
    granule_dir = find_granule_dir(safe_root)

    all_bands = list(bands)
    if include_scl:
        all_bands.append("SCL")

    # -- Reference grid: use a 10 m band for shape / transform / CRS --------
    ref_band = next((b for b in all_bands if BAND_RESOLUTION[b] == 10), None)
    if ref_band is None:
        # Fall back to first band available at 10 m in the product
        ref_band = "B02"
    ref_path = locate_band_file(granule_dir, ref_band)

    with rasterio.open(ref_path) as ref:
        target_height = ref.height
        target_width = ref.width
        target_transform = ref.transform
        target_crs = ref.crs
        target_dtype = ref.dtypes[0]

    # -- Read & resample every requested band --------------------------------
    arrays: list[np.ndarray] = []
    band_names: list[str] = []
    for band in all_bands:
        bp = locate_band_file(granule_dir, band)
        arr = read_band_resampled(
            bp, target_height, target_width, target_transform, target_crs
        )
        arrays.append(arr)
        band_names.append(band)
        print(f"  ✓ {band} ({BAND_RESOLUTION[band]}m → {TARGET_RES}m): "
              f"{bp.name}")

    stack = np.stack(arrays)  # shape: (n_bands, H, W)

    # -- Write COG -----------------------------------------------------------
    if output_path is None:
        output_path = safe_root.parent / f"{safe_root.stem}_stacked.tif"
    output_path = Path(output_path)

    profile = {
        "driver": "GTiff",
        "dtype": target_dtype,
        "width": target_width,
        "height": target_height,
        "count": len(all_bands),
        "crs": target_crs,
        "transform": target_transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(stack[i - 1], i)
            dst.set_band_description(i, name)

    print(f"\n  ➜ Wrote {output_path}  "
          f"({stack.shape[0]} bands, {target_height}×{target_width}, "
          f"{output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack Sentinel-2 L2A bands from a .SAFE product "
                    "into a multi-band GeoTIFF."
    )
    parser.add_argument("safe_path", help="Path to the .SAFE directory")
    parser.add_argument(
        "--bands", nargs="+", default=DEFAULT_BANDS,
        help="Band identifiers to include (default: B03 B08 B11)"
    )
    parser.add_argument(
        "--include-scl", action="store_true",
        help="Append the Scene Classification Layer as an extra band"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output GeoTIFF path (default: <safe_name>_stacked.tif)"
    )
    args = parser.parse_args()

    stack_bands(
        safe_path=args.safe_path,
        bands=args.bands,
        include_scl=args.include_scl,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

