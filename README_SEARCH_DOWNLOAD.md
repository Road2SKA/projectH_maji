# Sentinel-2 Search & Download Pipeline

## Overview

The `maji` package provides an end-to-end pipeline for discovering and
downloading Sentinel-2 Level-2A (L2A) imagery from the **Copernicus Data
Space Ecosystem (CDSE)**.  It searches the free CDSE STAC catalog,
selects the best available scenes per MGRS tile, and downloads them as
multi-band Cloud-Optimized GeoTIFFs ready for downstream analysis.

**Why CDSE S3 (not Sentinel Hub)?**  CDSE exposes the full Sentinel-2
archive as JP2 files in an S3-compatible object store.  This allows
band-level random access via `rasterio`'s `/vsis3/` driver without
needing a commercial Sentinel Hub subscription or per-request processing
fees.

---

## Architecture

```
 .env
  |
  v
Settings (maji/config.py)
  |
  |--- credentials + paths
  v
search_scenes(bbox, start, end, ...)        [maji/search.py]
  |
  |--- GeoDataFrame (scene_id, mgrs_tile, datetime, cloud_cover, geometry, assets)
  v
select_best_scenes(scenes, strategy=...)    [maji/search.py]
  |
  |--- filtered GeoDataFrame (1 row per tile)
  v
create_s3_session(access_key, secret_key)   [maji/download.py]
  |
  |--- AWSSession
  v
download_tiles(scenes, data_dir, session)   [maji/download.py]
  |
  |--- for each scene: download_tile() -> read bands from S3 -> stack -> write GeoTIFF
  v
data/<mgrs_tile>/<date>_S2L2A.tif           [7-band uint16 COG]
```

---

## Configuration (`maji/config.py`)

The `Settings` class uses `pydantic-settings` to load configuration from
environment variables or a `.env` file at the project root.

| Field                  | Type   | Default    | Description                                      |
|------------------------|--------|------------|--------------------------------------------------|
| `cdse_client_id`       | `str`  | `""`       | OAuth2 client ID (for authenticated STAC queries) |
| `cdse_client_secret`   | `str`  | `""`       | OAuth2 client secret                             |
| `cdse_s3_access_key`   | `str`  | `""`       | S3 access key for the CDSE eodata bucket         |
| `cdse_s3_secret_key`   | `str`  | `""`       | S3 secret key                                    |
| `data_dir`             | `Path` | `data/`    | Root directory for downloaded satellite data      |
| `output_dir`           | `Path` | `output/`  | Root directory for model outputs                 |
| `device`               | `str`  | `"cpu"`    | Compute device (`"cpu"` or `"cuda"`)             |

See [`env.example`](env.example) for a fully-commented template.

---

## Search Workflow (`maji/search.py`)

### STAC Endpoint

The pipeline queries the **free** CDSE STAC catalog at:

```
https://stac.dataspace.copernicus.eu/v1/
```

No authentication is required.  The `sentinel-2-l2a` collection is
searched.

### Query Parameters

- **bbox** — bounding box in EPSG:4326 `(west, south, east, north)`
- **datetime** — ISO range `start/end`
- **eo:cloud_cover** — `< max_cloud` filter
- **max_items** — pagination cap (handled internally by `pystac_client`)

### MGRS Tile Extraction

1. **Primary**: read `properties["grid:code"]` and strip the `MGRS-`
   prefix (e.g. `MGRS-37MFT` -> `37MFT`).
2. **Fallback**: scan underscore-delimited segments of the `title` or
   `id` for a 6-character token starting with `T` (e.g. `T36MZE` ->
   `36MZE`).

### Band-Asset Discovery

Asset keys in CDSE follow the `<band>_<resolution>` pattern (e.g.
`B03_10m`, `B8A_20m`).  Matching uses `key.startswith(band + "_")` to
avoid collisions — `B08` must **not** match `B8A_20m`.

Bands of interest: `B03`, `B04`, `B08`, `B8A`, `B11`, `B12`, `SCL`.

### DataFrame Schema

| Column        | Type           | Description                          |
|---------------|----------------|--------------------------------------|
| `scene_id`    | `str`          | STAC item ID                         |
| `mgrs_tile`   | `str`          | 5-char MGRS tile code                |
| `datetime`    | `Timestamp`    | Acquisition timestamp                |
| `cloud_cover` | `float`        | Cloud cover percentage (0-100)       |
| `geometry`    | `Polygon`      | Scene footprint (EPSG:4326)          |
| `assets`      | `dict`         | `{band: s3_href}` mapping            |

### Selection Strategies

`select_best_scenes(scenes, strategy=...)` picks one scene per tile:

| Strategy        | Logic                                      |
|-----------------|--------------------------------------------|
| `least_cloudy`  | Minimum `cloud_cover` per `mgrs_tile`      |
| `most_recent`   | Maximum `datetime` per `mgrs_tile`         |
| `all`           | No filtering — return every scene          |

---

## Download Workflow (`maji/download.py`)

### S3 Session Setup

`create_s3_session()` returns a `rasterio.session.AWSSession` configured
for the CDSE S3 endpoint (`https://eodata.dataspace.copernicus.eu`).
The session uses signed requests (`aws_unsigned=False`).

### Band Resolution Table

| Resolution | Bands                              |
|------------|-------------------------------------|
| 10 m       | B02, B03, B04, B08                  |
| 20 m       | B05, B06, B07, B8A, B11, B12, SCL  |
| 60 m       | B01, B09                            |

### Resampling Rules

All bands are resampled to 10 m (10 980 x 10 980 pixels):

- **Reflectance bands** (B03, B04, B08, B8A, B11, B12): **bilinear**
  interpolation.
- **SCL** (Scene Classification Layer): **nearest-neighbour** to
  preserve integer class values.

### Retry Logic

`_read_band_with_retry()` catches `RasterioIOError` and retries up to
`MAX_RETRIES` (3) times with exponential back-off starting at
`RETRY_BACKOFF` (2 s), doubling each attempt (2 s, 4 s, 8 s).

### Band-by-Band COG Writing

`download_tile()` writes each band individually to keep peak memory at
a single 10 980 x 10 980 `uint16` array (~230 MB).  The GeoTIFF profile
uses:

- Driver: `GTiff`
- Compression: `deflate`
- Internal tiling: `512 x 512`
- Data type: `uint16`

### Output File Naming & Format

```
<data_dir>/<mgrs_tile>/<YYYY-MM-DD>_S2L2A.tif
```

Example: `data/37MFT/2026-01-10_S2L2A.tif`

### `max_workers` Clamping

CDSE enforces a limit of **4 concurrent S3 connections** per credential
set.  `download_tiles()` clamps `max_workers` to 4 and emits a
`UserWarning` if the caller exceeds this value.

---

## Output Format

Each GeoTIFF is a **7-band `uint16` Cloud-Optimized GeoTIFF**:

| Band Index | Name | Native Resolution | Description              |
|------------|------|-------------------|--------------------------|
| 1          | B03  | 10 m              | Green                    |
| 2          | B04  | 10 m              | Red                      |
| 3          | B08  | 10 m              | NIR                      |
| 4          | B8A  | 20 m -> 10 m      | Narrow NIR               |
| 5          | B11  | 20 m -> 10 m      | SWIR 1                   |
| 6          | B12  | 20 m -> 10 m      | SWIR 2                   |
| 7          | SCL  | 20 m -> 10 m      | Scene Classification     |

- CRS: UTM zone of the MGRS tile (e.g. EPSG:32637)
- Dimensions: 10 980 x 10 980 pixels
- File path: `<data_dir>/<mgrs_tile>/<YYYY-MM-DD>_S2L2A.tif`

---

## Quick-Start Example

```python
from pathlib import Path
from maji.config import Settings
from maji.search import search_scenes, select_best_scenes
from maji.download import create_s3_session, download_tiles

# 1. Load settings from .env
settings = Settings()

# 2. Search for scenes
scenes = search_scenes(
    bbox=(36.0, -2.0, 37.0, -1.0),
    start="2026-01-01",
    end="2026-01-31",
    max_cloud=25.0,
)
print(f"Found {len(scenes)} scenes across {scenes['mgrs_tile'].nunique()} tiles")

# 3. Select the least cloudy scene per tile
best = select_best_scenes(scenes, strategy="least_cloudy")

# 4. Create an S3 session
session = create_s3_session(
    access_key=settings.cdse_s3_access_key,
    secret_key=settings.cdse_s3_secret_key,
)

# 5. Download
paths = download_tiles(best, data_dir=settings.data_dir, session=session)
for p in paths:
    print(f"Saved: {p}")
```

---

## Testing

All tests are fully offline — STAC and S3 interactions are mocked.

```bash
# Run the full suite (expects the 'maji' conda environment)
conda run -n maji python -m pytest tests/ -v
```

### What's Tested

| Module           | Tests | Coverage highlights                              |
|------------------|-------|--------------------------------------------------|
| `test_search.py` | 11    | MGRS extraction, band-asset matching, search & select |
| `test_download.py` | 9   | S3 session, retry logic, GeoTIFF writing, skip/overwrite, worker clamping |
| `conftest.py`    | —     | Shared fixtures (mock STAC items)                |

No network access is needed — all external calls (`pystac_client.Client`,
`rasterio.open`) are patched with `unittest.mock`.
