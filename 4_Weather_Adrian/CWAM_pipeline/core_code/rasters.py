"""Raster loading: IEM N0Q composites and MRMS EchoTop_18 GRIB2.

Both products are cropped to the pre-TRACON bounding box at load time and
returned as (data, affine_transform, shape) in lon/lat (EPSG:4326) space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window, from_bounds

KM_TO_FT = 3280.839895


@dataclass
class GeoGrid:
    data: np.ndarray        # 2D float array
    transform: Affine       # pixel -> lon/lat
    valid: np.ndarray       # bool mask of valid (non-missing) pixels


def _read_window(src, bounds):
    """Read band 1 of src cropped to lon/lat bounds, handling 0-360 rasters."""
    minx, miny, maxx, maxy = bounds
    # MRMS GRIB2 grids are often georeferenced with 0..360 longitudes
    if src.bounds.left > 180.0 or src.bounds.right > 180.0:
        minx, maxx = minx + 360.0, maxx + 360.0
        lon_shift = -360.0
    else:
        lon_shift = 0.0

    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    win = win.round_offsets().round_lengths()
    # Clamp to raster extent
    win = win.intersection(Window(0, 0, src.width, src.height))
    data = src.read(1, window=win)
    transform = src.window_transform(win)
    if lon_shift:
        transform = Affine(
            transform.a, transform.b, transform.c + lon_shift,
            transform.d, transform.e, transform.f,
        )
    return data, transform


def n0q_index_to_dbz(index: np.ndarray) -> np.ndarray:
    """IEM N0Q palette index (0-255) -> dBZ.

    dBZ = index / 2 - 32  (index 0 = no data / < -32 dBZ).
    Consistent with wx_grid_creator_3, where 45 dBZ -> index 154.
    """
    return index.astype(np.float32) / 2.0 - 32.0


def load_n0q(png_path, bounds) -> GeoGrid:
    """Load an N0Q png (+ sidecar .wld world file) cropped to bounds, in dBZ."""
    with rasterio.open(png_path) as src:
        idx, transform = _read_window(src, bounds)
    valid = idx > 0
    dbz = n0q_index_to_dbz(idx)
    dbz[~valid] = np.nan
    return GeoGrid(data=dbz, transform=transform, valid=valid)


def load_echo_top(path, bounds) -> GeoGrid:
    """Load MRMS EchoTop_18 (GRIB2 or GeoTIFF) cropped to bounds, in FEET MSL.

    MRMS encodes EchoTop_18 in km; missing/no-echo values are negative
    (-999 = missing, -99/-3 variants = no echo). No-echo is mapped to 0 ft
    (as in CWAM, no storm top). A units heuristic upgrades km -> m if the
    file appears to be in meters.
    """
    with rasterio.open(path) as src:
        raw, transform = _read_window(src, bounds)

    raw = raw.astype(np.float32)
    missing = raw <= -990.0          # sensor coverage missing
    no_echo = (raw < 0.0) & ~missing  # scanned, no 18 dBZ echo

    finite = raw[~missing & ~no_echo]
    if finite.size and np.nanpercentile(finite, 99) > 100.0:
        scale = KM_TO_FT / 1000.0    # values look like meters
    else:
        scale = KM_TO_FT             # values in km (standard MRMS)

    etop_ft = raw * scale
    etop_ft[no_echo] = 0.0
    etop_ft[missing] = np.nan
    return GeoGrid(data=etop_ft, transform=transform, valid=~missing)
