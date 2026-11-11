"""CWAM pre-TRACON weather grid -- importable pipeline.

Turns the Test_teresa.ipynb notebook into a callable module. One call fetches
the weather, tiles the pre-TRACON region, and writes ONE CSV per
(timestamp, flight level) containing coverage_pct, p_deviation and infeasible
for every cell.

Typical use from another file:

    from cwam_weather import run_cwam

    result = run_cwam(date="2026-09-03", start_time="19:00",
                      end_time="20:00", apt_code="DTW")
    print(result["summary"])        # one row per (timestamp, flight level)
    print(result["csv_files"][:3])  # paths that were written

    # keep the DataFrames in memory instead of re-reading the CSVs
    result = run_cwam("2026-09-03", "19:00", "20:00", "DTW", return_data=True)
    df = result["data"][(result["timestamps"][0], "FL350")]

From the command line:

    python cwam_weather.py --date 2026-09-03 --start 19:00 --end 20:00 --apt DTW
    python cwam_weather.py --date 2026-09-03 --start 19:00 --end 20:00 --apt DTW \
        --save-plots --plot-flight-levels FL350

date, start_time, end_time and apt_code are required. Everything else keeps the
notebook's values as defaults.

Output naming -- every file name fully defines the situation that produced it:

    <APT>_<YYYYMMDD>T<HHMM>Z_<FL>_r<radius>km_c<cell>km_<dbz>dBZ_<stat>_pt<thr>_<policy>.csv
    DTW_20260903T1900Z_FL350_r300km_c16km_35dBZ_p90_pt060_infeas.csv

The trailing parameter tag is applied to plots too, so a PNG can never be
confused with one from a different configuration. A JSON manifest listing every
parameter and every file written lands next to the CSVs.

Times are UTC throughout -- no timezone conversion happens anywhere.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
from affine import Affine
from pyproj import Geod
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import transform as shp_transform

KM_TO_FT = 3280.839895
_geod = Geod(ellps="WGS84")

# Paths are resolved against THIS file, not the caller's working directory, so
# importing the module from anywhere still finds ./data and ./output.
_HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _HERE / "data"
DEFAULT_OUTPUT_DIR = _HERE / "output"

# NASR / CIFP reference tables, expected inside data_dir
APT_BASE = "APT_BASE_Jul26.csv"
FIX_BASE = "FIX_BASE_Jul26.csv"
NAV_BASE = "NAV_BASE_Jul26.csv"
CIFP_INDEX = "IN_CIFP_Sep26.txt"

# Download sources
N0Q_BASE_URL = "https://mesonet.agron.iastate.edu/archive/data/{yyyy}/{mm}/{dd}/GIS/uscomp/"
MRMS_BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_PREFIX = "CONUS/EchoTop_18_00.50"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_MRMS_TS_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2")

# NWS reflectivity palette, used only by the optional plots
NWS = ["#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00", "#fdf802",
       "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000", "#f800fd", "#9854c6"]


def _write_with_retry(write_fn, path, retries=6, delay=0.4, backoff=1.6):
    """Call write_fn(path) with retries, absorbing transient file locks.

    On Windows, a path inside a synced folder (OneDrive, Dropbox, ...) can be
    briefly locked by the sync client right after being written -- surfacing
    as PermissionError / WinError 32 on the very next write, even though
    nothing is actually wrong with the file or its permissions. That lock
    normally clears within a second or two, so a short exponential-backoff
    retry is the practical fix rather than a real error to raise immediately.
    """
    wait = delay
    for attempt in range(retries):
        try:
            return write_fn(path)
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(wait)
            wait *= backoff


# ======================================================================
# Reference-table loaders (cached -- the CSVs are large and reread often)
# ======================================================================
@lru_cache(maxsize=4)
def _load_csv(path):
    return pd.read_csv(path, low_memory=False)


@lru_cache(maxsize=2)
def _star_index(cifp_index):
    """Parse the CIFP procedure index once: {ICAO: [STAR procedure idents]}.

    Tab separated with a header row:
        Arpt_ICAO / Procedure Type / Procedure Ident
    e.g.  KDTW  STAR  BONZZ2
    """
    idx = {}
    with open(cifp_index, encoding="latin-1") as f:
        next(f)                                     # skip header row
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3 and parts[1].strip() == "STAR":
                idx.setdefault(parts[0].strip(), set()).add(parts[2].strip())
    return {k: sorted(v) for k, v in idx.items()}


# ======================================================================
# STAR entry fixes -> TRACON / pre-TRACON geometry
# ======================================================================
def star_entry_fix_names(apt_code, cifp_index):
    """One entry fix per STAR: the procedure ident minus its revision digit.

    A STAR is named after the fix its transitions converge on -- BONZZ2 -> BONZZ,
    HANBL3 -> HANBL.
    """
    icao = apt_code.strip().upper()
    icao = icao if len(icao) == 4 else "K" + icao      # IATA -> ICAO (CONUS only)
    procs = _star_index(str(cifp_index)).get(icao, [])
    return sorted({re.sub(r"\d+$", "", p) for p in procs})


def star_fixes_from_nasr(apt_code, center_lat, center_lon, data_dir, verbose=True):
    """{NAME: [lon, lat]} -- entry fix names from CIFP, coordinates from NASR.

    Most STARs are named after an RNAV waypoint (FIX_BASE). A few are named
    after a VOR/VORTAC instead (3-letter identifiers, e.g. GEP -> Gopher
    VORTAC for MSP) -- those live in NAV_BASE, checked as a fallback for
    whatever FIX_BASE didn't resolve.
    """
    data_dir = Path(data_dir)
    names = star_entry_fix_names(apt_code, data_dir / CIFP_INDEX)
    fix = _load_csv(str(data_dir / FIX_BASE))
    hit = fix[fix["FIX_ID"].isin(names)].copy()

    # a fix name can repeat across regions; keep the one nearest the airport
    hit["_d"] = (hit["LAT_DECIMAL"] - center_lat) ** 2 + (hit["LONG_DECIMAL"] - center_lon) ** 2
    hit = hit.sort_values("_d").drop_duplicates("FIX_ID")
    out = {r.FIX_ID: [round(r.LONG_DECIMAL, 4), round(r.LAT_DECIMAL, 4)]
           for r in hit.itertuples()}

    missing = sorted(set(names) - set(out))
    if missing:
        nav = _load_csv(str(data_dir / NAV_BASE))
        nh = nav[nav["NAV_ID"].isin(missing)].copy()
        nh["_d"] = (nh["LAT_DECIMAL"] - center_lat) ** 2 + (nh["LONG_DECIMAL"] - center_lon) ** 2
        nh = nh.sort_values("_d").drop_duplicates("NAV_ID")
        out.update({r.NAV_ID: [round(r.LONG_DECIMAL, 4), round(r.LAT_DECIMAL, 4)]
                    for r in nh.itertuples()})
        missing = sorted(set(missing) - set(out))

    if missing and verbose:
        print(f"  note: {apt_code} STARs unresolved in FIX_BASE and NAV_BASE: {missing}")
    return out


def order_by_convex_hull(star_fixes):
    """Order the entry fixes onto their convex hull so the ring cannot self-intersect."""
    hull = MultiPoint([tuple(v) for v in star_fixes.values()]).convex_hull
    by_coord = {tuple(v): n for n, v in star_fixes.items()}
    return [by_coord[tuple(pt)] for pt in list(hull.exterior.coords)[:-1]]


def get_apt_data(apt_code, radius_km, data_dir, verbose=True):
    """Airport ARP coordinates plus its STAR entry fixes, hull-ordered."""
    data_dir = Path(data_dir)
    airport_code = apt_code.strip().upper()
    aptb = data_dir / APT_BASE
    apt = _load_csv(str(aptb))
    m = apt[(apt["ARPT_ID"] == airport_code) & (apt["SITE_TYPE_CODE"] == "A")]
    if m.empty:
        raise KeyError(f"{airport_code!r} not found as an airport in {aptb}")
    row = m.iloc[0]
    center_lat, center_lon = float(row["LAT_DECIMAL"]), float(row["LONG_DECIMAL"])

    star_fixes = star_fixes_from_nasr(apt_code, center_lat, center_lon, data_dir, verbose)
    if len(star_fixes) < 3:
        raise ValueError(f"{apt_code}: only {len(star_fixes)} STAR entry fixes "
                         f"resolved; need at least 3 for a polygon")
    ordered_fix_names = order_by_convex_hull(star_fixes)

    poly = Polygon([tuple(star_fixes[n]) for n in ordered_fix_names])
    if not poly.contains(Point(center_lon, center_lat)) and verbose:
        print(f"  WARNING: {apt_code} ARP falls outside its computed boundary")

    max_dist_km = max(_geod.inv(center_lon, center_lat, *star_fixes[n])[2] / 1000.0
                      for n in ordered_fix_names)
    if max_dist_km > 0.8 * radius_km and verbose:
        print(f"  WARNING: {apt_code} boundary reaches {max_dist_km:.0f} km, "
              f"close to or past radius_km={radius_km:g}")

    return center_lat, center_lon, star_fixes, ordered_fix_names


def pretracon_geometry(center_lat, center_lon, radius_km,
                       star_fixes, ordered_fix_names, n_points=361):
    """{circle, tracon, pre_tracon, bounds} in lon/lat."""
    tracon = Polygon([tuple(star_fixes[name]) for name in ordered_fix_names])
    az = np.linspace(0, 360, n_points)   # 361 points = azimuths in 1 degree steps
    lon2, lat2, _ = _geod.fwd(
        np.full(az.shape, center_lon), np.full(az.shape, center_lat),
        az, np.full(az.shape, radius_km * 1000.0),
    )
    circle = Polygon(np.column_stack([lon2, lat2]))
    return {"circle": circle, "tracon": tracon,
            "pre_tracon": circle.difference(tracon), "bounds": circle.bounds}


def get_roi_data(apt_code, radius_km, data_dir=None, verbose=True):
    """Airport + STAR fixes + pre-TRACON geometry in one call."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    center_lat, center_lon, star_fixes, ordered_fix_names = get_apt_data(
        apt_code, radius_km, data_dir, verbose)
    roi = pretracon_geometry(center_lat, center_lon, radius_km,
                             star_fixes, ordered_fix_names)
    return center_lat, center_lon, radius_km, star_fixes, ordered_fix_names, roi


# ======================================================================
# Timestamps
# ======================================================================
def build_timestamps(date, start_time, end_time, step_minutes):
    """List of UTC datetimes to process (inclusive of end_time)."""
    t0 = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
    if t1 < t0:
        raise ValueError(f"end_time {end_time} is before start_time {start_time}")
    out, t = [], t0
    while t <= t1:
        out.append(t)
        t += timedelta(minutes=step_minutes)
    return out


# ======================================================================
# Downloads
# ======================================================================
def n0q_filenames(dt):
    stamp = dt.strftime("%Y%m%d%H%M")
    return f"n0q_{stamp}.png", f"n0q_{stamp}.wld"


def download_n0q(data_dir, timestamps, verbose=False, session=None, workers=8):
    """Download N0Q png+wld for each timestamp. Returns {dt: png_path}."""
    session = session or requests.Session()

    save_dir = Path(data_dir) / "n0q"
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dt in timestamps:
        base = N0Q_BASE_URL.format(
            yyyy=dt.strftime("%Y"), mm=dt.strftime("%m"), dd=dt.strftime("%d")
        )
        for fname in n0q_filenames(dt):
            tasks.append((base + fname, save_dir / fname))

    def _get(url, path):
        if path.exists() and path.stat().st_size > 0:
            return f"cached  {path.name}"
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return f"FAILED  {path.name}: HTTP {r.status_code}"
        path.write_bytes(r.content)
        return f"fetched {path.name}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_get, url, path): path for url, path in tasks}
        for fut in as_completed(futures):
            msg = fut.result()
            # "cached" is the common case on a re-run -- only report real work
            if msg.startswith("FAILED") or (verbose and not msg.startswith("cached")):
                print("  [n0q]", msg)

    out = {}
    for dt in timestamps:
        png, wld = n0q_filenames(dt)
        p, w = save_dir / png, save_dir / wld
        if p.exists() and w.exists():
            out[dt] = p
        else:
            print(f"  [n0q] WARNING: missing files for {dt:%Y-%m-%d %H:%M}, skipping")
    return out


def list_mrms_keys(date, session=None):
    """List available EchoTop_18 object keys for a UTC date via the S3 REST API."""
    session = session or requests.Session()
    prefix = f"{MRMS_PREFIX}/{date:%Y%m%d}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(MRMS_BUCKET_URL, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for el in root.iter(f"{_S3_NS}Key"):
            keys.append(el.text)
        truncated = root.findtext(f"{_S3_NS}IsTruncated") == "true"
        token = root.findtext(f"{_S3_NS}NextContinuationToken")
        if not truncated or not token:
            break
    return keys


def parse_mrms_timestamp(key):
    m = _MRMS_TS_RE.search(key)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def pair_echo_tops(keys, timestamps, tolerance_min):
    """For each N0Q timestamp pick the nearest MRMS key within tolerance."""
    stamped = [(parse_mrms_timestamp(k), k) for k in keys]
    stamped = [(t, k) for t, k in stamped if t is not None]
    pairs = {}
    for dt in timestamps:
        if not stamped:
            break
        t_best, k_best = min(stamped, key=lambda tk: abs((tk[0] - dt).total_seconds()))
        if abs((t_best - dt).total_seconds()) <= tolerance_min * 60.0:
            pairs[dt] = (t_best, k_best)
        else:
            print(f"  [eet] WARNING: no EchoTop file within "
                  f"{tolerance_min} min of {dt:%H:%M}, skipping")
    return pairs


def download_echo_tops(data_dir, timestamps, tolerance_min=5,
                       verbose=False, session=None):
    """Download and gunzip paired EchoTop_18 GRIB2 files. Returns {dt: grib2_path}."""
    session = session or requests.Session()
    save_dir = Path(data_dir) / "echotop"
    save_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted({dt.date() for dt in timestamps})
    keys = []
    for d in dates:
        keys += list_mrms_keys(datetime(d.year, d.month, d.day), session)
    if verbose:
        print(f"  [eet] {len(keys)} EchoTop_18 objects listed for {dates}")

    pairs = pair_echo_tops(keys, timestamps, tolerance_min)

    out = {}
    for dt, (t_eet, key) in pairs.items():
        gz_path = save_dir / Path(key).name
        grib_path = gz_path.with_suffix("")  # strip .gz
        if not grib_path.exists():
            if not gz_path.exists():
                url = f"{MRMS_BUCKET_URL}/{key}"
                r = session.get(url, timeout=60)
                r.raise_for_status()
                gz_path.write_bytes(r.content)
            with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            gz_path.unlink(missing_ok=True)
        if verbose:
            print(f"  [eet] {dt:%H:%M} -> {grib_path.name} (dt={t_eet:%H:%M:%S}Z)")
        out[dt] = grib_path
    return out


# ======================================================================
# Raster loading (dBZ and feet MSL, cropped to the ROI bbox)
# ======================================================================
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
    win = win.intersection(Window(0, 0, src.width, src.height))   # clamp to extent
    data = src.read(1, window=win)
    transform = src.window_transform(win)
    if lon_shift:
        transform = Affine(
            transform.a, transform.b, transform.c + lon_shift,
            transform.d, transform.e, transform.f,
        )
    return data, transform


def n0q_index_to_dbz(index):
    """IEM N0Q palette index (0-255) -> dBZ.

    dBZ = index / 2 - 32  (index 0 = no data / < -32 dBZ).
    Consistent with wx_grid_creator_3, where 45 dBZ -> index 154.
    """
    return index.astype(np.float32) / 2.0 - 32.0


def load_n0q(png_path, bounds):
    """Load an N0Q png (+ sidecar .wld world file) cropped to bounds, in dBZ."""
    with rasterio.open(png_path) as src:
        idx, transform = _read_window(src, bounds)
    valid = idx > 0
    dbz = n0q_index_to_dbz(idx)
    dbz[~valid] = np.nan
    return {"data": dbz, "transform": transform, "valid": valid}


def load_echo_top(path, bounds):
    """Load MRMS EchoTop_18 (GRIB2 or GeoTIFF) cropped to bounds, in FEET MSL.

    MRMS encodes EchoTop_18 in km; missing/no-echo values are negative
    (-999 = missing, -99/-3 variants = no echo). No-echo is mapped to 0 ft
    (as in CWAM, no storm top). A units heuristic upgrades km -> m if the
    file appears to be in meters.
    """
    with rasterio.open(path) as src:
        raw, transform = _read_window(src, bounds)

    raw = raw.astype(np.float32)
    missing = raw <= -990.0           # sensor coverage missing
    no_echo = (raw < 0.0) & ~missing  # scanned, no 18 dBZ echo

    finite = raw[~missing & ~no_echo]
    if finite.size and np.nanpercentile(finite, 99) > 100.0:
        scale = KM_TO_FT / 1000.0     # values look like meters
    else:
        scale = KM_TO_FT              # values in km (standard MRMS)

    etop_ft = raw * scale
    etop_ft[no_echo] = 0.0
    etop_ft[missing] = np.nan

    return {"data": etop_ft, "transform": transform, "valid": ~missing}


# ======================================================================
# Cell grid and per-cell predictors
# ======================================================================
def build_cell_grid(pre_tracon, center_lat, center_lon,
                    cell_size_km, min_area_fraction=0.05):
    """Tile the pre-TRACON region with equal-size square cells.

    Squares are cut in a local azimuthal-equidistant projection centred on the
    airport, so each is a true cell_size_km x cell_size_km square on the ground,
    then transformed back to lon/lat for raster sampling. Cells are clipped to
    the region; those with less than min_area_fraction of a full square inside
    it are dropped.
    """
    crs_lonlat = pyproj.CRS("EPSG:4326")
    crs_local = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_local = pyproj.Transformer.from_crs(crs_lonlat, crs_local, always_xy=True).transform
    to_lonlat = pyproj.Transformer.from_crs(crs_local, crs_lonlat, always_xy=True).transform

    pre_tracon_local = shp_transform(to_local, pre_tracon)

    size = cell_size_km * 1000.0
    minx, miny, maxx, maxy = pre_tracon_local.bounds

    # Snap the grid origin outward to whole cells so the grid is stable
    x0 = np.floor(minx / size) * size
    y1 = np.ceil(maxy / size) * size
    n_cols = int(np.ceil((maxx - x0) / size))
    n_rows = int(np.ceil((y1 - miny) / size))

    full_area = size * size
    cells, cid = [], 0
    for r in range(n_rows):
        cy1 = y1 - r * size          # top edge of this row
        cy0 = cy1 - size
        for c in range(n_cols):
            cx0 = x0 + c * size
            cx1 = cx0 + size
            clipped = box(cx0, cy0, cx1, cy1).intersection(pre_tracon_local)
            if clipped.is_empty:
                continue
            frac = clipped.area / full_area
            if frac < min_area_fraction:
                continue
            cid += 1
            # Densify edges before reprojecting so square sides stay accurate
            geom_ll = shp_transform(to_lonlat, clipped.segmentize(size / 8.0))
            cen = shp_transform(to_lonlat, clipped.centroid)
            cells.append({
                "cell_id": cid,          # 1-based; 0 is reserved for "no cell"
                "row": r,                # 0 = northernmost
                "col": c,                # 0 = westernmost
                "geom_lonlat": geom_ll,
                "centroid_lon": cen.x,
                "centroid_lat": cen.y,
                "area_fraction": frac,
            })

    return {"cells": cells, "n_rows": n_rows, "n_cols": n_cols,
            "pre_tracon_local": pre_tracon_local}


def label_cells(cells, shape, transform):
    """Burn cell polygons onto a raster grid -> uint32 array of cell ids.

    all_touched=False means a pixel is claimed only if its CENTRE falls inside
    the polygon, which keeps cells a clean partition: every pixel belongs to at
    most one cell.
    """
    shapes = [(c["geom_lonlat"], c["cell_id"]) for c in cells]
    return rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype="uint32", all_touched=False)


def compute_cell_metrics(cells, n0q_data, n0q_valid, n0q_labels,
                         etop_data, eet_labels,
                         dbz_threshold=35.0, echo_top_stat="p90"):
    """Predictor 1 (coverage_pct) and predictor 2 (etop_stat_ft) per cell."""
    n_cells = len(cells)

    # ---- Predictor 1: % of N0Q pixels >= threshold ----
    lab = n0q_labels.ravel()
    dbz = n0q_data.ravel()
    inside = lab > 0
    # NOTE: index-0 "no echo" pixels stay in the denominator -- they are
    # legitimate below-threshold observations, not missing data.
    n_total = np.bincount(lab[inside], minlength=n_cells + 1)[1:]
    exceed = inside & n0q_valid.ravel() & (
        np.nan_to_num(dbz, nan=-99.0) >= dbz_threshold)
    n_exceed = np.bincount(lab[exceed], minlength=n_cells + 1)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_pct = np.where(n_total > 0, 100.0 * n_exceed / n_total, np.nan)

    # ---- Predictor 2: echo top statistic per cell ----
    lab_e = eet_labels.ravel()
    eet = etop_data.ravel()
    ok = (lab_e > 0) & np.isfinite(eet)           # NaN = no radar coverage
    counts = np.bincount(lab_e[ok], minlength=n_cells + 1)[1:]
    order = np.argsort(lab_e[ok], kind="stable")  # group pixels by cell id
    vals_sorted = eet[ok][order]
    offsets = np.concatenate([[0], np.cumsum(counts)])

    etop_stat_ft = np.zeros(n_cells, dtype=np.float64)
    for i in range(n_cells):
        seg = vals_sorted[offsets[i]:offsets[i + 1]]
        if seg.size == 0:
            etop_stat_ft[i] = np.nan          # no valid data -> no verdict
        elif echo_top_stat == "max":
            etop_stat_ft[i] = seg.max()
        else:
            etop_stat_ft[i] = np.percentile(seg, 90)

    return pd.DataFrame({
        "cell_id": np.arange(1, n_cells + 1),
        "row": [c["row"] for c in cells],
        "col": [c["col"] for c in cells],
        "centroid_lat": [c["centroid_lat"] for c in cells],
        "centroid_lon": [c["centroid_lon"] for c in cells],
        "min_lon": [c["geom_lonlat"].bounds[0] for c in cells],
        "min_lat": [c["geom_lonlat"].bounds[1] for c in cells],
        "max_lon": [c["geom_lonlat"].bounds[2] for c in cells],
        "max_lat": [c["geom_lonlat"].bounds[3] for c in cells],
        "area_fraction": [c["area_fraction"] for c in cells],
        "n_pixels_n0q": n_total,
        "n_pixels_exceed": n_exceed,
        "coverage_pct": coverage_pct,
        "etop_stat_ft": etop_stat_ft,
    })


# ======================================================================
# Probability table and the altitude sweep
# ======================================================================
def make_default_prob_table():
    """Analytic approximation of the CWAM2 deviation-probability surface.

        P = (coverage/100)^0.35 * sigmoid(-dz_kft / 3.5)

    Rows are dz bin centres in kft (aircraft altitude minus echo top;
    negative = flying below the tops), columns are coverage-% bin centres.
    UNCALIBRATED PLACEHOLDER -- shaped to match DeLaura et al. Fig 8c
    qualitatively, not fitted to encounter data.
    """
    dz = np.array([-30, -25, -20, -15, -10, -7.5, -5, -2.5, 0,
                   2.5, 5, 7.5, 10, 15, 20], dtype=float)
    cov = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    dzg, covg = np.meshgrid(dz, cov, indexing="ij")
    p = (covg / 100.0) ** 0.35 / (1.0 + np.exp(dzg / 3.5))
    return {"dz_kft": dz, "coverage_pct": cov, "p": np.round(p, 3)}


def load_prob_table(path):
    """Read a table CSV: first column = dz_kft, header row = coverage bins."""
    df = pd.read_csv(path)
    return {"dz_kft": df.iloc[:, 0].to_numpy(float),
            "coverage_pct": np.array([float(c) for c in df.columns[1:]]),
            "p": df.iloc[:, 1:].to_numpy(float)}


def save_prob_table(table, path):
    df = pd.DataFrame(table["p"], columns=[f"{c:g}" for c in table["coverage_pct"]])
    df.insert(0, "dz_kft", table["dz_kft"])
    _write_with_retry(lambda p: df.to_csv(p, index=False), path)


def lookup_probability(table, dz_ft, coverage_pct):
    """Bilinearly interpolated P(deviation). dz in FEET, coverage in %.

    Inputs of any shape. Clamped at the table edges; NaN in -> NaN out
    (a cell with no valid echo-top data gets no verdict).
    """
    dz_axis, cov_axis, p = table["dz_kft"], table["coverage_pct"], table["p"]

    dz_kft = np.asarray(dz_ft, dtype=float) / 1000.0
    cov = np.asarray(coverage_pct, dtype=float)
    nan_mask = ~np.isfinite(dz_kft) | ~np.isfinite(cov)

    x = np.clip(dz_kft, dz_axis[0], dz_axis[-1])
    y = np.clip(cov, cov_axis[0], cov_axis[-1])
    i = np.clip(np.searchsorted(dz_axis, x) - 1, 0, dz_axis.size - 2)
    j = np.clip(np.searchsorted(cov_axis, y) - 1, 0, cov_axis.size - 2)

    x0, x1 = dz_axis[i], dz_axis[i + 1]
    y0, y1 = cov_axis[j], cov_axis[j + 1]
    tx = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (y - y0) / (y1 - y0), 0.0)

    out = (p[i, j] * (1 - tx) * (1 - ty) + p[i + 1, j] * tx * (1 - ty)
           + p[i, j + 1] * (1 - tx) * ty + p[i + 1, j + 1] * tx * ty)
    return np.where(nan_mask, np.nan, out)


def altitudes(altitude_min_ft, altitude_max_ft, altitude_step_ft, altitudes_ft=None):
    """Altitude levels (ft) to evaluate, from explicit list or min/max/step."""
    if altitudes_ft:
        return sorted(float(a) for a in altitudes_ft)
    if altitude_max_ft < altitude_min_ft:
        raise ValueError("altitude_max_ft must be >= altitude_min_ft")
    if altitude_step_ft <= 0:
        raise ValueError("altitude_step_ft must be > 0")
    out, a = [], float(altitude_min_ft)
    while a <= altitude_max_ft + 1e-6:      # epsilon so the top level isn't lost
        out.append(a)
        a += altitude_step_ft
    return out


def flight_level(alt_ft):
    """35000 ft -> 'FL350'"""
    return f"FL{int(round(alt_ft / 100)):03d}"


def evaluate_altitude(base_df, prob_table, alt_ft,
                      mask_probability_threshold=0.6,
                      missing_data_policy="infeasible"):
    """Apply one flight level to the altitude-independent predictors."""
    df = base_df.copy()
    df["altitude_ft"] = alt_ft
    df["dz_ft"] = alt_ft - df["etop_stat_ft"]        # negative = below the tops
    df["p_deviation"] = lookup_probability(prob_table,
                                           df["dz_ft"].to_numpy(),
                                           df["coverage_pct"].to_numpy())
    # A NaN echo-top statistic means no verdict, not "clear".
    df["data_missing"] = df["dz_ft"].isna()
    df["infeasible"] = df["p_deviation"] >= mask_probability_threshold
    if missing_data_policy == "infeasible":
        df["infeasible"] = df["infeasible"] | df["data_missing"]
    return df


def cell_value_raster(df, labels, column, fill=np.nan, dtype=float):
    """Paint each pixel with its cell's value for `column` (fill outside)."""
    n_cells = int(df["cell_id"].max())
    lut = np.full(n_cells + 1, fill, dtype=dtype)
    lut[df["cell_id"].to_numpy(int)] = df[column].to_numpy(dtype)
    return lut[labels]


# ======================================================================
# File naming -- every name fully defines the run that produced it
# ======================================================================
def param_tag(radius_km, cell_size_km, dbz_threshold, echo_top_stat,
              mask_probability_threshold, missing_data_policy):
    """The configuration suffix shared by every file a run writes.

    e.g. r300km_c16km_35dBZ_p90_pt060_infeas
    """
    thr = f"pt{mask_probability_threshold:.2f}".replace(".", "")
    policy = "infeas" if missing_data_policy == "infeasible" else "pass"
    return (f"r{radius_km:g}km_c{cell_size_km:g}km_{dbz_threshold:g}dBZ_"
            f"{echo_top_stat}_{thr}_{policy}")


def time_stamp(dt):
    """datetime -> 20260903T1900Z (filename safe, UTC explicit)."""
    return f"{dt:%Y%m%dT%H%M}Z"


def csv_name(apt_code, dt, fl, tag):
    return f"{apt_code}_{time_stamp(dt)}_{fl}_{tag}.csv"


def plot_name(apt_code, kind, tag, dt=None, fl=None):
    parts = [apt_code]
    if dt is not None:
        parts.append(time_stamp(dt))
    if fl is not None:
        parts.append(fl)
    parts += [kind, tag]
    return "_".join(parts) + ".png"


# ======================================================================
# Optional plots -- saved to disk, never shown
# ======================================================================
def _save_field_plot(path, roi, grid, data, cmap, norm, label, title,
                     cells=None, grid_lw=0.5, grid_alpha=0.4, figsize=(6, 7)):
    """One field over the ROI, in the notebook's black-background style."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    def _extent(g):
        h, w = g["data"].shape
        x0, y0 = g["transform"] * (0, 0)
        x1, y1 = g["transform"] * (w, h)
        return [x0, x1, y1, y0]

    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(data, extent=_extent(grid), origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest")

    if cells:
        polys = []
        for c in cells:
            geom = c["geom_lonlat"]
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for part in parts:
                polys.append(np.asarray(part.exterior.coords))
        ax.add_collection(PolyCollection(
            polys, facecolors="none", edgecolors="white",
            linewidths=grid_lw, alpha=grid_alpha, zorder=3))

    for poly, color in ((roi["circle"], "deepskyblue"), (roi["tracon"], "red")):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=2, zorder=4)

    xmin, ymin, xmax, ymax = roi["circle"].bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title, color="white", fontsize=10); ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label, color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    _write_with_retry(
        lambda p: fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="black"), path)
    plt.close(fig)
    return path


def save_roi_plot(path, star_fixes, ordered_fix_names, center_lat, center_lon,
                  radius_km, title, figsize=(6, 7)):
    """The TRACON hull, its STAR fixes, and the pre-TRACON ROI circle."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ring = ordered_fix_names + [ordered_fix_names[0]]
    hx = [star_fixes[n][0] for n in ring]
    hy = [star_fixes[n][1] for n in ring]
    ax.plot(hx, hy, color="red", lw=2, zorder=3, label="TRACON hull (computed)")
    ax.fill(hx, hy, color="red", alpha=0.08, zorder=1)

    on_hull = set(ordered_fix_names)
    interior = [n for n in star_fixes if n not in on_hull]
    if interior:
        ax.scatter([star_fixes[n][0] for n in interior],
                   [star_fixes[n][1] for n in interior],
                   s=45, facecolors="none", edgecolors="gray", linewidths=1.4,
                   zorder=4, label=f"interior, dropped ({len(interior)})")
    ax.scatter([star_fixes[n][0] for n in ordered_fix_names],
               [star_fixes[n][1] for n in ordered_fix_names],
               s=60, c="red", marker="^", edgecolors="black", linewidths=0.6,
               zorder=5, label=f"hull vertices ({len(ordered_fix_names)})")
    for n, (lon, lat) in star_fixes.items():
        ax.annotate(n, (lon, lat), fontsize=7,
                    color="darkred" if n in on_hull else "gray",
                    xytext=(4, 3), textcoords="offset points", zorder=6)
    ax.scatter([center_lon], [center_lat], marker="*", s=170, c="deepskyblue",
               edgecolors="black", linewidths=0.6, zorder=7, label="airport (ARP)")

    az = np.linspace(0, 360, 361)
    lo, la, _ = _geod.fwd(np.full(az.shape, center_lon), np.full(az.shape, center_lat),
                          az, np.full(az.shape, radius_km * 1000.0))
    ax.plot(lo, la, color="deepskyblue", lw=1.5, ls="--", zorder=2,
            label=f"pre-TRACON ROI ({radius_km:g} km)")

    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right")
    _write_with_retry(lambda p: fig.savefig(p, dpi=130, bbox_inches="tight"), path)
    plt.close(fig)
    return path


def _frame_plots(plot_dir, apt_code, tag, dt, roi, n0q, etop, cells,
                 base_df, n0q_labels, dbz_threshold):
    """Altitude-independent plots for one timestamp: N0Q, EchoTop, coverage."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    written = []
    cmap_n0q = ListedColormap(NWS); cmap_n0q.set_bad((0, 0, 0, 0))
    cmap_etop = plt.get_cmap("turbo").copy(); cmap_etop.set_bad((0, 0, 0, 0))
    cmap_cov = plt.get_cmap("viridis").copy(); cmap_cov.set_bad((0, 0, 0, 0))

    dbz = np.ma.masked_less(np.ma.masked_invalid(n0q["data"]), 5.0)
    kft = np.ma.masked_less_equal(np.ma.masked_invalid(etop["data"] / 1000.0), 0.0)

    written.append(_save_field_plot(
        plot_dir / plot_name(apt_code, "n0q", tag, dt=dt), roi, n0q, dbz,
        cmap_n0q, BoundaryNorm(np.arange(5, 80, 5), cmap_n0q.N),
        "Reflectivity (dBZ)", f"{apt_code}  N0Q  {dt:%Y-%m-%d %H:%M} UTC",
        cells=cells))
    written.append(_save_field_plot(
        plot_dir / plot_name(apt_code, "echotop", tag, dt=dt), roi, etop, kft,
        cmap_etop, BoundaryNorm(np.arange(0, 65, 5), cmap_etop.N),
        "18 dBZ echo top (kft)", f"{apt_code}  EchoTop  {dt:%Y-%m-%d %H:%M} UTC",
        cells=cells))
    cov_raster = cell_value_raster(base_df, n0q_labels, "coverage_pct")
    written.append(_save_field_plot(
        plot_dir / plot_name(apt_code, "coverage", tag, dt=dt), roi, n0q,
        np.ma.masked_invalid(cov_raster), cmap_cov, Normalize(0, 100),
        "coverage (%)",
        f"{apt_code}  coverage >= {dbz_threshold:g} dBZ  {dt:%H:%M} UTC",
        cells=cells))
    return written


def _level_plots(plot_dir, apt_code, tag, dt, fl, roi, n0q, cells, df,
                 n0q_labels, mask_probability_threshold):
    """Per-flight-level plots: P(deviation) and the infeasibility mask."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize

    written = []
    cmap_p = plt.get_cmap("magma").copy(); cmap_p.set_bad((0, 0, 0, 0))
    p_raster = cell_value_raster(df, n0q_labels, "p_deviation")
    written.append(_save_field_plot(
        plot_dir / plot_name(apt_code, "pdev", tag, dt=dt, fl=fl), roi, n0q,
        np.ma.masked_invalid(p_raster), cmap_p, Normalize(0, 1),
        "P(deviation)", f"{apt_code}  CWAM WAF {fl}  {dt:%Y-%m-%d %H:%M} UTC",
        cells=cells))

    inf_raster = cell_value_raster(df, n0q_labels, "infeasible",
                                   fill=False, dtype=bool)
    written.append(_save_field_plot(
        plot_dir / plot_name(apt_code, "infeasible", tag, dt=dt, fl=fl), roi, n0q,
        inf_raster.astype(float), ListedColormap(["black", "red"]), Normalize(0, 1),
        "infeasible",
        f"{apt_code}  infeasible {fl}  (P >= {mask_probability_threshold:g})"
        f"  {dt:%H:%M} UTC",
        cells=cells))
    return written


# ======================================================================
# Main entry point
# ======================================================================
def run_cwam(date, start_time, end_time, apt_code,
             *,
             step_minutes=5,
             radius_km=300,
             cell_size_km=16,
             min_cell_area_fraction=0.05,
             dbz_threshold=35.0,
             echo_top_stat="p90",
             altitude_min_ft=11000,
             altitude_max_ft=40000,
             altitude_step_ft=2000,
             altitudes_ft=None,
             mask_probability_threshold=0.6,
             missing_data_policy="infeasible",
             echo_top_time_tolerance_min=5,
             download_workers=8,
             prob_table_csv=None,
             data_dir=None,
             output_dir=None,
             plot_dir=None,
             save_plots=False,
             plot_flight_levels=None,
             return_data=False,
             verbose=True):
    """Run the CWAM pre-TRACON weather pipeline and write one CSV per
    (timestamp, flight level).

    Required
    --------
    date        : "YYYY-MM-DD"  (UTC)
    start_time  : "HH:MM"       (UTC, inclusive)
    end_time    : "HH:MM"       (UTC, inclusive)
    apt_code    : airport id, e.g. "DTW" (also ORD, ATL, DFW, MSP)

    Everything else keeps the notebook's defaults. Notable options:
    altitudes_ft        explicit list of levels; overrides min/max/step
    prob_table_csv      path to a calibrated P table; default is the
                        UNCALIBRATED analytic placeholder
    save_plots          write PNGs as well as CSVs (default False)
    plot_flight_levels  which levels get P/infeasible plots; None = all
    return_data         also return every DataFrame in memory

    Returns a dict:
        params, timestamps, flight_levels, center_lat, center_lon, roi, cells,
        csv_files, plot_files, manifest_path, summary (DataFrame),
        skipped (timestamps with no weather data), and data (only if
        return_data=True), keyed by (timestamp, flight_level).
    """
    if missing_data_policy not in ("infeasible", "passable"):
        raise ValueError("missing_data_policy must be 'infeasible' or 'passable'")
    if echo_top_stat not in ("p90", "max"):
        raise ValueError("echo_top_stat must be 'p90' or 'max'")

    apt_code = apt_code.strip().upper()
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(plot_dir) if plot_dir else output_dir / "plots"
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    tag = param_tag(radius_km, cell_size_km, dbz_threshold, echo_top_stat,
                    mask_probability_threshold, missing_data_policy)

    # ---- 1. time window ------------------------------------------------
    timestamps = build_timestamps(date, start_time, end_time, step_minutes)

    # ---- 2. airport, STAR fixes, pre-TRACON geometry -------------------
    center_lat, center_lon, radius_km, star_fixes, ordered_fix_names, roi = \
        get_roi_data(apt_code, radius_km, data_dir, verbose)

    alts = altitudes(altitude_min_ft, altitude_max_ft, altitude_step_ft, altitudes_ft)
    fls = [flight_level(a) for a in alts]

    if verbose:
        print("=" * 72)
        print(f"Case      {date}   {timestamps[0]:%H:%M} to {timestamps[-1]:%H:%M} UTC "
              f"  (step {step_minutes} min)")
        print(f"Region    {apt_code} TRACON ({center_lat:.4f}, {center_lon:.4f})  "
              f"radius {radius_km:g} km")
        print(f"Frames    {len(timestamps)}   Levels  {len(alts)} "
              f"({fls[0]} to {fls[-1]})")
        print(f"Output    {output_dir}")
        print("=" * 72)

    # ---- 3. download the raw weather (cached -- cheap to re-run) -------
    n0q_files = download_n0q(data_dir, timestamps, verbose, workers=download_workers)
    eet_files = download_echo_tops(data_dir, timestamps,
                                   echo_top_time_tolerance_min, verbose)

    # ---- 4. the cell grid: geometry only, so build it ONCE -------------
    grid = build_cell_grid(roi["pre_tracon"], center_lat, center_lon,
                           cell_size_km, min_cell_area_fraction)
    cells = grid["cells"]
    if not cells:
        raise ValueError(f"{apt_code}: cell grid is empty -- check cell_size_km "
                         f"({cell_size_km}) against radius_km ({radius_km})")
    if verbose:
        print(f"{len(cells)} cells of {cell_size_km:g} km "
              f"({grid['n_rows']} rows x {grid['n_cols']} cols)")

    prob_table = load_prob_table(prob_table_csv) if prob_table_csv \
        else make_default_prob_table()

    # Rasterizing the cells is the slow part and depends only on the raster
    # geometry, which is identical across timestamps -- cache per (shape, transform).
    label_cache = {}

    def _labels(shape, transform):
        key = (shape, tuple(transform)[:6])
        if key not in label_cache:
            label_cache[key] = label_cells(cells, shape, transform)
        return label_cache[key]

    want_levels = set(fls) if plot_flight_levels is None else set(plot_flight_levels)

    csv_files, plot_files, skipped, summary_rows = [], [], [], []
    data_out = {} if return_data else None

    # ---- 5. per timestamp: metrics, then every flight level ------------
    for dt in timestamps:
        if dt not in n0q_files or dt not in eet_files:
            skipped.append(dt)
            if verbose:
                print(f"  skipping {dt:%Y-%m-%d %H:%M} UTC -- missing weather files")
            continue

        n0q = load_n0q(n0q_files[dt], roi["bounds"])
        etop = load_echo_top(eet_files[dt], roi["bounds"])

        n0q_labels = _labels(n0q["data"].shape, n0q["transform"])
        eet_labels = _labels(etop["data"].shape, etop["transform"])

        base_df = compute_cell_metrics(cells, n0q["data"], n0q["valid"], n0q_labels,
                                       etop["data"], eet_labels,
                                       dbz_threshold, echo_top_stat)

        if save_plots:
            plot_files += _frame_plots(plot_dir, apt_code, tag, dt, roi, n0q, etop,
                                       cells, base_df, n0q_labels, dbz_threshold)

        for alt in alts:
            fl = flight_level(alt)
            df = evaluate_altitude(base_df, prob_table, alt,
                                   mask_probability_threshold, missing_data_policy)

            # stamp each row so concatenated CSVs stay self-describing
            df.insert(0, "apt_code", apt_code)
            df.insert(1, "timestamp_utc", dt.strftime("%Y-%m-%d %H:%M:%S"))
            df.insert(2, "flight_level", fl)

            path = output_dir / csv_name(apt_code, dt, fl, tag)
            _write_with_retry(lambda p, df=df: df.to_csv(p, index=False), path)
            csv_files.append(path)

            if return_data:
                data_out[(dt, fl)] = df

            summary_rows.append({
                "timestamp_utc": dt,
                "flight_level": fl,
                "altitude_ft": alt,
                "n_cells": len(df),
                "max_p_deviation": float(np.nanmax(df["p_deviation"]))
                if df["p_deviation"].notna().any() else np.nan,
                "mean_p_deviation": float(np.nanmean(df["p_deviation"])),
                "n_infeasible": int(df["infeasible"].sum()),
                "n_data_missing": int(df["data_missing"].sum()),
                "csv_file": path.name,
            })

            if save_plots and fl in want_levels:
                plot_files += _level_plots(plot_dir, apt_code, tag, dt, fl, roi,
                                           n0q, cells, df, n0q_labels,
                                           mask_probability_threshold)

        if verbose:
            print(f"  {dt:%Y-%m-%d %H:%M} UTC  ->  {len(alts)} CSVs")

    if save_plots and ordered_fix_names:
        plot_files.append(save_roi_plot(
            plot_dir / plot_name(apt_code, "roi", tag),
            star_fixes, ordered_fix_names, center_lat, center_lon, radius_km,
            f"{apt_code} - TRACON hull and {radius_km:g} km pre-TRACON ROI"))

    summary = pd.DataFrame(summary_rows)

    # ---- 6. manifest: the full provenance of this run ------------------
    params = {
        "date": date, "start_time": start_time, "end_time": end_time,
        "apt_code": apt_code, "step_minutes": step_minutes,
        "radius_km": radius_km, "cell_size_km": cell_size_km,
        "min_cell_area_fraction": min_cell_area_fraction,
        "dbz_threshold": dbz_threshold, "echo_top_stat": echo_top_stat,
        "altitude_min_ft": altitude_min_ft, "altitude_max_ft": altitude_max_ft,
        "altitude_step_ft": altitude_step_ft, "altitudes_ft": altitudes_ft,
        "mask_probability_threshold": mask_probability_threshold,
        "missing_data_policy": missing_data_policy,
        "echo_top_time_tolerance_min": echo_top_time_tolerance_min,
        "prob_table_csv": str(prob_table_csv) if prob_table_csv else None,
        "prob_table_is_calibrated": bool(prob_table_csv),
        "param_tag": tag,
    }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "center_lat": center_lat, "center_lon": center_lon,
        "n_cells": len(cells),
        "grid_rows": grid["n_rows"], "grid_cols": grid["n_cols"],
        "star_fixes_on_hull": ordered_fix_names,
        "timestamps_utc": [dt.strftime("%Y-%m-%d %H:%M") for dt in timestamps],
        "timestamps_skipped": [dt.strftime("%Y-%m-%d %H:%M") for dt in skipped],
        "flight_levels": fls,
        "csv_files": [p.name for p in csv_files],
        "plot_files": [p.name for p in plot_files],
    }
    manifest_path = output_dir / f"manifest_{apt_code}_{date.replace('-', '')}_{tag}.json"
    _write_with_retry(
        lambda p: p.write_text(json.dumps(manifest, indent=2), encoding="utf-8"),
        manifest_path)

    if verbose:
        print("-" * 72)
        print(f"{len(csv_files)} CSVs -> {output_dir}")
        if plot_files:
            print(f"{len(plot_files)} PNGs -> {plot_dir}")
        if skipped:
            print(f"{len(skipped)} timestamps skipped (no weather data)")
        print(f"manifest -> {manifest_path.name}")
        if not params["prob_table_is_calibrated"]:
            print("NOTE: p_deviation uses the UNCALIBRATED placeholder table "
                  "(see make_default_prob_table).")

    result = {
        "params": params,
        "timestamps": [dt for dt in timestamps if dt not in skipped],
        "skipped": skipped,
        "flight_levels": fls,
        "center_lat": center_lat, "center_lon": center_lon,
        "roi": roi, "cells": cells, "star_fixes": star_fixes,
        "ordered_fix_names": ordered_fix_names,
        "csv_files": csv_files, "plot_files": plot_files,
        "manifest_path": manifest_path,
        "output_dir": output_dir, "plot_dir": plot_dir if save_plots else None,
        "summary": summary,
    }
    if return_data:
        result["data"] = data_out
    return result


# ======================================================================
# CLI
# ======================================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="CWAM pre-TRACON weather grid: one CSV per timestamp and "
                    "flight level, with coverage_pct, p_deviation and infeasible.")
    req = p.add_argument_group("required")
    req.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    req.add_argument("--start", required=True, dest="start_time", help="HH:MM (UTC)")
    req.add_argument("--end", required=True, dest="end_time", help="HH:MM (UTC)")
    req.add_argument("--apt", required=True, dest="apt_code",
                     help="airport id, e.g. DTW")

    p.add_argument("--step-minutes", type=int, default=5)
    p.add_argument("--radius-km", type=float, default=300)
    p.add_argument("--cell-size-km", type=float, default=16)
    p.add_argument("--min-cell-area-fraction", type=float, default=0.05)
    p.add_argument("--dbz-threshold", type=float, default=35.0)
    p.add_argument("--echo-top-stat", choices=("p90", "max"), default="p90")
    p.add_argument("--altitude-min-ft", type=float, default=11000)
    p.add_argument("--altitude-max-ft", type=float, default=40000)
    p.add_argument("--altitude-step-ft", type=float, default=2000)
    p.add_argument("--altitudes-ft", type=float, nargs="+", default=None,
                   help="explicit levels in ft; overrides min/max/step")
    p.add_argument("--mask-probability-threshold", type=float, default=0.6)
    p.add_argument("--missing-data-policy",
                   choices=("infeasible", "passable"), default="infeasible")
    p.add_argument("--echo-top-time-tolerance-min", type=float, default=5)
    p.add_argument("--prob-table-csv", default=None,
                   help="calibrated P table; default is the analytic placeholder")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--plot-dir", default=None)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--plot-flight-levels", nargs="+", default=None,
                   help="levels to plot P/infeasible for, e.g. FL350; default all")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    a = _build_arg_parser().parse_args(argv)
    run_cwam(
        date=a.date, start_time=a.start_time, end_time=a.end_time,
        apt_code=a.apt_code,
        step_minutes=a.step_minutes, radius_km=a.radius_km,
        cell_size_km=a.cell_size_km,
        min_cell_area_fraction=a.min_cell_area_fraction,
        dbz_threshold=a.dbz_threshold, echo_top_stat=a.echo_top_stat,
        altitude_min_ft=a.altitude_min_ft, altitude_max_ft=a.altitude_max_ft,
        altitude_step_ft=a.altitude_step_ft, altitudes_ft=a.altitudes_ft,
        mask_probability_threshold=a.mask_probability_threshold,
        missing_data_policy=a.missing_data_policy,
        echo_top_time_tolerance_min=a.echo_top_time_tolerance_min,
        prob_table_csv=a.prob_table_csv,
        data_dir=a.data_dir, output_dir=a.output_dir, plot_dir=a.plot_dir,
        save_plots=a.save_plots, plot_flight_levels=a.plot_flight_levels,
        verbose=not a.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
