"""Data acquisition.

* N0Q composite reflectivity: PNG + WLD pairs from the Iowa Environmental
  Mesonet archive (same source/layout as wx_grid_creator_3).
* Echo tops: NOAA MRMS EchoTop_18 (height of the 18 dBZ echo top, the same
  echo-top definition used by the MIT/LL CWAM papers), GRIB2 files from the
  public NOAA S3 archive (noaa-mrms-pds), ~2 minute cadence, CONUS mosaic.
"""

from __future__ import annotations

import gzip
import re
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

from .config import PipelineConfig

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_MRMS_TS_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2")


# ----------------------------------------------------------------------
# N0Q (IEM)
# ----------------------------------------------------------------------
def n0q_filenames(dt: datetime):
    stamp = dt.strftime("%Y%m%d%H%M")
    return f"n0q_{stamp}.png", f"n0q_{stamp}.wld"


def download_n0q(cfg: PipelineConfig, timestamps, session=None):
    """Download N0Q png+wld for each timestamp. Returns {dt: png_path}."""
    session = session or requests.Session()
    save_dir = cfg.data_dir / "n0q"
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dt in timestamps:
        base = cfg.sources.n0q_base_url.format(
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

    with ThreadPoolExecutor(max_workers=cfg.sources.download_workers) as ex:
        futures = {ex.submit(_get, url, path): path for url, path in tasks}
        for fut in as_completed(futures):
            print("  [n0q]", fut.result())

    out = {}
    for dt in timestamps:
        png, wld = n0q_filenames(dt)
        p, w = save_dir / png, save_dir / wld
        if p.exists() and w.exists():
            out[dt] = p
        else:
            print(f"  [n0q] WARNING: missing files for {dt:%Y-%m-%d %H:%M}, skipping")
    return out


# ----------------------------------------------------------------------
# MRMS EchoTop_18 (NOAA S3 archive)
# ----------------------------------------------------------------------
def list_mrms_keys(cfg: PipelineConfig, date: datetime, session=None):
    """List available EchoTop_18 object keys for a UTC date via the S3 REST API."""
    session = session or requests.Session()
    prefix = f"{cfg.sources.mrms_prefix}/{date:%Y%m%d}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(cfg.sources.mrms_bucket_url, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for el in root.iter(f"{_S3_NS}Key"):
            keys.append(el.text)
        truncated = root.findtext(f"{_S3_NS}IsTruncated") == "true"
        token = root.findtext(f"{_S3_NS}NextContinuationToken")
        if not truncated or not token:
            break
    return keys


def parse_mrms_timestamp(key: str):
    m = _MRMS_TS_RE.search(key)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def pair_echo_tops(keys, timestamps, tolerance_min: float):
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
            print(
                f"  [eet] WARNING: no EchoTop file within "
                f"{tolerance_min} min of {dt:%H:%M}, skipping"
            )
    return pairs


def download_echo_tops(cfg: PipelineConfig, timestamps, session=None):
    """Download and gunzip paired EchoTop_18 GRIB2 files. Returns {dt: grib2_path}."""
    session = session or requests.Session()
    save_dir = cfg.data_dir / "echotop"
    save_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted({dt.date() for dt in timestamps})
    keys = []
    for d in dates:
        keys += list_mrms_keys(cfg, datetime(d.year, d.month, d.day), session)
    print(f"  [eet] {len(keys)} EchoTop_18 objects listed for {dates}")

    pairs = pair_echo_tops(keys, timestamps, cfg.analysis.echo_top_time_tolerance_min)

    out = {}
    for dt, (t_eet, key) in pairs.items():
        gz_path = save_dir / Path(key).name
        grib_path = gz_path.with_suffix("")  # strip .gz
        if not grib_path.exists():
            if not gz_path.exists():
                url = f"{cfg.sources.mrms_bucket_url}/{key}"
                r = session.get(url, timeout=60)
                r.raise_for_status()
                gz_path.write_bytes(r.content)
            with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            gz_path.unlink(missing_ok=True)
        print(f"  [eet] {dt:%H:%M} -> {grib_path.name} (dt={t_eet:%H:%M:%S}Z)")
        out[dt] = grib_path
    return out


def local_echo_tops(local_dir, timestamps, tolerance_min: float):
    """Use pre-downloaded EchoTop files (*.grib2 / *.grib2.gz / *.tif) instead of S3.

    Files must contain an MRMS-style _YYYYMMDD-HHMMSS timestamp in the name,
    e.g. MRMS_EchoTop_18_00.50_20250403-040039.grib2
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        print(f"  [eet] WARNING: {local_dir} does not exist")
        return {}
    stamped = []
    for p in sorted(local_dir.iterdir()):
        m = re.search(r"_(\d{8})-(\d{6})", p.name)
        if m:
            stamped.append(
                (datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), p)
            )
    out = {}
    for dt in timestamps:
        if not stamped:
            break
        t_best, p_best = min(stamped, key=lambda tk: abs((tk[0] - dt).total_seconds()))
        if abs((t_best - dt).total_seconds()) <= tolerance_min * 60.0:
            out[dt] = p_best
        else:
            print(f"  [eet] WARNING: no local EchoTop file near {dt:%H:%M}")
    return out
