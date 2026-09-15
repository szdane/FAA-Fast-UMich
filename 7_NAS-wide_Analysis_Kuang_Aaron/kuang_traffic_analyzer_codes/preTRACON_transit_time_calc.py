#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch pre-TRACON transit time analysis, driven by a TOML config file
with multi-scenario support.

The config defines base settings and optional [[scenarios]] blocks that
each override a subset of settings. Overridable per scenario:
    name, data_dir, output_dir, csv_output_dir,
    center_lat, center_lon, ordered_fixes, star_fixes,
    dest_airports, include_airline_prefixes, exclude_airline_prefixes,
    landing_runway_prefixes, exclude_unknown_runway

Scenarios that share the same data_dir AND region geometry are grouped:
each group makes ONE pass over the data (RD read once per day, IFF read
once per day for the union of the group's join keys). Scenarios with a
different data_dir or region get their own pass automatically.

Usage:
    python pre_tracon_transit_configurable.py [path/to/config.toml]
If no argument is given, CONFIG_PATH below is used, so it also works
from Spyder's %runfile.
"""

import re
import sys
import tomllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon
from pyproj import Geod
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
CONFIG_PATH = Path("/Users/anomi/Documents/LATTICE/FAST/Codes/SHERLOCK_codes/preTRACON_time_calc_config.toml")  # default; CLI arg overrides

if len(sys.argv) > 1:
    CONFIG_PATH = Path(sys.argv[1])
if not CONFIG_PATH.exists():
    raise SystemExit(f"Config file not found: {CONFIG_PATH.resolve()}")

with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

# Region parameters that stay global (not per-scenario)
REGION_RADIUS_KM = float(cfg["region"]["radius_km"])
BOUNDARY_TOLERANCE_KM = float(cfg["region"]["boundary_tolerance_km"])
CHUNKSIZE = int(cfg.get("processing", {}).get("chunksize", 500_000))

# ---- Base settings (defaults for every scenario) ----
rdf = cfg.get("rd_filters", {})
BASE = {
    "name": cfg.get("name", CONFIG_PATH.stem),
    "data_dir": cfg["paths"]["data_dir"],
    "output_dir": cfg["paths"]["output_dir"],
    "csv_output_dir": cfg["paths"]["csv_output_dir"],
    "center_lat": cfg["region"]["center_lat"],
    "center_lon": cfg["region"]["center_lon"],
    "ordered_fixes": cfg["region"]["ordered_fixes"],
    "star_fixes": cfg["region"]["star_fixes"],
    "dest_airports": rdf.get("dest_airports", []),
    "include_airline_prefixes": rdf.get("include_airline_prefixes", []),
    "exclude_airline_prefixes": rdf.get("exclude_airline_prefixes", []),
    "landing_runway_prefixes": rdf.get("landing_runway_prefixes", []),
    "exclude_unknown_runway": rdf.get("exclude_unknown_runway", True),
}
UNKNOWN_RUNWAY_VALUES = {"?", "NONE", "", "nan", "NaN"}

raw_scenarios = cfg.get("scenarios") or [{}]
SCENARIOS = []
for i, s in enumerate(raw_scenarios):
    unknown = set(s) - set(BASE)
    if unknown:
        raise SystemExit(
            f"Scenario {i + 1} has unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(BASE)}"
        )
    scn = {**BASE, **s}  # scenario star_fixes REPLACES the base dict

    # Normalize + validate
    scn["data_dir"] = Path(scn["data_dir"])
    scn["output_dir"] = Path(scn["output_dir"])
    scn["csv_output_dir"] = Path(scn["csv_output_dir"])
    scn["center_lat"] = float(scn["center_lat"])
    scn["center_lon"] = float(scn["center_lon"])
    scn["ordered_fixes"] = list(scn["ordered_fixes"])
    scn["dest_airports"] = set(scn["dest_airports"])
    scn["include_airline_prefixes"] = tuple(scn["include_airline_prefixes"])
    scn["exclude_airline_prefixes"] = tuple(scn["exclude_airline_prefixes"])
    scn["landing_runway_prefixes"] = tuple(scn["landing_runway_prefixes"])

    missing = [n for n in scn["ordered_fixes"] if n not in scn["star_fixes"]]
    if missing:
        raise SystemExit(
            f"Scenario '{scn['name']}': ordered_fixes not found in "
            f"star_fixes: {missing}"
        )
    if len(scn["ordered_fixes"]) < 3:
        raise SystemExit(
            f"Scenario '{scn['name']}': need at least 3 ordered_fixes."
        )
    if not scn["dest_airports"]:
        raise SystemExit(f"Scenario '{scn['name']}': dest_airports is empty.")
    if not scn["data_dir"].is_dir():
        raise SystemExit(
            f"Scenario '{scn['name']}': data_dir not found: {scn['data_dir']}"
        )
    SCENARIOS.append(scn)

print(f"Config: {CONFIG_PATH.resolve()}")
print(f"Scenarios to run: {len(SCENARIOS)}")
for scn in SCENARIOS:
    print(f"  - {scn['name']}: dest={sorted(scn['dest_airports'])}, "
          f"center=({scn['center_lat']}, {scn['center_lon']}), "
          f"include={list(scn['include_airline_prefixes']) or 'all'}, "
          f"exclude={list(scn['exclude_airline_prefixes']) or 'none'}, "
          f"rwy={list(scn['landing_runway_prefixes']) or 'any'}, "
          f"unknown rwy "
          f"{'excluded' if scn['exclude_unknown_runway'] else 'kept'}")
print()

# ---- RD.csv column indices (zero-based) ----
RD_COL_MSN = 11      # col 12: Msn (joins to IFF fltKey)
RD_COL_ACID = 12     # col 13: AcId
RD_COL_AIRLINE = 20  # col 21: Airline
RD_COL_DEST = 25     # col 26: Dest
RD_COL_LDG_RWY = 27  # col 28: LNDRwy

# ---- IFF.csv column indices (zero-based) ----
IFF_COL_RECTYPE = 0  # col 1: recType (3 = track point)
IFF_COL_RECTIME = 1  # col 2: recTime (seconds since 1/1/1970 UTC)
IFF_COL_FLTKEY = 2   # col 3: fltKey (joins to RD Msn)
IFF_COL_LAT = 9      # col 10: coord1 (latitude)
IFF_COL_LON = 10     # col 11: coord2 (longitude)

# IFF rows are ragged (header/flight-plan/track records have different
# widths), so we declare a fixed schema wide enough for the widest
# record type. Short rows are padded with NaN.
IFF_MAX_COLS = 60
RD_MAX_COLS = 70

KM_PER_DEG_LAT = 110.574

# ============================================================
# REGION CONSTRUCTION (cached per unique geometry)
# ============================================================
def region_signature(scn):
    """Hashable signature of a scenario's region geometry."""
    fixes = tuple(
        (n, float(scn["star_fixes"][n][0]), float(scn["star_fixes"][n][1]))
        for n in scn["ordered_fixes"]
    )
    return (scn["center_lat"], scn["center_lon"], fixes)


_region_cache = {}

def build_region(scn):
    """Build (and cache) the region geometry for a scenario."""
    sig = region_signature(scn)
    if sig in _region_cache:
        return _region_cache[sig]

    center_lat, center_lon = scn["center_lat"], scn["center_lon"]
    # Waypoints are [lat, lon] in the config; shapely wants (lon, lat).
    star_fixes_lonlat = {
        name: (float(lon), float(lat))
        for name, (lat, lon) in scn["star_fixes"].items()
    }
    tracon_polygon = Polygon(
        [star_fixes_lonlat[n] for n in scn["ordered_fixes"]]
    )

    geod = Geod(ellps="WGS84")
    azimuths = np.linspace(0, 360, 361)
    lons, lats = [], []
    for az in azimuths:
        lon2, lat2, _ = geod.fwd(
            center_lon, center_lat, az, REGION_RADIUS_KM * 1000.0
        )
        lons.append(lon2)
        lats.append(lat2)
    geodesic_circle = Polygon(zip(lons, lats))

    km_per_deg_lon = 111.320 * np.cos(np.radians(center_lat))
    tracon_km = Polygon(
        [
            ((lo - center_lon) * km_per_deg_lon,
             (la - center_lat) * KM_PER_DEG_LAT)
            for lo, la in tracon_polygon.exterior.coords
        ]
    )

    region = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "star_fixes_lonlat": star_fixes_lonlat,
        "ordered_fix_names": list(scn["ordered_fixes"]),
        "tracon_polygon": tracon_polygon,
        "geodesic_circle": geodesic_circle,
        "pre_TRACON_area": geodesic_circle.difference(tracon_polygon),
        "tracon_km": tracon_km,
        "km_per_deg_lon": km_per_deg_lon,
    }
    _region_cache[sig] = region
    return region

# ============================================================
# HELPERS
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    earth_radius_km = 6371.0088
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return earth_radius_km * 2 * np.arcsin(np.sqrt(a))


FNAME_RE = re.compile(r"^(IFF|RD)_([A-Z0-9]+)_(\d{8})_", re.IGNORECASE)


def find_file_pairs(folder):
    """
    Pair IFF and RD csv files on (facility, YYYYMMDD) parsed from the name.
    Returns a sorted list of (date_str, iff_path, rd_path).
    """
    iff_files, rd_files = {}, {}
    for path in sorted(folder.glob("*.csv")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        kind, facility, date = m.group(1).upper(), m.group(2).upper(), m.group(3)
        key = (facility, date)
        (iff_files if kind == "IFF" else rd_files)[key] = path
    pairs = []
    for key in sorted(iff_files):
        if key in rd_files:
            pairs.append((key[1], iff_files[key], rd_files[key]))
        else:
            print(f"WARNING: no RD file for {iff_files[key].name} - skipped")
    for key in sorted(set(rd_files) - set(iff_files)):
        print(f"WARNING: no IFF file for {rd_files[key].name} - skipped")
    return pairs


def read_rd(rd_path):
    """Read one RD file (read once per day, filtered per scenario)."""
    # No usecols here: the C parser rejects names wider than the file's
    # widest row when usecols is also given. Read all, then select.
    rd = pd.read_csv(
        rd_path,
        header=None,
        names=range(RD_MAX_COLS),
        low_memory=False,
    )
    rd = rd[[RD_COL_MSN, RD_COL_ACID, RD_COL_AIRLINE, RD_COL_DEST,
             RD_COL_LDG_RWY]]
    rd = rd.rename(
        columns={
            RD_COL_MSN: "Msn",
            RD_COL_ACID: "AcId",
            RD_COL_AIRLINE: "Airline",
            RD_COL_DEST: "Dest",
            RD_COL_LDG_RWY: "LNDRwy",
        }
    )
    for col in ("AcId", "Airline", "Dest", "LNDRwy"):
        rd[col] = rd[col].astype(str).str.strip()
    return rd


def filter_rd(rd, scn):
    """Apply one scenario's RD filters. Returns (msn key set, AcId map)."""
    if scn["include_airline_prefixes"]:
        rd = rd[rd["Airline"].str.startswith(scn["include_airline_prefixes"])]
    if scn["exclude_airline_prefixes"]:
        rd = rd[~rd["Airline"].str.startswith(scn["exclude_airline_prefixes"])]
    rd = rd[rd["Dest"].isin(scn["dest_airports"])]
    if scn["exclude_unknown_runway"]:
        rd = rd[~rd["LNDRwy"].isin(UNKNOWN_RUNWAY_VALUES)]
    if scn["landing_runway_prefixes"]:
        rd = rd[rd["LNDRwy"].str.startswith(scn["landing_runway_prefixes"])]

    msn = pd.to_numeric(rd["Msn"], errors="coerce")
    keys = set(msn.dropna().astype(np.int64))
    acid_lookup = dict(zip(msn.astype("Int64"), rd["AcId"]))
    return keys, acid_lookup


def get_in_region_points(iff_path, msn_keys, region):
    """IFF: recType 3, fltKey in msn_keys, inside the pre-TRACON area."""
    chunks = []
    usecols = [IFF_COL_RECTYPE, IFF_COL_RECTIME, IFF_COL_FLTKEY,
               IFF_COL_LAT, IFF_COL_LON]
    # No usecols in read_csv (see note in read_rd); select per chunk.
    for chunk in pd.read_csv(
        iff_path,
        header=None,
        names=range(IFF_MAX_COLS),
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        chunk = chunk[usecols]
        chunk = chunk.rename(
            columns={
                IFF_COL_RECTYPE: "recType",
                IFF_COL_RECTIME: "recTime",
                IFF_COL_FLTKEY: "fltKey",
                IFF_COL_LAT: "lat",
                IFF_COL_LON: "lon",
            }
        )
        chunk["recType"] = pd.to_numeric(chunk["recType"], errors="coerce")
        chunk = chunk[chunk["recType"] == 3]
        chunk["fltKey"] = pd.to_numeric(chunk["fltKey"], errors="coerce")
        chunk = chunk.dropna(subset=["fltKey"])
        chunk["fltKey"] = chunk["fltKey"].astype(np.int64)
        chunk = chunk[chunk["fltKey"].isin(msn_keys)]
        chunk["recTime"] = pd.to_numeric(chunk["recTime"], errors="coerce")
        chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
        chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
        chunk = chunk.dropna(subset=["recTime", "lat", "lon"])
        if not chunk.empty:
            inside = shapely.contains_xy(
                region["pre_TRACON_area"],
                chunk["lon"].to_numpy(),
                chunk["lat"].to_numpy(),
            )
            chunk = chunk[inside]
        if not chunk.empty:
            chunks.append(chunk[["recTime", "fltKey", "lat", "lon"]])
    if not chunks:
        return pd.DataFrame(columns=["recTime", "fltKey", "lat", "lon"])
    return pd.concat(chunks, ignore_index=True)


def compute_transit(iff, acid_lookup, date_str):
    """Entry/exit record and transit time per flight."""
    iff = iff.sort_values(["fltKey", "recTime"])
    grouped = iff.groupby("fltKey")
    first = grouped.first()
    last = grouped.last()
    transit = pd.DataFrame(
        {
            "entry_time_epoch": first["recTime"],
            "exit_time_epoch": last["recTime"],
            "entry_lat": first["lat"],
            "entry_lon": first["lon"],
            "exit_lat": last["lat"],
            "exit_lon": last["lon"],
            "num_points": grouped.size(),
        }
    )
    transit["transit_seconds"] = (
        transit["exit_time_epoch"] - transit["entry_time_epoch"]
    )
    transit["transit_minutes"] = transit["transit_seconds"] / 60.0
    transit["AcId"] = transit.index.map(acid_lookup)
    transit["entry_timestamp_utc"] = pd.to_datetime(
        transit["entry_time_epoch"], unit="s", utc=True
    )
    transit["exit_timestamp_utc"] = pd.to_datetime(
        transit["exit_time_epoch"], unit="s", utc=True
    )
    transit["file_date"] = date_str
    transit = transit.reset_index()
    transit["flight_uid"] = date_str + "_" + transit["fltKey"].astype(str)
    # Drop degenerate flights with a single in-region point
    return transit[transit["num_points"] > 1]


def finalize_scenario(scn, region, daily_transits):
    """Full-trajectory filter, stats, CSVs, and plots for one scenario."""
    name = scn["name"]
    output_dir = scn["output_dir"]
    csv_output_dir = scn["csv_output_dir"]
    center_lat = region["center_lat"]
    center_lon = region["center_lon"]

    print(f"\n===== {name} =====")
    if not daily_transits:
        print("  No transit data produced - skipped.")
        return

    transit = pd.concat(daily_transits, ignore_index=True)
    n_days = transit["file_date"].nunique()
    print(f"Combined flights across {n_days} days: {len(transit):,}")

    # ---- Full-trajectory filter ----
    transit["entry_distance_km"] = haversine_km(
        transit["entry_lat"], transit["entry_lon"], center_lat, center_lon
    )
    transit["starts_at_boundary"] = (
        transit["entry_distance_km"]
        >= REGION_RADIUS_KM - BOUNDARY_TOLERANCE_KM
    )
    exit_pts_km = shapely.points(
        (transit["exit_lon"].to_numpy() - center_lon)
        * region["km_per_deg_lon"],
        (transit["exit_lat"].to_numpy() - center_lat) * KM_PER_DEG_LAT,
    )
    transit["exit_distance_to_tracon_km"] = shapely.distance(
        region["tracon_km"], exit_pts_km
    )
    transit["ends_at_tracon"] = (
        transit["exit_distance_to_tracon_km"] <= BOUNDARY_TOLERANCE_KM
    )
    transit["full_trajectory"] = (
        transit["starts_at_boundary"] & transit["ends_at_tracon"]
    )

    n_all = len(transit)
    n_full = int(transit["full_trajectory"].sum())
    print(f"  entering at the region boundary: "
          f"{int(transit['starts_at_boundary'].sum()):,}")
    print(f"  ending at the TRACON boundary:   "
          f"{int(transit['ends_at_tracon'].sum()):,}")
    print(f"  full trajectories (both):        {n_full:,} (kept)")
    print(f"  partial trajectories:            {n_all - n_full:,} (excluded)")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    transit_out = csv_output_dir / "pre_tracon_transit_times_all_days.csv"
    transit.sort_values("transit_minutes").to_csv(transit_out, index=False)
    print(f"Saved per-flight transit times to: {transit_out}")

    kept = transit[transit["full_trajectory"]]
    if kept.empty:
        print("  No full trajectories - skipping stats/plots for this "
              "scenario (try increasing boundary_tolerance_km).")
        return

    # ---- Box-and-whisker statistics ----
    data = kept["transit_minutes"].dropna().to_numpy()
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mean = data.mean()
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    whisker_min = data[data >= lo_fence].min()
    whisker_max = data[data <= hi_fence].max()
    outliers = data[(data < whisker_min) | (data > whisker_max)]

    stats = {
        "analysis": name,
        "n_days": n_days,
        "n_flights": len(data),
        "mean_min": mean,
        "std_min": data.std(ddof=1),
        "data_min_min": data.min(),
        "whisker_min_min": whisker_min,
        "q1_min": q1,
        "median_min": median,
        "q3_min": q3,
        "whisker_max_min": whisker_max,
        "data_max_min": data.max(),
        "iqr_min": iqr,
        "n_outliers": len(outliers),
    }
    print("\n  Transit time statistics (minutes):")
    for k, v in stats.items():
        print(f"    {k:>18}: {v:,.2f}" if isinstance(v, float)
              else f"    {k:>18}: {v}")
    pd.DataFrame([stats]).to_csv(
        csv_output_dir / "pre_tracon_transit_stats_all_days.csv", index=False
    )

    per_day = (
        kept.groupby("file_date")["transit_minutes"]
        .agg(n="size", mean="mean", median="median",
             q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
        .round(2)
    )
    per_day.to_csv(csv_output_dir / "pre_tracon_transit_per_day_summary.csv")

    # ---- Boxplot ----
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.boxplot(
        data,
        vert=True,
        showmeans=True,
        meanline=True,
        showfliers=False,  # outliers hidden so the plot stays readable
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", alpha=0.8),
        medianprops=dict(color="firebrick", linewidth=2),
        meanprops=dict(color="darkgreen", linewidth=2, linestyle="--"),
    )
    labels = [
        (whisker_max, f"Max whisker: {whisker_max:.1f}"),
        (q3, f"Q3 (75th): {q3:.1f}"),
        (mean, f"Mean: {mean:.1f}"),
        (median, f"Median: {median:.1f}"),
        (q1, f"Q1 (25th): {q1:.1f}"),
        (whisker_min, f"Min whisker: {whisker_min:.1f}"),
    ]
    for yval, text in labels:
        ax.annotate(
            text,
            xy=(1.19, yval),
            xytext=(1.28, yval),
            va="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )
    ax.set_xlim(0.5, 2.1)
    ax.set_xticks([])
    ax.set_ylabel("Transit time (minutes)")
    ax.set_title(
        f"Pre-TRACON transit times - {name}\n"
        f"{'/'.join(sorted(scn['dest_airports']))} arrivals, "
        f"full trajectories only (+/-{BOUNDARY_TOLERANCE_KM:.0f} km)\n"
        f"{n_days} days, n = {len(data):,} flights, "
        f"{len(outliers):,} outliers not shown "
        f"(mean {mean:.1f} min, median {median:.1f} min)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_out = output_dir / "pre_tracon_transit_boxplot_all_days.png"
    fig.savefig(plot_out, dpi=150)
    print(f"\n  Saved plot to: {plot_out}")
    plt.show()

    # ---- Verification plot ----
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    circle_x, circle_y = region["geodesic_circle"].exterior.xy
    tracon_x, tracon_y = region["tracon_polygon"].exterior.xy
    ax2.fill(circle_x, circle_y, color="lightsteelblue", alpha=0.35,
             zorder=1, label="Pre-TRACON region")
    ax2.fill(tracon_x, tracon_y, color="white", zorder=2)
    ax2.fill(tracon_x, tracon_y, color="navajowhite", alpha=0.6, zorder=2)
    ax2.plot(circle_x, circle_y, color="steelblue", lw=1.5, zorder=3)
    ax2.plot(tracon_x, tracon_y, color="darkorange", lw=2, zorder=4,
             label="TRACON boundary")
    for fix_name in region["ordered_fix_names"]:
        fx_lon, fx_lat = region["star_fixes_lonlat"][fix_name]
        ax2.plot(fx_lon, fx_lat, "^", color="darkorange", ms=6, zorder=5)
        ax2.annotate(fix_name, (fx_lon, fx_lat), textcoords="offset points",
                     xytext=(5, 4), fontsize=7)
    excluded = transit[~transit["full_trajectory"]]
    ax2.scatter(excluded["entry_lon"], excluded["entry_lat"], s=14,
                marker="x", color="gray", alpha=0.6, zorder=6,
                label=f"Excluded entries (n={len(excluded):,})")
    ax2.scatter(excluded["exit_lon"], excluded["exit_lat"], s=14,
                marker="+", color="dimgray", alpha=0.6, zorder=6,
                label=f"Excluded exits (n={len(excluded):,})")
    ax2.scatter(kept["entry_lon"], kept["entry_lat"], s=10, color="green",
                alpha=0.5, zorder=7,
                label=f"Entry points, kept (n={len(kept):,})")
    ax2.scatter(kept["exit_lon"], kept["exit_lat"], s=10, color="red",
                alpha=0.5, zorder=8,
                label=f"Exit points, kept (n={len(kept):,})")
    ax2.plot(center_lon, center_lat, "k*", ms=12, zorder=8,
             label="Region center")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_title(
        f"Verification: entry (green) and exit (red) points - {name}\n"
        "Expected: entries near the outer circle, exits near the "
        "TRACON boundary"
    )
    ax2.set_aspect(1 / np.cos(np.radians(center_lat)))
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    verif_out = output_dir / "pre_tracon_entry_exit_verification_all_days.png"
    fig2.savefig(verif_out, dpi=150)
    print(f"  Saved verification plot to: {verif_out}")
    plt.show()


# ============================================================
# GROUP SCENARIOS BY (data_dir, region) AND RUN ONE PASS EACH
# ============================================================
groups = {}
for i, scn in enumerate(SCENARIOS):
    key = (str(scn["data_dir"]), region_signature(scn))
    groups.setdefault(key, []).append(i)

print(f"Scenario groups (shared data pass): {len(groups)}\n")

scenario_daily = {i: [] for i in range(len(SCENARIOS))}

for (data_dir_str, _), scn_indices in groups.items():
    group_scns = [SCENARIOS[i] for i in scn_indices]
    region = build_region(group_scns[0])
    data_dir = Path(data_dir_str)

    print(f"### Data pass: {data_dir}")
    print(f"    scenarios: {[s['name'] for s in group_scns]}")

    pairs = find_file_pairs(data_dir)
    if not pairs:
        print(f"WARNING: no IFF/RD csv file pairs found in {data_dir} - "
              f"group skipped\n")
        continue
    print(f"Found {len(pairs)} IFF/RD file pairs\n")

    for date_str, iff_path, rd_path in pairs:
        print(f"--- {date_str} ---")
        rd = read_rd(rd_path)

        # RD filters per scenario; union of keys drives one IFF read
        per_scn = {}
        union_keys = set()
        for i in scn_indices:
            keys, acid_lookup = filter_rd(rd, SCENARIOS[i])
            per_scn[i] = (keys, acid_lookup)
            union_keys |= keys
            print(f"  [{SCENARIOS[i]['name']}] RD flights after filters: "
                  f"{len(keys):,}")

        if not union_keys:
            print("  skipped (no flights survived any scenario's filters)")
            continue

        iff = get_in_region_points(iff_path, union_keys, region)
        print(f"  in-region track points (union):  {len(iff):,} "
              f"({iff['fltKey'].nunique():,} flights)")
        if iff.empty:
            print("  skipped (no in-region track points)")
            continue

        for i in scn_indices:
            keys, acid_lookup = per_scn[i]
            if not keys:
                continue
            sub = iff[iff["fltKey"].isin(keys)]
            if sub.empty:
                continue
            day = compute_transit(sub, acid_lookup, date_str)
            if not day.empty:
                scenario_daily[i].append(day)
    print()

# ============================================================
# FINALIZE EACH SCENARIO
# ============================================================
for i, scn in enumerate(SCENARIOS):
    finalize_scenario(scn, build_region(scn), scenario_daily[i])

print("\nAll scenarios complete.")