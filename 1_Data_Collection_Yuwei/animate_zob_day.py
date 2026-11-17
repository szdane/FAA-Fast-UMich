r"""Render a converted ARTCC IFF CSV as a 1600x900 H.264 MP4.

From the repository root:
    python -m pip install -r "1_Data_Collection_Yuwei\requirements_animation.txt"
    python "1_Data_Collection_Yuwei\animate_zob_day.py" INPUT.csv OUTPUT.mp4 --date 2023-12-22

Input is the output of convert_iff_to_filtered_rows.py, not a raw IFF file.
Naive recTime values are interpreted in --timezone (America/Detroit by default).
For DST-transition days, convert the IFF to UTC first and use --timezone UTC.
The selected day is [00:00, next 00:00); neighboring points are retained only
to interpolate positions at midnight. alt is converted from hundreds of feet.

Defaults: 30 simulated seconds per frame, 24 fps, 10-minute trails, and a final
24:00 frame (2881 frames / about 120 seconds). Markers interpolate between
observations only when the gap is <=120 seconds. No extrapolation is used.
Faint observed points ensure even legs shorter than one frame are visible.
Flights are grouped by fltKey, never by callsign or route. Identical duplicate
observations are removed with a log message; conflicting duplicates are errors.

To replace the flight list with cumulative fleet fuel, first run
compute_fuel_history.py (see its --help and requirements_fuel.txt), then add
--fuel-history OUTPUT_DIR. The saved source hash, date and timezone must match.
The curve shows only fuel accumulated up to the displayed clock, in tonnes;
all trajectories remain visible, including intervals excluded from fuel.
The past curve is displayed at one-minute resolution, with an exact current
value interpolated from the saved piecewise-linear cumulative history.
Without --fuel-history, the original flight-list sidebar is unchanged.

The ARTCC is detected from Source, including sensor prefixes such as 0/ZOB.
Mixed reporting facilities are rejected: split them before rendering. Source
does not identify the timestamp timezone; keep --timezone matched to the CSV.
The old script filename is retained for compatibility, but ZOB is not hardcoded.

FAA AIS enroute high-chart and low-chart ARTCC polygons are drawn separately.
These are 2D chart footprints, not altitude-specific control assignments or
TRACON boundaries. Tracks outside the contours are retained. Missing/invalid
ARTCC geometry is an error; no outline is inferred from observed trajectories.
Unknown-airport flights omitted by the converter cannot appear here.

On first use, the current FAA ARTCC GeoJSON and effective-period metadata are
cached under LOCALAPPDATA\FAA-Fast-UMich (or ~/.cache/FAA-Fast-UMich). The cache
is a fixed snapshot, not automatically refreshed. Its edition is shown in the
video; a warning identifies observations outside its effective period.
Use --boundaries PATH.geojson for an archived/offline WGS84 FAA-schema snapshot.
Required feature properties: IDENT, NAME, TYPE_CODE=ARTCC, LOCAL_TYPE=ARTCC_H/L.
Optional top-level faa_metadata: effective_from/effective_to (ISO dates).
Unspecified boundary dates are visibly marked, not assumed historically valid.
FAA source: https://adds-faa.opendata.arcgis.com/datasets/67885972e4e940b2aa6d74024901c561_0/about

Natural Earth basemap/airport data are also downloaded and cached on first use.
The imageio-ffmpeg package supplies FFmpeg; no PATH setup is needed. Existing
output files are never overwritten. Besides caches, only the MP4 is written.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from html import unescape
import json
import logging
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity


LOGGER = logging.getLogger(__name__)
DAY_SECONDS = 86400
MAX_GAP_SECONDS = 120
GEOGRAPHIC = ccrs.PlateCarree()
FAA_ITEM_URL = "https://www.arcgis.com/sharing/rest/content/items/67885972e4e940b2aa6d74024901c561?f=json"
FAA_LAYER_URL = "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/Boundary_Airspace/FeatureServer/0"
DEFAULT_BOUNDARIES = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "FAA-Fast-UMich" / "artcc_boundaries.geojson"
BOUNDARY_STYLES = {"ARTCC_H": ("#ffffff", "-", "High-chart ARTCC"),
                   "ARTCC_L": ("#f18aed", "--", "Low-chart ARTCC")}


@dataclass
class ArtccBoundary:
    facility: str
    name: str
    contours: list[tuple[str, BaseGeometry]]
    effective_from: Optional[date]
    effective_to: Optional[date]


def reporting_facility(source: str) -> str:
    match = re.fullmatch(r"(?:\d+/)?([A-Z][A-Z0-9]{2,3})", source.strip().upper())
    if match is None:
        raise ValueError(f"Cannot identify a reporting facility from Source={source!r}.")
    return match.group(1)


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        data = json.load(response)
    if not isinstance(data, dict) or "error" in data:
        raise ValueError(f"FAA boundary service error: {data}")
    return data


def read_boundary_dataset(path: Optional[Path] = None) -> dict:
    target = path if path is not None else DEFAULT_BOUNDARIES
    if path is None and not target.exists():
        LOGGER.info("Downloading FAA chart ARTCC polygons to %s", target)
        item = fetch_json(FAA_ITEM_URL)
        description = unescape(re.sub(r"<[^>]*>", " ", item.get("description", "")))
        period = re.search(
            r"Current Effective Date:\s*\d{4}Z\s+(\d{1,2}\s+\w{3}\s+\d{4})"
            r"\s+to\s+\d{4}Z\s+(\d{1,2}\s+\w{3}\s+\d{4})", description,
        )
        if period is None:
            raise ValueError("FAA metadata has no recognizable effective period; supply --boundaries.")
        query = urlencode({"where": "TYPE_CODE = 'ARTCC'",
                           "outFields": "IDENT,NAME,TYPE_CODE,LOCAL_TYPE,ICAO_ID,LEVEL_",
                           "returnGeometry": "true", "outSR": "4326", "f": "geojson"})
        data = fetch_json(f"{FAA_LAYER_URL}/query?{query}")
        if data.get("type") != "FeatureCollection" or not data.get("features") or data.get("exceededTransferLimit"):
            raise ValueError("FAA returned an empty, incomplete, or invalid boundary collection.")
        data["faa_metadata"] = {
            "source": FAA_LAYER_URL,
            "effective_from": datetime.strptime(period.group(1), "%d %b %Y").date().isoformat(),
            "effective_to": datetime.strptime(period.group(2), "%d %b %Y").date().isoformat(),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".artcc-", dir=target.parent) as directory:
            temporary = Path(directory) / "boundaries.geojson"
            temporary.write_text(json.dumps(data), encoding="utf-8")
            if not target.exists():
                temporary.rename(target)
    with target.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if (data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list)
            or data.get("exceededTransferLimit")):
        raise ValueError("Expected a complete GeoJSON FeatureCollection of FAA ARTCC polygons.")
    crs = data.get("crs")
    if crs is not None and crs.get("properties", {}).get("name") not in (
        "EPSG:4326", "urn:ogc:def:crs:EPSG::4326", "urn:ogc:def:crs:OGC:1.3:CRS84",
    ):
        raise ValueError("ARTCC GeoJSON must use WGS84 longitude/latitude (EPSG:4326).")
    return data


def load_artcc_boundary(facility: str, path: Optional[Path] = None) -> ArtccBoundary:
    data = read_boundary_dataset(path)
    features = [feature for feature in data["features"]
                if feature.get("properties", {}).get("IDENT") == facility
                and feature["properties"].get("TYPE_CODE") == "ARTCC"]
    if not features:
        raise ValueError(
            f"No ARTCC outline for Source={facility!r}. TRACON/merged feeds cannot be "
            "treated as ARTCCs; provide matching ARTCC data with --boundaries."
        )
    contours, names = [], set()
    for feature in features:
        properties = feature["properties"]
        chart = properties.get("LOCAL_TYPE")
        geometry = feature.get("geometry")
        if chart not in BOUNDARY_STYLES or not properties.get("NAME"):
            raise ValueError(f"Missing/unsupported ARTCC chart classification or name for {facility}.")
        if not geometry or geometry.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"ARTCC {facility} requires polygon boundary geometry.")
        polygon = shape(geometry)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError(f"Invalid FAA polygon for {facility} ({chart}): {explain_validity(polygon)}. Supply a valid --boundaries snapshot.")
        west, south, east, north = polygon.bounds
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError(f"ARTCC {facility} geometry is not WGS84 longitude/latitude.")
        contours.append((chart, polygon))
        names.add(properties["NAME"])
    metadata = data.get("faa_metadata", {})
    first, last = metadata.get("effective_from"), metadata.get("effective_to")
    if (first is None) != (last is None):
        raise ValueError("Boundary metadata must provide both effective_from and effective_to.")
    effective_from = date.fromisoformat(first) if first is not None else None
    effective_to = date.fromisoformat(last) if last is not None else None
    if effective_from is not None and effective_to is not None and effective_to <= effective_from:
        raise ValueError("Boundary effective_to must follow effective_from.")
    LOGGER.info("Matched Source=%s to %s ARTCC: %s contour(s), edition %s.",
                facility, ", ".join(sorted(names)), len(contours), first or "unspecified")
    return ArtccBoundary(facility, ", ".join(sorted(names)), contours, effective_from, effective_to)


@dataclass
class Flight:
    key: str
    callsign: str
    departure: str
    arrival: str
    samples: np.ndarray  # elapsed seconds, longitude, latitude, altitude in feet
    facility: str
    kinematics: Optional[np.ndarray] = None  # ground speed in knots, vertical rate in ft/min


@dataclass
class Playback:
    flights: list[Flight]
    points: np.ndarray  # time, projected x/y, altitude, flight index; sorted by time
    segments: np.ndarray  # t0/t1, x0/y0, x1/y1, alt0/alt1, flight index; sorted by t1
    extent: tuple[float, float, float, float]
    day_point_count: int
    boundary: ArtccBoundary
    projection: ccrs.Projection


def load_flights(path: Path, day: date, *, include_kinematics: bool = False,
                 full_history: bool = False) -> list[Flight]:
    start = datetime.combine(day, time())
    grouped: dict[str, list[list[float]]] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    facility = None
    required = {"recType", "recTime", "fltKey", "Source", "acId",
                "coord1", "coord2", "alt", "dep", "arrival"}
    if include_kinematics:
        required.update(("groundSpeed", "rateOfClimb"))
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for line_number, row in enumerate(reader, 2):
            if row["recType"] != "3":
                raise ValueError(f"Line {line_number}: expected a type-3 track point.")
            source = reporting_facility(row["Source"])
            if facility is not None and facility != source:
                raise ValueError(f"Multiple reporting facilities ({facility}, {source}); split the CSV by Source before rendering.")
            facility = source
            instant = datetime.fromisoformat(row["recTime"])
            if instant.tzinfo is not None:
                raise ValueError("Expected naive timestamps from the IFF converter.")
            elapsed = (instant - start).total_seconds()
            if not full_history and not -MAX_GAP_SECONDS <= elapsed <= DAY_SECONDS + MAX_GAP_SECONDS:
                continue
            key = row["fltKey"]
            info = (row["acId"].split("_", 1)[0], row["dep"], row["arrival"])
            if not key or not all(info):
                raise ValueError(f"Line {line_number}: missing flight identity or airport metadata.")
            if key in metadata and metadata[key] != info:
                raise ValueError(f"Line {line_number}: conflicting metadata for fltKey {key}.")
            metadata[key] = info
            lon, lat, altitude = float(row["coord2"]), float(row["coord1"]), float(row["alt"]) * 100
            if not np.isfinite((lon, lat, altitude)).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f"Line {line_number}: invalid position or altitude.")
            values = [elapsed, lon, lat, altitude]
            if include_kinematics:
                speed, vertical_rate = float(row["groundSpeed"]), float(row["rateOfClimb"])
                if not np.isfinite((speed, vertical_rate)).all():
                    raise ValueError(f"Line {line_number}: nonfinite speed or vertical rate.")
                values.extend((speed, vertical_rate))
            grouped.setdefault(key, []).append(values)

    if facility is None:
        raise ValueError(f"No observed flights on {day}.")
    flights = []
    duplicate_count = 0
    for key, rows in grouped.items():
        samples = np.asarray(rows, dtype=float)
        samples = samples[np.argsort(samples[:, 0], kind="stable")]
        repeated = np.diff(samples[:, 0]) == 0
        if np.any(np.any(samples[1:, 1:] != samples[:-1, 1:], axis=1) & repeated):
            raise ValueError(f"Conflicting positions at the same time for fltKey {key}.")
        duplicate_count += int(repeated.sum())
        samples = samples[np.r_[True, ~repeated]]
        if full_history or np.any((samples[:, 0] >= 0) & (samples[:, 0] < DAY_SECONDS)):
            kinematics = samples[:, 4:6] if include_kinematics else None
            flights.append(Flight(key, *metadata[key], samples[:, :4], facility, kinematics))
    if not flights:
        raise ValueError(f"No observed flights on {day}.")
    if duplicate_count:
        LOGGER.warning("Removed %s identical duplicate observations.", f"{duplicate_count:,}")
    return flights


def prepare_playback(flights: list[Flight], boundary: ArtccBoundary) -> Playback:
    if {flight.facility for flight in flights} != {boundary.facility}:
        raise ValueError("Reporting facility and ARTCC boundary do not match.")
    point_parts, segment_parts = [], []
    geographic = np.concatenate([flight.samples[:, 1:3] for flight in flights])
    bounds = np.asarray([polygon.bounds for _, polygon in boundary.contours])
    west = min(geographic[:, 0].min(), bounds[:, 0].min()) - 0.4
    east = max(geographic[:, 0].max(), bounds[:, 2].max()) + 0.4
    south = min(geographic[:, 1].min(), bounds[:, 1].min()) - 0.4
    north = max(geographic[:, 1].max(), bounds[:, 3].max()) + 0.4
    if east - west >= 180 or south <= -89 or north >= 89:
        raise ValueError("Polar/antimeridian-spanning coverage needs a dedicated map projection; it cannot be drawn with this regional renderer.")
    center_lat = (south + north) / 2
    projection = ccrs.LambertConformal(central_longitude=(west + east) / 2,
                                     central_latitude=center_lat,
                                     standard_parallels=(center_lat - 6, center_lat + 6))
    for index, flight in enumerate(flights):
        samples = flight.samples
        xy = projection.transform_points(GEOGRAPHIC, samples[:, 1], samples[:, 2])[:, :2]
        points = np.column_stack((samples[:, 0], xy, samples[:, 3], np.full(len(samples), index)))
        point_parts.append(points[(points[:, 0] >= 0) & (points[:, 0] < DAY_SECONDS)])
        before, after = points[:-1], points[1:]
        valid = ((after[:, 0] - before[:, 0] <= MAX_GAP_SECONDS)
                 & (after[:, 0] >= 0) & (before[:, 0] < DAY_SECONDS))
        segment_parts.append(np.column_stack((
            before[valid, 0], after[valid, 0], before[valid, 1:3], after[valid, 1:3],
            before[valid, 3], after[valid, 3], before[valid, 4],
        )))
    points = np.concatenate(point_parts)
    segments = np.concatenate(segment_parts)
    points = points[np.argsort(points[:, 0], kind="stable")]
    segments = segments[np.argsort(segments[:, 1], kind="stable")]
    return Playback(flights, points, segments, (west, east, south, north), len(points), boundary, projection)


def frame_data(playback: Playback, elapsed: float, trail_seconds: float):
    lower = max(0, elapsed - trail_seconds)
    points, segments = playback.points, playback.segments
    left, right = np.searchsorted(points[:, 0], (lower, elapsed), side="right")
    if lower == 0:
        left = 0
    observed = points[left:right]
    left = np.searchsorted(segments[:, 1], lower, side="left")
    right = np.searchsorted(segments[:, 1], elapsed + MAX_GAP_SECONDS, side="right")
    candidates = segments[left:right]
    candidates = candidates[candidates[:, 0] <= elapsed]
    span = candidates[:, 1] - candidates[:, 0]
    first = np.maximum(candidates[:, 0], lower)
    last = np.minimum(candidates[:, 1], elapsed)
    begin_fraction = (first - candidates[:, 0]) / span
    end_fraction = (last - candidates[:, 0]) / span
    movement = candidates[:, 4:6] - candidates[:, 2:4]
    begin = candidates[:, 2:4] + begin_fraction[:, None] * movement
    end = candidates[:, 2:4] + end_fraction[:, None] * movement
    altitude = candidates[:, 6] + end_fraction * (candidates[:, 7] - candidates[:, 6])
    trail_mask = last > first
    trails = np.stack((begin[trail_mask], end[trail_mask]), axis=1)
    fade = 0.1 + 0.65 * (last[trail_mask] - lower) / trail_seconds
    active = candidates[:, 1] >= elapsed
    head_ids = candidates[active, 8].astype(int)
    _, unique = np.unique(head_ids, return_index=True)
    heads = np.column_stack((end[active][unique], altitude[active][unique], head_ids[unique]))
    return trails, altitude[trail_mask], fade, observed, heads


def frame_times(seconds_per_frame: int, trail_seconds: float) -> np.ndarray:
    if seconds_per_frame <= 0 or not np.isfinite(trail_seconds) or trail_seconds < seconds_per_frame:
        raise ValueError("Trail duration must be finite and at least one positive frame step.")
    return np.r_[np.arange(0, DAY_SECONDS, seconds_per_frame), DAY_SECONDS]


def make_figure(playback: Playback, day: date, timezone_name: str, trail_seconds: float,
                fuel_curve: Optional[np.ndarray] = None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#e6edf7",
                         "axes.labelcolor": "#c4cfdf", "xtick.color": "#aab9cf",
                         "ytick.color": "#aab9cf"})
    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0b1424")
    boundary = playback.boundary
    ax = figure.add_axes((0.035, 0.14, 0.735, 0.76), projection=playback.projection)
    ax.set_extent(playback.extent, crs=GEOGRAPHIC)
    ax.set_facecolor("#142b41")
    for feature, options in [
        (cfeature.LAND, {"facecolor": "#1e3040", "edgecolor": "none"}),
        (cfeature.LAKES, {"facecolor": "#142b41", "edgecolor": "#4c6a7b"}),
        (cfeature.COASTLINE, {"edgecolor": "#607b8b"}),
        (cfeature.BORDERS, {"edgecolor": "#8fa1af", "linestyle": ":"}),
        (cfeature.STATES, {"edgecolor": "#425969"}),
    ]:
        ax.add_feature(feature.with_scale("50m"), linewidth=0.5, **options)
    ax.gridlines(color="#678094", alpha=0.22, linewidth=0.5, linestyle="--")
    legend = {}
    for chart, polygon in boundary.contours:
        color, style, label = BOUNDARY_STYLES[chart]
        outline = ax.add_geometries([polygon], GEOGRAPHIC, facecolor="none", edgecolor=color,
                                    linewidth=1.8, linestyle=style, zorder=4)
        outline.set_gid(f"ARTCC_{boundary.facility}_{chart}")
        legend[chart] = Line2D([], [], color=color, linestyle=style, linewidth=1.8,
                              label=f"{boundary.facility} {label}")
    ax.legend(handles=[legend[key] for key in sorted(legend)], loc="upper left",
              fontsize=9, facecolor="#0b1424", edgecolor="#607b8b", framealpha=0.9)
    airport_counts = Counter(code for flight in playback.flights
                             for code in (flight.departure, flight.arrival))
    airport_path = shapereader.natural_earth("10m", "cultural", "airports")
    west, east, south, north = playback.extent
    airports = []
    for record in shapereader.Reader(airport_path).records():
        code = record.attributes.get("gps_code")
        lon, lat = record.geometry.x, record.geometry.y
        if code in airport_counts and west <= lon <= east and south <= lat <= north:
            airports.append((airport_counts[code], code, lon, lat))
    for _, code, lon, lat in sorted(airports, reverse=True)[:12]:
        ax.plot(lon, lat, marker="s", markersize=3, color="#e2c482", transform=GEOGRAPHIC, zorder=4)
        ax.text(lon + 0.07, lat + 0.07, code, fontsize=7, color="#e2c482",
                transform=GEOGRAPHIC, zorder=4)
    figure.text(0.045, 0.945, f"{boundary.facility} | {boundary.name} ARTCC     {day:%B %d, %Y}", fontsize=21, weight="bold")
    figure.text(0.045, 0.913,
                f"{len(playback.flights):,} flight legs  |  {playback.day_point_count:,} observed points"
                "  |  ARTCC detected from Source; tracks outside outlines retained", fontsize=10, color="#aab9cf")
    norm, cmap = Normalize(0, 50000, clip=True), plt.get_cmap("turbo")
    trails = LineCollection([], linewidths=0.65, zorder=5, animated=True)
    ax.add_collection(trails)
    observations = ax.scatter([], [], c=[], s=2, alpha=0.35, cmap=cmap, norm=norm, zorder=5, animated=True)
    heads = ax.scatter([], [], c=[], s=15, cmap=cmap, norm=norm, edgecolors="#eff7ff",
                       linewidths=0.25, zorder=6, animated=True)
    head_labels = [
        ax.annotate("", (0, 0), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color="#ffffff", animated=True, zorder=7,
                    bbox={"facecolor": "#0b1424", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        for _ in range(8 if fuel_curve is None else 0)
    ]
    scale_ax = figure.add_axes((0.075, 0.102, 0.45, 0.014))
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=scale_ax, orientation="horizontal")
    colorbar.set_label("Altitude (feet)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    clock = figure.text(0.795, 0.85, "", fontsize=22, color="#81e3ed", animated=True)
    count_text = figure.text(0.795, 0.765, "", fontsize=13, animated=True)
    fuel_artists = []
    if fuel_curve is None:
        figure.text(0.795, 0.72, "LOWEST 8 ACTIVE FLIGHTS", fontsize=10, color="#aab9cf")
        route_text = figure.text(0.795, 0.685, "", fontsize=9, va="top", linespacing=1.6, animated=True)
        figure.text(0.795, 0.17, "Moving dots: interpolated positions\n"
                    f"Faint tracks: last {trail_seconds / 60:g} minutes\n"
                    "Gaps > 2 min are not connected\n"
                    "No extrapolation beyond observations", fontsize=8, color="#aab9cf", linespacing=1.6)
    else:
        figure.text(0.795, 0.715, "CUMULATIVE FLEET FUEL", fontsize=11, color="#aab9cf")
        route_text = figure.text(0.795, 0.675, "", fontsize=21, color="#81e3ed", animated=True)
        fuel_ax = figure.add_axes((0.82, 0.32, 0.16, 0.30), facecolor="#142b41")
        fuel_ax.set_xlim(0, 24)
        fuel_ax.set_ylim(0, max(1.0, fuel_curve[-1, 1] / 1000 * 1.08))
        fuel_ax.set_xticks([0, 6, 12, 18, 24], ["00", "06", "12", "18", "24"])
        fuel_ax.set_xlabel("Hour of selected day", fontsize=8)
        fuel_ax.set_ylabel("Fuel used (tonnes)", fontsize=9)
        fuel_ax.tick_params(labelsize=8)
        fuel_ax.grid(color="#678094", alpha=0.25, linewidth=0.5)
        fuel_line, = fuel_ax.plot([], [], color="#81e3ed", linewidth=2, animated=True,
                                 gid="daily_cumulative_fuel")
        fuel_marker, = fuel_ax.plot([], [], "o", color="#ffffff", markersize=4, animated=True)
        fuel_artists = [fuel_line, fuel_marker]
        plot_times = np.arange(0, DAY_SECONDS + 1, 60)
        plot_fuel = np.interp(plot_times, fuel_curve[:, 0], fuel_curve[:, 1])
        figure.text(0.795, 0.25, "All flights: Boeing 787-8; 80% MTOW\n"
                    "Ground speed approximates TAS; no wind\n"
                    "Observed segments only; not full journeys\n"
                    "Gaps / speeds outside 100-650 kt excluded\n"
                    "Above-ceiling states: model extrapolation\n"
                    f"Map trails: last {trail_seconds / 60:g} min; no extrapolation",
                    va="top", fontsize=8, color="#aab9cf", linespacing=1.5)
    timeline = figure.add_axes((0.58, 0.085, 0.36, 0.03), facecolor="#0b1424")
    timeline.set_xlim(0, 24)
    timeline.set_ylim(0, 1)
    timeline.set_yticks([])
    timeline.set_xticks([0, 6, 12, 18, 24], ["00:00", "06:00", "12:00", "18:00", "24:00"])
    timeline.tick_params(labelsize=8)
    progress = timeline.axvline(0, color="#81e3ed", linewidth=2, animated=True)
    edition = boundary.effective_from.isoformat() if boundary.effective_from is not None else "unspecified"
    figure.text(0.045, 0.037, "Data: NASA Sherlock IFF | Basemap: Natural Earth"
                f" | FAA AIS boundary edition: {edition} | Clock: {timezone_name}", fontsize=8, color="#8194ad")
    if boundary.effective_from is None or boundary.effective_to is None:
        boundary_note = "Boundary effective dates unspecified; historical suitability is unverified."
    elif not boundary.effective_from <= day < boundary.effective_to:
        boundary_note = f"Boundary edition does NOT cover {day}; this is not a verified historical airspace reconstruction."
    else:
        boundary_note = "High/low-chart outlines are 2D footprints, not altitude-specific control assignments."
    figure.text(0.045, 0.017, boundary_note + " Not a TRACON boundary. Not for navigation.",
                fontsize=8, color="#f1b3df")
    artists = (trails, observations, heads, clock, count_text, route_text, progress,
               *head_labels, *fuel_artists)

    def update(elapsed):
        lines, altitudes, fade, points, aircraft = frame_data(playback, elapsed, trail_seconds)
        colors = cmap(norm(altitudes))
        colors[:, 3] = fade
        trails.set_segments(lines)
        trails.set_color(colors)
        observations.set_offsets(points[:, 1:3])
        observations.set_array(points[:, 3])
        heads.set_offsets(aircraft[:, :2])
        heads.set_array(aircraft[:, 2])
        instant = datetime.combine(day, time(), ZoneInfo(timezone_name)) + timedelta(seconds=float(elapsed))
        clock.set_text(f"{instant:%H:%M:%S}\n{instant:%b %d %Z}")
        count_text.set_text(f"{len(aircraft):,} active tracked flights")
        if fuel_curve is None:
            selected = aircraft[np.argsort(aircraft[:, 2], kind="stable")[:8]]
            labels = []
            for number, (x, y, altitude, flight_index) in enumerate(selected, 1):
                flight = playback.flights[int(flight_index)]
                labels.append(f"{number}. {flight.callsign}  [{flight.key}]\n"
                              f"{flight.departure} > {flight.arrival}   {altitude:,.0f} ft")
                head_labels[number - 1].xy = (x, y)
                head_labels[number - 1].set_text(str(number))
            for number, label in enumerate(head_labels):
                label.set_visible(number < len(selected))
            route_text.set_text("\n".join(labels) if labels else "No active interpolated positions")
        else:
            current_fuel = np.interp(elapsed, fuel_curve[:, 0], fuel_curve[:, 1])
            past = np.searchsorted(plot_times, elapsed, side="left")
            fuel_line.set_data(np.r_[plot_times[:past], elapsed] / 3600,
                               np.r_[plot_fuel[:past], current_fuel] / 1000)
            fuel_marker.set_data([elapsed / 3600], [current_fuel / 1000])
            route_text.set_text(f"{current_fuel / 1000:,.1f} tonnes")
        progress.set_xdata([elapsed / 3600] * 2)

    return figure, artists, update


def render_day(input_path: Path, output_path: Path, day: date, *,
               seconds_per_frame: int = 30, fps: int = 24,
               trail_minutes: float = 10, timezone_name: str = "America/Detroit",
               boundary_path: Optional[Path] = None,
               fuel_history_path: Optional[Path] = None) -> tuple[int, int]:
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    times = frame_times(seconds_per_frame, trail_minutes * 60)
    zone = ZoneInfo(timezone_name)
    start = datetime.combine(day, time(), zone)
    if start.utcoffset() != (start + timedelta(days=1)).utcoffset():
        raise ValueError("Naive local times are ambiguous on DST-transition days; use a UTC-converted CSV and --timezone UTC.")
    fuel_curve = None
    if fuel_history_path is not None:
        from compute_fuel_history import load_daily_fuel

        fuel_curve = load_daily_fuel(fuel_history_path, input_path, day, timezone_name)
    imageio_ffmpeg.get_ffmpeg_exe()
    flights = load_flights(input_path, day)
    boundary = load_artcc_boundary(flights[0].facility, boundary_path)
    if (boundary.effective_from is None or boundary.effective_to is None
            or not boundary.effective_from <= day < boundary.effective_to):
        LOGGER.warning("Boundary edition %s does not establish historical geometry for %s; the limitation will be shown in the MP4.",
                       boundary.effective_from or "unspecified", day)
    playback = prepare_playback(flights, boundary)
    LOGGER.info("%s flight legs; %s in-day observations; %s frames; %.2f seconds of video.",
                f"{len(playback.flights):,}", f"{playback.day_point_count:,}", len(times), len(times) / fps)
    figure, artists, update = make_figure(playback, day, timezone_name, trail_minutes * 60, fuel_curve)
    try:
        figure.canvas.draw()
        background = figure.canvas.copy_from_bbox(figure.bbox)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".artcc-animation-", dir=output_path.parent) as directory:
            temporary = Path(directory) / "day.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(temporary), figure.canvas.get_width_height(), fps=fps,
                codec="libx264", pix_fmt_in="rgba", pix_fmt_out="yuv420p",
                macro_block_size=1, output_params=["-crf", "20", "-preset", "medium",
                                                  "-threads", "2", "-movflags", "+faststart"],
            )
            writer.send(None)
            try:
                for index, elapsed in enumerate(times):
                    update(elapsed)
                    figure.canvas.restore_region(background)
                    for artist in artists:
                        artist.axes.draw_artist(artist) if artist.axes is not None else figure.draw_artist(artist)
                    writer.send(figure.canvas.buffer_rgba())
                    if index % max(1, len(times) // 10) == 0:
                        LOGGER.info("Rendered %s/%s frames.", index + 1, len(times))
            finally:
                writer.close()
            encoded_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(temporary))
            if encoded_frames != len(times):
                raise ValueError(f"Encoder produced {encoded_frames} frames; expected {len(times)}.")
            if output_path.exists():
                raise FileExistsError(f"Output already exists: {output_path}")
            temporary.rename(output_path)
    finally:
        plt.close(figure)
    return len(times), len(playback.flights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--seconds-per-frame", type=int, default=30)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--trail-minutes", type=float, default=10)
    parser.add_argument("--timezone", default="America/Detroit")
    parser.add_argument("--boundaries", type=Path,
                        help="Offline/archived WGS84 FAA-schema GeoJSON; otherwise use/download the cached current FAA edition.")
    parser.add_argument("--fuel-history", type=Path,
                        help="Results directory from compute_fuel_history.py; replace flight IDs with cumulative fleet fuel.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        frames, flights = render_day(args.input, args.output, args.date,
                                    seconds_per_frame=args.seconds_per_frame, fps=args.fps,
                                    trail_minutes=args.trail_minutes, timezone_name=args.timezone,
                                    boundary_path=args.boundaries, fuel_history_path=args.fuel_history)
    except (OSError, ValueError, csv.Error, ZoneInfoNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {args.output}: {frames:,} frames, {flights:,} flight legs.")


if __name__ == "__main__":
    main()
