#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pyproj import Geod
from shapely.geometry import Point, Polygon
import pandas as pd

import matplotlib.pyplot as plt
from shapely.geometry import Polygon



#%%

"""
Created on Wed Jun 17 10:13:57 2026

@author: anomi
"""

COORD_RE = re.compile(
    r"([NS])(\d{2})(\d{2})(\d{4})([EW])(\d{3})(\d{2})(\d{4})"
)


def parse_cifp_coord(coord: str) -> Optional[Tuple[float, float]]:
    """
    Parse CIFP coordinate format:

        NDDMMSSssWDDDMMSSss

    Returns:
        (lat, lon)
    """

    m = COORD_RE.search(coord)

    if not m:
        return None

    ns, lat_d, lat_m, lat_s100, ew, lon_d, lon_m, lon_s100 = m.groups()

    lat = (
        int(lat_d)
        + int(lat_m) / 60.0
        + (int(lat_s100) / 100.0) / 3600.0
    )

    lon = (
        int(lon_d)
        + int(lon_m) / 60.0
        + (int(lon_s100) / 100.0) / 3600.0
    )

    if ns == "S":
        lat *= -1

    if ew == "W":
        lon *= -1

    return lat, lon


def classify_waypoint_record(line: str) -> Optional[str]:
    """
    Identify useful waypoint/fix records.

    Keeps:
      - Enroute waypoints:
            section E, subsection A

      - Terminal waypoints/fixes:
            section P, terminal waypoint subsection C
    """

    if len(line) < 51 or not line.startswith("S"):
        return None

    if line[4] == "E" and line[5] == "A":
        return "ENROUTE_WAYPOINT"

    if line[4] == "P" and len(line) > 12 and line[12] == "C":
        return "TERMINAL_WAYPOINT"

    return None


def extract_waypoint(line: str, line_number: int) -> Optional[dict]:
    """
    Extract waypoint information from a CIFP fixed-width record.
    """

    wp_class = classify_waypoint_record(line)

    if wp_class is None:
        return None

    coord_raw = line[32:51]
    parsed = parse_cifp_coord(coord_raw)

    if parsed is None:
        return None

    lat, lon = parsed

    ident = line[13:18].strip()

    if not ident:
        ident = line[98:103].strip()

    if not ident:
        return None

    return {
        "line_number": line_number,
        "record_type": line[:5],
        "area_code": line[1:4].strip(),
        "section_code": line[4:5],
        "subsection_code": line[5:6] if line[4] == "E" else line[12:13],
        "waypoint_class": wp_class,
        "airport_or_enroute": line[6:10].strip(),
        "ident": ident,
        "lat": lat,
        "lon": lon,
        "coord_raw": coord_raw,
        "raw_record": line,
    }


def main() -> None:
    input_path = Path(
        r"/Users/XXXX/Documents/LATTICE/FAST/Misc/FAACIFP18"
    )

    db_path = Path(
        r"/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/faacifp18.sqlite"
    )

    csv_path = Path(
        r"/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/all_waypoints.csv"
    )

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode = OFF")
    cur.execute("PRAGMA synchronous = OFF")

    cur.execute("""
        CREATE TABLE waypoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_number INTEGER NOT NULL,
            record_type TEXT,
            area_code TEXT,
            section_code TEXT,
            subsection_code TEXT,
            waypoint_class TEXT,
            airport_or_enroute TEXT,
            ident TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            coord_raw TEXT,
            raw_record TEXT NOT NULL
        )
    """)

    wp_batch = []

    with input_path.open("r", encoding="ascii", errors="replace", newline="") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")

            wp = extract_waypoint(line, line_number)

            if wp is not None:
                wp_batch.append(
                    (
                        wp["line_number"],
                        wp["record_type"],
                        wp["area_code"],
                        wp["section_code"],
                        wp["subsection_code"],
                        wp["waypoint_class"],
                        wp["airport_or_enroute"],
                        wp["ident"],
                        wp["lat"],
                        wp["lon"],
                        wp["coord_raw"],
                        wp["raw_record"],
                    )
                )

            if len(wp_batch) >= 10_000:
                cur.executemany("""
                    INSERT INTO waypoints
                    (
                        line_number,
                        record_type,
                        area_code,
                        section_code,
                        subsection_code,
                        waypoint_class,
                        airport_or_enroute,
                        ident,
                        lat,
                        lon,
                        coord_raw,
                        raw_record
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, wp_batch)

                wp_batch.clear()

    if wp_batch:
        cur.executemany("""
            INSERT INTO waypoints
            (
                line_number,
                record_type,
                area_code,
                section_code,
                subsection_code,
                waypoint_class,
                airport_or_enroute,
                ident,
                lat,
                lon,
                coord_raw,
                raw_record
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, wp_batch)

    cur.execute("CREATE INDEX idx_wp_ident ON waypoints(ident)")
    cur.execute("CREATE INDEX idx_wp_lat_lon ON waypoints(lat, lon)")
    cur.execute("CREATE INDEX idx_wp_class ON waypoints(waypoint_class)")

    rows = cur.execute("""
        SELECT ident,
               ROUND(AVG(lat), 8) AS lat,
               ROUND(AVG(lon), 8) AS lon,
               GROUP_CONCAT(DISTINCT waypoint_class) AS waypoint_classes,
               GROUP_CONCAT(DISTINCT area_code) AS area_codes,
               COUNT(*) AS record_count
        FROM waypoints
        GROUP BY ident, ROUND(lat, 8), ROUND(lon, 8)
        ORDER BY ident
    """).fetchall()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "ident",
                "lat",
                "lon",
                "waypoint_classes",
                "area_codes",
                "record_count",
            ]
        )

        writer.writerows(rows)

    conn.commit()

    total_wp = cur.execute("SELECT COUNT(*) FROM waypoints").fetchone()[0]
    unique_wp = len(rows)

    conn.close()

    print(f"Created SQLite database: {db_path}")
    print(f"Created waypoint CSV: {csv_path}")
    print(f"Waypoint records extracted: {total_wp}")
    print(f"Unique waypoint coordinate entries: {unique_wp}")


if __name__ == "__main__":
    main()
    
#%% Extract pre-TRACON waypoints

def build_pre_tracon_area(
    center_lat,
    center_lon,
    boundary_fixes,
    ordered_fix_names,
    radius_m=300_000
):
    """
    Build pre-TRACON region as:

        geodesic circle around airport - TRACON boundary polygon

    Coordinates:
        boundary_fixes should be {fix_name: (lat, lon)}
    """

    # TRACON polygon must be built as (lon, lat) for Shapely
    tracon_polygon = Polygon([
        (boundary_fixes[name][1], boundary_fixes[name][0])
        for name in ordered_fix_names
    ])

    geod = Geod(ellps="WGS84")
    azimuths = np.linspace(0, 360, 361)

    circle_points = []

    for az in azimuths:
        lon2, lat2, _ = geod.fwd(
            center_lon,
            center_lat,
            az,
            radius_m
        )

        circle_points.append((lon2, lat2))

    geodesic_circle = Polygon(circle_points)

    return geodesic_circle.difference(tracon_polygon)


def extract_waypoints_for_region(
    db_path,
    center_lat,
    center_lon,
    boundary_fixes,
    ordered_fix_names,
    radius_m=300_000,
    output_csv=None,
    manual_waypoints=None,
    include_manual_outside_region=True
):
    """
    Extract waypoints from SQLite database that lie within the
    pre-TRACON region.

    Parameters
    ----------
    manual_waypoints : dict or None
        Optional manually defined waypoints in the format:

            {
                "FIXA": (lat, lon),
                "FIXB": (lat, lon)
            }

    include_manual_outside_region : bool
        If True, manually defined waypoints are included even if they are
        outside the pre-TRACON region.

        If False, manually defined waypoints are only included if they fall
        inside the pre-TRACON region.
    """

    region = build_pre_tracon_area(
        center_lat=center_lat,
        center_lon=center_lon,
        boundary_fixes=boundary_fixes,
        ordered_fix_names=ordered_fix_names,
        radius_m=radius_m
    )

    min_lon, min_lat, max_lon, max_lat = region.bounds

    conn = sqlite3.connect(db_path)

    candidate_df = pd.read_sql_query(
        """
        SELECT *
        FROM waypoints
        WHERE lat BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
        """,
        conn,
        params=(min_lat, max_lat, min_lon, max_lon)
    )

    conn.close()

    def inside_region(row):
        point = Point(row["lon"], row["lat"])
        return region.contains(point) or region.touches(point)

    mask = candidate_df.apply(inside_region, axis=1)

    result_df = candidate_df.loc[mask].copy()

    result_df = (
        result_df
        .groupby(["ident", "lat", "lon"], as_index=False)
        .agg({
            "waypoint_class": lambda x: ",".join(sorted(set(x.dropna()))),
            "area_code": lambda x: ",".join(sorted(set(x.dropna()))),
            "record_type": "first",
            "section_code": "first",
            "subsection_code": "first",
            "airport_or_enroute": "first",
            "coord_raw": "first",
            "raw_record": "first",
            "line_number": "first"
        })
        .sort_values("ident")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Add manually defined waypoints
    # --------------------------------------------------
    if manual_waypoints is not None and len(manual_waypoints) > 0:

        manual_rows = []

        for ident, coords in manual_waypoints.items():
            lat, lon = coords

            point = Point(lon, lat)
            in_region = region.contains(point) or region.touches(point)

            if include_manual_outside_region or in_region:
                manual_rows.append({
                    "ident": ident,
                    "lat": lat,
                    "lon": lon,
                    "waypoint_class": "MANUAL_WAYPOINT",
                    "area_code": "MANUAL",
                    "record_type": "MANUAL",
                    "section_code": "MANUAL",
                    "subsection_code": "MANUAL",
                    "airport_or_enroute": "MANUAL",
                    "coord_raw": "",
                    "raw_record": "",
                    "line_number": -1
                })

        if manual_rows:
            manual_df = pd.DataFrame(manual_rows)

            result_df = pd.concat(
                [result_df, manual_df],
                ignore_index=True
            )

            result_df = (
                result_df
                .groupby(["ident", "lat", "lon"], as_index=False)
                .agg({
                    "waypoint_class": lambda x: ",".join(sorted(set(x.dropna()))),
                    "area_code": lambda x: ",".join(sorted(set(x.dropna()))),
                    "record_type": "first",
                    "section_code": "first",
                    "subsection_code": "first",
                    "airport_or_enroute": "first",
                    "coord_raw": "first",
                    "raw_record": "first",
                    "line_number": "first"
                })
                .sort_values("ident")
                .reset_index(drop=True)
            )

    if output_csv is not None:
        result_df.to_csv(output_csv, index=False)

    return result_df, region

db_path = "/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/faacifp18.sqlite"

msp_lat, msp_lon = 44.8820, -93.2217

msp_fixes = {
    "BAINY": (45.7536, -93.6994),

    "MUSCL": (45.0288, -91.7768),

    "KASPR": (43.9655, -93.2470),

    "TORGY": (44.6436, -94.3753),
}

msp_ordered_fixes = [
    "BAINY",
    "MUSCL",
    "KASPR",
    "TORGY",
]

manual_msp_waypoints = {
    "GEP":   (45.145694, -93.373194),

    "KKILR": (44.852839, -92.185589),

    "BLUEM": (44.181350, -93.223797),

    "NITZR": (44.186244, -93.466072),
    
}

msp_waypoints, msp_region = extract_waypoints_for_region(
    db_path=db_path,
    center_lat=msp_lat,
    center_lon=msp_lon,
    boundary_fixes=msp_fixes,
    ordered_fix_names=msp_ordered_fixes,
    radius_m=300_000,
    output_csv="/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/msp_pre_tracon_graph.csv",
    manual_waypoints=manual_msp_waypoints,
    include_manual_outside_region=True
)

#%% Plot and overlay on pre-TRACON map

# --------------------------------------------------
# Load extracted fixes from CSV
# --------------------------------------------------

csv_path = r"/Users/XXXX/Documents/LATTICE/FAST/CSV/Graph_waypoint/msp_pre_tracon_graph.csv"

fix_df = pd.read_csv(csv_path)

# --------------------------------------------------
# Build TRACON boundary from ordered msp fixes
# msp_fixes format: {fix_name: (lat, lon)}
# Shapely/matplotlib plotting uses x=lon, y=lat
# --------------------------------------------------

tracon_lons = [msp_fixes[name][1] for name in msp_ordered_fixes]
tracon_lats = [msp_fixes[name][0] for name in msp_ordered_fixes]

# Close polygon by repeating first point
tracon_lons.append(tracon_lons[0])
tracon_lats.append(tracon_lats[0])

# --------------------------------------------------
# Build true 3-degree geodesic circle around msp
# --------------------------------------------------

geod = Geod(ellps="WGS84")

radius_m = 300_000  # 300 km

theta = np.linspace(0, 360, 361)

circle_lons = []
circle_lats = []

for az in theta:
    lon2, lat2, _ = geod.fwd(
        msp_lon,
        msp_lat,
        az,
        radius_m
    )

    circle_lons.append(lon2)
    circle_lats.append(lat2)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 12))

# Extracted fixes from CSV
ax.scatter(
    fix_df["lon"],
    fix_df["lat"],
    s=12,
    alpha=0.7,
    label=f"Extracted fixes ({len(fix_df)})"
)

# TRACON boundary from ordered fixes
ax.plot(
    tracon_lons,
    tracon_lats,
    linewidth=2,
    linestyle="--",
    color="red",
    label="TRACON Boundary"
)

# Plot msp boundary fixes as individual points
for name in msp_ordered_fixes:
    lat, lon = msp_fixes[name]

    ax.scatter(
        lon,
        lat,
        s=60,
        marker="o"
    )

    ax.text(
        lon,
        lat,
        f" {name}",
        fontsize=9
    )

# 3-degree circle
ax.plot(
    circle_lons,
    circle_lats,
    linewidth=2,
    linestyle="--",
    color="black",
    label="3-degree radius circle"
)

# Airport center
ax.scatter(
    msp_lon,
    msp_lat,
    s=120,
    marker="X",
    label="MSP"
)

ax.text(
    msp_lon,
    msp_lat,
    "MSP",
    fontsize=10,
    fontweight="bold"
)

# Formatting
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("MSP Extracted Fixes, TRACON Boundary, and 3-Degree Circle")

ax.grid(True)
ax.legend()
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------

airport_id = "MSP"

plot_dir = Path(
    "/Users/XXXX/Documents/LATTICE/FAST/Figures/Graph_waypoint"
)

plot_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_plot = plot_dir / f"{airport_id}_pre_tracon_waypoints.png"

print(f"Plot saved to:\n{output_plot}")

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
