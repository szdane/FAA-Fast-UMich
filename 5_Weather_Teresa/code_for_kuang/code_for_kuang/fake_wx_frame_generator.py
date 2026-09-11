# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:31:49 2026

@author: anomi
"""

# This code creates fake frames of weather based on the infeasible_regions CSV created by the wx_grid_creator python code

import pandas as pd
import numpy as np
import os
from pyproj import Geod

# -----------------------------
# USER SETTINGS
# -----------------------------

csv_path = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\wx_csv\infeasible_regions.csv"

out_root = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\wx_csv\fake\east\dt_8min\wx_8"

speeds_knots = [10]
directions = ["East"]

dt_minutes = 8
n_steps = 6
include_t0 = True

# -----------------------------
# SETUP
# -----------------------------

geod = Geod(ellps="WGS84")
meters_per_nm = 1852.0

DIR_TO_AZIMUTH = {
    "East": 90.0
}

df0 = pd.read_csv(csv_path)

# -----------------------------
# GEODESIC MOVE FUNCTION
# -----------------------------


def move_geodesic(lon, lat, dist_m, azimuth_deg):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    az = np.full_like(lon, azimuth_deg)
    dist = np.full_like(lon, dist_m)

    lon2, lat2, _ = geod.fwd(lon, lat, az, dist)
    return lon2, lat2

# -----------------------------
# MAIN LOOP
# -----------------------------


for direction in directions:

    az = DIR_TO_AZIMUTH[direction]

    for speed in speeds_knots:

        meters_per_step = speed * meters_per_nm * (dt_minutes / 60.0)

        start_k = 0 if include_t0 else 1

        for k in range(start_k, n_steps + 1):

            t_min = k * dt_minutes
            dist = k * meters_per_step

            df = df0.copy()

            # Move SW corner
            sw_lon, sw_lat = move_geodesic(
                df["min_lon"].to_numpy(),
                df["min_lat"].to_numpy(),
                dist,
                az
            )

            # Move NE corner
            ne_lon, ne_lat = move_geodesic(
                df["max_lon"].to_numpy(),
                df["max_lat"].to_numpy(),
                dist,
                az
            )

            # Move centroid
            c_lon, c_lat = move_geodesic(
                df["centroid_lon"].to_numpy(),
                df["centroid_lat"].to_numpy(),
                dist,
                az
            )

            df["min_lon"], df["min_lat"] = sw_lon, sw_lat
            df["max_lon"], df["max_lat"] = ne_lon, ne_lat
            df["centroid_lon"], df["centroid_lat"] = c_lon, c_lat

            df["t_minutes"] = t_min
            df["shift_meters"] = dist

            # ✅ TIME ONLY in filename
            filename = f"infeasible_regions_t{t_min:02d}min.csv"

            save_dir = os.path.dirname(out_root)
            save_path = os.path.join(save_dir, filename)

            df.to_csv(save_path, index=False)

print("All direction/speed cases generated successfully.")
