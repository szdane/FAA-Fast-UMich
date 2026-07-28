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

# Load infeasible regions

csv_path = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\wx_csv\infeasible_regions.csv"

df0 = pd.read_csv(csv_path)


# IMPORTANT: Sets geodesic shape and calculation of earth

geod = Geod(ellps="WGS84")

# wind speed
speed_knots = 150

# at what interval are frames required
dt_minutes = 5

# number of intervals
n_steps = 6

# 50 knots conversion to m/s
meters_per_nm = 1852.0

meters_per_step = speed_knots * meters_per_nm * (dt_minutes / 60)


def move_north(lon, lat, dist_m):
    lon = np.asarray(lon, dtype=float)

    lat = np.asarray(lat, dtype=float)

    # Make azimuth and distance arrays matching lon/lat length

    az = np.full_like(lon, 0.0)   # 0 degrees (north) for every point

    dist = np.full_like(lon, float(dist_m))  # same distance for every point

    lon2, lat2, _ = geod.fwd(lon, lat, az, dist)

    return lon2, lat2


frames = []

# Include t=0 frame if you want:
include_t0 = True

start_k = 0 if include_t0 else 1

end_k = n_steps if include_t0 else n_steps

for k in range(start_k, n_steps + 1):

    t_min = k * dt_minutes

    dist = k * meters_per_step

    df = df0.copy()

    # Move bbox corners by moving its SW and NE corners

    # SW corner: (min_lon, min_lat)
    sw_lon, sw_lat = move_north(
        df["min_lon"].to_numpy(), df["min_lat"].to_numpy(), dist)

    # NE corner: (max_lon, max_lat)
    ne_lon, ne_lat = move_north(
        df["max_lon"].to_numpy(), df["max_lat"].to_numpy(), dist)

    # Move centroid
    c_lon, c_lat = move_north(
        df["centroid_lon"].to_numpy(), df["centroid_lat"].to_numpy(), dist)

    # Write back
    df["min_lon"], df["min_lat"] = sw_lon, sw_lat

    df["max_lon"], df["max_lat"] = ne_lon, ne_lat

    df["centroid_lon"], df["centroid_lat"] = c_lon, c_lat

    df["t_minutes"] = t_min

    df["shift_meters"] = dist

    frames.append(df)

out_dir = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\wx_csv\infeasible_regions_fake_150kt"

os.makedirs(out_dir, exist_ok=True)

for df in frames:
    t = int(df["t_minutes"].iloc[0])
    df.to_csv(os.path.join(out_dir, f"infeasible_regions_t{
              t:02d}sec.csv"), index=False)
