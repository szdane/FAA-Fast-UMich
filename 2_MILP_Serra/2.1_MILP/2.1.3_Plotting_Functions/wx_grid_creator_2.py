# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:46:56 2025

@author: anomi
"""

from datetime import datetime, timedelta
from PIL import Image
import imageio.v2 as imageio
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon, box, MultiPolygon

from pathlib import Path
import rasterio
from rasterio.features import geometry_mask, rasterize

from affine import Affine
from scipy.ndimage import label

import pandas as pd
from rasterio.transform import array_bounds
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

# %% This downloads images from the IEM server


def download_single_n0q(dt, ext, save_dir):
    """Download a single n0q file (png or wld) for a given datetime."""
    dt_str = dt.strftime("%Y%m%d%H%M")
    date_url = dt.strftime("%Y/%m/%d")
    base_url = f"https://mesonet.agron.iastate.edu/archive/data/{
        date_url}/GIS/uscomp/"

    fname = f"n0q_{dt_str}.{ext}"
    url = base_url + fname
    save_path = os.path.join(save_dir, fname)

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return f"Downloaded {fname} → {save_path}"
        else:
            return f"Failed to download {fname}: HTTP {r.status_code}"
    except Exception as e:
        return f"Error downloading {fname}: {e}"


def download_n0q_for_hour(date_str, hour, save_dir, max_workers=8):
    """
    Download n0q PNG + WLD files every 5 minutes for a given hour.
    Saves files into save_dir and uses concurrency for speed.
    """
    os.makedirs(save_dir, exist_ok=True)

    base_dt = datetime.strptime(f"{date_str} {hour:02d}:00", "%Y-%m-%d %H:%M")

    # Build list of (datetime, ext) tasks
    tasks = []
    for i in range(12):  # 60 minutes / 5 minutes = 12 timestamps
        dt = base_dt + timedelta(minutes=5 * i)
        for ext in ['png', 'wld']:
            tasks.append((dt, ext))

    print(f"Starting downloads for {
          len(tasks)} files using {max_workers} workers...")

    # Run downloads concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_single_n0q, dt, ext, save_dir): (dt, ext)
            for dt, ext in tasks
        }

        for future in as_completed(future_to_task):
            msg = future.result()
            print(msg)


# Example usage:
download_n0q_for_hour(
    "2025-04-03",
    2,
    r"mesonet_img",
    max_workers=8,
)

# %% Creates the ROI for future use

# Center (lat, lon) of DTW
dtw_lat, dtw_lon = 42.2125, -83.3534

# Coordinates of STAR waypoints
star_fixes = {"BONZZ": (-82.7972, 41.7483), "CRAKN": (-82.9405, 41.6730), "CUUGR": (-83.0975, 42.3643), "FERRL": (-82.6093, 42.4165), "GRAYT": (-83.6020, 42.9150), "HANBL": (-84.1773, 41.7375), "HAYLL": (-84.2975, 41.9662),
              "HTROD": (-83.3442, 42.0278), "KKISS": (-83.7620, 42.5443), "KLYNK": (-82.9888, 41.8793), "LAYKS": (-83.5498, 42.8532), "LECTR": (-84.0217, 41.9183), "RKCTY": (-83.9603, 42.6869), "VCTRZ": (-84.0670, 41.9878)}

ordered_fix_names = ["CRAKN", "BONZZ", "FERRL",
                     "LAYKS", "GRAYT", "RKCTY", "HAYLL", "HANBL"]

# Boundary construction of TRACON region
tracon_polygon = Polygon([star_fixes[name] for name in ordered_fix_names])

radius = 300000  # 300 KM

# Create a Geod object for WGS84 Earth
geod = Geod(ellps="WGS84")

# Sample bearings from 0 to 360 degrees
azimuths = np.linspace(0, 360, 361)

lons = []
lats = []

for az in azimuths:
    lon2, lat2, back_az = geod.fwd(dtw_lon, dtw_lat, az, radius)
    lons.append(lon2)
    lats.append(lat2)

# Build a polygon from the resulting coordinates (lon, lat)
geodesic_circle = Polygon(zip(lons, lats))

# Build pre-tracon area by subtracting
pre_TRACON_area = geodesic_circle.difference(tracon_polygon)

# %% Removes all pixels except for ROI and saves photos again

# provide folder path for png and wld files
input_folder = Path(
    "mesonet_img")

# provide folder path for saved geoTIFF files

output_folder = Path("mesonet_img/masked_data")

os.makedirs(output_folder, exist_ok=True)

# Get all the png images
png_files = sorted(input_folder.glob("*.png"))

# Get wld files as well
for png_path in png_files:

    wld_path = png_path.with_suffix(".wld")

    # Check for ensuring .wld file exists
    if not wld_path.exists():

        print(f"Skipping {png_path.name}: no matching world file")

        continue

    print(f"Processing {png_path.name}...")

    # Georeferenced Image processing starts here!

    # Open the georeferenced image (png + wld file)
    with rasterio.open(png_path) as src:

        data = src.read(1)

        transform = src.transform

        profile = src.profile

        # Build mask for NEXRAD radar
        mask = geometry_mask(
            [pre_TRACON_area],
            out_shape=data.shape,
            transform=transform,
            invert=True  # True = mask is True inside region
        )

        masked_data = np.where(mask, data, np.nan)

     # Create masked array with a nodata value

        nodata_value = profile.get("nodata")

     # If nodata is missing or None, choose a sensible default
        if nodata_value is None:
            nodata_value = 0  # or another value you like

        masked_data = data.copy()
        masked_data[~mask] = nodata_value

        nodata_value = nodata_value  # fallback to 0 if none
        masked_data = data.copy()
        masked_data[~mask] = nodata_value        # outside ROI → nodata

        # Update profile for output GeoTIFF
        profile.update(
            driver="GTiff",
            dtype=masked_data.dtype,
            count=1,
            nodata=nodata_value
        )

        out_path = output_folder / (png_path.stem + "_masked.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(masked_data, 1)

        print(f"Saved {out_path}")

# %% Binary Mask creation

input_tif = r"mesonet_img/masked_data/n0q_202504030205_masked.tif"

out_dir = Path.home() / "mesonet_img/masked_data"
out_dir.mkdir(parents=True, exist_ok=True)

output_mask_tif = str(out_dir / "n0q_202504030205_binary_mask.tif")

# output_mask_tif = r"/mesonet_img/masked_data/n0q_202504030205_binary_mask.tif"

with rasterio.open(input_tif) as src:
    pix = src.read(1)          # 0–255 grayscale reflectivity
    transform = src.transform
    profile = src.profile

    boundary_shapes = [
        (tracon_polygon.boundary, 1),
        (geodesic_circle.boundary, 2),
    ]

    boundary_raster = rasterize(
        boundary_shapes,
        out_shape=pix.shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    # ROI mask: inside pre_TRACON_area (circle minus TRACON)
    roi_mask = geometry_mask(
        [pre_TRACON_area],
        out_shape=pix.shape,
        transform=transform,
        invert=True
    )

    # Find indices where ROI is True
    rows, cols = np.where(roi_mask)

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    print("ROI pixel bounds:")
    print("  rows:", row_min, "to", row_max)
    print("  cols:", col_min, "to", col_max)

    # 45 dBZ threshold → pixel >= 154
    threshold_pixel = 154

    # Base infeasibility from weather alone
    weather_mask = pix >= threshold_pixel

    # Combine with ROI: only care about pixels inside ROI
    infeasible_mask = roi_mask & weather_mask

    # Convert to 0/1 for MILP, etc.
    binary_mask = infeasible_mask.astype(np.uint8)

    # Crop your data and masks to this rectangle
    pix_roi = pix[row_min:row_max+1, col_min:col_max+1]
    binary_mask_roi = binary_mask[row_min:row_max+1, col_min:col_max+1]
    roi_mask_roi = roi_mask[row_min:row_max+1, col_min:col_max+1]

    print("mask unique values:", np.unique(binary_mask))
    print("number of 1 pixels:", int(binary_mask.sum()))
    print("fraction of 1s:", binary_mask.mean())

    transform_roi = transform * Affine.translation(col_min, row_min)

    out_profile = profile.copy()
    out_profile.update(
        height=binary_mask_roi.shape[0],
        width=binary_mask_roi.shape[1],
        transform=transform_roi,
        dtype="uint8",
        count=1,
        nodata=0,
        compress="LZW",  # make it small
    )

os.makedirs(os.path.dirname(output_mask_tif), exist_ok=True)

# Save GeoTIFF mask
with rasterio.open(output_mask_tif, "w", **out_profile) as dst:
    dst.write((binary_mask_roi * 255).astype(np.uint8), 1)

print("Saved cropped ROI mask:", output_mask_tif)

# Save cropped ROI binary mask for Gurobi as .npz
npz_output = "n0q_202504030205_binary_mask_roi.npz"


# %% Used to export a CSV that contains region_id, numba_of_pixels, area_of_region, min_lat, max_lat, min_lon, max_lon, centroid_lat, centroid_lon

# Array to affine conversion

if isinstance(transform_roi, np.ndarray):
    T = Affine(*transform_roi)
else:
    T = transform_roi  # already an Affine

# Connected-component labeling: find distinct infeasible "regions"
structure = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=int)  # 4-connected
region_id, num_regions = label(binary_mask_roi, structure=structure)
print("Found", num_regions, "regions")

# Optional: ignore tiny regions
min_pixels = 5  # set to e.g. 5 or 10 if you want to filter specks

rows_out = []

for r in range(1, num_regions + 1):
    ys, xs = np.where(region_id == r)
    if ys.size < min_pixels:
        continue

    n_pixels = ys.size

    # Pixel-space bounding box
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Convert corners to lat/lon (approx: min/max over pixel centers)
    lon_min, lat_max = T * (x_min + 0.5, y_min + 0.5)
    lon_max, lat_min = T * (x_max + 0.5, y_max + 0.5)

    # Centroid in pixel space
    yc = ys.mean()
    xc = xs.mean()
    lon_c, lat_c = T * (xc + 0.5, yc + 0.5)

    rows_out.append([
        r,
        int(n_pixels),
        float(lat_min),
        float(lat_max),
        float(lon_min),
        float(lon_max),
        float(lat_c),
        float(lon_c),
    ])

rows_out = np.array(rows_out, dtype=float)

output_csv = r"infeasible_regions.csv"
header = "region_id,n_pixels,min_lat,max_lat,min_lon,max_lon,centroid_lat,centroid_lon"

np.savetxt(
    output_csv,
    rows_out,
    delimiter=",",
    header=header,
    comments="",
    fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"]
)

print("Saved regions CSV:", output_csv)

# %% create overlay of infeasible regions

# --- inputs ---
# cropped ROI mask tif ( 0/1)
regions_csv = r"infeasible_regions.csv"

out_png = r"mesonet_img/final_masked/overlay_t30_highfreq.png"


def plot_boundary(ax, poly, color, lw=2):
    x, y = poly.exterior.xy
    ax.plot(x, y, color=color, linewidth=lw)


def fill_geom(ax, geom, color="white", alpha=1.0):
    """Fill Polygon or MultiPolygon on matplotlib axes."""
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, linewidth=0)
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            x, y = g.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha, linewidth=0)


df = pd.read_csv(regions_csv)

# Use the circle bounds as viewport
xmin, ymin, xmax, ymax = geodesic_circle.bounds

fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
ax.set_facecolor("black")

# --- Draw clipped filled regions ---
for _, r in df.iterrows():
    rect = box(r["min_lon"], r["min_lat"], r["max_lon"], r["max_lat"])
    clipped = rect.intersection(pre_TRACON_area)
    fill_geom(ax, clipped, color="white", alpha=1.0)

# --- Overlay boundaries ---
plot_boundary(ax, geodesic_circle, color="blue", lw=2)
plot_boundary(ax, tracon_polygon,  color="red",  lw=2)


ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.axis("off")
plt.tight_layout()
plt.savefig(out_png, dpi=250, bbox_inches="tight",
            pad_inches=0, facecolor=fig.get_facecolor())
plt.close()

print("Saved:", out_png)

# %% create overlay of optimized MILP waypoints

# --- Inputs ---
base_png = r"mesonet_img/final_masked/overlay_t30_highfreq.png"   # your saved PNG
waypoints_csv = "../../2.3_Outputs_and_Results/weathertrialhighfreq.csv"
out_png = r"mesonet_img/final_masked/overlay_dynamict30_waypoints_highfreq.png"

# --- Load PNG ---
img = mpimg.imread(base_png)

# --- Load waypoint CSV ---
df = pd.read_csv(waypoints_csv)

# Expect columns: lat, lon (and optionally name)
lats = df["f1_lat"].values
lons = df["f1_lon"].values
names = df["name"].values if "name" in df.columns else None

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
ax.set_facecolor("black")

# IMPORTANT:
# We must use the SAME lon/lat bounds that were used when the PNG was created.
# If you used src.bounds earlier, hard-code or reuse those values here.
#

# Use the circle bounds as your viewport (or use your raster bounds if you prefer)
xmin, ymin, xmax, ymax = geodesic_circle.bounds

ax.imshow(
    img,
    extent=[xmin, xmax, ymin, ymax],
    origin="upper"
)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# --- Plot waypoints ---
ax.scatter(
    lons, lats,
    c="cyan",
    s=30,
    edgecolors="black",
    zorder=5
)

# Optional: label waypoints
if names is not None:
    for lon, lat, name in zip(lons, lats, names):
        ax.text(
            lon, lat, name,
            fontsize=8,
            color="cyan",
            ha="left",
            va="bottom",
            zorder=6
        )

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")

plt.tight_layout()
plt.savefig(out_png, dpi=250, bbox_inches="tight",
            pad_inches=0, facecolor=fig.get_facecolor())
plt.close()

print("Saved:", out_png)
