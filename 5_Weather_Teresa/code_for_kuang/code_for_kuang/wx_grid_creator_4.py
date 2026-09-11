#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:57:35 2026

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

import pyart

from rasterio.warp import reproject, Resampling

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
    4,
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Weather_stuff\mesonet_img",
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
    "/Users/anomi/Documents/LATTICE/FAST/Weather_stuff/mesonet_img"
)

# provide folder path for saved geoTIFF files
output_folder = input_folder / "masked_data"
output_folder.mkdir(exist_ok=True)

# Get all the png images
png_files = sorted(input_folder.glob("*.png"))

for png_path in png_files:
    wld_path = png_path.with_suffix(".wld")

    # Check for ensuring .wld file exists
    if not wld_path.exists():
        print(f"Skipping {png_path.name}: no matching world file")
        continue

    print(f"Processing {png_path.name}...")

    # Open the georeferenced image (png + wld file)
    with rasterio.open(png_path) as src:
        data = src.read(1)          # original grayscale reflectivity values
        transform = src.transform
        profile = src.profile

        # Build mask for ROI
        mask = geometry_mask(
            [pre_TRACON_area],
            out_shape=data.shape,
            transform=transform,
            invert=True   # True inside ROI
        )

        # Find bounding box of ROI in pixel coordinates
        rows, cols = np.where(mask)
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()

        print(f"ROI pixel bounds: rows {row_min} to {
              row_max}, cols {col_min} to {col_max}")

        # Crop original reflectivity and mask to ROI bounding box
        cropped_data = data[row_min:row_max+1, col_min:col_max+1].copy()
        cropped_mask = mask[row_min:row_max+1, col_min:col_max+1]

        # Choose nodata value
        nodata_value = profile.get("nodata")
        if nodata_value is None:
            nodata_value = 0

        # Set pixels outside actual ROI but inside bounding box to nodata
        cropped_data[~cropped_mask] = nodata_value

        # Update transform for cropped window
        cropped_transform = transform * Affine.translation(col_min, row_min)

        # --------------------------------------------------------
        # Ensure CRS exists
        # PNG + WLD often provides transform but no CRS.
        # The IEM mosaics are in EPSG:4326.
        # --------------------------------------------------------
        
        if profile.get("crs") is None:
            profile.update(crs="EPSG:4326")
        
        
        # --------------------------------------------------------
        # Update profile for cropped GeoTIFF
        # --------------------------------------------------------
        
        profile.update(
            driver="GTiff",
            height=cropped_data.shape[0],
            width=cropped_data.shape[1],
            transform=cropped_transform,
            crs="EPSG:4326",
            dtype=cropped_data.dtype,
            count=1,
            nodata=nodata_value,
            compress="LZW"
        )

        out_path = output_folder / f"{png_path.stem}_reflectivity_roi.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(cropped_data, 1)

        print(f"Saved cropped reflectivity raster: {out_path}")
        
#%% Binary Mask Creation

input_tif = r"/Users/anomi/Documents/LATTICE/FAST/Weather_stuff/mesonet_img/masked_data/n0q_202504030205_reflectivity_roi.tif"

npz_output = r"/Users/anomi/Documents/LATTICE/FAST/Weather_stuff/n0q_202504030205_5km_grid_scores_roi.npz"

# Each pixel is 1 km.
# Therefore, a 5 km grid is 5 pixels by 5 pixels.
GRID_SIZE_PIXELS = 5

# 45 dBZ threshold corresponds to grayscale pixel value 154
threshold_pixel = 154

# Load reflectivity raster

with rasterio.open(input_tif) as src:
    # Read the first band.
    # This is assumed to be 0–255 grayscale reflectivity.
    pix = src.read(1)

    # Save geospatial metadata.
    transform = src.transform
    profile = src.profile
    n0q_crs = src.crs

    #print("Original raster shape:", pix.shape)
    
    # Create ROI mask
    
    roi_mask = geometry_mask(
        [pre_TRACON_area],
        out_shape=pix.shape,
        transform=transform,
        invert=True
    )

    #print("ROI mask shape:", roi_mask.shape)

    # Find bounding box around ROI
    
    rows, cols = np.where(roi_mask)

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    #print("ROI pixel bounds:")
    
    #print("  rows:", row_min, "to", row_max)
    
    #print("  cols:", col_min, "to", col_max)
    
    # Crop reflectivity and ROI mask to ROI bounding box
    
    reflectivity_roi = pix[row_min:row_max + 1, col_min:col_max + 1]

    roi_mask_roi = roi_mask[row_min:row_max + 1, col_min:col_max + 1]

    #print("Cropped reflectivity shape:", reflectivity_roi.shape)
    
    #print("Cropped ROI mask shape:", roi_mask_roi.shape)

    # Create weather threshold mask
    
    weather_mask_roi = reflectivity_roi >= threshold_pixel
    
    infeasible_mask_roi = roi_mask_roi & weather_mask_roi

    print("Number of threshold-exceeding ROI pixels:",
          int(infeasible_mask_roi.sum()))

    print("Fraction of ROI pixels above threshold:",
          infeasible_mask_roi.sum() / roi_mask_roi.sum())
    
    # Pad ROI dimensions so that they are divisible by 5
    
    height, width = infeasible_mask_roi.shape

    grid_height_pixels = int(np.ceil(height / GRID_SIZE_PIXELS) * GRID_SIZE_PIXELS)
    grid_width_pixels = int(np.ceil(width / GRID_SIZE_PIXELS) * GRID_SIZE_PIXELS)

    pad_height = grid_height_pixels - height
    pad_width = grid_width_pixels - width
    
    infeasible_mask_crop = np.pad(
    infeasible_mask_roi,
    pad_width=((0, pad_height), (0, pad_width)),
    mode="constant",
    constant_values=False
    )
    
    roi_mask_crop = np.pad(
    roi_mask_roi,
    pad_width=((0, pad_height), (0, pad_width)),
    mode="constant",
    constant_values=False
    )


    #print("Grid-compatible cropped shape:",infeasible_mask_crop.shape)
    
    #Reshape into 5 × 5 grid cells
    
    grid_rows = grid_height_pixels // GRID_SIZE_PIXELS
    grid_cols = grid_width_pixels // GRID_SIZE_PIXELS
    
    grid_transform = (
    transform
    * Affine.translation(col_min, row_min)
    * Affine.scale(GRID_SIZE_PIXELS, GRID_SIZE_PIXELS)
    )
    
    grid_transform=np.array(grid_transform)

    infeasible_blocks = infeasible_mask_crop.reshape(
        grid_rows,
        GRID_SIZE_PIXELS,
        grid_cols,
        GRID_SIZE_PIXELS
    )

    roi_blocks = roi_mask_crop.reshape(
        grid_rows,
        GRID_SIZE_PIXELS,
        grid_cols,
        GRID_SIZE_PIXELS
    )
    
    # Compute fraction of bad pixels per 5 km grid cell
    
    bad_pixel_count = infeasible_blocks.sum(axis=(1, 3))
    
    valid_roi_pixel_count = roi_blocks.sum(axis=(1, 3))
    
    grid_score = np.divide(
        bad_pixel_count,
        valid_roi_pixel_count,
        out=np.zeros_like(bad_pixel_count, dtype=np.float32),
        where=valid_roi_pixel_count > 0
    ).astype(np.float32)
    
    #binary grid decision

    GRID_BLOCKAGE_THRESHOLD = 0.50

    grid_blocked = grid_score >= GRID_BLOCKAGE_THRESHOLD

    print("Blocked grid cells:", int(grid_blocked.sum()))
    
    valid_grid_cells = valid_roi_pixel_count > 0
    
    print(
    "Fraction blocked inside ROI:",
    float(grid_blocked[valid_grid_cells].mean())
)
    

#%% Inclusion of NET data

#Load the NET GeoTIFF

net_tif = r"/Users/anomi/Documents/LATTICE/FAST/Weather_stuff/net_geoTIFF/KDTX_NET_20250403_020700_null_sweep-00.tif"

with rasterio.open(net_tif) as src:
    net_data = src.read(1).astype(np.float32)
    net_transform = src.transform
    net_crs = src.crs

if net_crs is None:
    net_crs = "EPSG:4326"

if n0q_crs is None:
    n0q_crs = "EPSG:4326"

aligned_net = np.full(
    shape=reflectivity_roi.shape,
    fill_value=np.nan,
    dtype=np.float32
)

#Reproject/resample NET onto the N0Q ROI grid.

net_data = np.where(np.isfinite(net_data), net_data, np.nan)

reproject(
    source=net_data,
    destination=aligned_net,
    src_transform=net_transform,
    src_crs=net_crs,
    dst_transform=transform * Affine.translation(col_min, row_min),
    dst_crs=n0q_crs,
    src_nodata=np.nan,
    dst_nodata=np.nan,
    resampling=Resampling.nearest
)

#Pad N0Q and NET together to the same 5-pixel multiple

aligned_net_crop = np.pad(
    aligned_net,
    pad_width=((0, pad_height), (0, pad_width)),
    mode="constant",
    constant_values=np.nan
)

#Reshape NET into the same 5 × 5 blocks

net_blocks = aligned_net_crop.reshape(
    grid_rows,
    GRID_SIZE_PIXELS,
    grid_cols,
    GRID_SIZE_PIXELS
)

#Store max echo top per 5 km cell

max_echo_top = np.nanmax(net_blocks, axis=(1, 3)).astype(np.float32)

net_valid_count = np.isfinite(net_blocks).sum(axis=(1, 3))
max_echo_top[net_valid_count == 0] = np.nan

# --------------------------------------------------------
# Cells satisfying BOTH conditions:
#
# 1. N0Q fraction above threshold
# 2. NET echo tops >= 30 kft
# --------------------------------------------------------

N0Q_SCORE_THRESHOLD = 0.05
NET_THRESHOLD = 0   # assuming NET units are kft

combined_grid_mask = (
    (grid_score >= N0Q_SCORE_THRESHOLD) &
    (max_echo_top >= NET_THRESHOLD)
)

print(
    "Cells meeting BOTH conditions:",
    int(combined_grid_mask.sum())
)


# --------------------------------------------------------
# Create visualization
# --------------------------------------------------------

# Black background
display_grid = np.zeros(
    combined_grid_mask.shape,
    dtype=np.uint8
)

# White squares where BOTH conditions are true
display_grid[combined_grid_mask] = 255


# --------------------------------------------------------
# Plot
# --------------------------------------------------------

plt.figure(figsize=(12, 10))

plt.imshow(
    display_grid,
    origin="upper",
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "5 km Cells:\n"
    "Reflectivity > 45 dBZ AND Echo Tops > 30 kft"
)

plt.xlabel("5 km grid column")
plt.ylabel("5 km grid row")

plt.show()


#Save alongside existing N0Q grid score

np.savez_compressed(
    npz_output,
    n0q_score=grid_score,
    max_echo_top=max_echo_top,
    grid_blocked=grid_blocked.astype(np.uint8),
    bad_pixel_count=bad_pixel_count.astype(np.uint8),
    valid_roi_pixel_count=valid_roi_pixel_count.astype(np.uint8),
    net_valid_count=net_valid_count.astype(np.uint8),
    threshold_pixel=threshold_pixel,
    grid_size_pixels=GRID_SIZE_PIXELS,
    grid_transform=grid_transform,
)

print("Saved 5 km grid scores:", npz_output)
print("N0Q CRS:", n0q_crs)
print("NET CRS:", net_crs)


#%%

# Visual diagnostic for 5 kn discretization

valid_grid_cells = valid_roi_pixel_count > 0
partial_grid_cells = (valid_roi_pixel_count > 0) & (
    valid_roi_pixel_count < GRID_SIZE_PIXELS * GRID_SIZE_PIXELS
)

plt.figure(figsize=(12, 10))

# Show fractional weather score
plt.imshow(grid_score, origin="upper")
plt.colorbar(label="Fraction of pixels >= threshold")

plt.title("5 km Grid Weather Score")
plt.xlabel("5 km grid column")
plt.ylabel("5 km grid row")

plt.show()

# Show blocked cells only

plt.figure(figsize=(12, 10))

plt.imshow(grid_blocked, origin="upper")

plt.title("Blocked 5 km Cells")
plt.xlabel("5 km grid column")
plt.ylabel("5 km grid row")

plt.show()


#%%



