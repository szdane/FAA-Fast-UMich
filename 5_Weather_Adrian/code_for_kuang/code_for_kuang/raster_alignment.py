# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:39:54 2026

@author: anomi
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# %%
# Master grid: cropped reflectivity raster
reflectivity_path = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Weather_stuff\mesonet_img\masked_data\n0q_202504030205_reflectivity_roi.tif"

# Source grid: NET GeoTIFF
net_path = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Weather_stuff\NET_files\KDTX_EET_20250403_020200_null_sweep-00.tif"

# Output: NET aligned to cropped reflectivity raster
net_aligned_path = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Weather_stuff\NET_files\net_aligned_to_reflectivity_roi_20250403_020200.tif"

with rasterio.open(reflectivity_path) as ref_src:
    ref = ref_src.read(1)
    dst_height = ref_src.height
    dst_width = ref_src.width
    dst_transform = ref_src.transform
    dst_profile = ref_src.profile.copy()

    # If CRS is missing, assume geographic lat/lon
    dst_crs = ref_src.crs if ref_src.crs is not None else "EPSG:4326"
    ref_nodata = ref_src.nodata

with rasterio.open(net_path) as net_src:
    src_crs = net_src.crs if net_src.crs is not None else "EPSG:4326"
    src_nodata = net_src.nodata

    # Output array on reflectivity grid
    net_aligned = np.full((dst_height, dst_width), -9999.0, dtype=np.float32)

    reproject(
        source=rasterio.band(net_src, 1),
        destination=net_aligned,
        src_transform=net_src.transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=-9999.0,
        resampling=Resampling.nearest
    )

    # Optional: also mask NET anywhere reflectivity raster is nodata
    if ref_nodata is not None:
        net_aligned[ref == ref_nodata] = -9999.0

    dst_profile.update(
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        transform=dst_transform,
        crs=dst_crs,
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="LZW"
    )

    with rasterio.open(net_aligned_path, "w", **dst_profile) as dst:
        dst.write(net_aligned, 1)

print("Saved aligned NET raster to:")
print(net_aligned_path)

# %%

with rasterio.open(reflectivity_path) as src:
    refl = src.read(1)
    refl_nodata = src.nodata

with rasterio.open(net_aligned_path) as src:
    net = src.read(1)
    net_nodata = src.nodata

plt.figure(figsize=(8, 6))
plt.imshow(refl, origin="upper")
plt.contour(net, levels=[20, 25, 30, 35, 40, 45, 50], linewidths=1)
plt.title("Reflectivity with NET contours")
plt.show()
