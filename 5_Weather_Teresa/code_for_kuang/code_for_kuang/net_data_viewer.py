#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:56:18 2026

@author: anomi
"""

import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Path to GeoTIFF
tif_path = "/Users/anomi/Documents/LATTICE/FAST/Weather_stuff/net_geoTIFF/KDTX_NET_20250403_020700_null_sweep-00.tif"

# Open GeoTIFF
with rasterio.open(tif_path) as src:
    data = src.read(1)

    #print("Shape:", data.shape)
    #print("Min:", np.nanmin(data))
    #print("Max:", np.nanmax(data))
    #print("CRS Used", src.crs)
    #print("Pixel Size", src.transform)
    #print("Geographic Extent", src.bounds)
    print(src.tags())
    print(np.unique(data[~np.isnan(data)])[:20])

# Plot
plt.figure(figsize=(10, 10))
plt.imshow(data, origin="upper")
plt.colorbar(label="Value")
plt.title("GeoTIFF Data")
plt.show()