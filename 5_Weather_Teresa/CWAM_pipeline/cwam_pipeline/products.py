"""Output products per timestamp.

* cells CSV        cell-level predictors + probability (the "probability table
                   dependent on 2 variables", evaluated per grid square)
* probability tif  full-resolution WAF raster on the N0Q grid (EPSG:4326)
* binary mask npz  P >= threshold, for the MILP (same as wx_grid_creator_3)
* regions CSV      connected infeasible regions (id, pixels, bounds, centroid)
* overlay PNG      black-background plot with TRACON (red) / pre-TRACON (blue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from scipy.ndimage import label as cc_label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .rasters import GeoGrid


def probability_raster(cell_df: pd.DataFrame, n0q_labels: np.ndarray) -> np.ndarray:
    """Paint each N0Q pixel with its cell's deviation probability (NaN outside).

    NaN means either "outside the pre-TRACON grid" (pixel never belonged to
    any cell) or "inside the grid but the cell had no valid EchoTop data" —
    this array alone can't tell the two apart. Use `cell_bool_raster(...,
    "data_missing")` to distinguish a real data gap from outside-ROI.
    """
    n_cells = int(cell_df["cell_id"].max())
    lut = np.full(n_cells + 1, np.nan, dtype=np.float32)
    lut[cell_df["cell_id"].to_numpy(int)] = cell_df["p_deviation"].to_numpy(float)
    return lut[n0q_labels]


def cell_bool_raster(cell_df: pd.DataFrame, n0q_labels: np.ndarray,
                     column: str) -> np.ndarray:
    """Paint each N0Q pixel with a boolean per-cell attribute.

    Pixels outside any cell (n0q_labels == 0, outside the pre-TRACON grid)
    default to False regardless of the column's own default.
    """
    n_cells = int(cell_df["cell_id"].max())
    lut = np.zeros(n_cells + 1, dtype=bool)
    lut[cell_df["cell_id"].to_numpy(int)] = cell_df[column].to_numpy(bool)
    return lut[n0q_labels]


def write_geotiff(path, data: np.ndarray, transform: Affine, nodata=np.nan):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
        "compress": "LZW",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def binary_mask_and_regions(
    p_raster: np.ndarray,
    transform: Affine,
    p_threshold: float,
    min_region_pixels: int,
    missing_raster: np.ndarray | None = None,
    missing_data_policy: str = "infeasible",
):
    """Threshold the WAF into a binary mask and extract connected regions.

    `missing_raster` (bool, from `cell_bool_raster(..., "data_missing")`)
    marks pixels inside the grid whose cell had no valid EchoTop data.
    Without it, NaN cells (missing data OR outside the grid, indistinguishable
    here) are always treated as passable — pass `missing_raster` explicitly
    to apply `missing_data_policy` instead of silently defaulting to safe.
    """
    hit = np.isfinite(p_raster) & (p_raster >= p_threshold)
    if missing_raster is not None and missing_data_policy == "infeasible":
        hit = hit | missing_raster
    mask = hit.astype(np.uint8)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)  # 4-connected
    region_id, n_regions = cc_label(mask, structure=structure)

    rows = []
    for r in range(1, n_regions + 1):
        ys, xs = np.where(region_id == r)
        if ys.size < min_region_pixels:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        lon_min, lat_max = transform * (x0 + 0.5, y0 + 0.5)
        lon_max, lat_min = transform * (x1 + 0.5, y1 + 0.5)
        lon_c, lat_c = transform * (xs.mean() + 0.5, ys.mean() + 0.5)
        rows.append(
            [r, int(ys.size), float(lat_min), float(lat_max),
             float(lon_min), float(lon_max), float(lat_c), float(lon_c)]
        )
    regions = pd.DataFrame(
        rows,
        columns=["region_id", "n_pixels", "min_lat", "max_lat",
                 "min_lon", "max_lon", "centroid_lat", "centroid_lon"],
    )
    return mask, regions


def _plot_boundary(ax, poly, color, lw=2):
    if poly.geom_type == "MultiPolygon":
        for g in poly.geoms:
            _plot_boundary(ax, g, color, lw)
        return
    x, y = poly.exterior.xy
    ax.plot(x, y, color=color, linewidth=lw)


def overlay_png(
    out_png,
    p_raster: np.ndarray,
    transform: Affine,
    roi,
    p_threshold: float | None = None,
    missing_raster: np.ndarray | None = None,
):
    """Black-background overlay in the style of wx_grid_creator_3.

    Continuous WAF shown in grayscale (or binary if p_threshold given),
    pre-TRACON circle in blue, TRACON polygon in red. Pixels inside the grid
    with no valid EchoTop data (`missing_raster`, from
    `cell_bool_raster(..., "data_missing")`) are painted mid-gray (0.4) so a
    genuine data gap is never visually indistinguishable from confirmed
    clear weather (which renders black, 0.0).
    """
    h, w = p_raster.shape
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    extent = [x0, x1, y1, y0]  # left, right, bottom, top

    img = np.nan_to_num(p_raster, nan=0.0)
    if p_threshold is not None:
        img = (img >= p_threshold).astype(float)
    if missing_raster is not None:
        img = np.where(missing_raster, 0.4, img)

    xmin, ymin, xmax, ymax = roi.geodesic_circle.bounds
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(img, extent=extent, origin="upper", cmap="gray",
              vmin=0.0, vmax=1.0, interpolation="nearest")
    _plot_boundary(ax, roi.geodesic_circle, "blue", 2)
    _plot_boundary(ax, roi.tracon_polygon, "red", 2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("auto")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight", pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
