"""Per-cell CWAM predictors from the two rasters.

Predictor 1 (coverage_pct): % of valid N0Q pixels in the cell whose
reflectivity >= dbz_threshold.

Predictor 2 building block (etop_stat_ft): the per-cell echo top statistic
(90th percentile by default, per CWAM2) from MRMS EchoTop_18. dz is computed
per altitude level in run.py: dz = altitude_ft - etop_stat_ft (both predictors
here are altitude-independent, so they are computed once per timestamp).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.features import rasterize

from .rasters import GeoGrid


def label_cells(cells, grid: GeoGrid) -> np.ndarray:
    """Rasterize cell polygons onto a raster grid -> uint32 label array (0 = none)."""
    shapes = [(c.geom_lonlat, c.cell_id) for c in cells]
    return rasterize(
        shapes,
        out_shape=grid.data.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint32",
        all_touched=False,
    )


def compute_cell_metrics(
    cells,
    n0q: GeoGrid,
    etop: GeoGrid,
    dbz_threshold: float,
    echo_top_stat: str = "p90",
) -> pd.DataFrame:
    """Compute altitude-independent cell predictors. Returns (DataFrame, labels)."""
    n0q_labels = label_cells(cells, n0q)
    eet_labels = label_cells(cells, etop)

    n_cells = len(cells)
    ids = np.arange(1, n_cells + 1)

    # ---- Predictor 1: coverage % from N0Q ----
    lab = n0q_labels.ravel()
    dbz = n0q.data.ravel()
    # NOTE: N0Q index-0 pixels ("no echo") stay in the denominator — they are
    # legitimate below-threshold observations, not missing data.
    inside = lab > 0
    n_total = np.bincount(lab[inside], minlength=n_cells + 1)[1:]
    exceed = inside & n0q.valid.ravel() & (np.nan_to_num(dbz, nan=-99.0) >= dbz_threshold)
    n_exceed = np.bincount(lab[exceed], minlength=n_cells + 1)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_pct = np.where(n_total > 0, 100.0 * n_exceed / n_total, np.nan)

    # ---- Predictor 2: echo top stat per cell -> dz ----
    lab_e = eet_labels.ravel()
    eet = etop.data.ravel()
    ok = (lab_e > 0) & np.isfinite(eet)
    etop_stat_ft = np.zeros(n_cells, dtype=np.float64)
    counts = np.bincount(lab_e[ok], minlength=n_cells + 1)[1:]
    order = np.argsort(lab_e[ok], kind="stable")
    vals_sorted = eet[ok][order]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    for i in range(n_cells):
        seg = vals_sorted[offsets[i]:offsets[i + 1]]
        if seg.size == 0:
            etop_stat_ft[i] = np.nan
        elif echo_top_stat == "max":
            etop_stat_ft[i] = seg.max()
        else:  # p90 (CWAM2)
            etop_stat_ft[i] = np.percentile(seg, 90)

    df = pd.DataFrame(
        {
            "cell_id": ids,
            "row": [c.row for c in cells],
            "col": [c.col for c in cells],
            "centroid_lat": [c.centroid_lat for c in cells],
            "centroid_lon": [c.centroid_lon for c in cells],
            "min_lon": [c.geom_lonlat.bounds[0] for c in cells],
            "min_lat": [c.geom_lonlat.bounds[1] for c in cells],
            "max_lon": [c.geom_lonlat.bounds[2] for c in cells],
            "max_lat": [c.geom_lonlat.bounds[3] for c in cells],
            "area_fraction": [c.area_fraction for c in cells],
            "n_pixels_n0q": n_total,
            "n_pixels_exceed": n_exceed,
            "coverage_pct": coverage_pct,
            "etop_stat_ft": etop_stat_ft,
        }
    )
    return df, n0q_labels
