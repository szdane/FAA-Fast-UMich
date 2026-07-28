#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:15:25 2026

@author: anomi

Render one raw N0Q reflectivity image and one raw NET/EchoTop image
for a single timestamp, cropped to the pre-TRACON circle, with the
region boundaries overlaid — for comparison with pipeline outputs and
for use as PPT illustrations.

Place this file in the cwam_wx_pipeline folder (next to config.yaml).

Run from a terminal:
    python plot_raw_rasters.py --config config.yaml --time "2025-04-03 04:05"

Or in Spyder: edit CONFIG/TIME below and press F5.

Outputs (in <output_dir>/raw/):
    n0q_raw_<stamp>.png       NWS-style reflectivity colors, dBZ colorbar
    echotop_raw_<stamp>.png   echo top heights in kft, colorbar
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from cwam_pipeline.config import load_config
from cwam_pipeline.fetch import download_echo_tops, download_n0q, local_echo_tops
from cwam_pipeline.rasters import load_echo_top, load_n0q
from cwam_pipeline.roi import PreTraconROI

# Defaults for Spyder (F5) use; command-line arguments override these
CONFIG = HERE / "config.yaml"
TIME = "2025-04-03 02:05"
ECHOTOP_LOCAL_DIR = None

# Standard NWS radar reflectivity palette, 5 dBZ steps from 5 to 75
NWS_COLORS = [
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6",
]
NWS_LEVELS = np.arange(5, 80, 5)


def _extent(grid):
    h, w = grid.data.shape
    x0, y0 = grid.transform * (0, 0)
    x1, y1 = grid.transform * (w, h)
    return [x0, x1, y1, y0]


def _draw(grid, roi, data, cmap, norm, cbar_label, title, out_png):
    xmin, ymin, xmax, ymax = roi.geodesic_circle.bounds
    fig, ax = plt.subplots(figsize=(7.2, 6), facecolor="black")
    ax.set_facecolor("black")

    im = ax.imshow(data, extent=_extent(grid), origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest")

    for poly, color in ((roi.geodesic_circle, "blue"),
                        (roi.tracon_polygon, "red")):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("auto")   # same convention as pipeline overlays
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=11)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")

    plt.savefig(out_png, dpi=250, bbox_inches="tight", pad_inches=0.1,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved:", out_png)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--time", default=TIME,
                    help='UTC timestamp "YYYY-MM-DD HH:MM" (5-min multiple)')
    ap.add_argument("--echotop-local-dir", default=ECHOTOP_LOCAL_DIR)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dt = datetime.strptime(args.time, "%Y-%m-%d %H:%M")
    stamp = dt.strftime("%Y%m%d%H%M")

    roi = PreTraconROI(cfg.region)
    bounds = roi.bounds_lonlat

    # Reuses cached files in data_dir if already downloaded
    n0q_files = download_n0q(cfg, [dt])
    if args.echotop_local_dir:
        eet_files = local_echo_tops(args.echotop_local_dir, [dt],
                                    cfg.analysis.echo_top_time_tolerance_min)
    else:
        eet_files = download_echo_tops(cfg, [dt])
    if dt not in n0q_files or dt not in eet_files:
        sys.exit("Missing input data for that timestamp.")

    out_dir = cfg.output_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Raw N0Q reflectivity ----
    n0q = load_n0q(n0q_files[dt], bounds)
    dbz = np.ma.masked_invalid(n0q.data)
    dbz = np.ma.masked_less(dbz, 5.0)          # below 5 dBZ -> black
    cmap = ListedColormap(NWS_COLORS)
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(NWS_LEVELS, cmap.N)
    _draw(n0q, roi, dbz, cmap, norm, "Reflectivity (dBZ)",
          f"IEM N0Q composite reflectivity — {dt:%Y-%m-%d %H:%M} UTC",
          out_dir / f"n0q_raw_{stamp}.png")

    # ---- Raw NET / EchoTop_18 ----
    etop = load_echo_top(eet_files[dt], bounds)
    kft = np.ma.masked_invalid(etop.data / 1000.0)
    kft = np.ma.masked_less_equal(kft, 0.0)    # no echo -> black
    cmap2 = plt.get_cmap("turbo").copy()
    cmap2.set_bad((0, 0, 0, 0))
    norm2 = BoundaryNorm(np.arange(0, 65, 5), cmap2.N)
    _draw(etop, roi, kft, cmap2, norm2, "18 dBZ echo top (kft)",
          f"NOAA NET echo tops — {dt:%Y-%m-%d %H:%M} UTC",
          out_dir / f"echotop_raw_{stamp}.png")


if __name__ == "__main__":
    main()