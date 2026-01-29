#!/usr/bin/env python3
# Plot multiple trajectories
"""
plot_n_trajectories.py  –  colour‑coded by time
------------------------------------------------

Visualise N aircraft trajectories with point colours that encode time *t*.

Usage
-----
python plot_multiple.py --solution_csv sol.csv --n 5  # or specify N
"""
import argparse
import re
from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm                       # colour maps
from mpl_toolkits.mplot3d import Axes3D         # noqa: F401 – 3‑D backend


# ---------------------------------------------------------------------------
def detect_flights(columns):
    """
    Return the highest flight index found in columns such as f1_lat, f2_lon …
    """
    flight_pattern = re.compile(r"f(\d+)_lat")
    indices = [
        int(m.group(1))
        for col in columns
        if (m := flight_pattern.match(col)) is not None
    ]
    return max(indices, default=0)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution_csv", required=True,
                    help="CSV with columns t,f1_lat,f1_lon,f1_alt_ft…")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of flights; if omitted the script infers it")
    args = ap.parse_args()

    df = pd.read_csv(args.solution_csv)

    # ‑‑ determine how many flights we have ‑‑
    n_flights = args.n or detect_flights(df.columns)
    if n_flights == 0:
        raise ValueError("No flight columns (f1_lat etc.) found in the CSV")

    # ---- normalise t to [0,1] for colour‑map -------------------------------
    t_norm = (df.t - df.t.min()) / (df.t.max() - df.t.min())
    cmap   = cm.plasma                      # perceptually uniform & colour‑blind‑friendly
    colors = cmap(t_norm.to_numpy())        # RGBA array, one row per time step

    # ---- figure layout ------------------------------------------------------
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])

    ax3d = fig.add_subplot(gs[0, :], projection="3d")   # big 3‑D axis
    ax_xz = fig.add_subplot(gs[1, 0])                   # side view (lat–alt)
    ax_xy = fig.add_subplot(gs[1, 1])                   # top view (lat–lon)

    # ---- define marker cycle so flights look different ---------------------
    marker_cycle = cycle(["o", "^", "s", "x", "D", "P", "v", "<", ">", "*"])

    # ---- plot each flight ---------------------------------------------------
    for i in range(1, n_flights + 1):
        lat_col  = f"f{i}_lat"
        lon_col  = f"f{i}_lon"
        alt_col  = f"f{i}_alt_ft"

        if not all(c in df.columns for c in (lat_col, lon_col, alt_col)):
            print(f"Warning: missing columns for flight {i}; skipping")
            continue

        marker = next(marker_cycle)

        # 3‑D scatter colour‑coded by time
        ax3d.scatter(df[lat_col], df[lon_col], df[alt_col],
                     c=colors, s=14, marker=marker, depthshade=False)

        # 2‑D projections (plain lines)
        ax_xz.plot(df[lat_col], df[alt_col], marker=marker, markevery=[0])
        ax_xy.plot(df[lat_col], df[lon_col], marker=marker, markevery=[0])

    # ---- axis labels, legends, colour bar -----------------------------------
    ax3d.set_xlabel("latitude")
    ax3d.set_ylabel("longitude")
    ax3d.set_zlabel("altitude (ft)")
    ax3d.set_title("3‑D trajectories (colour = time)")
    # ax3d.legend(loc="upper left")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=df.t.min(),
                                                  vmax=df.t.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, pad=0.12)
    cbar.set_label("time t (s)")

    ax_xz.set_xlabel("latitude")
    ax_xz.set_ylabel("altitude (ft)")
    ax_xz.set_title("Side view (lat–alt)")
    # ax_xz.legend()

    ax_xy.set_xlabel("latitude")
    ax_xy.set_ylabel("longitude")
    ax_xy.set_title("Top view (lat–lon)")
    # ax_xy.legend()

    plt.show()


if __name__ == "__main__":
    main()
