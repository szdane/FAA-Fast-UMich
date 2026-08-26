#!/usr/bin/env python3
"""
plot_two_trajectories.py  –  colour‑coded by time
-------------------------------------------------

Visualise two aircraft trajectories produced by `milp_two_flights.py`
with point colours that encode time *t*.

Usage
-----
python plot_two_trajectories.py --solution_csv sol.csv
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm                       # colour maps
from mpl_toolkits.mplot3d import Axes3D         # noqa: F401 – 3‑D backend

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution_csv", required=True,
                    help="CSV with columns t,x1,y1,z1,x2,y2,z2")
    args = ap.parse_args()

    df = pd.read_csv(args.solution_csv)

    # ---- normalise t to [0,1] for colour‑map -------------------------------
    t_norm = (df.t - df.t.min()) / (df.t.max() - df.t.min())
    cmap   = cm.plasma                      # perceptually uniform & colour‑blind‑friendly
    colors = cmap(t_norm.to_numpy())        # RGBA array, one row per time step

    # ---- figure layout ------------------------------------------------------
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])

    ax3d = fig.add_subplot(gs[0, :], projection="3d")   # big 3‑D axis
    ax_xz = fig.add_subplot(gs[1, 0])                   # side view (x–z)
    ax_xy = fig.add_subplot(gs[1, 1])                   # top view (x–y)

    # ---- 3‑D scatter, colour = time ----------------------------------------
    ax3d.scatter(df.f1_lat, df.f1_lon, df.f1_alt_ft, c=colors, s=14, label="Flight 1", depthshade=False)
    ax3d.scatter(df.f2_lat, df.f2_lon, df.f2_alt_ft, c=colors, s=14, marker='^',
                 label="Flight 2", depthshade=False)

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z (ft)")
    ax3d.set_title("3‑D trajectories (colour = time)")
    ax3d.legend(loc="upper left")

    # add colour bar keyed to time
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=df.t.min(),
                                                  vmax=df.t.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, pad=0.12)
    cbar.set_label("time t (s)")

    # ---- 2‑D projections (plain lines) -------------------------------------
    ax_xz.plot(df.f1_lat, df.f1_alt_ft, label="Flight 1")
    ax_xz.plot(df.f2_lat, df.f2_alt_ft, label="Flight 2")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z (ft)")
    ax_xz.set_title("Side view (x–z)")
    ax_xz.legend()

    ax_xy.plot(df.f1_lat, df.f1_lon, label="Flight 1")
    ax_xy.plot(df.f2_lat, df.f2_lon, label="Flight 2")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("Top view (x–y)")
    ax_xy.legend()

    plt.show()


if __name__ == "__main__":
    main()
