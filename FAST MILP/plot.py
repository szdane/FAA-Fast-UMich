#!/usr/bin/env python3
"""
plot_trajectories_3d.py
-----------------------

Visualise own-ship and intruder trajectories in 3-D (x, y, z) space.

Usage
-----
python plot_trajectories_3d.py \
    --solution_csv solution1.csv \
    --intruder_glob "intruder*.csv" \
    --sep_nm 3.0 \
    --show_buffers
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D

# ---------------------------------------------------------------------- helpers
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if not set(["t", "x", "y", "z"]).issubset(df.columns):
            raise ValueError(f"{path} must have columns t,x,y,z")
        return df
    except Exception as exc:
        sys.exit(f"Error loading {path}: {exc}")


def nm_to_plot_units(nm: float) -> float:
    """If your x, y are nautical-miles already, this is identity.
    Adapt if you switched to km/lat-lon etc."""
    return nm * 6076.12
    # return nm


def sphere(ax, x0, y0, z0, r, **kwargs):
    """Draw a translucent sphere of radius r centred at (x0,y0,z0)."""
    # coarse mesh is fine for visual cue
    u, v = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
    x = x0 + r * np.cos(u) * np.sin(v)
    y = y0 + r * np.sin(u) * np.sin(v)
    z = z0 + r * np.cos(v)
    ax.plot_surface(x, y, z, **kwargs, linewidth=0, antialiased=False)


# --------------------------------------------------------------------------- ui
parser = argparse.ArgumentParser()
parser.add_argument("--solution_csv", default="solution1.csv", help="own-ship file")
parser.add_argument(
    "--intruder_glob", default="intruder*.csv", help='glob for intruders ("*.csv")'
)
parser.add_argument("--sep_nm", type=float, default=3.0, help="lateral sep (NM)")
parser.add_argument(
    "--show_buffers", action="store_true", help="draw sep spheres on intruders"
)
parser.add_argument(
    "--save_png", metavar="FILE", help="save figure instead of interactive show"
)
args = parser.parse_args()

own_df = load_csv(args.solution_csv)
# own_df['x'] = nm_to_plot_units(own_df['x'])
# own_df['y'] = nm_to_plot_units(own_df['y'])
intr_paths = glob.glob(args.intruder_glob)
intr_dfs = {os.path.basename(p): load_csv(p) for p in intr_paths}

# -------------------------------------------------------------------- plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# own-ship trajectory: colour-code by time
norm_t = (own_df["t"] - own_df["t"].min()) / (own_df["t"].max() - own_df["t"].min())
for i in range(len(own_df) - 1):
    ax.plot(
        own_df["x"].iloc[i : i + 2],
        own_df["y"].iloc[i : i + 2],
        own_df["z"].iloc[i : i + 2],
        color=plt.cm.viridis(norm_t.iloc[i]),
        linewidth=2,
    )
ax.plot(
    [own_df["x"].iloc[0]],
    [own_df["y"].iloc[0]],
    [own_df["z"].iloc[0]],
    marker="o",
    color="black",
    label="Entry / start",
)
ax.plot(
    [own_df["x"].iloc[-1]],
    [own_df["y"].iloc[-1]],
    [own_df["z"].iloc[-1]],
    marker="X",
    color="black",
    label="Landing / end",
)

# intruders
# for name, df in intr_dfs.items():
#     ax.plot(df["x"], df["y"], df["z"], "--", label=f"Intruder ({name})")
#     if args.show_buffers:
#         radius = nm_to_plot_units(args.sep_nm)
#         for x, y, z in zip(df["x"], df["y"], df["z"]):
#             sphere(
#                 ax,
#                 nm_to_plot_units(x),
#                 nm_to_plot_units(y),
#                 z,
#                 radius,
#                 alpha=0.08,
#                 color="red",
#                 zorder=0,
#             )

# cosmetics
ax.set_xlabel("X (NM)")
ax.set_ylabel("Y (NM)")
ax.set_zlabel("Altitude (ft)")
ax.set_title("3-D Trajectories with Separation Buffers")
ax.legend()
ax.grid(True)

if args.save_png:
    plt.savefig(args.save_png, dpi=200, bbox_inches="tight")
    print(f"Saved {args.save_png}")
else:
    plt.show()
