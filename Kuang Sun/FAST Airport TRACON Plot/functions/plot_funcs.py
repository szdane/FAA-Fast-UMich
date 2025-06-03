import matplotlib
import matplotlib.pyplot as plt
import adjustText

import numpy as np

from functions import computation_funcs


def init_plot_style():  # Define global plot formatting
    matplotlib.rc("font", size=11)
    matplotlib.rc("font", family="Arial")
    matplotlib.rc("lines", linewidth=2, markersize=8)
    matplotlib.rc("grid", color="darkgray", linestyle=":")

def plot_2d_trajectory(df1=None, df2=None, label1="Trajectory 1", label2="Trajectory 2", plot_trajectory_endpoints = False,
                       tracon_polygon=None, star_fixes = None, pretracon_circle=None, tracon_label="TRACON", pretracon_label="Pre-TRACON", 
                       lat0=42.2125, lon0=-83.3534, airport_name = None, lat_lon_grid=False,
                       waypoints=None, waypoints_plot = False,
                       preTRACON_entry=None, preTRACON_exit=None, pre_TRACON_endpoints = False,
                       title="2D Flight Trajectory (X-Y Frame)"):
    """
    Plot 2D trajectories with optional coordinate annotations and overlays.
    """
    init_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot df1 trajectory
    if df1 is not None:
        ax.plot(df1["x"] / 1000, df1["y"] / 1000, label=label1, color="#E42320")
        if plot_trajectory_endpoints:
            ax.scatter(df1.iloc[0]["x"] / 1000, df1.iloc[0]["y"] / 1000, color="#E42320", s=50, label=f'{label1} Origin')
            ax.scatter(df1.iloc[-1]["x"] / 1000, df1.iloc[-1]["y"] / 1000, color="#E42320", s=50, label=f'{label1} Destination')

    # Plot df2 if given
    if df2 is not None:
        ax.plot(df2["x"] / 1000, df2["y"] / 1000, label=label2, color='#6A8EC9')
        if plot_trajectory_endpoints:
            ax.scatter(df2.iloc[0]["x"] / 1000, df2.iloc[0]["y"] / 1000, color='#6A8EC9', s=50, label=f'{label2} Origin')
            ax.scatter(df2.iloc[-1]["x"] / 1000, df2.iloc[-1]["y"] / 1000, color='#6A8EC9', s=50, label=f'{label2} Destination')

    # DTW reference
    if tracon_polygon or pretracon_circle:
        ax.scatter(0, 0, color='#CC5B45', s=50, label=airport_name, zorder=5)
        ax.text(0 / 1000 + 2, 0 / 1000 + 2, airport_name, fontsize=8, color='black')

    # TRACON polygon
    if tracon_polygon is not None:
        x, y = tracon_polygon
        ax.plot(np.array(x) / 1000, np.array(y) / 1000, color='#018B38', linestyle='--',  marker='None', markersize=4, label=tracon_label)

    # All Star Fixes
    if star_fixes is not None:
        x, y, star_labels = star_fixes
        ax.plot(np.array(x) / 1000, np.array(y) / 1000, color='#018B38', linestyle='None', marker='o', markersize=4, label=tracon_label)
       
        texts = []
        for xi, yi, label in zip(x, y, star_labels):
            texts.append(ax.text(xi / 1000, yi / 1000, label, fontsize=8))
        adjustText.adjust_text(
            texts,
            ax=ax,
            expand = (3, 3), # Increase spacing between texts
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
        ) #add star fix name label next to star fix points

    # Pre-TRACON circle
    if pretracon_circle is not None:
        x, y = pretracon_circle
        ax.plot(np.array(x) / 1000, np.array(y) / 1000, color='#57AF37', linestyle='--', label=pretracon_label)
        
    # Lat/lon grid (optional)
    if lat_lon_grid:
        lat_lines = np.arange(lat0 - 1.5, lat0 + 1.75, 0.5)
        lon_lines = np.arange(lon0 - 2, lon0 + 2.25, 0.5)

        for lat in lat_lines:
            lons = np.linspace(lon_lines.min(), lon_lines.max(), 200)
            lats = np.full_like(lons, lat)
            x, y = computation_funcs.proj_with_defined_origin(lats, lons, lat0, lon0)
            ax.plot(x / 1000, y / 1000, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
            ax.text(x[0] / 1000, y[0] / 1000, f"{lat:.1f}°", va="center", ha="right", fontsize=7, color="#848484")

        for lon in lon_lines:
            lats = np.linspace(lat_lines.min(), lat_lines.max(), 200)
            lons = np.full_like(lats, lon)
            x, y = computation_funcs.proj_with_defined_origin(lats, lons, lat0, lon0)
            ax.plot(x / 1000, y / 1000, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
            ax.text(x[-1] / 1000, y[-1] / 1000, f"{lon:.1f}°", va="bottom", ha="center", fontsize=7, color="#848484")

    # Waypoints (optional)
    if waypoints_plot:
        if waypoints is not None:
            waypoint_lat = [pt[0] for pt in waypoints]
            waypoint_lon = [pt[1] for pt in waypoints]
            x_wp, y_wp = computation_funcs.proj_with_defined_origin(waypoint_lat, waypoint_lon, lat0, lon0)
            ax.scatter(x_wp / 1000, y_wp / 1000, color='#4E5689', marker='^', s=30, label="Waypoints")

    # Plot Pre-TRACON Entry & Exit Points (optional)
    if pre_TRACON_endpoints:
        for point, label in [(preTRACON_entry, "Pre-TRACON Entry"), (preTRACON_exit, "Pre-TRACON Exit")]:
            if point is not None:
                x, y = point
                ax.scatter(x / 1000, y / 1000, color='#F5A216', s=50, marker='x', zorder=6, label=label)
                # Convert (x, y) to (lat, lon)
                lat, lon = computation_funcs.proj_with_defined_origin(y, x, lat0, lon0, inverse=True)
                ax.text(x / 1000 + 10, y / 1000, f"({lat:.4f}, {lon:.4f})", color='#F5A216', fontsize=8, ha='left', va='bottom')

    # Axis labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]+10, 0, "x (km)", va="center", ha="left", fontsize=12)
    ax.text(10, ylim[1], "y (km)", va="center", ha="left", fontsize=12)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.legend()

    # Center axes like cross
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.tight_layout()