import matplotlib
from matplotlib import dates
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import numpy as np
import pandas as pd

from functions import computation_funcs

def init_plot_style():  # Define global plot formatting
    matplotlib.rc("font", size=11)
    matplotlib.rc("font", family="Arial")
    matplotlib.rc("lines", linewidth=2, markersize=8)
    matplotlib.rc("grid", color="darkgray", linestyle=":")

def format_ax(ax, time_axis=True):  # Define axis formatting with optional time formatting
    if time_axis:
        ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_label_coords(-0.1, 1.03)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("left")
    ax.grid()
    

def plot_2d_trajectory(df1=None, df2=None, label1="Trajectory 1", label2="Trajectory 2", plot_trajectory_endpoints = False,
                       tracon_polygon=None, pretracon_circle=None, tracon_label="TRACON", pretracon_label="Pre-TRACON",
                       lat0=42.2125, lon0=-83.3534, lat_lon_grid=False,
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
        ax.scatter(0, 0, color='#458A74', s=50, label='DTW', zorder=5)

    # TRACON polygon
    if tracon_polygon is not None:
        x, y = tracon_polygon
        ax.plot(np.array(x) / 1000, np.array(y) / 1000, color='#018B38', linestyle='-', marker='o', markersize=4, label=tracon_label)

    # Pre-TRACON circle
    if pretracon_circle is not None:
        x, y = pretracon_circle
        ax.plot(np.array(x) / 1000, np.array(y) / 1000, color='#57AF37', linestyle='--', label=pretracon_label)
        
    # Lat/lon grid (optional)
    if lat_lon_grid:
        lat_lines = np.arange(lat0 - 3, lat0 + 3.5, 0.5)
        lon_lines = np.arange(lon0 - 4, lon0 + 4.5, 0.5)

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

def plot_3d_trajectory(df1=None, df2=None, label1="Trajectory 1", label2="Trajectory 2", title="3D Flight Trajectory"):
    """
    Plot one or two 3D trajectories.

    Parameters:
    - df1, df2 (pd.DataFrame): DataFrames with columns "x", "y", "alt", "coord1", "coord2"
    - label1, label2 (str): Labels for the two trajectories
    - title (str): Plot title
    """

    # Create new figure for 3D trajectory
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot first trajectory
    ax.plot(df1["x"] / 1000, df1["y"] / 1000, df1["alt"], label=label1, lw=2, color='#E42320')
    ax.scatter(df1["x"].iloc[0] / 1000, df1["y"].iloc[0] / 1000, df1["alt"].iloc[0],
               color='#018B38', s=50, label=f'{label1} Origin', marker='^')
    ax.scatter(df1["x"].iloc[-1] / 1000, df1["y"].iloc[-1] / 1000, df1["alt"].iloc[-1],
               color='#F5A216', s=50, label=f'{label1} Destination', marker='^')

    # Plot second trajectory if provided
    if df2 is not None:
        ax.plot(df2["x"] / 1000, df2["y"] / 1000, df2["alt"], label=label2, lw=2, color='#6A8EC9')
        ax.scatter(df2["x"].iloc[0] / 1000, df2["y"].iloc[0] / 1000, df2["alt"].iloc[0],
                   color='#018B38', s=50, label=f'{label2} Origin', marker='s')
        ax.scatter(df2["x"].iloc[-1] / 1000, df2["y"].iloc[-1] / 1000, df2["alt"].iloc[-1],
                   color='#41B9C1', s=50, label=f'{label2} Destination', marker='s')

    # Set axis labels and title
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Altitude (ft)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

def plot_altitude_groundspeed_rateOfClimb(df1=None, df2=None, label1="Trajectory 1", label2="Trajectory 2", title="Flight Parameters"):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    ax1.plot(df1.recTime, df1.alt, label=label1, color='#E42320')
    ax2.plot(df1.recTime, df1.groundSpeed, label=label1, color='#E42320')
    ax3.plot(df1.recTime, df1.rateOfClimb, label=label1, color='#E42320')

    if df2 is not None:
        ax1.plot(df2.recTime, df2.alt, label=label2, linestyle='--', color='#6A8EC9')
        ax2.plot(df2.recTime, df2.groundSpeed, label=label2, color='#6A8EC9')
        ax3.plot(df2.recTime, df2.rateOfClimb, label=label2, color='#6A8EC9')

    ax1.set_ylabel("altitude (ft)")
    ax2.set_ylabel("groundspeed (kts)")
    ax3.set_ylabel("vertical rate (ft/min)")
    ax3.set_xlabel("Time")

    for ax in (ax1, ax2, ax3):
        format_ax(ax)
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()

def plot_fuel_flow_and_usage(df1, df2=None, label1="Trajectory 1", label2="Trajectory 2", title="Fuel Flow and Usage"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # --- Plot Fuel Flow ---
    axs[0].plot(df1.recTime, df1.fuel_flow, label=label1, color='#E42320')
    if df2 is not None:
        axs[0].plot(df2.recTime, df2.fuel_flow, label=label2, color='#6A8EC9')
    axs[0].set_ylabel("fuel flow (kg/s)")
    format_ax(axs[0])
    axs[0].legend()

    # --- Plot Cumulative Fuel Used ---
    axs[1].plot(df1.recTime, df1.fuel_used, label=label1, color='#E42320')
    if df2 is not None:
        axs[1].plot(df2.recTime, df2.fuel_used, label=label2, color='#6A8EC9')

    axs[1].set_ylabel("fuel used (kg)")
    format_ax(axs[1])
    axs[1].legend()

    # --- Mark final fuel usage points ---
    final_time_1 = df1.recTime.iloc[-1]
    final_fuel_1 = df1.fuel_used.iloc[-1]
    axs[1].scatter(final_time_1, final_fuel_1, color='#E42320', zorder=3)
    axs[1].annotate(f"{label1}: {final_fuel_1:.0f} kg",
                    xy=(final_time_1, final_fuel_1),
                    xytext=(final_time_1 - pd.Timedelta(seconds=100), final_fuel_1 + 200),
                    fontsize=9, color='#E42320')

    if df2 is not None:
        final_time_2 = df2.recTime.iloc[-1]
        final_fuel_2 = df2.fuel_used.iloc[-1]
        axs[1].scatter(final_time_2, final_fuel_2, color='#6A8EC9', zorder=3)
        axs[1].annotate(f"{label2}: {final_fuel_2:.0f} kg",
                        xy=(final_time_2, final_fuel_2),
                        xytext=(final_time_2 - pd.Timedelta(seconds=100), final_fuel_2 + 200),
                        fontsize=9, color='#6A8EC9')

    axs[1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

