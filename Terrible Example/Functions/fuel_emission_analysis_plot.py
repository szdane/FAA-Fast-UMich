import os
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
from matplotlib import dates
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import numpy as np
import pandas as pd

from Functions import fuel_emission_analysis_computation

def print_altitude_comparison(aircraft_list, dic_hist_pretracon, dic_opt_pretracon,
                               dic_waypoints, star_fixes_csv_path, output_csv_path):
    """Print altitude-at-arrival comparison table and save to CSV."""
    import math
    _sf_csv = pd.read_csv(str(star_fixes_csv_path))
    _fix_alt = {row["name"]: (row["alt_max_ft"], row["alt_min_ft"]) for _, row in _sf_csv.iterrows()}

    _rows = []
    for ac in aircraft_list:
        acId  = ac["acId"]
        df_h  = dic_hist_pretracon.get(acId)
        df_o  = dic_opt_pretracon.get(acId)
        df_wp = dic_waypoints.get(acId)
        if df_h is None or df_o is None or df_wp is None:
            continue
        dest_lat, dest_lon = df_wp.iloc[-1]["lat"], df_wp.iloc[-1]["lon"]
        _sf_lookup = {row["name"]: (row["lat"], row["lon"]) for _, row in _sf_csv.iterrows()}
        closest_fix = min(_sf_lookup, key=lambda n: (
            (_sf_lookup[n][0] - dest_lat)**2 + (_sf_lookup[n][1] - dest_lon)**2
        ))
        alt_max, alt_min = _fix_alt.get(closest_fix, (None, None))
        _rows.append({
            "acId":               acId,
            "chosen_fix":         closest_fix,
            "fix_lat":            round(dest_lat, 4),
            "fix_lon":            round(dest_lon, 4),
            "fix_alt_max_ft":     alt_max,
            "fix_alt_min_ft":     alt_min,
            "milp_fix_alt_ft":    round(df_wp.iloc[-1]["alt_ft"], 0),
            "hist_arrival_alt_ft":round(float(df_h.iloc[-1]["alt"]), 0),
            "opt_arrival_alt_ft": round(float(df_o.iloc[-1]["alt"]), 0),
        })

    df_cmp = pd.DataFrame(_rows)
    from pathlib import Path
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_cmp.to_csv(output_csv_path, index=False)

    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  ALTITUDE AT ARRIVAL COMPARISON")
    print(SEP)
    print(df_cmp.to_string(index=False))
    print(f"{SEP}")
    print(f"  Saved → {output_csv_path}\n")


def print_fuel_time_summary(aircraft_list, dic_hist_pretracon, dic_opt_pretracon):
    """Print historic vs optimized time and fuel summary table."""
    W = 38
    SEP  = "=" * 70
    HSEP = f"  {'':─<{W}}  {'──────────':>10}  {'─────────':>9}  {'────────':>8}  {'──────────':>10}  {'─────────':>9}  {'────────':>8}"
    print(f"\n{SEP}")
    print(f"  SUMMARY — Historic vs Optimized")
    print(SEP)
    print(f"  {'Flight':<{W}}  {'Hist time':>10}  {'Opt time':>9}  {'Δtime':>8}  {'Hist fuel':>10}  {'Opt fuel':>9}  {'Δfuel':>8}")
    print(f"  {'':─<{W}}  {'(s)':>10}  {'(s)':>9}  {'(s)':>8}  {'(kg)':>10}  {'(kg)':>9}  {'(kg)':>8}")
    total_hist_t = total_opt_t = total_hist_f = total_opt_f = 0.0
    n_valid = 0
    for ac in aircraft_list:
        acId = ac["acId"]
        df_h = dic_hist_pretracon.get(acId)
        df_o = dic_opt_pretracon.get(acId)
        if df_h is None or df_o is None:
            continue
        hist_t = (pd.to_datetime(df_h.iloc[-1]["recTime"]) - pd.to_datetime(df_h.iloc[0]["recTime"])).total_seconds()
        opt_t  = float(df_o.iloc[-1]["t"])
        hist_f = float(df_h["fuel_used"].iloc[-1])
        opt_f  = float(df_o["fuel_used"].iloc[-1])
        total_hist_t += hist_t;  total_opt_t += opt_t
        total_hist_f += hist_f;  total_opt_f += opt_f
        n_valid += 1
        print(f"  {acId:<{W}}  {hist_t:>10.0f}  {opt_t:>9.0f}  {opt_t-hist_t:>+8.0f}  {hist_f:>10.1f}  {opt_f:>9.1f}  {opt_f-hist_f:>+8.1f}")
    print(HSEP)
    print(f"  {'TOTAL':<{W}}  {total_hist_t:>10.0f}  {total_opt_t:>9.0f}  {total_opt_t-total_hist_t:>+8.0f}  {total_hist_f:>10.1f}  {total_opt_f:>9.1f}  {total_opt_f-total_hist_f:>+8.1f}")
    if n_valid > 0:
        avg_hist_t = total_hist_t / n_valid;  avg_opt_t = total_opt_t / n_valid
        avg_hist_f = total_hist_f / n_valid;  avg_opt_f = total_opt_f / n_valid
        print(f"  {'AVERAGE':<{W}}  {avg_hist_t:>10.0f}  {avg_opt_t:>9.0f}  {avg_opt_t-avg_hist_t:>+8.0f}  {avg_hist_f:>10.1f}  {avg_opt_f:>9.1f}  {avg_opt_f-avg_hist_f:>+8.1f}")
    print(f"{SEP}\n")


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


def _plot_wp_times(ax, df, label_color, dx_km=5, dy_km=5):
    """
    Annotate each waypoint with its exact `recTime` value (HH:MM AM/PM),
    using the row indices stored in df.attrs['wp_node_indices'].
    Also label the start and end points of the trajectory.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pandas.DataFrame        # must have cols 'x', 'y', 'recTime'
    label_color : str            # text colour to match the trajectory
    dx_km, dy_km : float         # offset for the label in km
    """

    def _fmt(ts):
        """Return 'HH:MM AM/PM' (ts may be Timestamp or str)."""
        ts = pd.to_datetime(ts)
        suffix = "PM" if ts.hour >= 12 else "AM"
        return ts.strftime("%I:%M:%S ").lstrip("0") + suffix

    # # === Waypoints (from metadata) ===
    # for wp_no, row_idx in df.attrs.get("wp_node_indices", {}).items():
    #     if row_idx >= len(df):
    #         continue
    #     row = df.iloc[row_idx]
    #     x_km, y_km = row["x"] / 1000, row["y"] / 1000
    #     ax.text(x_km + dx_km, y_km + dy_km, _fmt(row["recTime"]),
    #             fontsize=10, color=label_color, ha="left", va="bottom")

    # === Start point ===
    start = df.iloc[0]
    sx, sy = start["x"] / 1000, start["y"] / 1000
    ax.plot(sx, sy, color=label_color, markersize=10)
    ax.text(sx + dx_km, sy + dy_km, _fmt(start["recTime"]),
            fontsize=10, color=label_color, ha="left", va="bottom")

    # === End point ===
    end = df.iloc[-1]
    ex, ey = end["x"] / 1000, end["y"] / 1000
    ax.plot(ex, ey, color=label_color, markersize=10)
    ax.text(ex + dx_km, ey + dy_km, _fmt(end["recTime"]),
            fontsize=10, color=label_color, ha="left", va="bottom")

def plot_2d_trajectories(
    dic=None, labels=None, colors=None, plot_trajectory_endpoints=False,
    tracon_polygon=None, tracon_label="TRACON", preTracon_circle=None, preTracon_label="Pre-TRACON",
    plot_waypoints=False, waypoints=None, waypoints_tolerance=3000, plot_waypoint_tol_zone=False,
    lat0=42.2125, lon0=-83.3534, plot_lat_lon_grid=False,
    title="2D Flight Trajectories (X-Y Frame)"
):
    """
    Plot 2D trajectories for multiple aircraft from a dictionary of DataFrames.
    """
    init_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each trajectory
    if dic is not None:
        acIds = list(dic.keys())
        dfs = list(dic.values())

        # Defaults
        if labels is None:
            labels = acIds
        if colors is None:
            colors = plt.cm.tab10.colors

        for i, (acId, df) in enumerate(dic.items()):
            color = colors[i % len(colors)]
            
            name_to_check = labels[i].lower() if labels else acId.lower()
            if "historic" in name_to_check:
                ls = '--'
            else:
                ls = '-'

            label = labels[i]

            ax.plot(df["x"] / 1000, df["y"] / 1000,
                    label=label, color=color, linestyle=ls)

            if plot_trajectory_endpoints:
                ax.scatter(df.iloc[0]["x"] / 1000, df.iloc[0]["y"] / 1000,
                        color=color, s=50, label=f"{label} Origin")
                ax.scatter(df.iloc[-1]["x"] / 1000, df.iloc[-1]["y"] / 1000,
                        color=color, s=50, label=f"{label} Destination")

    # Plot DTW reference
    ax.scatter(0, 0, color='#458A74', s=50, label='DTW', zorder=5)

    # Plot TRACON polygon
    x, y, *_fix_named = tracon_polygon
    ax.plot(np.array(x) / 1000, np.array(y) / 1000,
            color='#018B38', linestyle='-', marker='o', markersize=4, label=tracon_label)
    if _fix_named:
        for fix_name, fx, fy in _fix_named[0]:
            ax.text(fx / 1000, fy / 1000, fix_name, fontsize=7, color='#018B38',
                    ha='left', va='bottom')

    # Plot Pre-TRACON circle
    x, y = preTracon_circle
    ax.plot(np.array(x) / 1000, np.array(y) / 1000,
            color='#57AF37', linestyle='--', label=preTracon_label)

    # Plot Lat/lon grid (optional)
    if plot_lat_lon_grid == True:
        lat_lines = np.arange(lat0 - 3, lat0 + 3.5, 0.5)
        lon_lines = np.arange(lon0 - 4, lon0 + 4.5, 0.5)

        for lat in lat_lines:
            lons = np.linspace(lon_lines.min(), lon_lines.max(), 200)
            lats = np.full_like(lons, lat)
            x, y = fuel_emission_analysis_computation.proj_with_defined_origin(lats, lons, lat0, lon0)
            ax.plot(x / 1000, y / 1000, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
            ax.text(x[0] / 1000, y[0] / 1000, f"{lat:.1f}°", va="center", ha="right", fontsize=7, color="#848484")

        for lon in lon_lines:
            lats = np.linspace(lat_lines.min(), lat_lines.max(), 200)
            lons = np.full_like(lats, lon)
            x, y = fuel_emission_analysis_computation.proj_with_defined_origin(lats, lons, lat0, lon0)
            ax.plot(x / 1000, y / 1000, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
            ax.text(x[-1] / 1000, y[-1] / 1000, f"{lon:.1f}°", va="bottom", ha="center", fontsize=7, color="#848484")

    # Plot Waypoints with Tolerance Zone (optional)
    if plot_waypoints == True:
        for acId, df_wp in waypoints.items():
            waypoint_lat = df_wp["lat"].tolist()
            waypoint_lon = df_wp["lon"].tolist()
            x_wp, y_wp = fuel_emission_analysis_computation.proj_with_defined_origin(waypoint_lat, waypoint_lon, lat0, lon0)
            ax.scatter(x_wp / 1000, y_wp / 1000,
                       color="black", marker='x', s=40, linewidth=1,
                       label=f"Waypoints ({acId})", zorder=6)
            
        if plot_waypoint_tol_zone == True:
            for lat, lon, alt, t in waypoints:
                x_wp, y_wp = fuel_emission_analysis_computation.proj_with_defined_origin(lat, lon, lat0, lon0)
                ax.add_patch(plt.Circle((x_wp / 1000, y_wp / 1000), waypoints_tolerance / 1000, fill=False, color='black', linestyle=':', label='Waypoint Constraint Zone'))

        label_plotted = False  # this block is to ensure only one label in legend
        for acId, df_wp in waypoints.items():
            for _, row in df_wp.iterrows():
                lat, lon, alt, t = row["lat"], row["lon"], row["alt_ft"], row["t"]
                x_wp, y_wp = fuel_emission_analysis_computation.proj_with_defined_origin(lat, lon, lat0, lon0)
                circle = plt.Circle((x_wp / 1000, y_wp / 1000), waypoints_tolerance / 1000, fill=True, linewidth = 1.5, color='#F5A216', zorder=5.5) #, linestyle=':')
                if not label_plotted:
                    circle.set_label("Waypoint Constraint Zone")
                    label_plotted = True
                ax.add_patch(circle)  
        
    # Title and legend
    ax.set_aspect(0.8)
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(0.85, 1.05), borderaxespad=0.)
    plt.tight_layout()
    return fig, ax



def plot_3d_trajectories(
    dic=None, labels=None, colors=None,
    waypoints=None, plot_trajectory=True,
    plot_waypoint_lines=True,
    lat0=42.2125, lon0=-83.3534,
    title="3D Flight Trajectories",
    show_legend=True
):
    """
    Plot 3D trajectories for multiple aircraft from a dictionary of DataFrames.
    """
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    acIds = list(dic.keys())
    dfs = list(dic.values())

    # Defaults
    if labels is None:
        labels = acIds
    if colors is None:
        colors = plt.cm.tab10.colors

    # Plot each trajectory
    for i, (acId, df) in enumerate(dic.items()):
        color = colors[i % len(colors)]

        name_to_check = labels[i].lower() if labels else acId.lower()
        is_historic = "historic" in name_to_check
        ls = '--' if is_historic else '-'
        lw = 2.5 if is_historic else 1.5

        if not plot_trajectory and not is_historic:
            continue

        label = labels[i]

        ax.plot(df["x"] / 1000, df["y"] / 1000, df["alt"], label=label, lw=lw, color=color, linestyle=ls) # Plot line
        ax.scatter(df.iloc[0]["x"] / 1000, df.iloc[0]["y"] / 1000, df.iloc[0]["alt"], color=color, marker='^', s=60, label='_nolegend_') # Plot origin 
        ax.scatter(df.iloc[-1]["x"] / 1000, df.iloc[-1]["y"] / 1000, df.iloc[-1]["alt"],
                color=color, marker='s', s=60, label='_nolegend_') # Plot destination

    # Plot waypoints if provided
    if waypoints is not None:
        for idx, (acId, wps) in enumerate(waypoints.items()):
            lat_list = wps["lat"].tolist()
            lon_list = wps["lon"].tolist()
            alt_list = wps["alt_ft"].tolist()
            x_wp, y_wp = fuel_emission_analysis_computation.proj_with_defined_origin(lat_list, lon_list, lat0, lon0)
            
            color = colors[idx % len(colors)]
            if plot_waypoint_lines:
                ax.plot(np.array(x_wp) / 1000, np.array(y_wp) / 1000, alt_list,
                        color=color, lw=1.5, linestyle='-', zorder=5)
            ax.scatter(np.array(x_wp) / 1000, np.array(y_wp) / 1000, alt_list,
                        color=color, marker='x', s=40, label='_nolegend_')
    # Format
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Altitude (ft)")
    ax.set_title(title)
    if show_legend:
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.subplots_adjust(right=0.75)
    return fig, ax


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

def plot_fuel_flow_and_usage(df1=None, df2=None,
                             color1='#E42320', color2='#E42320',
                            linestyle1 = '--' , linestyle2 = '-', 
                            label1="Trajectory 1", label2="Trajectory 2", title="Fuel Flow and Usage"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # --- Plot Fuel Flow ---
    if df1 is not None:
        axs[0].plot(df1.recTime, df1.fuel_flow, label=label1, color=color1, linestyle = linestyle1)
    if df2 is not None:
        axs[0].plot(df2.recTime, df2.fuel_flow, label=label2, color=color2, linestyle = linestyle2)
    axs[0].set_ylabel("fuel flow (kg/s)")
    format_ax(axs[0])
    axs[0].legend()

    # --- Plot Cumulative Fuel Used ---
    axs[1].plot(df1.recTime, df1.fuel_used, label=label1, color=color1, linestyle = linestyle1)
    if df2 is not None:
        axs[1].plot(df2.recTime, df2.fuel_used, label=label2, color=color2, linestyle = linestyle2)

    axs[1].set_ylabel("fuel used (kg)")
    format_ax(axs[1])
    axs[1].legend()

    # --- Mark final fuel usage points ---
    final_time_1 = df1.recTime.iloc[-1]
    final_fuel_1 = df1.fuel_used.iloc[-1]
    axs[1].scatter(final_time_1, final_fuel_1, color=color1, zorder=3)
    axs[1].annotate(f"{final_fuel_1:.0f} kg",
                    xy=(final_time_1, final_fuel_1),
                    xytext=(final_time_1 - pd.Timedelta(seconds=10), final_fuel_1 + 20),
                    fontsize=9, color=color1)

    if df2 is not None:
        final_time_2 = df2.recTime.iloc[-1]
        final_fuel_2 = df2.fuel_used.iloc[-1]
        axs[1].scatter(final_time_2, final_fuel_2, color=color2, zorder=3)
        axs[1].annotate(f"{final_fuel_2:.0f} kg",
                        xy=(final_time_2, final_fuel_2),
                        xytext=(final_time_2 - pd.Timedelta(seconds=10), final_fuel_2 + 20),
                        fontsize=9, color=color2)

    axs[1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_NOx_flow_and_emission(df1=None, df2=None,
                               color1='#E42320', color2='#6A8EC9',
                               linestyle1='--', linestyle2='-',
                               label1="Trajectory 1", label2="Trajectory 2",
                               title="NOx Emission Flow and Cumulative NOx Emitted"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # --- Plot NOx Flow ---
    if df1 is not None:
        axs[0].plot(df1.recTime, df1.NOx_flow, label=label1, color=color1, linestyle=linestyle1)
    if df2 is not None:
        axs[0].plot(df2.recTime, df2.NOx_flow, label=label2, color=color2, linestyle=linestyle2)
    axs[0].set_ylabel("NOx flow (kg/s)")
    format_ax(axs[0])
    axs[0].legend()

    # --- Plot Cumulative NOx Emitted ---
    axs[1].plot(df1.recTime, df1.NOx_emitted, label=label1, color=color1, linestyle=linestyle1)
    if df2 is not None:
        axs[1].plot(df2.recTime, df2.NOx_emitted, label=label2, color=color2, linestyle=linestyle2)

    axs[1].set_ylabel("NOx emitted (kg)")
    format_ax(axs[1])
    axs[1].legend()

    # --- Mark final NOx emission points ---
    final_time_1 = df1.recTime.iloc[-1]
    final_NOx_1 = df1.NOx_emitted.iloc[-1]
    axs[1].scatter(final_time_1, final_NOx_1, color=color1, zorder=3)
    axs[1].annotate(f"{final_NOx_1:.2f} kg",
                    xy=(final_time_1, final_NOx_1),
                    xytext=(final_time_1 - pd.Timedelta(seconds=10), final_NOx_1 + 200),
                    fontsize=9, color=color1)

    if df2 is not None:
        final_time_2 = df2.recTime.iloc[-1]
        final_NOx_2 = df2.NOx_emitted.iloc[-1]
        axs[1].scatter(final_time_2, final_NOx_2, color=color2, zorder=3)
        axs[1].annotate(f"{final_NOx_2:.2f} kg",
                        xy=(final_time_2, final_NOx_2),
                        xytext=(final_time_2 - pd.Timedelta(seconds=10), final_NOx_2 + 200),
                        fontsize=9, color=color2)

    axs[1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def animate_milp_trajectories(df_wide, flights, star_fixes_xyz, output_path, fps=1, dpi=100):
    """Step-by-step GIF of MILP waypoint trajectories — each frame is one MILP timestep."""
    from matplotlib.animation import FuncAnimation
    from pathlib import Path

    n_f   = len([c for c in df_wide.columns if c.endswith('_x') and c.startswith('f')])
    steps = sorted(df_wide['t_step'].unique())
    palette = ["#E42320","#2171B5","#2CA25F","#F16913","#7B2D8B","#D4AC0D","#1B7837","#D6604D","#8B4513"]
    colors  = [palette[i % len(palette)] for i in range(n_f)]
    labels  = [flights.iloc[i]['acId'].split('_')[0] if i < len(flights) else f'f{i+1}'
               for i in range(n_f)]

    fix_x  = [v[0]/1000 for v in star_fixes_xyz.values()]
    fix_y  = [v[1]/1000 for v in star_fixes_xyz.values()]
    fix_nm = list(star_fixes_xyz.keys())
    th     = np.linspace(0, 2*np.pi, 361)
    pre_x, pre_y = 333*np.cos(th), 333*np.sin(th)   # ~3° radius pre-TRACON circle
    _ord   = ["CRAKN","BONZZ","FERRL","LAYKS","GRAYT","RKCTY","HAYLL","HANBL"]
    tr_x   = [star_fixes_xyz[n][0]/1000 for n in _ord if n in star_fixes_xyz]
    tr_y   = [star_fixes_xyz[n][1]/1000 for n in _ord if n in star_fixes_xyz]
    tr_x  += [tr_x[0]]; tr_y += [tr_y[0]]   # close polygon

    fig, ax = plt.subplots(figsize=(10, 10))
    stat_cols = [f'f{i}_status' for i in range(1, n_f+1) if f'f{i}_status' in df_wide.columns]
    first_k   = int(df_wide[df_wide[stat_cols].max(axis=1) >= 1]['t_step'].min())
    frames    = [i for i, k in enumerate(steps) if k >= max(0, first_k - 1)]

    def draw_frame(frame_idx):
        k   = steps[frame_idx]
        row = df_wide[df_wide['t_step'] == k].iloc[0]
        ax.clear()
        ax.plot(pre_x, pre_y, 'g--', lw=0.8, alpha=0.35)
        ax.plot(tr_x,  tr_y,  color='#018B38', lw=1.5, alpha=0.7, label='TRACON')
        ax.scatter([0],[0], color='#018B38', s=120, marker='*', zorder=10)
        ax.text(3, 3, 'DTW', color='#018B38', fontsize=9, fontweight='bold')
        ax.scatter(fix_x, fix_y, color='darkgreen', s=50, marker='^', zorder=8)
        for nm, fx, fy in zip(fix_nm, fix_x, fix_y):
            ax.text(fx+3, fy+3, nm, color='darkgreen', fontsize=7)

        n_fly = n_arr = 0
        for i in range(n_f):
            fi = i+1
            cx  = f'f{fi}_x'; cy_col = f'f{fi}_y'; cs = f'f{fi}_status'
            if cx not in df_wide.columns: continue
            color  = colors[i]; status = int(row[cs])
            past   = df_wide[(df_wide['t_step'] <= k) & (df_wide[cs] == 1)]
            if len(past):
                ax.plot(past[cx]/1000, past[cy_col]/1000, color=color, lw=1.5, alpha=0.6, zorder=5)
            px, py = row[cx]/1000, row[cy_col]/1000
            if status == 0:
                ax.scatter([px],[py], s=30, color='lightgrey', edgecolors=color, lw=0.8, zorder=6)
            elif status == 1:
                ax.scatter([px],[py], s=90, color=color, marker='>', edgecolors='white', lw=0.5, zorder=7)
                ax.text(px+4, py+4, labels[i], color=color, fontsize=8, fontweight='bold'); n_fly += 1
            else:
                ax.scatter([px],[py], s=140, color=color, marker='*', edgecolors='white', lw=0.5, zorder=8)
                ax.text(px+4, py-8, labels[i]+'✓', color=color, fontsize=8); n_arr += 1

        t_str = str(row.get('t_utc', f'step {k}')).replace('T',' ').replace('Z','')
        ax.set_title(f'MILP Trajectories  |  step {k}  ({t_str})', fontsize=11, fontweight='bold')
        ax.text(0.02, 0.98, f'Flying: {n_fly}   Arrived: {n_arr}', transform=ax.transAxes,
                fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
        ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()

    anim = FuncAnimation(fig, draw_frame, frames=frames, interval=1000//fps, repeat=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved → {output_path}\n")

