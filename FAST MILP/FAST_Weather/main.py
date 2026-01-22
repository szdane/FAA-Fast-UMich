# input data processing package
import pandas as pd
from datetime import timedelta
import time
import numpy as np
from openap.extra.aero import fpm, ft, kts

# input custom helper functions
from functions import computation, plot
from functions.trajectory import Cruise_with_Multi_Waypoints
from functions.single_flight import single_flight_optimization
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# input plotting functions
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import os

start_time = time.time()  # Record start time

############
# 1. Input #
############
# 1.1 DTW location
dtw_coord = (42.2125, -83.3534)
dtw_lat, dtw_lon = dtw_coord

# 1.2 Arbitrary Size of pre-TRACON area
preTRACON_radius = 3 # degrees
   
df_csv = pd.read_csv("/Users/sdane/Desktop/FAA Fast/code/FAST MILP/filtered_rows.csv",
                     usecols=lambda col: col in ["recTime", "acId", "groundSpeed", "alt", "rateOfClimb", "coord1", "coord2"],
                     dtype={"acId": str})  # Load the flight data into pandaframe df

# Group by acId to create a dictionary of small DataFrames
dic_by_acid = {acid: group for acid, group in df_csv.groupby('acId')} # dic_by_acid is a dictionary of small DataFrames

# 1.4 Data preprocessing
for acId in dic_by_acid.keys():     
    df = dic_by_acid[acId]
    df["recTime"] = pd.to_datetime(df["recTime"], format='mixed') #Convert recTime to datetime format and compute time difference 
    df["d_ts"] = df["recTime"].diff().dt.total_seconds() # Compute d_ts only inside this group
    df.iloc[0, df.columns.get_loc("d_ts")] = df.iloc[1]["d_ts"] # Replace first row's d_ts with second row's value
    df["alt"] = df["alt"] * 100 # Convert altitude from 100ft to ft
    dic_by_acid[acId] = df

# 2.1 STAR Fixes
# all STAR fixes
star_fixes = {
    "BONZZ": (41.7483, -82.7972), "CRAKN": (41.6730, -82.9405), "CUUGR": (42.3643, -83.0975),
    "FERRL": (42.4165, -82.6093), "GRAYT": (42.9150, -83.6020), "HANBL": (41.7375, -84.1773),
    "HAYLL": (41.9662, -84.2975), "HTROD": (42.0278, -83.3442), "KKISS": (42.5443, -83.7620),
    "KLYNK": (41.8793, -82.9888), "LAYKS": (42.8532, -83.5498), "LECTR": (41.9183, -84.0217),
    "RKCTY": (42.6869, -83.9603), "VCTRZ": (41.9878, -84.0670) # (lat, lon)
}

# select a part of fixes to plot a simple TRACON region
selected_fix_names = ["CRAKN", "BONZZ", "FERRL", "LAYKS", "GRAYT", "RKCTY", "HAYLL", "HANBL"] 
selected_fix_coords = [star_fixes[name] for name in selected_fix_names]

# lat, lon of selected star fixes
selected_fix_lat_list = [lat for lat, lon in selected_fix_coords]
selected_fix_lon_list = [lon for lat, lon in selected_fix_coords]

# lat, lon, x, y of selected star fixes with dtw at origin
selected_fix_x_list, selected_fix_y_list = computation.proj_with_defined_origin(selected_fix_lat_list, selected_fix_lon_list, dtw_lat, dtw_lon)
selected_fix_df = pd.DataFrame({
    "lat": selected_fix_lat_list,
    "lon": selected_fix_lon_list,
    "x": selected_fix_x_list,
    "y": selected_fix_y_list
})

# 2.2 TRACON Area
# Create Shapely Polygon from TRACON boundary
fix_points = list(zip(selected_fix_df["x"], selected_fix_df["y"]))
TRACON_polygon = Polygon(fix_points) #filled polygon
TRACON_polygon_x, TRACON_polygon_y = TRACON_polygon.exterior.xy

# 2.3 Pre-TRACON Area
# Create a circle with 3 degrees radius around DTW (0, 0)
preTRACON_circle_latlon = Point(dtw_lat, dtw_lon).buffer(preTRACON_radius)  # buffer radius in degrees #filled circle
x_raw, y_raw = preTRACON_circle_latlon.exterior.xy  # x = lon, y = lat
lat_array = np.array(x_raw)
lon_array = np.array(y_raw)
preTRACON_circle_x, preTRACON_circle_y = computation.proj_with_defined_origin(lat_array, lon_array, dtw_lat, dtw_lon)
preTRACON_circle_xy = Polygon(zip(preTRACON_circle_x, preTRACON_circle_y))

##############
# 2. Compute #
##############
# aircraft_list = [
#     {"acId": "FFT3987_KMCOtoKDTW", "acType": "A20N", "waypoint_csv": "FAST_Extended_Abstract/data/flight1.csv"},
#     {"acId": "DAL8952_KPHLtoKDTW", "acType": "B752", "waypoint_csv": "FAST_Extended_Abstract/data/flight2.csv"},
#     {"acId": "SWA3255_KDENtoKDTW", "acType": "B38M", "waypoint_csv": "FAST_Extended_Abstract/data/flight3.csv"},
#     {"acId": "SWA3201_KBWItoKDTW", "acType": "B737", "waypoint_csv": "FAST_Extended_Abstract/data/flight4.csv"},
#     {"acId": "AAL419_KCLTtoKDTW", "acType": "B738", "waypoint_csv": "FAST_Extended_Abstract/data/flight5.csv"},
#     {"acId": "UPS1480_KSDFtoKDTW", "acType": "B763", "waypoint_csv": "FAST_Extended_Abstract/data/flight7.csv"},
#     {"acId": "DAL1296_KDENtoKDTW", "acType": "B739", "waypoint_csv": "FAST_Extended_Abstract/data/flight8.csv"},
#     {"acId": "FFT4886_KPHXtoKDTW", "acType": "A20N", "waypoint_csv": "FAST_Extended_Abstract/data/flight9.csv"},
#     {"acId": "DAL1445_KPHXtoKDTW", "acType": "A321", "waypoint_csv": "FAST_Extended_Abstract/data/flight10.csv"},
# ]

# aircraft_list = [
#     {"acId": "DAL8952_KPHLtoKDTW", "acType": "B752", "waypoint_csv": "FAST_Extended_Abstract/data/flight2.csv"},
#     {"acId": "SWA3201_KBWItoKDTW", "acType": "B737", "waypoint_csv": "FAST_Extended_Abstract/data/flight4.csv"},
#     {"acId": "AAL419_KCLTtoKDTW", "acType": "B738", "waypoint_csv": "FAST_Extended_Abstract/data/flight5.csv"},
#     {"acId": "DAL1296_KDENtoKDTW", "acType": "B739", "waypoint_csv": "FAST_Extended_Abstract/data/flight8.csv"},
#     {"acId": "DAL1445_KPHXtoKDTW", "acType": "A321", "waypoint_csv": "FAST_Extended_Abstract/data/flight10.csv"},
# ]

aircraft_list = [
    {"acId": "DAL1066_KTPAtoKDTW", "acType": "B738", "waypoint_csv": "/Users/sdane/Desktop/FAA Fast/code/FAA-Fast-UMich/FAST MILP/res/weathertrial.csv"},

]

dic_hist = {}
dic_wp = {}
dic_hist_preTracon = {}
dic_opt_preTracon = {}

for ac in aircraft_list:
    print("Processing aircraft:", ac["acId"])
    acID = ac["acId"]
    acType = ac["acType"]
    waypoint_csv = ac["waypoint_csv"]

    df_hist, df_wp, df_hist_preTracon, df_opt_preTRACON = single_flight_optimization(
        acId=acID,
        acType=acType,
        dic_by_acid=dic_by_acid,
        waypoint_csv=waypoint_csv,
        dtw_lat=42.2125,
        dtw_lon=-83.3534,
        preTRACON_radius=3,
        preTRACON_circle_xy=preTRACON_circle_xy,
        TRACON_polygon=TRACON_polygon,
    )
    dic_hist[acID] = df_hist
    dic_wp[acID] = df_wp
    dic_hist_preTracon[acID] = df_hist_preTracon
    dic_opt_preTracon[acID] = df_opt_preTRACON

#########################
# 3. Plot & Comparation #
#########################
pd.set_option("display.max_rows", 40)

# 5.1 Print historic and optimized dataframes for all aircraft
output_dir = "FAST_Extended_Abstract/data"
os.makedirs(output_dir, exist_ok=True)
for ac in aircraft_list:
    acId = ac["acId"]
    print(f"Historic pre-TRACON trajectory for {acId}")
    print(dic_hist_preTracon[acId])
    print(f"Optimized pre-TRACON trajectory for {acId}")
    print(dic_opt_preTracon[acId])

    hist_path = os.path.join(output_dir, f"{acId}_pretracon_historic.csv")
    opt_path = os.path.join(output_dir, f"{acId}_pretracon_optimized.csv")
    dic_hist_preTracon[acId].to_csv(hist_path, index=False)
    dic_opt_preTracon[acId].to_csv(opt_path, index=False)
    print(f"Saved CSVs for {acId}:\n  {hist_path}\n  {opt_path}")


# # 5.2 Plot Historic trajectory
# plot.plot_2d_trajectories(dic=dic_hist, labels=[f"Historic Trajectory for \n {ac['acId']}" for ac in aircraft_list], colors=["#E42320", "#6A8EC9", "#B46DA9"], plot_trajectory_endpoints=False,
#                         tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y),  preTracon_label="Pre-TRACON",
#                         plot_waypoints=False, waypoints=None, waypoints_tolerance=None, plot_waypoint_tol_zone=False,
#                         lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=True,
#                         title="2D Historic Flight Trajectories")

# 5.2 plot original & cleaned waypoints
plot.plot_2d_trajectories(dic=None, labels=None, colors=None, plot_trajectory_endpoints=False, 
                        tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y),  preTracon_label="Pre-TRACON",
                        plot_waypoints=True, waypoints=dic_wp, waypoints_tolerance=3000, plot_waypoint_tol_zone=False,
                        lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=False,
                        title="Optimized Waypoints")

# 5.3 Plot 2D trajectory
combined_dic = {}
for acId, df_hist in dic_hist_preTracon.items():
    combined_dic[f"{acId}_historic"] = df_hist
for acId, df_opt in dic_opt_preTracon.items():
    combined_dic[f"{acId}_optimized"] = df_opt
#print(combined_dic)
plot.plot_2d_trajectories(
    dic=combined_dic, labels=[f"Historic {ac['acId']}" for ac in aircraft_list] + [f"Optimized {ac['acId']}" for ac in aircraft_list],
    colors=["#E42320", "#6A8EC9", "#E42320", "#6A8EC9"], plot_trajectory_endpoints=False,
    tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y), preTracon_label="Pre-TRACON",
    plot_waypoints=True, waypoints=dic_wp, waypoints_tolerance=3000, plot_waypoint_tol_zone=False,
    lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=True,
    title="2D Historic and Optimized Flight Trajectories")
    #["#E42320", "#6A8EC9","#B46DA9", "#E42320", "#6A8EC9","#B46DA9"]

# 5.4 Plot 3D Trajectory
combined_dic = {}
for acId, df_hist in dic_hist_preTracon.items():
    combined_dic[f"{acId}_historic"] = df_hist
for acId, df_opt in dic_opt_preTracon.items():
    combined_dic[f"{acId}_optimized"] = df_opt
plot.plot_3d_trajectories(dic=combined_dic, labels=[f"Historic {ac['acId']}" for ac in aircraft_list] + [f"Optimized {ac['acId']}" for ac in aircraft_list], colors=["#E42320", "#6A8EC9", "#E42320", "#6A8EC9"],
                        waypoints=dic_wp, waypoints_tolerance_radius=3000, waypoints_tolerance_height=600, plot_waypoint_tol_vol=True,
                        lat0=dtw_lat, lon0=dtw_lon,
                        title="3D Historic & Optimized Routes")

# 5.5 Plot fuel flow/usage and NOx emissions
for idx, ac in enumerate(aircraft_list):
    acId = ac["acId"]

    df_hist = dic_hist_preTracon[acId]       # historic trajectory in pre-TRACON
    df_opt = dic_opt_preTracon[acId]         # optimized trajectory in pre-TRACON

    # Choose colors per aircraft
    color_list = ["#E42320", "#6A8EC9", "#E42320", "#6A8EC9"] #["#E42320", "#6A8EC9","#B46DA9", "#E42320", "#6A8EC9","#B46DA9"]
    color = color_list[idx % len(color_list)]

    # i) Fuel Flow and Usage
    plot.plot_fuel_flow_and_usage(
        df1=df_hist, df2=df_opt,
        color1=color, color2=color,
        linestyle1='--', linestyle2='-',
        label1=f"{acId} Historic Route",
        label2=f"{acId} Optimized Route",
        title=f"{acId} Fuel Flow and Usage"
    )

    # ii) NOx Flow and Emission
    plot.plot_NOx_flow_and_emission(
        df1=df_hist, df2=df_opt,
        color1=color, color2=color,
        linestyle1='--', linestyle2='-',
        label1=f"{acId} Historic Route",
        label2=f"{acId} Optimized Route",
        title=f"{acId} NOx Flow and Emission"
    )

    # iii) Fuel usage and NOx emission in the same figure
    plot.plot_fuel_and_NOx_usage(
        df1=df_hist, df2=df_opt,
        color1="#E42320", color2="#6A8EC9",
        linestyle1='--', linestyle2='-',
        label1=f"{acId} Historic Route",
        label2=f"{acId} Optimized Route",
        title=f"{acId} Fuel Used and NOx Emitted"
    )


end_time = time.time()  # Record end time
elapsed = end_time - start_time
print(f"\nPlotting completed, total execution time: {elapsed:.2f} seconds")

plt.show()