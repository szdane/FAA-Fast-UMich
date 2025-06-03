# Fuel Estimation For A Single Flight
# Kuang Sun, May 2025

# input data processing package
import pandas as pd
from datetime import timedelta
import time

# input custom helper functions
from functions import computation_funcs, plot_funcs
from functions.trajectory_interpolation_from_waypoints import Cruise_with_Multi_Waypoints

# input plotting functions
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def main():
    start_time = time.time()  # Record start time

    ############
    # 1. Input #
    ############
    # 1.1 Aircraft No. and Type
    print("Please enter acId: (e.g. DAL1296_KDENtoKDTW)")
    # target_acId = input()  # For now this is manually inputted. However, it would be better if this info in incoporated into Eric's output file
    target_acId = "DAL1296_KDENtoKDTW"
    print("Please enter the aircraft type: (e.g. A321)")
    # target_acType = input() # For now this is manually inputted. However, it would be better if this info in incoporated into Yuwei's output file
    target_acType = "A321"

    # 1.2 DTW location
    dtw_coord = (42.2125, -83.3534)
    dtw_lat, dtw_lon = dtw_coord

    # 1.3 Arbitrary Size of pre-TRACON area
    print("Please enter radius of pre-TRACON area in km: (e.g. 20)")
    # preTRACON_radius = input() # For now this is manually inputted. However, it would be better if this info in incoporated into Eric's output file
    preTRACON_radius = 270

    # 1.4 Assumed initial mass factor
    print("Please enter how much percent of MTOW does aircraft weigh wheen entering pre-TRACON: e.g. 0.8 (which means 80%) ")
    # m0 = input()
    m0 = 0.8 # assume weight when entering the pre-TRACON region is 80% MTOW when entering pre-TRACON
   
    # 1.5 Input Yuwei's csv and find the target flight (ref: https://openap.dev/fuel_emission.html) (coord1, coord2 = lat, lon)
    # i) first dataframe containing data for all the time for all the flights
    df_csv = pd.read_csv("data/filtered_rows.csv",
                     usecols=lambda col: col in [
                         "recTime", "acId", "groundSpeed", "alt", "rateOfClimb", "coord1", "coord2"],
                     dtype={"acId": str})  # Load the flight data into pandaframe df
    df_csv["recTime"] = pd.to_datetime(df_csv["recTime"], format='mixed')
    df_csv = df_csv.assign(d_ts=lambda d: d.recTime.diff().dt.total_seconds().bfill()) # recTime Setup in pandaframe df
    df_csv["alt"] = df_csv["alt"] * 100 # Altitude conversion pandaframe df
    df_by_acid = {acid: group for acid, group in df_csv.groupby('acId')} # Group by acId pandaframe df

    # ii) second dataframe containing data for all the time for the target flight
    df_target_hist = df_by_acid[target_acId]  # Select target aircraft from pandaframe df

    # iii) compute x, y coordinates of historic flight
    df_target_hist_x_column, df_target_hist_y_column = computation_funcs.proj_with_defined_origin(df_target_hist["coord1"], df_target_hist["coord2"], dtw_lat, dtw_lon)
    df_target_hist['x'] = df_target_hist_x_column
    df_target_hist['y'] = df_target_hist_y_column

    # 1.6 Eric's Optimized Waypoints, which includes entering and exiting location for pre-TRACON
    print("Please enter Waypoints in (lat, lon) form (e.g. (-84.5078, 44.9031), (-84.5078, 44.8262)...)")
    #target_optWaypoints = input() # Eric need to convert output to tuple list format of [(float, float), (float, float), ...]
    target_optWaypoints = [
        (42.5298, -86.6081), (42.5192, -86.1996), (42.5192, -86.1227),
        (42.4423, -86.0458), (42.3654, -85.9689), (42.2885, -85.892),
        (42.2116, -85.8151), (42.1347, -85.7382), (42.0578, -85.6613),
        (41.9809, -85.5844), (41.9809, -85.5075), (41.9809, -85.4306),
        (41.9809, -85.3537), (41.9809, -85.2768), (41.9809, -85.1999),
        (41.9809, -85.123), (41.9809, -85.0461), (41.9809, -84.9692),
        (41.9809, -84.8923), (41.9809, -84.8154), (41.9809, -84.7385),
        (41.9809, -84.6616), (41.9809, -84.5847), (41.9809, -84.5078),
        (41.9809, -84.4309), (41.9809, -84.354), (41.9662, -84.2975)
    ] #the first one should be the entry point (intersection) and the last one should be a star fix

    print("Input Success")

    ############################################
    # 2. Compute Pre-TRACON Entry & Exit Point #
    ############################################
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
    selected_fix_x_list, selected_fix_y_list = computation_funcs.proj_with_defined_origin(selected_fix_lat_list, selected_fix_lon_list, dtw_lat, dtw_lon)
    selected_fix_df = pd.DataFrame({
        "lat": selected_fix_lat_list,
        "lon": selected_fix_lon_list,
        "x": selected_fix_x_list,
        "y": selected_fix_y_list
    })
    print(selected_fix_df)
  
    # 2.2 TRACON Area
    # Create Shapely Polygon from TRACON boundary
    fix_points = list(zip(selected_fix_df["x"], selected_fix_df["y"]))
    TRACON_polygon = Polygon(fix_points) #filled polygon
    TRACON_polygon_x, TRACON_polygon_y = TRACON_polygon.exterior.xy

    # 2.3 Pre-TRACON Area
    # Create a circle with 300 km radius around DTW (0, 0)
    preTRACON_circle = Point(0, 0).buffer(preTRACON_radius*1000)  # buffer radius in meters #filled circle
    preTRACON_circle_x, preTRACON_circle_y = preTRACON_circle.exterior.xy

    # 2.4 Find Trajectory Intersection
    # Pre-TRACON circle & Trajectory Intersection
    preTRACON_entry, pre_idx = computation_funcs.find_trajectory_intersection(df_target_hist, preTRACON_circle) # find intersection (entry point) x-y coordinate
    preTRACON_entry_lat, pre_entry_lon = computation_funcs.proj_with_defined_origin(preTRACON_entry[0], preTRACON_entry[1], dtw_lat, dtw_lon) # translate intersection x-y coordinate to lat-lon coordinate
    interpolated_entry_row = computation_funcs.interpolate_row_from_xy(df_target_hist, preTRACON_entry[0], preTRACON_entry[1]) # Interpolate entry time, altitude, groundspeed, rateOfClimb

    # TRACON polygon & Trajectory Intersection
    preTRACON_exit, tracon_idx = computation_funcs.find_trajectory_intersection(df_target_hist, TRACON_polygon) # find intersection (exit point) x-y coordinate
    preTRACON_exit_lat, pre_exit_lon = computation_funcs.proj_with_defined_origin(preTRACON_exit[0], preTRACON_exit[1], dtw_lat, dtw_lon) # translate intersection x-y coordinate to lat-lon coordinate
    interpolated_exit_row = computation_funcs.interpolate_row_from_xy(df_target_hist, preTRACON_exit[0], preTRACON_exit[1]) # Interpolate entry time, altitude, groundspeed, rateOfClimb

    # 2.5 Find the segment of historic trajectory within pre-TRACON, and append the entry and exit rows
    df_target_preTRACON_hist = df_target_hist[
    (df_target_hist["recTime"] >= interpolated_entry_row["recTime"]) &
    (df_target_hist["recTime"] <= interpolated_exit_row["recTime"])] # include the rows within pre-TRACON time period
    
    df_target_preTRACON_hist = pd.concat(
        [pd.DataFrame([interpolated_entry_row]), df_target_preTRACON_hist,
         pd.DataFrame([interpolated_exit_row])]
    ).sort_values("recTime").reset_index(drop=True) # include the two rows entering and exiting the pre-TRACON

    print("Compute Pre-TRACON Entry & Exit Point Success")

    ##########################################
    # 3. Historic Trajectory Fuel Estimation #
    ##########################################
    # 3.1 Fuel Computation (ref: https://openap.dev/fuel_emission.html)
    df_target_preTRACON_hist = computation_funcs.compute_fuel_flow(df_target_preTRACON_hist, target_acType, m0) 
    # this calculates the fuel flow and fuel used within the preTRACON area, and add the fuel flow and fuel used as three new columns
    
    # 3.2 add two more columns for x, y coordinates
    x_hist, y_hist = computation_funcs.proj_with_defined_origin(df_target_preTRACON_hist["coord1"], df_target_preTRACON_hist["coord2"], dtw_lat, dtw_lon)
    df_target_preTRACON_hist["x"] = x_hist
    df_target_preTRACON_hist["y"] = y_hist

    print("Historic Trajectory Fuel Estimation Success")

    ###########################################
    # 4. Optimized Trajectory Fuel Estimation #
    ###########################################
    # 4.1 Extract origin and destination, and intermediate waypoints from Eric's waypoint list
    origin = target_optWaypoints[0]
    destination = target_optWaypoints[-1]
    middle_waypoints = target_optWaypoints[1:-1]

    # 4.2 Data cleaning by reducing number of waypoints
    middle_waypoints_cleaned = computation_funcs.waypoint_cleaning(origin, middle_waypoints, destination, threshold_deg=10)
    
    # 4.3 Interpolate trajectory
    waypoint_proximity = 500 # proximity for how close trajecotry should be near the waypoints (unit: m)

    cruise_flight = Cruise_with_Multi_Waypoints(
        target_acType, origin, destination, m0=m0
    ).trajectory(
        middle_waypoints=middle_waypoints_cleaned,  # pass waypoints here
        middle_radius=waypoint_proximity        # single radius for all waypoints
    )

    # 4.4 Data pre-processing in dataframe containing flight parameters of the points in the interpolated trajectory
    target_enter_time = interpolated_entry_row['recTime']
    rectime_opt = [target_enter_time + timedelta(seconds=dt) for dt in cruise_flight["ts"]] # helper_functions trajectory_interpolation_from_waypoints.py trajectory(), and then #helper_functions/base.py to_trajectory()

    df_target_preTRACON_opt = pd.DataFrame({
        "acId": target_acId,
        "coord1": cruise_flight["latitude"],
        "coord2": cruise_flight["longitude"],
        "alt": cruise_flight["altitude"],                   # from 'altitude' (in feet)
        "groundSpeed": cruise_flight["tas"],                # from 'tas' (in knots) #assume there's no wind, then true airspeed would be the same as ground speed
        "rateOfClimb": cruise_flight["vertical_rate"],      # from 'vertical_rate' (in ft/min)
        "recTime":rectime_opt,
        "fuel_flow": cruise_flight["fuel_flow"],            # fuel flow and fuel has been computed when interpolating the trajecotry
        "fuel_used": cruise_flight["fuel_used"]             # helper_functions trajectory_interpolation_from_waypoints.py trajectory(), and then #helper_functions/base.py to_trajectory()
    })

    # 4.5 add two more columns for x, y coordinates
    x_opt, y_opt = computation_funcs.proj_with_defined_origin(df_target_preTRACON_opt["coord1"], df_target_preTRACON_opt["coord2"], dtw_lat, dtw_lon)
    df_target_preTRACON_opt["x"] = x_opt
    df_target_preTRACON_opt["y"] = y_opt

    print("Optimized Trajectory Fuel Estimation Success")

    #########################
    # 5. Plot & Comparation #
    #########################
    # 5.1 Historic trajectory
    plot_funcs.plot_2d_trajectory(df1=df_target_hist, df2=None, label1="Historic Trajectory", label2=None, plot_trajectory_endpoints = False,
                       tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), pretracon_circle=(preTRACON_circle_x, preTRACON_circle_y), tracon_label="TRACON", pretracon_label="Pre-TRACON",
                       lat0=dtw_lat, lon0=dtw_lon, lat_lon_grid=True,
                       waypoints=None, waypoints_plot = False,
                       preTRACON_entry=preTRACON_entry, preTRACON_exit=preTRACON_exit, pre_TRACON_endpoints = True,
                       title="2D Historic Flight Trajectory")
    
    # 5.2 plot original & cleaned waypoints
    plot_funcs.plot_2d_trajectory(df1=None, df2=None, label1=None, label2=None, plot_trajectory_endpoints = False,
                       tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), pretracon_circle=(preTRACON_circle_x, preTRACON_circle_y), tracon_label="TRACON", pretracon_label="Pre-TRACON",
                       lat0=dtw_lat, lon0=dtw_lon, lat_lon_grid=True,
                       waypoints=[origin] + middle_waypoints + [destination], waypoints_plot = True,
                       preTRACON_entry=None, preTRACON_exit=None, pre_TRACON_endpoints = False,
                       title="Optimal Waypoints") # original waypoints
    
    plot_funcs.plot_2d_trajectory(df1=None, df2=None, label1=None, label2=None, plot_trajectory_endpoints = False,
                       tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), pretracon_circle=(preTRACON_circle_x, preTRACON_circle_y), tracon_label="TRACON", pretracon_label="Pre-TRACON",
                       lat0=dtw_lat, lon0=dtw_lon, lat_lon_grid=True,
                       waypoints=[origin] + middle_waypoints_cleaned + [destination], waypoints_plot = True,
                       preTRACON_entry=None, preTRACON_exit=None, pre_TRACON_endpoints = False,
                       title="Cleaned Optimal Waypoints") # cleaned waypoints
    
    # 5.3 plot Historic & optimized trajectory with in pre-TRACON region in 2D & 3D
    plot_funcs.plot_2d_trajectory(df1=df_target_preTRACON_hist, df2=df_target_preTRACON_opt, label1="Historic Trajectory", label2="Optimized Trajectory", plot_trajectory_endpoints = False,
                       tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), pretracon_circle=(preTRACON_circle_x, preTRACON_circle_y), tracon_label="TRACON", pretracon_label="Pre-TRACON",
                       lat0=dtw_lat, lon0=dtw_lon, lat_lon_grid=True,
                       waypoints=[origin] + middle_waypoints_cleaned + [destination], waypoints_plot = True,
                       preTRACON_entry=None, preTRACON_exit=None, pre_TRACON_endpoints = False,
                       title="Historic & Optimal Trajectory")
    
    plot_funcs.plot_3d_trajectory(df_target_preTRACON_hist, df_target_preTRACON_opt, label1="Historic", label2="Optimized")

    # 5.4 Plot Altitude, GroundSpeed & RateOfClimb
    plot_funcs.plot_altitude_groundspeed_rateOfClimb(df1=df_target_preTRACON_hist, df2=df_target_preTRACON_opt, label1="Historic", label2="Optimized")

    # 5.5 Plot Fuel Flow & Fuel Used
    plot_funcs.plot_fuel_flow_and_usage(df1=df_target_preTRACON_hist, df2=df_target_preTRACON_opt, label1="Historic", label2="Optimized")

    end_time = time.time()  # Record end time
    elapsed = end_time - start_time
    print(f"\nPlotting compted, total execution time: {elapsed:.2f} seconds")

    plt.show()



if __name__ == "__main__":
    main()
