# Fuel Estimation For A Single Flight with Serra's Waypoints
# Kuang Sun, June 2025
# Modified to accept df_wide from MILP optimization as input

"""
Usage with MILP optimization output (df_wide):

    from milp_multiple_Debug import *  # Run optimization
    from main import analyze_optimized_trajectory
    
    # After optimization in milp_multiple_Debug.py produces df_wide:
    aircraft_list = [
        {"acId": "DAL1208_KORDtoKDTW", "acType": "B737"},
        {"acId": "DAL1066_KTPAtoKDTW", "acType": "B737"}
    ]
    
    results = analyze_optimized_trajectory(df_wide, aircraft_list)
    # results contains: 'historic', 'optimized', 'waypoints'
"""

# input data processing package
import pandas as pd
from datetime import timedelta
import time
import numpy as np
from openap.extra.aero import fpm, ft, kts
from pathlib import Path

# input custom helper functions
from Functions import fuel_emission_analysis_computation, fuel_emission_analysis_plot
from Functions.fuel_emission_analysis_trajectory import Cruise_with_Multi_Waypoints

# input plotting functions
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def analyze_optimized_trajectory(df_wide, aircraft_list=None, final_times=None):
    """
    Analyze and visualize optimized trajectories from MILP optimization.
    
    Args:
        df_wide (pd.DataFrame): Optimized trajectory data with columns ['t', 'f1_lat', 'f1_lon', 'f1_alt_ft', ...]
        aircraft_list (list): List of dicts with 'acId' and 'acType'. If None, uses default.
        final_times (dict): Dictionary mapping acId to final time in seconds from MILP is_end variable. Optional.
    
    Returns:
        dict: Dictionary containing historic and optimized trajectory data
    """
    start_time = time.time()  # Record start time
    
    # Get the directory where main.py is located
    script_dir = Path(__file__).parent

    ############
    # 1. Input #
    ############
    print("Program Start...  \n")
    # 1.1 Aircraft No. and Type for target flight and intruder flight
    if aircraft_list is None:
        aircraft_list = [
            {"acId": "DAL1066_KTPAtoKDTW", "acType": "B737"},
            {"acId": "DAL498_KRSWtoKDTW", "acType": "B737"}]
            #{"acId": "EDV5018_CYULtoKDTW", "acType": "B737"}]


    # 1.2 DTW location
    dtw_coord = (42.2125, -83.3534)
    dtw_lat, dtw_lon = dtw_coord

    # 1.3 Arbitrary Size of pre-TRACON area
    preTRACON_radius = 3 # degrees

    # 1.4 Assumed initial mass factor
    m0 = 0.8 # assume weight when entering the pre-TRACON region is 70% MTOW when entering pre-TRACON
   
    # 1.5 Input Yuwei's csv and find selected flights (ref: https://openap.dev/fuel_emission.html) (coord1, coord2 = lat, lon)
    # i) first dataframe containing data for all the time for all the flights
    df_csv = pd.read_csv(str(script_dir.parent / "Input" / "filtered_rows.csv"),
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

    # ii) dictionary of small DataFrames containing data for all the time for the flights, including computed x, y, z coordinates
    dic_hist_flights = {}
    for ac in aircraft_list:
        acId = ac["acId"]
        # Select the DataFrame for this aircraft from dic_by_acid
        df_hist = dic_by_acid[acId]
        # Add computed x, y, z columns
        x_col, y_col = fuel_emission_analysis_computation.proj_with_defined_origin(
            df_hist["coord1"], df_hist["coord2"], dtw_lat, dtw_lon
        )
        df_hist['x'] = x_col
        df_hist['y'] = y_col
        df_hist['z'] = df_hist["alt"] * ft
        # Store it in a dictionary keyed by acId
        dic_hist_flights[acId] = df_hist # a dictionary of small DataFrames
        #print(dic_hist_flights[acId])
    """
    Dictionary dic_hist_flights includes small DataFrames of the following format:
    HISTORIC FLIGHT DATAFRAME
    recTime | acID | coord1(Lat) | coord2(Lon) | alt(ft) | groundSpeed(kts) | rateOfClimb(ft/min) | d_ts(s) | x(m) | y(m) | z(m) |
    """

    # 1.6 Optimized Waypoints from MILP optimization (df_wide input)
    # df_wide has columns: ['t', 'f1_lat', 'f1_lon', 'f1_alt_ft', 'f2_lat', 'f2_lon', 'f2_alt_ft', ...]
    df_Serra_optWaypoints = df_wide.copy()  # Use the input DataFrame directly

    # Create a dictionary to hold smaller DataFrames, one per aircraft
    dic_waypoints = {}

    # Loop over aircraft and extract columns
    for i, ac in enumerate(aircraft_list, start=1):
        acId = ac["acId"]

        # Column names for this aircraft in the CSV
        lat_col = f"f{i}_lat"
        lon_col = f"f{i}_lon"
        alt_col = f"f{i}_alt_ft"

        # Build a small DataFrame for this aircraft
        df_wp = df_Serra_optWaypoints[["t", lat_col, lon_col, alt_col]].copy()
        df_wp.columns = ["t", "lat", "lon", "alt_ft"]  # rename columns

        dic_waypoints[acId] = pd.DataFrame(df_wp) # save waypoints in dataframe format

    """
    Dictionary dic_waypoints includes small DataFrames of the following format:
    INPUT WAYPOINTS DATAFRAME
    t (s) | lat | lon | alt_ft |
    """

    print("--- Input Success --- \n")

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
    selected_fix_x_list, selected_fix_y_list = fuel_emission_analysis_computation.proj_with_defined_origin(selected_fix_lat_list, selected_fix_lon_list, dtw_lat, dtw_lon)
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
    preTRACON_circle_x, preTRACON_circle_y = fuel_emission_analysis_computation.proj_with_defined_origin(lat_array, lon_array, dtw_lat, dtw_lon)
    preTRACON_circle_xy = Polygon(zip(preTRACON_circle_x, preTRACON_circle_y))

    # 2.4 Find Trajectory Intersection
    # Initialize dictionaries for results
    dic_hist_flights_preTracon = {}   # pre-TRACON trajectory segments per aircraft

    for ac in aircraft_list:
        acId = ac["acId"]
        df_hist_flight = dic_hist_flights[acId]

        # Pre-TRACON entry intersection
        entry_xy,  entry_idx = fuel_emission_analysis_computation.find_trajectory_intersection(df_hist_flight, preTRACON_circle_xy)
        entry_lat, entry_lon = fuel_emission_analysis_computation.proj_with_defined_origin(entry_xy[0], entry_xy[1], dtw_lat, dtw_lon, inverse=True)
        entry_row = fuel_emission_analysis_computation.interpolate_row_from_xy(df_hist_flight, entry_xy[0], entry_xy[1])

        # Pre-TRACON exit intersection
        exit_xy, exit_idx = fuel_emission_analysis_computation.find_trajectory_intersection(df_hist_flight, TRACON_polygon)
        exit_lat, exit_lon = fuel_emission_analysis_computation.proj_with_defined_origin(exit_xy[0], exit_xy[1], dtw_lat, dtw_lon, inverse=True)
        exit_row = fuel_emission_analysis_computation.interpolate_row_from_xy(df_hist_flight, exit_xy[0], exit_xy[1])

        # Extract the segment of trajectory within pre-TRACON
        df_segment = df_hist_flight[
            (df_hist_flight["recTime"] >= entry_row["recTime"]) &
            (df_hist_flight["recTime"] <= exit_row["recTime"])
        ]

        # Add the interpolated entry/exit rows
        df_segment = pd.concat(
            [pd.DataFrame([entry_row]), df_segment, pd.DataFrame([exit_row])]
        ).sort_values("recTime").reset_index(drop=True)

        # Save results
        dic_hist_flights_preTracon[acId] = df_segment
        print(f"Computed pre-TRACON entry/exit for {acId}:\nEntry ({entry_lat:.4f}, {entry_lon:.4f}, {entry_row['alt']:.4f})\nExit ({exit_lat:.4f}, {exit_lon:.4f}, {exit_row['alt']:.4f})\n...")
        print("--- Compute Pre-TRACON Entry & Exit Point Success --- \n")


    ##########################################
    # 3. Historic Trajectory Fuel Estimation #
    ##########################################
    # 3.1 Fuel Computation (ref: https://openap.dev/fuel_emission.html)
    for ac in aircraft_list:
        acId = ac["acId"]
        acType = ac["acType"]

        # Get the pre-TRACON segment for this aircraft
        df_pre = dic_hist_flights_preTracon[acId]

        # Compute fuel and emissions for this segment
        df_pre_update = fuel_emission_analysis_computation.compute_fuel_and_emission(df_pre, acType, m0)

        # Cover the previous df in the dictionary
        dic_hist_flights_preTracon[acId] = df_pre_update

    """
    Dictionary dic_hist_flights_preTracon includes small DataFrames of the following format:
    HISTORIC FLIGHT DATAFRAME (within pre-TRACON region)
    recTime | acID | coord1(Lat) | coord2(Lon) | alt(ft) | groundSpeed(kts) | rateOfClimb(ft/min) | d_ts(s) | x(m) | y(m) | z(m) |
        fuel_flow(kg/s) | fuel_used(kg) | CO2_flow | Soot_flow | SOx_flow | NOx_flow | CO_flow | HC_flow |
        CO2_emitted | H2O_emitted | Soot_emitted | SOx_emitted | NOx_emitted | CO_emitted | HC_emitted
    """

    print("--- Historic Trajectory Fuel Estimation Success --- \n")

    ###########################################
    # 4. Optimized Trajectory Fuel Estimation #
    ###########################################
    # 4.1 Initialization
    dic_opt_flights_preTracon = {} # dictionary to hold small optimized DataFrames for each aircraft
    dic_waypoints_cleaned = {}  # dictionary to hold cleaned waypoints for each aircraft
    waypoint_proximity = 20_000  # radius in meters

    for i, ac in enumerate(aircraft_list, start=1):
        acId = ac["acId"]
        acType = ac["acType"]

        # Waypoints for this aircraft
        df_wp = dic_waypoints[acId]

        # 4.2 Extract origin and destination, and intermediate waypoints from Serra's waypoint list
        origin = tuple(df_wp.iloc[0][["lat", "lon", "alt_ft", "t"]])
        destination = tuple(df_wp.iloc[-1][["lat", "lon", "alt_ft", "t"]])
        middle_waypoints = df_wp.iloc[1:-1][["lat", "lon", "alt_ft", "t"]].values.tolist()

        # 4.3 Data cleaning by reducing number of waypoints
        middle_waypoints_cleaned = middle_waypoints
        # middle_waypoints_cleaned = fuel_emission_analysis_computation.waypoint_cleaning(
        #     origin, middle_waypoints, destination,
        #     lat0=dtw_lat, lon0=dtw_lon, threshold_deg=10
        # )

        dic_waypoints_cleaned[acId] = pd.DataFrame([origin] + middle_waypoints_cleaned + [destination], columns=["lat", "lon", "alt_ft", "t"])  # save cleaned waypoints in dataframe format

        """
        Dictionary dic_waypoints_cleaned includes small DataFrames of the following format:
        INPUT WAYPOINTS DATAFRAME (CLEANED）
        | lat | lon | alt_ft | t (s) |
        """
        
        # 4.4 Interpolate trajectory
        # Get final time for this aircraft if available
        final_time_sec = final_times.get(acId) if final_times else None
        if final_time_sec is not None:
            print(f"Using final time constraint for {acId}: {final_time_sec} seconds")
        
        cruise_flight = Cruise_with_Multi_Waypoints(
            acType, origin, destination, m0=m0
        ).trajectory(
            objective = "fuel",
            middle_waypoints=middle_waypoints_cleaned,
            middle_radius=waypoint_proximity,
            middle_alt_margin=200 * ft,  # altitude tolerance
            forbidden_region=TRACON_polygon,
            final_time=final_time_sec  # Pass final time from MILP optimization
        )

        # Compute recTime for optimized points
        enter_time = dic_hist_flights_preTracon[acId].iloc[0]["recTime"] # Retrieve entry time
        recTime_opt = [enter_time + timedelta(seconds=dt) for dt in cruise_flight["ts"]]

        # Compute d_ts
        cruise_flight["d_ts"] = cruise_flight["ts"].diff().fillna(0)
        cruise_flight[0, "d_ts"] = cruise_flight["d_ts"].iloc[1]  # Fill first value with the first difference

        # Build DataFrame for optimized trajectory
        df_opt = pd.DataFrame({
            "acId": acId,
            "coord1": cruise_flight["latitude"],
            "coord2": cruise_flight["longitude"],
            "alt": cruise_flight["altitude"],
            "groundSpeed": cruise_flight["tas"],
            "rateOfClimb": cruise_flight["vertical_rate"],
            "recTime": recTime_opt,
            "d_ts": cruise_flight["d_ts"],
            "t": cruise_flight["ts"]
        })

        # Compute x,y,z
        x_opt, y_opt = fuel_emission_analysis_computation.proj_with_defined_origin(
            df_opt["coord1"], df_opt["coord2"], dtw_lat, dtw_lon
        )
        df_opt["x"] = x_opt
        df_opt["y"] = y_opt
        df_opt["z"] = df_opt["alt"] * ft

        # 4.5 Estimate fuel and emissions
        df_opt = fuel_emission_analysis_computation.compute_fuel_and_emission(df_opt, acType, m0)

        # 4.6 Save to dictionary
        dic_opt_flights_preTracon[acId] = df_opt

    """
    Dictionary dic_opt_flights_preTracon includes small DataFrames of the following format:
    OPTIMIZED FLIGHT DATAFRAME
    acID | coord1(Lat) | coord2(Lon) | alt(ft) | groundSpeed(kts) | rateOfClimb(ft/min) | recTime | t(s) | d_ts | x(m) | y(m) | z(m) |
        fuel_flow(kg/s) | fuel_used(kg) | CO2_flow | Soot_flow | SOx_flow | NOx_flow | CO_flow | HC_flow |
        CO2_emitted | H2O_emitted | Soot_emitted | SOx_emitted | NOx_emitted | CO_emitted | HC_emitted
    """

    print("--- Optimized Trajectory Fuel Estimation Success --- \n")

    #########################
    # 5. Plot & Comparation #
    #########################
    pd.set_option("display.max_rows", 80)

    # 5.1 Print historic and optimized dataframes for all aircraft
    for ac in aircraft_list:
        acId = ac["acId"]
        print(f"Historic pre-TRACON trajectory for {acId}")
        print(dic_hist_flights_preTracon[acId])
        print(f"Optimized pre-TRACON trajectory for {acId}")
        print(dic_opt_flights_preTracon[acId])

    # 5.2 Plot Historic trajectory
    fuel_emission_analysis_plot.plot_2d_trajectories(dic=dic_hist_flights, labels=[f"Historic Trajectory for \n {ac['acId']}" for ac in aircraft_list], colors=["#E42320", "#6A8EC9", "#B46DA9"], plot_trajectory_endpoints=False,
                            tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y),  preTracon_label="Pre-TRACON",
                            plot_waypoints=False, waypoints=None, waypoints_tolerance=None, plot_waypoint_tol_zone=False,
                            lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=True,
                            title="2D Historic Flight Trajectories")

    # 5.2 plot original & cleaned waypoints
    fuel_emission_analysis_plot.plot_2d_trajectories(dic=None, labels=None, colors=None, plot_trajectory_endpoints=False, 
                            tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y),  preTracon_label="Pre-TRACON",
                            plot_waypoints=True, waypoints=dic_waypoints, waypoints_tolerance=3000, plot_waypoint_tol_zone=False,
                            lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=False,
                            title="Optimal Waypoints")
    
    fuel_emission_analysis_plot.plot_2d_trajectories(dic=None, labels=None, colors=None, plot_trajectory_endpoints=False, 
                            tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y),  preTracon_label="Pre-TRACON",
                            plot_waypoints=True, waypoints=dic_waypoints_cleaned, waypoints_tolerance=3000, plot_waypoint_tol_zone=False,
                            lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=False,
                            title="Cleaned Optimal Waypoints")

    # 5.3 Plot 2D trajectory
    combined_dic = {}
    for acId, df_hist in dic_hist_flights_preTracon.items():
        combined_dic[f"{acId}_historic"] = df_hist
    for acId, df_opt in dic_opt_flights_preTracon.items():
        combined_dic[f"{acId}_optimized"] = df_opt
    #print(combined_dic)
    fuel_emission_analysis_plot.plot_2d_trajectories(
        dic=combined_dic, labels=[f"Historic {ac['acId']}" for ac in aircraft_list] + [f"Optimized {ac['acId']}" for ac in aircraft_list],
        colors=["#E42320", "#6A8EC9", "#E42320", "#6A8EC9"], plot_trajectory_endpoints=False,
        tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), tracon_label="TRACON", preTracon_circle=(preTRACON_circle_x, preTRACON_circle_y), preTracon_label="Pre-TRACON",
        plot_waypoints=True, waypoints=dic_waypoints_cleaned, waypoints_tolerance=3000, plot_waypoint_tol_zone=False,
        lat0=dtw_lat, lon0=dtw_lon, plot_lat_lon_grid=False,
        title="2D Historic and Optimized Flight Trajectories")
        #["#E42320", "#6A8EC9","#B46DA9", "#E42320", "#6A8EC9","#B46DA9"]
    
    # 5.4 Plot 3D Trajectory
    combined_dic = {}
    for acId, df_hist in dic_hist_flights_preTracon.items():
        combined_dic[f"{acId}_historic"] = df_hist
    for acId, df_opt in dic_opt_flights_preTracon.items():
        combined_dic[f"{acId}_optimized"] = df_opt
    fuel_emission_analysis_plot.plot_3d_trajectories(dic=combined_dic, labels=[f"Historic {ac['acId']}" for ac in aircraft_list] + [f"Optimized {ac['acId']}" for ac in aircraft_list], colors=["#E42320", "#6A8EC9", "#E42320", "#6A8EC9"],
                            waypoints=dic_waypoints_cleaned,
                            lat0=dtw_lat, lon0=dtw_lon,
                            title="3D Historic & Optimized Routes")

    # 5.5 Plot fuel flow/usage and NOx emissions
    for idx, ac in enumerate(aircraft_list):
        acId = ac["acId"]

        df_hist = dic_hist_flights_preTracon[acId]       # historic trajectory in pre-TRACON
        df_opt = dic_opt_flights_preTracon[acId]         # optimized trajectory in pre-TRACON

        # Choose colors per aircraft
        color_list = ["#E42320", "#6A8EC9", "#E42320", "#6A8EC9"] #["#E42320", "#6A8EC9","#B46DA9", "#E42320", "#6A8EC9","#B46DA9"]
        color = color_list[idx % len(color_list)]

        # i) Fuel Flow and Usage
        fuel_emission_analysis_plot.plot_fuel_flow_and_usage(
            df1=df_hist, df2=df_opt,
            color1=color, color2=color,
            linestyle1='--', linestyle2='-',
            label1=f"{acId} Historic Route",
            label2=f"{acId} Optimized Route",
            title=f"{acId} Fuel Flow and Usage"
        )

        # ii) NOx Flow and Emission
        fuel_emission_analysis_plot.plot_NOx_flow_and_emission(
            df1=df_hist, df2=df_opt,
            color1=color, color2=color,
            linestyle1='--', linestyle2='-',
            label1=f"{acId} Historic Route",
            label2=f"{acId} Optimized Route",
            title=f"{acId} NOx Flow and Emission"
        )


    end_time = time.time()  # Record end time
    elapsed = end_time - start_time
    print(f"\nPlotting completed, total execution time: {elapsed:.2f} seconds")

    plt.show()
    
    # Return results for further analysis
    return {
        'historic': dic_hist_flights_preTracon,
        'optimized': dic_opt_flights_preTracon,
        'waypoints': dic_waypoints_cleaned
    }

if __name__ == "__main__":
    # Example: Run with CSV file (backward compatibility)
    print("Loading optimized waypoints from CSV file...")
    df_wide = pd.read_csv("FAST_MILP_Evaluation/data/solution24.csv")
    
    aircraft_list = [
        {"acId": "DAL1066_KTPAtoKDTW", "acType": "B737"},
        {"acId": "DAL498_KRSWtoKDTW", "acType": "B737"}
    ]
    
    results = analyze_optimized_trajectory(df_wide, aircraft_list)
    print("\nAnalysis complete!")