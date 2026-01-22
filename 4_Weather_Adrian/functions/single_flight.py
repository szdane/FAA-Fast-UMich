# input data processing package
import pandas as pd
from datetime import timedelta
import time
import numpy as np
from openap.extra.aero import fpm, ft, kts

# input custom helper functions
from functions import computation, plot
from functions.trajectory import Cruise_with_Multi_Waypoints

# input plotting functions
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def single_flight_optimization(acId="FFT3987_KMCOtoKDTW", acType="B737", dic_by_acid= None, waypoint_csv=None,
                               dtw_lat=42.2125, dtw_lon=-83.3534, preTRACON_radius=3, preTRACON_circle_xy=None, TRACON_polygon=None):
    ############
    # 1. Input #
    ############
    # 1.1 Assumed initial mass factor
    m0 = 0.8

    # 1.2 Input Yuwei's csv and find selected flights (coord1, coord2 = lat, lon)
    df_hist = dic_by_acid[acId]
    x_col, y_col = computation.proj_with_defined_origin(
        df_hist["coord1"], df_hist["coord2"], dtw_lat, dtw_lon
    )
    df_hist['x'] = x_col
    df_hist['y'] = y_col
    df_hist['z'] = df_hist["alt"] * ft

    """
    Dictionary dic_hist includes small DataFrames of the following format:
    HISTORIC FLIGHT DATAFRAME
    recTime | acID | coord1(Lat) | coord2(Lon) | alt(ft) | groundSpeed(kts) | rateOfClimb(ft/min) | d_ts(s) | x(m) | y(m) | z(m) |
    """

    # 1.3 Serra's Optimized Waypoints, which includes entering and exiting location for pre-TRACON
    df_Serra_optWaypoints = pd.read_csv(waypoint_csv)  # Load the flight data into pandaframe df
    df_wp = df_Serra_optWaypoints[["t", "f_lat", "f_lon", "f_alt_ft"]].copy()
    df_wp.columns = ["t", "lat", "lon", "alt_ft"]  # rename columns

    df_wp = pd.DataFrame(df_wp) # save waypoints in dataframe format

    """
    Dictionary dic_wp includes small DataFrames of the following format:
    INPUT WAYPOINTS DATAFRAME
    t (s) | lat | lon | alt_ft |
    """

    ############################################
    # 2. Compute Pre-TRACON Entry & Exit Point #
    ############################################
    # 2.1 Find Trajectory Intersection
    # Pre-TRACON entry intersection
    entry_xy,  entry_idx = computation.find_trajectory_intersection(df_hist, preTRACON_circle_xy)
    entry_lat, entry_lon = computation.proj_with_defined_origin(entry_xy[0], entry_xy[1], dtw_lat, dtw_lon, inverse=True)
    entry_row = computation.interpolate_row_from_xy(df_hist, entry_xy[0], entry_xy[1])

    # Pre-TRACON exit intersection
    exit_xy, exit_idx = computation.find_trajectory_intersection(df_hist, TRACON_polygon)
    exit_lat, exit_lon = computation.proj_with_defined_origin(exit_xy[0], exit_xy[1], dtw_lat, dtw_lon, inverse=True)
    exit_row = computation.interpolate_row_from_xy(df_hist, exit_xy[0], exit_xy[1])

    # Extract the segment of trajectory within pre-TRACON
    df_hist_preTracon = df_hist[
        (df_hist["recTime"] >= entry_row["recTime"]) &
        (df_hist["recTime"] <= exit_row["recTime"])
    ]

    # Add the interpolated entry/exit rows
    df_hist_preTracon = pd.concat(
        [pd.DataFrame([entry_row]), df_hist_preTracon, pd.DataFrame([exit_row])]
    ).sort_values("recTime").reset_index(drop=True)

    ##########################################
    # 3. Historic Trajectory Fuel Estimation #
    ##########################################
    # 3.1 Fuel Computation (ref: https://openap.dev/fuel_emission.html)
    # Compute fuel and emissions for this segment and over the previous df in the dictionary
    df_hist_preTracon = computation.compute_fuel_and_emission(df_hist_preTracon, acType, m0)
    """
    Dictionary dic_hist_flights_preTracon includes small DataFrames of the following format:
    HISTORIC FLIGHT DATAFRAME (within pre-TRACON region)
    recTime | acID | coord1(Lat) | coord2(Lon) | alt(ft) | groundSpeed(kts) | rateOfClimb(ft/min) | d_ts(s) | x(m) | y(m) | z(m) |
        fuel_flow(kg/s) | fuel_used(kg) | CO2_flow | Soot_flow | SOx_flow | NOx_flow | CO_flow | HC_flow |
        CO2_emitted | H2O_emitted | Soot_emitted | SOx_emitted | NOx_emitted | CO_emitted | HC_emitted
    """

    ###########################################
    # 4. Optimized Trajectory Fuel Estimation #
    ###########################################
    # 4.1 Initialization
    waypoint_proximity = 3000  # radius in meters
    
    # 4.2 Extract origin and destination, and intermediate waypoints from Serra's waypoint list
    origin = tuple(df_wp.iloc[0][["lat", "lon", "alt_ft", "t"]])
    destination = tuple(df_wp.iloc[-1][["lat", "lon", "alt_ft", "t"]])
    middle_waypoints = df_wp.iloc[1:-1][["lat", "lon", "alt_ft", "t"]].values.tolist()

    # 4.4 Interpolate trajectory
    cruise_flight = Cruise_with_Multi_Waypoints(
        acType, origin, destination, m0=m0
    ).trajectory(
        middle_waypoints=middle_waypoints,
        middle_radius=waypoint_proximity,
        middle_alt_margin=600 * ft,  # altitude tolerance
        forbidden_region=TRACON_polygon
    )

    # Compute recTime for optimized points
    enter_time = df_hist_preTracon.iloc[0]["recTime"] # Retrieve entry time
    recTime_opt = [enter_time + timedelta(seconds=dt) for dt in cruise_flight["ts"]]

    # Compute d_ts
    cruise_flight["d_ts"] = cruise_flight["ts"].diff().fillna(0)
    cruise_flight[0, "d_ts"] = cruise_flight["d_ts"].iloc[1]  # Fill first value with the first difference

    # Build DataFrame for optimized trajectory
    df_opt_preTRACON = pd.DataFrame({
        "recTime": recTime_opt,
        "acId": acId,
        "coord1": cruise_flight["latitude"],
        "coord2": cruise_flight["longitude"],
        "alt": cruise_flight["altitude"],
        "groundSpeed": cruise_flight["tas"],
        "rateOfClimb": cruise_flight["vertical_rate"],
        "d_ts": cruise_flight["d_ts"],
        "t": cruise_flight["ts"]
    })

    # Compute x,y,z
    x_opt, y_opt = computation.proj_with_defined_origin(
        df_opt_preTRACON["coord1"], df_opt_preTRACON["coord2"], dtw_lat, dtw_lon
    )
    df_opt_preTRACON["x"] = x_opt
    df_opt_preTRACON["y"] = y_opt
    df_opt_preTRACON["z"] = df_opt_preTRACON["alt"] * ft

    # 4.5 Estimate fuel and emissions
    df_opt_preTRACON = computation.compute_fuel_and_emission(df_opt_preTRACON, acType, m0)

    """
    Dictionary df_opt_preTRACON includes small DataFrames of the following format:
    OPTIMIZED FLIGHT DATAFRAME
    acID | coord1(Lat) | coord2(Lon) | alt(ft) | groundSpeed(kts) | rateOfClimb(ft/min) | recTime | t(s) | d_ts | x(m) | y(m) | z(m) |
        fuel_flow(kg/s) | fuel_used(kg) | CO2_flow | Soot_flow | SOx_flow | NOx_flow | CO_flow | HC_flow |
        CO2_emitted | H2O_emitted | Soot_emitted | SOx_emitted | NOx_emitted | CO_emitted | HC_emitted
    """

    return df_hist, df_wp, df_hist_preTracon, df_opt_preTRACON