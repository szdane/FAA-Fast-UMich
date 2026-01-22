import openap
from openap import prop

import numpy as np
import pandas as pd

from datetime import timedelta

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

def proj_with_defined_origin(lat, lon, lat0, lon0, inverse=False): #lon0, lat0 are the coordinates of the selected origin
    if not inverse: # from lat, lon to x, y
        bearings = openap.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159 #Lin/site-packages/openap/extra/aero
        distances = openap.aero.distance(lat0, lon0, lat, lon)
        x = distances * np.sin(bearings)
        y = distances * np.cos(bearings)

        return x, y
    
    else: # from x, to lat, lon
        x, y = lon, lat
        distances = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(x, y) * 180 / 3.14159
        lat, lon = openap.aero.latlon(lat0, lon0, distances, bearing)

        return lat, lon
    

def find_trajectory_intersection(df_traj, shape: BaseGeometry):
    """
    General function to find the first intersection between a trajectory and a Shapely shape.

    Parameters:
    - df_traj: Pandas DataFrame with 'x' and 'y' columns (in meters)
    - shape: A Shapely geometry object (e.g. Polygon or LineString)

    Returns:
    - intersection_point: tuple (x, y) if intersection found, else None
    - segment_index: index i such that the intersection occurs between points i and i+1
    """
    boundary = shape.boundary if hasattr(shape, "boundary") else shape  # handle Polygon or LineString

    for i in range(len(df_traj) - 1):
        p1 = (df_traj.iloc[i]["x"], df_traj.iloc[i]["y"])
        p2 = (df_traj.iloc[i + 1]["x"], df_traj.iloc[i + 1]["y"])
        segment = LineString([p1, p2])
        
        intersection = segment.intersection(boundary)

        if not intersection.is_empty:
            if intersection.geom_type == "Point":
                return (intersection.x, intersection.y), i
            elif intersection.geom_type == "MultiPoint":
                # Return the closest point to p1
                points = sorted(intersection.geoms, key=lambda pt: Point(p1).distance(pt))
                return (points[0].x, points[0].y), i

    return None, None

def interpolate_row_from_xy(df, x_target, y_target):
    """
    Interpolate a trajectory row given a coordinate (x, y).
    
    Parameters:
    - df: trajectory DataFrame with columns ['x', 'y', 'recTime', 'alt', 'groundSpeed', 'rateOfClimb', 'coord1', 'coord2']
    - x_target, y_target: coordinates in meters
    
    Returns:
    - interpolated_row: pd.Series representing the estimated state at (x, y)
    """
    point = Point(x_target, y_target)

    for i in range(len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        p1 = Point(row1["x"], row1["y"])
        p2 = Point(row2["x"], row2["y"])
        segment = LineString([p1, p2])

        if segment.distance(point) < 1e-6 and segment.length > 0:  # point is on segment
            dist_total = p1.distance(p2)
            dist_to_p1 = p1.distance(point)
            ratio = dist_to_p1 / dist_total

            interpolated = row1.copy()
            for col in ["alt", "groundSpeed", "rateOfClimb", "coord1", "coord2", "x", "y"]: #coord1 = lat,coord2 = lon
                interpolated[col] = row1[col] + ratio * (row2[col] - row1[col])

            # Interpolate time
            t1 = pd.to_datetime(row1["recTime"])
            t2 = pd.to_datetime(row2["recTime"])
            interpolated_time = t1 + (t2 - t1) * ratio
            interpolated["recTime"] = interpolated_time.round("s")

            # Keep acId and any other relevant metadata
            interpolated["acId"] = row1["acId"]
            return interpolated

    raise ValueError("Target point does not lie on any segment of the trajectory.")



def compute_fuel_flow(df, target_acType, m0): # (ref: https://openap.dev/fuel_emission.html)
    """
    Computes fuel flow and fuel usage over time for a given flight.

    Parameters:
        df (pd.DataFrame): Flight data containing alt, groundSpeed, rateOfClimb, d_ts.
        fuelflow_model (openap.FuelFlow): Initialized OpenAP FuelFlow object.
        initial_mass (float): Initial aircraft mass at the beginning of this segment (kg).

    Returns:
        df (pd.DataFrame): DataFrame with additional columns 'fuel_flow' and 'fuel'.
        fuel_used_till_step (list): Cumulative fuel used over time.
    """
    target_acData = prop.aircraft(target_acType) # Gather aircraft parameters from OpenAP
    mass_takeoff_assumed = m0 * target_acData['mtow'] #assume weight when entering the pre-TRACON region is m0 * MTOW when entering pre-TRACON, here m0 is a fraction factor
    initial_mass = mass_takeoff_assumed

    fuelflow = openap.FuelFlow(target_acType) #Fuel Flow class setup fuelflow is the model for computation
    fuelflow_model = fuelflow

    mass_current = initial_mass
    fuel_used_current = 0

    fuelflow_every_step = []
    #fuel_every_step = []
    fuel_used_till_step = []

    for _, row in df.iterrows():
        ff = fuelflow_model.enroute(
            mass=mass_current,
            tas=row.groundSpeed,
            alt=row.alt,
            vs=row.rateOfClimb,
        )
        fuel = ff * row.d_ts
        fuelflow_every_step.append(ff)
        #fuel_every_step.append(fuel)
        mass_current -= fuel
        fuel_used_current += fuel
        fuel_used_till_step.append(fuel_used_current)

    df = df.copy()  # to avoid modifying input DataFrame directly
    df["fuel_flow"] = fuelflow_every_step
    #df["fuel_remaining"] = fuel_every_step
    df["fuel_used"] = fuel_used_till_step

    return df

def identify_destination(df_target_optWaypoints):
    """
    Identify the destination from Serra's waypoint list.
    Specifically, return the first row of the final repeated (lat, lon, alt) sequence.
    """
    waypoints_subset = df_target_optWaypoints[["lat", "lon", "alt"]]
    
    # Reverse the DataFrame to find the *last* group of identical rows
    reversed_subset = waypoints_subset[::-1]
    
    # Get the first (lat, lon, alt) triplet from the end
    final_row_values = tuple(reversed_subset.iloc[0])
    
    # Now search for the *first* index in the original DataFrame matching this final row
    mask = (waypoints_subset == final_row_values).all(axis=1)
    
    # Find the first index where this repeated pattern starts
    first_repeat_idx = mask.idxmax()

    # Extract the corresponding full row (with time t)
    destination_row = df_target_optWaypoints.iloc[first_repeat_idx]

    return tuple(destination_row[["lat", "lon", "alt", "t"]])

def waypoint_cleaning(origin,
                      middle_waypoints,
                      destination,
                      lat0=None,
                      lon0=None,
                      threshold_deg=None):
    """
    Keeps turning points (angle > threshold_deg) while ensuring no more than
    three consecutive middle-waypoints are removed.

    Returns
    -------
    list
        Cleaned middle-waypoints.
    """

    def angle_between(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return np.nan
        return np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1)))

    original = [origin] + middle_waypoints + [destination]
    m = len(middle_waypoints)

    # ------------- 1.  Decide which mid-points meet the angle test --------------
    keep = [False] * m                                              # index 0 → first middle-waypoint
    for k in range(1, len(original) - 1):                           # k is index in *original*
        lat1, lon1, alt1, _ = original[k - 1]
        lat2, lon2, alt2, _ = original[k]
        lat3, lon3, alt3, _ = original[k + 1]

        x1, y1 = proj_with_defined_origin(lat1, lon1, lat0, lon0)
        x2, y2 = proj_with_defined_origin(lat2, lon2, lat0, lon0)
        x3, y3 = proj_with_defined_origin(lat3, lon3, lat0, lon0)

        p1 = np.array([x1, y1, alt1 * 0.3048])
        p2 = np.array([x2, y2, alt2 * 0.3048])
        p3 = np.array([x3, y3, alt3 * 0.3048])

        if angle_between(p2 - p1, p3 - p2) > threshold_deg:
            keep[k - 1] = True                                      # (k-1) maps to middle list

    # ------------- 2.  Enforce “no 4 in a row” removal rule ---------------------
    consecutive_removed = 0
    for i in range(m):
        if keep[i]:
            consecutive_removed = 0
        else:
            # candidate for removal
            if consecutive_removed == 4:        # last 3 were already removed ###########################
                keep[i] = True                  # KEEP this one
                consecutive_removed = 0         # reset counter
            else:
                consecutive_removed += 1

    # ------------- 3.  Build cleaned list ---------------------------------------
    middle_waypoints_cleaned = [
        wp for wp, flag in zip(middle_waypoints, keep) if flag
    ]

    return middle_waypoints_cleaned

def compute_fuel_and_emission(df, target_acType, m0): # (ref: https://openap.dev/fuel_emission.html)
    """
    Computes fuel flow and fuel usage over time for a given flight.

    Parameters:
        df (pd.DataFrame): Flight data containing alt, groundSpeed, rateOfClimb, d_ts.
        fuelflow_model (openap.FuelFlow): Initialized OpenAP FuelFlow object.
        initial_mass (float): Initial aircraft mass at the beginning of this segment (kg).

    Returns:
        df (pd.DataFrame): DataFrame with additional columns 'fuel_flow' and 'fuel'.
        fuel_used_till_step (list): Cumulative fuel used over time.
    """
    target_acData = prop.aircraft(target_acType) # Gather aircraft parameters from OpenAP
    mass_takeoff_assumed = m0 * target_acData['mtow'] #assume weight when entering the pre-TRACON region is m0 * MTOW when entering pre-TRACON, here m0 is a fraction factor
    initial_mass = mass_takeoff_assumed

    fuelflow = openap.FuelFlow(target_acType) #Fuel Flow class setup, fuelflow is the model for computation
    fuelflow_model = fuelflow

    mass_current = initial_mass
    fuel_used_current = 0
    CO2_emitted_till_current = 0
    H2O_emitted_till_current = 0
    Soot_emitted_till_current = 0
    SOx_emitted_till_current = 0
    NOx_emitted_till_current = 0
    CO_emitted_till_current = 0
    HC_emitted_till_current = 0

    fuelflow_every_step = []
    CO2_every_step = [] # g/s
    H2O_every_step = [] # g/s
    Soot_every_step = [] # g/s
    SOx_every_step = [] # g/s
    NOx_every_step = [] # g/s
    CO_every_step = [] # g/s
    HC_every_step = [] # g/s

    fuel_used_till_step = []
    CO2_emitted_till_step = []
    H2O_emitted_till_step = []
    Soot_emitted_till_step = []
    SOx_emitted_till_step = []
    NOx_emitted_till_step = []
    CO_emitted_till_step = []
    HC_emitted_till_step = []

    for _, row in df.iterrows():
        ff = fuelflow_model.enroute(
            mass=mass_current,
            tas=row.groundSpeed,
            alt=row.alt,
            vs=row.rateOfClimb,
        )
        fuel = ff * row.d_ts # kg
        fuelflow_every_step.append(ff) # kg/s
        mass_current -= fuel
        fuel_used_current += fuel # cumulative kg
        fuel_used_till_step.append(fuel_used_current) # cumulative kg

        emission = openap.Emission(target_acType) #Emission class setup, "emission" is the model for computation
        CO2 = emission.co2(ff)  # g/s
        H2O = emission.h2o(ff)  # g/s
        Soot = emission.soot(ff) # g/s
        SOx = emission.sox(ff) # g/s
        NOx = emission.nox(ff, tas=row.groundSpeed, alt=row.alt)  # g/s
        CO = emission.co(ff, tas=row.groundSpeed, alt=row.alt)  # g/s
        HC = emission.hc(ff, tas=row.groundSpeed, alt=row.alt)  # g/s

        # Append per-step values
        CO2_every_step.append(CO2) # g/s
        H2O_every_step.append(H2O) # g/s
        Soot_every_step.append(Soot) # g/s
        SOx_every_step.append(SOx) # g/s
        NOx_every_step.append(NOx) # g/s
        CO_every_step.append(CO) # g/s
        HC_every_step.append(HC) # g/s

        # Emissions per step
        CO2_step = CO2 * row.d_ts # g
        H2O_step = H2O * row.d_ts # g
        Soot_step = Soot * row.d_ts # g
        SOx_step = SOx * row.d_ts # g
        NOx_step = NOx * row.d_ts # g
        CO_step = CO * row.d_ts # g
        HC_step = HC * row.d_ts # g

        # Update cumulative values
        CO2_emitted_till_current += CO2_step # cumulative g
        H2O_emitted_till_current += H2O_step # cumulative g
        Soot_emitted_till_current += Soot_step # cumulative g
        SOx_emitted_till_current += SOx_step # cumulative g
        NOx_emitted_till_current += NOx_step # cumulative g
        CO_emitted_till_current += CO_step # cumulative g
        HC_emitted_till_current += HC_step # cumulative g

        # Append cumulative values
        CO2_emitted_till_step.append(CO2_emitted_till_current) # cumulative g
        H2O_emitted_till_step.append(H2O_emitted_till_current) # cumulative g
        Soot_emitted_till_step.append(Soot_emitted_till_current) # cumulative g
        SOx_emitted_till_step.append(SOx_emitted_till_current) # cumulative g
        NOx_emitted_till_step.append(NOx_emitted_till_current) # cumulative g
        CO_emitted_till_step.append(CO_emitted_till_current) # cumulative g
        HC_emitted_till_step.append(HC_emitted_till_current) # cumulative g

    df = df.copy()  # to avoid modifying input DataFrame directly
    df["fuel_flow"] = fuelflow_every_step
    df["fuel_used"] = fuel_used_till_step

    df["CO2_flow"] = CO2_every_step
    df["H2O_flow"] = H2O_every_step
    df["Soot_flow"] = Soot_every_step
    df["SOx_flow"] = SOx_every_step
    df["NOx_flow"] = NOx_every_step
    df["CO_flow"] = CO_every_step
    df["HC_flow"] = HC_every_step

    df["CO2_emitted"] = CO2_emitted_till_step
    df["H2O_emitted"] = H2O_emitted_till_step
    df["Soot_emitted"] = Soot_emitted_till_step
    df["SOx_emitted"] = SOx_emitted_till_step
    df["NOx_emitted"] = NOx_emitted_till_step
    df["CO_emitted"] = CO_emitted_till_step
    df["HC_emitted"] = HC_emitted_till_step

    return df