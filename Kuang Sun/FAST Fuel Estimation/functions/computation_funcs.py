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
    
    else: # from x,  to lat, lon
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

def waypoint_cleaning(origin, middle_waypoints, destination, threshold_deg=10): # reduce number of waypoints by detecting and picking out turning points
    """
    Identifies turning points along a sequence of waypoints based on direction changes.

    Parameters:
        origin (tuple): The starting point (lat, lon).
        waypoints (list): List of intermediate waypoints [(lat, lon), ...].
        destination (tuple): The end point (lat, lon).
        threshold_deg (float): Minimum angle (in degrees) to be considered a turning point.

    Returns:
        middle_waypoints_cleaned (list): Waypoints at turning points (i.e., significant direction changes).
    """
    def angle_between(v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot)
        return np.degrees(angle)

    # Add origin and destination to full path
    original_waypoints = [origin] + middle_waypoints + [destination]

    turning_points = []

    for i in range(1, len(original_waypoints) - 1):
        p_prev = np.array(original_waypoints[i - 1])
        p_curr = np.array(original_waypoints[i])
        p_next = np.array(original_waypoints[i + 1])
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        angle = angle_between(v1, v2)
        
        if angle > threshold_deg:
            turning_points.append((i, angle))  # i corresponds to index in original_waypoints

    # Convert index back to index in original waypoints list
    middle_waypoints_cleaned = [middle_waypoints[i - 1] for i, _ in turning_points]

    return middle_waypoints_cleaned