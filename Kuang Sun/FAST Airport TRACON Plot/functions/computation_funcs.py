import openap

import numpy as np
import pandas as pd

from shapely.geometry import Point, MultiPoint

def get_convex_hull_star_fixes(df_cleaned):
    # Step 1: Extract (lon, lat) points from the DataFrame
    points = [Point(lon, lat) for lat, lon in zip(df_cleaned["Star Fix Lat"], df_cleaned["Star Fix Lon"])]
    names = df_cleaned["Star Fix Name"].tolist()

    # Step 2: Compute convex hull
    multi = MultiPoint(points)
    hull = multi.convex_hull

    # Step 3: Match convex hull coordinates back to STAR fix names
    hull_coords = list(hull.exterior.coords)
    outer_names = []
    outer_lats = []
    outer_lons = []

    for lon, lat in hull_coords:
        for i, (lat0, lon0) in enumerate(zip(df_cleaned["Star Fix Lat"], df_cleaned["Star Fix Lon"])):
            if abs(lat - lat0) < 1e-6 and abs(lon - lon0) < 1e-6:
                outer_names.append(names[i])
                outer_lats.append(lat0)
                outer_lons.append(lon0)
                break

    # Step 5: Return as DataFrame
    df_outer_star_fixes = pd.DataFrame({
        "Outer Star Fix": outer_names,
        "Lat": outer_lats,
        "Lon": outer_lons
    })

    return df_outer_star_fixes

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