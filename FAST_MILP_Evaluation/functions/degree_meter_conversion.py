import numpy as np

def bearing_func(lat1, lon1, lat2, lon2):
    """Compute the bearing between two (or two series) of coordinates.

    Args:
        lat1 (float or ndarray): Starting latitude (in degrees).
        lon1 (float or ndarray): Starting longitude (in degrees).
        lat2 (float or ndarray): Ending latitude (in degrees).
        lon2 (float or ndarray): Ending longitude (in degrees).

    Returns:
        float or ndarray: Bearing (in degrees). Between 0 and 360.

    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    return bearing

def distance(lat1, lon1, lat2, lon2, h=0):
    """Compute distance between two (or two series) of coordinates using
    Haversine formula.

    Args:
        lat1 (float or ndarray): Starting latitude (in degrees).
        lon1 (float or ndarray): Starting longitude (in degrees).
        lat2 (float or ndarray): Ending latitude (in degrees).
        lon2 (float or ndarray): Ending longitude (in degrees).
        h (float or ndarray): Altitude (in meters). Defaults to 0.

    Returns:
        float or ndarray: Distance (in meters).

    """
    r_earth = 6371000.0  # m, average earth radius

    # convert decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = c * (r_earth + h)  # meters, radius of earth
    return dist

def latlon(lat1, lon1, d, brg, h=0):
    """Get lat/lon given current point, distance and bearing.

    Args:
        lat1 (float or ndarray): Starting latitude (in degrees).
        lon1 (float or ndarray): Starting longitude (in degrees).
        d (float or ndarray): distance from point 1 (meters)
        brg (float or ndarray): bearing at point 1 (in degrees)
        h (float or ndarray): Altitude (in meters). Defaults to 0.

    Returns:
        lat2: Point latitude.
        lon2: Point longitude

    """
    r_earth = 6371000.0  # m, average earth radius
    
    # convert decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    brg = np.radians(brg)

    # haversine formula
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d / (r_earth + h))
        + np.cos(lat1) * np.sin(d / (r_earth + h)) * np.cos(brg)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(d / (r_earth + h)) * np.cos(lat1),
        np.cos(d / (r_earth + h)) - np.sin(lat1) * np.sin(lat2),
    )
    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)
    return lat2, lon2

def proj_with_defined_origin(lat, lon, lat0, lon0, inverse=False): #lon0, lat0 are the coordinates of the selected origin
    if not inverse: # from lat, lon to x, y
        bearings = bearing_func(lat0, lon0, lat, lon) / 180 * 3.14159 #Lin/site-packages/openap/extra/aero
        distances = distance(lat0, lon0, lat, lon)
        x = distances * np.sin(bearings)
        y = distances * np.cos(bearings)

        return x, y
    
    else: # from x, to lat, lon
        x, y = lon, lat
        distances = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(x, y) * 180 / 3.14159
        lat, lon = latlon(lat0, lon0, distances, bearing)

        return lat, lon

if __name__ == "__main__":
    # Origin (Duderstadt Center)
    origin_lat = 42.29112977072469
    origin_lon = -83.71573402535259

    # Test point (FXB Building)
    lat = 42.29357213486808
    lon = -83.71201342666508

    # Compute XY projection (meters)
    x, y = proj_with_defined_origin(lat, lon, origin_lat, origin_lon)

    # Compute great-circle distance (meters)
    dist = distance(origin_lat, origin_lon, lat, lon)

    print(f"Origin (deg):     lat={origin_lat:.6f}, lon={origin_lon:.6f}")
    print(f"Test point (deg): lat={lat:.6f}, lon={lon:.6f}")
    print(f"Projected XY (m): x={x:.3f}, y={y:.3f}")
    print(f"Distance (m):     {dist:.3f}")