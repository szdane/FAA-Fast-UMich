import pandas as pd

# Paths
WAYPOINTS_CSV = "res/weathertrial.csv"
REGIONS_CSV   = "infeasible_regions.csv"

# Column names (edit if yours differ)
LAT_COL = "f1_lat"
LON_COL = "f1_lon"

MIN_LAT = "min_lat"
MAX_LAT = "max_lat"
MIN_LON = "min_lon"
MAX_LON = "max_lon"

def points_in_any_region(waypoints_csv: str, regions_csv: str):
    wp = pd.read_csv(waypoints_csv)
    rg = pd.read_csv(regions_csv)

    # # Ensure numeric
    # wp[LAT_COL] = pd.to_numeric(wp[LAT_COL], errors="coerce")
    # wp[LON_COL] = pd.to_numeric(wp[LON_COL], errors="coerce")
    # for c in [MIN_LAT, MAX_LAT, MIN_LON, MAX_LON]:
    #     rg[c] = pd.to_numeric(rg[c], errors="coerce")

    violating_points = []

    # Loop over points (simple + readable)
    for i, row in wp.iterrows():
        lat = row[LAT_COL]
        lon = row[LON_COL]

        hit = rg[
            (rg[MIN_LAT] <= lat) & (lat <= rg[MAX_LAT]) &
            (rg[MIN_LON] <= lon) & (lon <= rg[MAX_LON])
        ]

        if not hit.empty:
            violating_points.append({
                "point_index": i,
                "lat": lat,
                "lon": lon,
                "hit_regions_count": len(hit),
                # keep which rectangles matched (row indices in regions CSV)
                "region_rows": hit.index.tolist(),
            })

    ok = (len(violating_points) == 0)
    return ok, violating_points


if __name__ == "__main__":
    ok, violations = points_in_any_region(WAYPOINTS_CSV, REGIONS_CSV)

    if ok:
        print("All trajectory points are outside infeasible regions.")
    else:
        print(f"{len(violations)} trajectory point(s) fall inside infeasible regions:")
        for v in violations[:50]:
            print(f"  - point {v['point_index']} at ({v['lat']}, {v['lon']}) "
                  f"hits {v['hit_regions_count']} region(s) rows={v['region_rows']}")
